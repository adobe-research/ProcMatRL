# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os, sys
import os.path as pth
import copy
import json
import re
import shutil
import math
import subprocess
import time
import traceback
from argparse import Namespace
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.distributed import get_rank, get_world_size
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from lpips import LPIPS

matplotlib.use('Agg')

from .model_def import CondMatFormer
from .model.clip_image_encoder import CLIPEncoder
from .simple_graph_dataset import SimpleGraphDataset, SimpleGraphIterableDataset, RealImageDataset
from .sequencer import NodeSequencer, SlotSequencer, EdgeSequencer, ParamSequencer
from .sequencer.sequences import convert_node_types, add_auxiliary_node_types, add_auxiliary_tokens, remove_auxiliary_tokens
from .simple_graph.convert_simple_graph_parameters import unconvert_simple_graph_parameters
from .simple_graph.convert_and_filter_simple_graph_parameters import unconvert_clamped_simple_graph_parameters
from .render_graph import render_graph_output
from .rl_dataset import RewardFunction
from .swd import swd
from .utils import load_node_types, collate_data_features, prepare_batch, gen_comp_grid

from .diffsbs.sbs_graph import SBSGraph
from .diffsbs.sbs_utils import read_image, write_image


# worker device for MATch v2 optimization
match_device = None


# compile model arguments
def get_model_args(args, validate=False):
    # set default arguments
    default_args = {
        'image_encoder_type': 'clip',
        'image_input': 'render',
        'image_ext': 'png',
        'pil_image_loader': False,
    }
    overwrite_args = {
        'semantic_validate': args.semantic_validate,
        'devices': args.devices,
        'use_alpha': args.use_alpha,
        'model_dir': args.model_dir,
        'use_fast_attn': getattr(args, 'use_fast_attn', False),
    }
    for k in ('temperature', 'prob_k', 'nucleus_top_p'):
        if getattr(args, k, None) is not None:
            overwrite_args[k] = getattr(args, k)

    # initialize model arguments
    node_args, edge_args, param_args = None, None, None

    if 'gen' in args.eval_modes:

        # extract node generator arguments
        node_args = Namespace(**default_args)
        with open(os.path.join(args.model_dir, args.node_exp_name, 'args.json')) as f:
            vars(node_args).update({
                **json.load(f),
                **overwrite_args,
                'exp_name': args.node_exp_name,
                'model_step': args.node_model_step,
                'max_gen_nodes': getattr(args, 'max_gen_nodes', None)
            })

        # extract edge generator arguments
        edge_args = Namespace(**default_args)
        with open(os.path.join(args.model_dir, args.edge_exp_name, 'args.json')) as f:
            vars(edge_args).update({
                **json.load(f),
                **overwrite_args,
                'exp_name': args.edge_exp_name,
                'model_step': args.edge_model_step
            })

    # extract parameter generator arguments
    param_args = Namespace(**default_args, full=False)
    with open(os.path.join(args.model_dir, args.param_exp_name, 'args.json')) as f:
        vars(param_args).update({
            **json.load(f),
            **overwrite_args,
            'exp_name': args.param_exp_name,  # overwrite for fast inference
            'model_step': args.param_model_step
        })

    # run diagnosis on the model arguments
    messages = []

    if validate and 'gen' in args.eval_modes:

        # check if the models are trained on the same dataset
        dataset_fields = [k for k in ('data_dir', 'data_chunksize', 'node_type_list', 'max_num_nodes', 'max_num_parents')
                            if not getattr(node_args, k) == getattr(edge_args, k) == getattr(param_args, k)]
        if dataset_fields:
            messages.append('Node, edge, and parameter generators are trained on different datasets. '
                            'Related fields: ' + ', '.join(dataset_fields))

        # check node orders
        no = (node_args.node_order, edge_args.node_order, param_args.node_order)

        if not (no[0] == no[1] == no[2]):
            if no[0] == no[1] == 'reverse_breadth_first_no_auxiliary_nodes' and no[2] == 'reverse_breadth_first_flipped_no_auxiliary_nodes':
                messages.append('Auxiliary Nodes: Node/Edge/Param=No/No/No, Reversed Param: Yes')
            # elif no[0] == 'reverse_breadth_first' and no[1] == no[2] == 'reverse_breadth_first_no_auxiliary_nodes':
            #     messages.append('Auxiliary Nodes: Node/Edge/Param=Yes/No/No, Reversed Param: No')
            # elif no[0] == 'reverse_breadth_first' and no[1] == 'reverse_breadth_first_no_auxiliary_nodes' and no[2] == 'reverse_breadth_first_flipped_no_auxiliary_nodes':
            #     messages.append('Auxiliary Nodes: Node/Edge/Param=Yes/No/No, Reversed Param: Yes')
            else:
                raise RuntimeError('Unsupported node order combinations: ' + ', '.join(no))

        # check legacy json loading
        if any(getattr(a, 'legacy_json_loading', False) for a in (node_args, edge_args, param_args)):
            raise RuntimeError('Legacy JSON loading is not allowed.')

    # assign node type lists
    if getattr(args, 'custom_node_type_list', None) is not None:
        for a in (node_args, edge_args, param_args):
            if a is not None:
                a.node_type_list = args.custom_node_type_list

    # display messages
    if messages:
        print('Warning: see the following diagnostic messages from validation')
        for msg in messages:
            print(f'  - {msg}')

    return {'args': args, 'node': node_args, 'edge': edge_args, 'param': param_args}


# prepare node types for node, edge, and parameter generators
def get_node_types(node_type_list, max_num_parents, node_order_for_nodes, node_order_for_edges, node_order_for_params):
    node_types = load_node_types(node_type_list)
    node_types = convert_node_types(node_types)
    node_types_wo_aux_nodes = copy.deepcopy(node_types)

    node_orders = node_order_for_nodes, node_order_for_edges, node_order_for_params
    node_types_w_aux_nodes = add_auxiliary_node_types(node_types, max_num_parents=max_num_parents, node_order=next(filter(None, node_orders)))

    node_types_per_generator = tuple(
        (node_types_wo_aux_nodes if no.endswith('no_auxiliary_nodes') else node_types_w_aux_nodes)
        if no is not None else None
        for no in node_orders
    )
    return node_types_per_generator


class GraphReformer:
    def __init__(self, node_order, node_types, max_num_nodes, out_dir=None, revert_to_v0=False):
        self.node_order = node_order
        self.node_types = node_types
        self.max_num_nodes = max_num_nodes
        self.out_dir = out_dir
        self.revert_to_v0 = revert_to_v0

    def __call__(self, ordered_nodes):
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(10000, recursion_limit))

        new_ordered_nodes = ordered_nodes.copy()
        need_auxiliary_nodes = not self.node_order.endswith('no_auxiliary_nodes')

        for k in range(len(ordered_nodes)):
            # print('Before: ', [str(node) for node in node_samples[k]])
            graph = copy.deepcopy(ordered_nodes[k]).to_graph()
            # remove all auxiliary nodes in case that they are not correctly connected
            if need_auxiliary_nodes:
                remove_auxiliary_tokens(graph)

            # delete unreachable nodes
            graph.prune(out_dir=self.out_dir)

            # add auxiliary nodes back
            if need_auxiliary_nodes:
                # node_type_names is useless if not child_end node exists; we revert to v0 because this is deprecated code
                add_auxiliary_tokens(graph=graph, node_order=self.node_order, node_types=self.node_types, node_type_names=None,
                                    revert_to_v0=self.revert_to_v0, sorted_by_name=True, fixed_output_order=True)
            new_on = SimpleGraphDataset.get_ordered_nodes(
                graph=graph, node_order=self.node_order, rng=None, sorted_by_name=True, fixed_output_order=True)

            # print('After: ', [str(node) for node in new_node_sample])
            if len(new_on) <= self.max_num_nodes:
                new_ordered_nodes[k] = new_on
            else:
                print('Warn: The number of nodes in the reformed graph is out of the predefined maximum number of nodes. '
                    'Use the original nodes instead.')

            sys.setrecursionlimit(recursion_limit)

        return new_ordered_nodes


class ModelBuilder:
    """Build a conditional MatFormer model for policy evaluation or value computation.
    """
    def __init__(self, model_args, device=None, train_from_scratch=False):
        args, node_args, edge_args, param_args = model_args['args'], model_args['node'], model_args['edge'], model_args['param'],
        device = device if device is not None else args.devices[0]

        # load node types
        node_types = {}
        node_types['node'], node_types['edge'], node_types['param'] = get_node_types(
            (node_args or param_args).node_type_list,
            (node_args or param_args).max_num_parents,
            *(getattr(a, 'node_order', None) for a in (node_args, edge_args, param_args))
        )

        # check all node types are under a predefined maximum number of input slots
        if node_types['node'] is not None:
            exclude_node_type_inds = []
            for node_type_idx, node_type in enumerate(node_types['node']):
                if len(node_type['input_names']) > edge_args.max_num_parents or len(node_type['output_names']) > edge_args.max_num_output_slots:
                    exclude_node_type_inds.append(node_type_idx)
            if exclude_node_type_inds:
                raise RuntimeError('The number of input slots of some node types are larger than predefined.')

        # create sequencers
        node_sequencer, slot_sequencer, edge_sequencer, param_node_sequencer, param_sequencer = [None] * 5
        generator_types = []

        if any(mode in args.eval_modes for mode in ['node_edge', 'gen', 'loss']):
            node_sequencer = NodeSequencer(max_num_nodes=node_args.max_num_nodes, node_types=node_types['node'], exclude_node_type_inds=exclude_node_type_inds,
                                           max_num_slots=edge_args.max_num_slots)
            slot_sequencer = SlotSequencer(max_num_nodes=edge_args.max_num_nodes, node_types=node_types['edge'], max_num_slots=edge_args.max_num_slots,
                                           max_num_output_slots=edge_args.max_num_output_slots, max_num_parents=edge_args.max_num_parents)
            edge_sequencer = EdgeSequencer(max_num_edges=edge_args.max_num_edges)
            generator_types.extend(['node', 'edge'])

        if any(mode in args.eval_modes for mode in ['param', 'gen', 'param_loss', 'loss']):
            param_node_sequencer = NodeSequencer(max_num_nodes=param_args.max_num_nodes, node_types=node_types['param'], use_start_token=False)
            param_sequencer = ParamSequencer(max_full_seq_len=param_args.max_full_seq_len,
                                             max_num_params=param_args.max_num_node_params, max_vec_dim=param_args.max_node_param_vec_dim,
                                             max_seq_len=param_args.max_node_param_seq_len, quant_steps=param_args.node_param_quant_steps,
                                             node_types=node_types['param'], use_alpha=args.use_alpha)
            generator_types.append('param')

        # create graph reformer
        graph_reformer = GraphReformer(node_order=param_args.node_order, node_types=node_types['param'], max_num_nodes=param_args.max_num_nodes)

        # create model in evaluation mode
        self.model_args = {k: model_args[k] for k in generator_types}
        self.sequencers = node_sequencer, slot_sequencer, edge_sequencer, param_node_sequencer, param_sequencer

        # save model arguments
        self.build_kwargs = {'graph_reformer': graph_reformer, 'device': device}

        # trainable flags
        self.trainable = {k: k in args.trainable for k in generator_types} if hasattr(args, 'trainable') else True

        # whether to train from scratch
        self.train_from_scratch = train_from_scratch

    def __call__(self, is_value_network=False, bypass_clip=False, requires_grad=True, distributed=False):
        # create model in evaluation mode
        kwargs = {
            'is_value_network': is_value_network,
            'bypass_clip': bypass_clip,
            'distributed': distributed,
            **self.build_kwargs
        }
        model = CondMatFormer(self.model_args, *self.sequencers, **kwargs)
        model.eval()

        # load model state
        if not self.train_from_scratch:
            device = kwargs['device']
            model.load_model_state(self.model_args, device=device, exclude_lm_head=is_value_network)

        # set model trainable flags
        model.set_trainable(requires_grad if requires_grad is not None else self.trainable)

        return model


def load_dataset(model_args, data_dir, data_list, dataset_features, batch_size,
                 node_sequencer, slot_sequencer, edge_sequencer, param_node_sequencer, param_sequencer,
                 real_data_list=None, real_data_ratio=1.0, is_map_dataset=None, shuffle=None, pre_shuffle=None, distributed=None,
                 rank=None, world_size=None):
    args, node_args, param_args = model_args['args'], model_args['node'], model_args['param']
    shuffle = shuffle if shuffle is not None else True
    distributed = distributed if distributed is not None else args.distributed
    dataset, real_dataset = None, None

    # prepare dataset options
    dataset_kwargs = {
        'node_sequencer': node_sequencer,
        'slot_sequencer': slot_sequencer,
        'edge_sequencer': edge_sequencer,
        'param_node_sequencer': param_node_sequencer,
        'param_sequencer': param_sequencer,
        'param_per_node': False,
        'max_num_param_sets': None,
        'node_order': (node_args or param_args).node_order,
        'param_node_order': param_args.node_order,
        'image_ext': getattr(args, 'image_ext', None) or (node_args or param_args).image_ext,
        **{name: getattr((node_args or param_args), name, None) for name in ('use_alpha', 'pil_image_loader', 'legacy_json_loading')},
        **{name: getattr(args, name, None) for name in ('seed', 'revert_to_v0', 'data_chunksize', 'graph_suffix')}
    }

    # create real image dataset
    if real_data_list is not None and real_data_ratio > 0:
        is_map_dataset = True

        real_dataset_kwargs = {
            **dataset_kwargs,
            'shuffle': pre_shuffle,
            'shuffle_mode': getattr(args, 'real_pre_shuffle_mode', None),
            'augment_image': getattr(args, 'real_augment_image', None),
            'augment_graph': getattr(args, 'real_augment_graph', None),
            'mp_rank': rank,
            'mp_world_size': world_size,
        }
        real_dataset = RealImageDataset(
            image_dir=args.real_image_dir, image_list=real_data_list, nn_file=args.real_nn_file,
            data_dir=data_dir, feature_list=dataset_features, **real_dataset_kwargs)

    # create regular dataset
    if real_data_list is None or real_data_ratio < 1:
        is_map_dataset = (is_map_dataset if is_map_dataset is not None
                          else not (data_dir.endswith('.h5') or pth.isfile(pth.join(data_dir, 'all_variations.json'))))

        dataset_kwargs.update({
            'shuffle': pre_shuffle if is_map_dataset else shuffle,
            'image_input': getattr(args, 'image_input', None) or (node_args or param_args).image_input,
            'mp_rank': get_rank() if distributed and not is_map_dataset else rank,
            'mp_world_size': get_world_size() if distributed and not is_map_dataset else world_size,
            **({
                'target_shuffle_queue_size': args.target_shuffle_queue_size,
                'batch_alignment': batch_size if distributed else None,
                'pre_shuffle': pre_shuffle
            } if not is_map_dataset else {})
        })
        Dataset = SimpleGraphDataset if is_map_dataset else SimpleGraphIterableDataset
        dataset = Dataset(data_dir=data_dir, data_list=data_list, feature_list=dataset_features, **dataset_kwargs)

    # combine real and synthetic datasets
    if real_dataset is not None and dataset is not None:
        num_syn = len(dataset)
        num_real = int(num_syn / (1 - real_data_ratio) + 0.5) - num_syn
        real_dataset.extend(num_real)
        dataset = ConcatDataset([real_dataset, dataset])
        print(f'Combined {num_real} real samples with {num_syn} synthetic samples.')
    else:
        dataset = dataset or real_dataset

    # create data loader
    data_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle if not distributed and is_map_dataset else None,
        'num_workers': args.num_workers,
        'collate_fn': collate_data_features,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0
    }
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=args.seed) if distributed and is_map_dataset else None
    dataloader = DataLoader(dataset, sampler=sampler, **data_loader_kwargs)

    return dataloader


def compute_loss(model, data, vis_per_token_loss=None, per_token_loss_dir=None, start_idx=0):
    seq_names_dict = {
        'node': ['type', 'depth'],
        'edge': [],
        'param': ['id', 'val']
    }

    # evaluate model
    model_outputs = model(data, return_logits=True)
    model_outputs = {k: v for k, v in model_outputs.items() if k in seq_names_dict}

    # helper function for calculating sequence loss
    loss_dict = {}
    loss_seq_dict = {k: {} for k in seq_names_dict}

    def compute_sequence_loss(gen_type):
        loss_seq, logits_seq = model_outputs.get(gen_type, (None, None))[:2]
        if loss_seq is None:
            raise RuntimeError(f'Loss of {gen_type} generator is not available.')

        seq_mask = data[f'{gen_type}_seq_mask'][:, 1:].bool()
        seq_names = seq_names_dict[gen_type]

        # losses and accuracies from multiple output sequences
        if seq_names:
            seq_mask = seq_mask[:, None]
            avg_loss = loss_seq[seq_mask.expand_as(loss_seq)].mean().item()
            losses = [avg_loss, *((loss_seq * seq_mask).sum(dim=(0, 2)) / seq_mask.count_nonzero(dim=(0, 2))).tolist()]

            acc_top1, acc_top5 = [], []
            for i_seq, name in enumerate(seq_names):
                labels = data[f'{gen_type}_{name}_seq'][:, 1:]
                logits = logits_seq[i_seq][:, :-1]
                act1 = (logits.argmax(dim=-1) == labels).float()
                act5 = (logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1).float()
                acc_top1.append(act1[seq_mask.squeeze(1)].mean().item())
                acc_top5.append(act5[seq_mask.squeeze(1)].mean().item())

        # loss and accuracy from single output sequence
        else:
            avg_loss = loss_seq[seq_mask].mean().item()
            losses = [avg_loss]

            labels = data[f'{gen_type}_seq'][:, 1:]
            logits = logits_seq[:, :-1]
            acc_top1 = (logits.argmax(dim=-1) == labels).float()
            acc_top1 = [acc_top1[seq_mask].mean().item()]
            acc_top5 = (logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1).float()
            acc_top5 = [acc_top5[seq_mask].mean().item()]

        # record loss-related info for per-token loss visualization
        if vis_per_token_loss is not None:
            if seq_names:
                for i_seq, name in enumerate(seq_names):
                    loss_seq_dict[gen_type][name] = (
                        loss_seq[:, i_seq], logits_seq[i_seq][:, :-1], seq_mask[:, 0],
                        data[f'{gen_type}_{name}_seq'][:, 1:]
                    )
            else:
                loss_seq_dict[gen_type][''] = (
                    loss_seq, logits_seq[:, :-1], seq_mask,
                    data[f'{gen_type}_seq'][:, 1:]
                )

        return {
            'loss': dict(zip(['tot', *seq_names], losses)),
            'acc_top1': dict(zip(seq_names or ['seq'], acc_top1)),
            'acc_top5': dict(zip(seq_names or ['seq'], acc_top5))
        }

    # compute sequence losses
    for gen_type in model_outputs:
        loss_dict[gen_type] = compute_sequence_loss(gen_type)

    # visualize per-token losses
    if vis_per_token_loss is not None:
        stage_name, seq_name = vis_per_token_loss
        loss_seq, logits_seq, seq_mask, labels = loss_seq_dict[stage_name][seq_name]
        sample_names = data['name']

        for i_sample, sample_name in enumerate(sample_names):
            sample_name = pth.basename(pth.dirname(sample_name)) + '_' + pth.basename(sample_name)[-5:]
            sample_file = pth.join(per_token_loss_dir, f'{start_idx + i_sample}_{sample_name}_{stage_name}_{seq_name}.png')
            vis_per_token_loss(loss_seq[i_sample], logits_seq[i_sample], seq_mask[i_sample], labels[i_sample], sample_file)

    return loss_dict


# visualize per-token loss
def vis_per_token_loss(loss_seq, logits_seq, seq_mask, labels, output_filename, max_seq_len=500, max_ranks=20):
    # create a matplotlib figure with two subplots stacked vertically (one for loss, one for logits)
    seq_len = min(max_seq_len, seq_mask.count_nonzero().item())
    fig, (ax_loss, ax_logits) = plt.subplots(2, 1, figsize=(seq_len * 0.1, 5), height_ratios=[1, 4])

    # visualize loss using a bar plot
    ax_loss.bar(np.arange(0.5, seq_len + 0.5), loss_seq[:seq_len].numpy(), width=0.9, align='center', color='C0')
    ax_loss.set_xlim(0, seq_len)
    ax_loss.set_ylim(bottom=0)
    ax_loss.set_xlabel('token index')
    ax_loss.set_ylabel('loss')

    # sort logits by their values and clip to the max ranks
    logits_seq, logits_args = torch.sort(logits_seq[:seq_len], dim=-1, descending=True)
    logits_seq, logits_args = logits_seq[:, :max_ranks], logits_args[:, :max_ranks]

    # visualize logits using a regular mesh grid, where each column is a token and each row is a class, and the color is the logit value
    # the mesh plot is centered at the token index, and the mesh height is the number of classes
    mesh_x, mesh_y = np.meshgrid(np.arange(seq_len + 1), np.arange(max_ranks + 1))
    ax_logits.pcolormesh(mesh_x, mesh_y, logits_seq.numpy().T, cmap='Blues', vmin=0, vmax=1)
    ax_logits.set_xlim(0, seq_len)
    ax_logits.set_ylim(0, max_ranks)
    ax_logits.set_xlabel('token index')
    ax_logits.set_ylabel('ranked logits')
    ax_logits.set_yticks(np.arange(0, max_ranks + 2, 2))

    # draw scatter plot to show the ground truth labels
    match_label = logits_args.numpy() == labels[:seq_len, None].numpy()
    match_label, label_rank = np.any(match_label, axis=-1), np.argmax(match_label, axis=-1) + 0.5
    ax_logits.scatter(np.arange(0.5, seq_len + 0.5)[match_label], label_rank[match_label], marker='_', color='C1', s=20)

    # save the plot
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)


# calculate token sequence losses for generators
def eval_token_loss(model_args, model, sequencers, rank=None, world_size=None, queue=None):
    args, param_args = model_args['args'], model_args['param']
    device = args.devices[rank if rank is not None else 0]
    is_main_rank = rank is None or rank == 0

    # create auxiliary directories to save intermediate results
    suffix = args.eval_suffix or ''
    images_dir = os.path.join(args.result_dir, f'{args.exp_name}{suffix}', args.label)
    os.makedirs(images_dir, exist_ok=True)

    if args.vis_per_token_loss:
        per_token_loss_dir = os.path.join(images_dir, 'per_token_loss')
        os.makedirs(per_token_loss_dir, exist_ok=True)
    else:
        per_token_loss_dir = None

    # collect dataset features
    dataset_features = ['prerendered']

    # parameter loss mode requires ground-truth tokenized sequences
    dataset_features += ['param_node_type_seq', 'param_node_depth_seq', 'param_node_seq_mask',
                         'param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq', 'param_array_elm_idx_seq',
                         'param_idx_seq', 'param_seq_mask', 'param_node_inds']

    # fetch ground-truth node-edge index when applying GCN
    if param_args.edge_condition == 'node_edge_gnn':
        dataset_features += ['edge_node_inds']

    # full loss evaluation modes also requires other ground-truth tokenized sequences
    if 'loss' in args.eval_modes:
        dataset_features += ['node_type_seq', 'node_idx_seq', 'node_depth_seq', 'node_seq_mask']
        dataset_features += ['slot_node_type_seq', 'slot_node_idx_seq', 'slot_node_depth_seq', 'slot_id_seq', 'slot_idx_seq', 'slot_seq_mask',
                             'edge_seq', 'edge_idx_seq', 'edge_elm_seq', 'edge_seq_mask']

    print('dataset features: ', dataset_features)

    # load dataset (do not use DDP)
    dataloader = load_dataset(
        model_args, args.custom_data_dir, args.test_dataset, dataset_features, args.batch_size, *sequencers,
        shuffle=False, distributed=False, rank=rank, world_size=world_size)

    # start main loop
    loss_stats = {}
    batch_size = args.batch_size

    for i_batch, data in enumerate(tqdm(dataloader, desc='Token loss', disable=not is_main_rank)):
        data = prepare_batch(param_args, data, dataset_features, device)
        batch_loss_dict = compute_loss(
            model, data, vis_per_token_loss=args.vis_per_token_loss, per_token_loss_dir=per_token_loss_dir,
            start_idx=i_batch * batch_size)

        # accumulate loss stats
        for gen_type in batch_loss_dict:
            if gen_type not in loss_stats:
                loss_stats[gen_type] = {k: {d_k: [] for d_k in d} for k, d in batch_loss_dict[gen_type].items()}
            for k, d in batch_loss_dict[gen_type].items():
                for d_k, d_v in d.items():
                    loss_stats[gen_type][k][d_k].append(d_v)

    # compute average loss
    for gen_type, stats in loss_stats.items():
        for k, d in stats.items():
            stats[k] = {d_k: np.mean(d_v) for d_k, d_v in d.items()}

    if not is_main_rank:
        # send loss stats to the main rank
        queue.put((rank, loss_stats))

        print('Finished evaluating token losses.')
        return

    # collect loss stats from other ranks (if applicable)
    if world_size is not None and world_size > 1:
        for gen_type, stats in loss_stats.items():
            for k, d in stats.items():
                stats[k] = {d_k: [d_v] for d_k, d_v in d.items()}

        for _ in range(1, world_size):
            i_rank, rank_loss_stats = queue.get()
            for gen_type, stats in rank_loss_stats.items():
                for k, d in stats.items():
                    for d_k, d_v in d.items():
                        loss_stats[gen_type][k][d_k].append(d_v)
            print(f'Collected loss stats from rank {i_rank}.')

        for gen_type, stats in loss_stats.items():
            for k, d in stats.items():
                stats[k] = {d_k: np.mean(d_v) for d_k, d_v in d.items()}

    # save loss stats
    with open(os.path.join(images_dir, 'token_loss_stats.json'), 'w') as f:
        json.dump(loss_stats, f, indent=4)

    # print loss stats
    print('Token loss stats:')
    for gen_type, stats in loss_stats.items():
        print(f" - Avg. loss of {gen_type} generator: "
              f"{', '.join(f'{name} = {loss:.6f}' for name, loss in stats['loss'].items())}")
        print(f" - Avg. top-1 accuracy of {gen_type} generator: "
              f"{', '.join(f'{name} = {accu:.6f}' for name, accu in stats['acc_top1'].items())}")
        print(f" - Avg. top-5 accuracy of {gen_type} generator: "
              f"{', '.join(f'{name} = {accu:.6f}' for name, accu in stats['acc_top5'].items())}")

    print('Finished evaluating token losses.')


# generate procedural material graphs from input images
def generate_graphs(model_args, model, sequencers, rank=None, world_size=None, queue=None):
    args, param_args = model_args['args'], model_args['param']
    device = args.devices[rank if rank is not None else 0]
    is_main_rank = rank is None or rank == 0

    # check distributed mode with fixed budget
    if args.distributed and args.num_gen_samples > 0:
        raise RuntimeError('Distributed mode is not supported when generating a fixed number of samples.')

    # create auxiliary directories to save intermediate results
    suffix = args.eval_suffix or ''
    images_dir = os.path.join(args.result_dir, f'{args.exp_name}{suffix}', args.label)
    os.makedirs(images_dir, exist_ok=True)

    # load node types
    node_types_unconv = load_node_types(param_args.node_type_list)
    node_types_conv = convert_node_types(node_types_unconv)

    # collect dataset features
    dataset_features = ['index', 'name', 'prerendered']

    # parameter only mode requires ground-truth uninitialized nodes
    if 'param' in args.eval_modes and 'gen' not in args.eval_modes:
        dataset_features += ['nodes', 'node_depths']

    print('dataset features: ', dataset_features)

    # load dataset (do not use DDP)
    has_real_data = getattr(args, 'real_image_dir', None) is not None
    dataloader = load_dataset(
        model_args, args.custom_data_dir, args.test_dataset, dataset_features, args.batch_size, *sequencers,
        shuffle=False, pre_shuffle=args.pre_shuffle, real_data_list=args.real_test_dataset if has_real_data else None,
        is_map_dataset=True, distributed=False, rank=rank, world_size=world_size)

    # define sample methods (random or deterministic)
    sample_rules = [True] if args.deterministic else []
    sample_rules += [False] * args.k_subsamples
    num_samples_per_graph = len(sample_rules)

    # determine batch size and number of batches
    num_gen_samples = 0
    aug_factor = args.real_augment_graph if has_real_data and args.real_augment_graph > 0 else 1
    if args.num_gen_samples > 0:
        num_gen_samples = args.num_gen_samples * aug_factor

    if num_gen_samples > 0:
        batch_size = min(num_gen_samples, args.batch_size)
        n_batch = int(np.ceil(num_gen_samples / batch_size))
    else:
        batch_size = args.batch_size
        n_batch = len(dataloader)

    # start main loop
    nn_suffix_pattern = re.compile(r'_nn(\d+)$')
    all_image_names = []
    t_start = time.time()

    for i_batch, data in enumerate(dataloader):
        print(f'Generating samples of {i_batch + 1}/{n_batch} batch (time elapsed: {time.time() - t_start:.2f} s)')

        # truncate the last batch if necessary
        if num_gen_samples > 0 and i_batch == n_batch - 1 and num_gen_samples % batch_size:
            data = [v[:num_gen_samples % batch_size] for v in data]

        # save target images
        target_imgs = data[dataset_features.index('prerendered')].unbind(dim=0)

        # move batched data to GPU
        data = prepare_batch(param_args, data, dataset_features, device)

        # pre-calculate CLIP embeddings
        data['prerendered'] = model.clip_encoder(data['prerendered'])

        # generate multiple sub-samples
        generated_nodes = [[] for _ in range(len(data['name']))]

        for k, deterministic in enumerate(sample_rules):
            print(f'Generating {k + 1}/{len(sample_rules)} subsamples')

            # generate graph
            ordered_nodes = model.generate(data, bypass_clip=True, deterministic=deterministic)[0]

            for i, on in enumerate(ordered_nodes):
                generated_nodes[i].append(on)

        # save generated nodes as JSON graphs
        for ordered_nodes, img, name, idx in zip(generated_nodes, target_imgs, data['name'], data['index'].tolist()):
            # remove the nearest neighbor suffix
            nn_suffix = nn_suffix_pattern.search(name)
            nn_idx = int(nn_suffix.group(1)) if nn_suffix else 0
            name = nn_suffix_pattern.sub('', name)

            # convert name to folder name
            folder_name = name
            if pth.sep in folder_name:
                segs = folder_name.split(pth.sep)
                folder_name = '_'.join([segs[0], segs[-1]])

            # create a directory for each image
            sample_dir = os.path.join(images_dir, f'{idx}_{folder_name}')
            os.makedirs(sample_dir, exist_ok=True)

            # save the target image as a PNG file
            if not nn_idx:
                write_image(pth.join(sample_dir, 'target.jpg'), img, process=True)
                all_image_names.append((idx, name))

            # save the generated graphs as JSON files
            for k, on in enumerate(ordered_nodes):
                json_nodes = on.to_graph().save_json(node_types_conv, use_alpha=args.use_alpha)
                json_filename = pth.join(sample_dir, f'graph_{nn_idx * num_samples_per_graph + k:05d}_quantized.json')
                with open(json_filename, 'w') as f:
                    json.dump(json_nodes, f)

        # stop upon reaching the number of batches
        if num_gen_samples > 0 and i_batch == n_batch - 1:
            break

    print(f'Generate {n_batch} batches of samples in total (time elapsed: {time.time() - t_start:.2f} s)')

    if not is_main_rank:
        # send image names to the main rank
        queue.put((rank, all_image_names))

        print('Finished generating samples.')
        return

    # collect image names from other ranks (if applicable)
    if world_size is not None and world_size > 1:
        for _ in range(1, world_size):
            i_rank, rank_image_names = queue.get()
            all_image_names.extend(rank_image_names)
            print(f'Collected image names from rank {i_rank}.')

        all_image_names = sorted(all_image_names)

    # save all image names
    with open(pth.join(images_dir, 'src_graphs.txt'), 'w') as f:
        f.write('\n'.join(name for _, name in sorted(all_image_names, key=lambda x: x[0])))

    print('Finished generating samples.')


# prepare options for each step of material rendering
def get_render_kwargs(model_args):
    args, param_args = model_args['args'], model_args['param']

    # compile arguments for rendering
    resource_dirs = {'sbs': pth.join(args.sat_dir, 'resources', 'packages'), 'sbsrc': args.sbsrc_dir}
    convert_kwargs = {
        'node_types': load_node_types(param_args.node_type_list),
        'step_count': param_args.node_param_quant_steps,
        'clamped': param_args.node_param_quant_steps != 4096,
        'use_alpha': args.use_alpha
    }
    load_json_kwargs = {
        'resource_dirs': resource_dirs,
        'use_alpha': args.use_alpha,
        'res': [9, 9],
        'output_usages': None,
        'prune_inactive_nodes': False,
        'expand_unsupported_nodes': False,
        'expand_unsupported_fnodes': False,
        'allow_unsupported_nodes': True,
        'condition_active_node_params': False
    }
    render_worker_kwargs = {
        'sat_dir': args.sat_dir,
        'resource_dirs': resource_dirs,
        'output_usages': None,
        'randomize_generators': False,
        'generators_only': False,
        'image_format': 'jpg',
        'center_normals': True,
        'engine': 'sse2',
        'use_networkx': False,
        'write_output_channels': args.render_channels
    }

    return convert_kwargs, load_json_kwargs, render_worker_kwargs


# convert JSON nodes to SBS graphs
def convert_sbs(json_nodes_path, convert_kwargs, load_json_kwargs, save_sbs_path=None):
    # load JSON nodes
    with open(json_nodes_path) as f:
        json_nodes = json.load(f)

    # convert JSON nodes to a SBS graph
    convert_kwargs = convert_kwargs.copy()
    clamped = convert_kwargs.pop('clamped')
    if clamped:
        unconvert_clamped_simple_graph_parameters(json_nodes=json_nodes, **convert_kwargs)
    else:
        unconvert_simple_graph_parameters(json_nodes=json_nodes, **convert_kwargs)

    # read SBS graph
    graph_sbs = SBSGraph.load_json(graph_name=pth.basename(json_nodes_path), json_data=json_nodes, **load_json_kwargs)
    graph_sbs.force_directx_normals()
    graph_sbs.update_node_dtypes(harmonize_signatures=True)

    # save the SBS graph
    if save_sbs_path:
        graph_sbs.save_sbs(
            save_sbs_path,
            resolve_resource_dirs={'sbsrc': load_json_kwargs['resource_dirs']['sbsrc']},
            package_dependencies_dir=pth.basename(save_sbs_path).replace('.sbs', '_dependencies'),
            use_networkx=False)

    return graph_sbs


# render a JSON format graph
def render_json_graph(graph_filename, convert_kwargs, load_json_kwargs, render_worker_kwargs):
    try:
        # read JSON nodes and convert to SBS graph
        graph_sbs = convert_sbs(json_nodes_path=f'{graph_filename}_quantized.json', convert_kwargs=convert_kwargs,
                                load_json_kwargs=load_json_kwargs)

        # invoke SAT to render the graph (with time limit)
        channels = graph_sbs.run_sat(graph_filename=f'{graph_filename}.sbs', output_name=graph_filename,
                                     timeout=60, **render_worker_kwargs)
        output_image = render_graph_output(output_channels=channels, normal_format='dx')

        # save the rendered image
        write_image(f'{graph_filename}_rendered.jpg', output_image.squeeze(dim=0).cpu(), process=True)

    except RuntimeError as e:
        print(f'[WARNING] Rendering {graph_filename} failed (error message: {e.args[0]}).')
        return False, (graph_filename, e.args[0])

    return True, None


# render generated graphs into images
def render_graphs(model_args):
    args = model_args['args']

    # create output directory
    output_dir = os.path.join(args.result_dir, f"{args.exp_name}{args.eval_suffix or ''}", args.label)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, pth.join(output_dir, 'args.json'))

    # collect all graph folders
    folder_pattern = re.compile(r'^\d+_')
    all_folders = [fn for fn in os.listdir(output_dir)
                   if pth.isdir(pth.join(output_dir, fn)) and folder_pattern.match(fn)]

    # collect all sample names to render
    all_sample_names = []
    suffix_len = len('_quantized.json')

    for fn in sorted(all_folders, key=lambda x: int(x.split('_')[0])):
        output_subdir = pth.join(output_dir, fn)
        all_sample_names.extend(
            sorted([pth.join(fn, sub_fn[:-suffix_len]) for sub_fn in os.listdir(output_subdir)
                    if sub_fn.endswith('.json')]))

    # save all sample names to render
    with open(pth.join(output_dir, 'all_sample_names.txt'), 'w') as f:
        f.write('\n'.join(all_sample_names))
        print(f'Found {len(all_sample_names)} samples to render.')

    # compile arguments for rendering
    convert_kwargs, load_json_kwargs, render_worker_kwargs = get_render_kwargs(model_args)

    # render all samples
    render_func = partial(render_json_graph, convert_kwargs=convert_kwargs, load_json_kwargs=load_json_kwargs,
                          render_worker_kwargs=render_worker_kwargs)
    chunksize = int(math.ceil(len(all_sample_names) / args.num_processes))
    chunksize = min(max(chunksize, 1), 32)
    unrenderable_graphs = []

    with Pool(args.num_processes) as pool:
        worker_map = pool.imap_unordered(
            render_func, (pth.join(output_dir, fn) for fn in all_sample_names), chunksize=chunksize)
        for flag, ret_val in tqdm(worker_map, desc='Rendering', total=len(all_sample_names)):
            if not flag:
                unrenderable_graphs.append(ret_val)

    # save unrenderable graphs
    if unrenderable_graphs:
        with open(pth.join(output_dir, 'unrenderable_graphs.txt'), 'w') as f:
            f.write('\n'.join(f'{gn}: {msg}' for gn, msg in sorted(unrenderable_graphs, key=lambda x: x[0])))

    print('Finished rendering graphs.')


# simple image dataset class for retrieving rendered images
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, all_sample_names):
        super().__init__()
        self.image_dir = image_dir
        self.all_sample_names = all_sample_names
        self.white_image = torch.ones(3, 512, 512)

    def __len__(self):
        return len(self.all_sample_names)

    def __getitem__(self, index):
        sample_name = self.all_sample_names[index]
        target = read_image(pth.join(self.image_dir, sample_name.split(pth.sep)[0], 'target.jpg'), process=True)

        try:
            if any(sample_name.endswith(k) for k in ('.jpg', '.png')):
                pred = read_image(pth.join(self.image_dir, sample_name), process=True)
            else:
                pred = read_image(pth.join(self.image_dir, sample_name + '_rendered.jpg'), process=True)
        except Exception:
            pred = self.white_image

        return target, pred, index


# collect image metrics for rendered outputs
def collect_metrics(model_args, samples_file_name='all_sample_names.txt', metrics_dir_name='metrics'):
    args, param_args = model_args['args'], model_args['param']
    device = args.devices[0]

    # read all sample names
    eval_dir = os.path.join(args.result_dir, f"{args.exp_name}{args.eval_suffix or ''}", args.label)
    with open(pth.join(eval_dir, samples_file_name)) as f:
        all_sample_names = f.read().splitlines()

    # create image dataset and data loader
    dataset = SimpleImageDataset(eval_dir, all_sample_names)
    dataloader = DataLoader(dataset, batch_size=args.metric_batch_size, shuffle=False, num_workers=args.metric_num_workers)

    # initialize metrics dictionary
    metrics = {
        'vgg': np.full_like(all_sample_names, np.inf, dtype=np.float32),
        'ds_l1': np.full_like(all_sample_names, np.inf, dtype=np.float32),
        'ds_lab_l1': np.full_like(all_sample_names, np.inf, dtype=np.float32),
        'lpips': np.full_like(all_sample_names, np.inf, dtype=np.float32),
        'swd': np.full_like(all_sample_names, np.inf, dtype=np.float32),
        'clip': np.full_like(all_sample_names, np.inf, dtype=np.float32),
    }

    # create metric functions
    reward_fn = RewardFunction(
        vgg_coeff=1.0, vgg_td_level=args.rank_vgg_td_level, ds_l1_coeff=1.0, ds_lab_l1_coeff=1.0,
        lab_weights=args.rank_lab_weights, device=device)
    lpips_fn = LPIPS(net='vgg').requires_grad_(False).to(device)
    clip_fn = CLIPEncoder(param_args.clip_model, device=device)

    # start main loop
    img_size = reward_fn.img_size
    interp_kwargs = dict(mode='bicubic', antialias=True, align_corners=False)

    for targets, preds, indices in tqdm(dataloader, desc='Metrics'):
        targets = targets.to(device, non_blocking=True)
        preds = preds.to(device, non_blocking=True)

        # resize images if necessary
        if targets.shape[-2:] != (img_size, img_size):
            targets = F.interpolate(targets, size=(img_size, img_size), **interp_kwargs).clamp_(0.0, 1.0)
        if preds.shape[-2:] != (img_size, img_size):
            preds = F.interpolate(preds, size=(img_size, img_size), **interp_kwargs).clamp_(0.0, 1.0)

        # calculate metrics
        stats = reward_fn.get_image_reward(targets, preds, reduce=False)[1]
        stats['lpips'] = lpips_fn(targets * 2 - 1, preds * 2 - 1).flatten()
        stats['clip'] = F.cosine_similarity(clip_fn(targets), clip_fn(preds))
        stats['swd'] = swd(targets, preds)

        # update metrics
        for metric, values in metrics.items():
            src_values = stats[metric]
            values[indices] = (src_values.cpu().numpy() if isinstance(src_values, torch.Tensor)
                                else np.array(src_values, dtype=np.float32) if isinstance(src_values, list)
                                else src_values)

    # save metrics
    metrics_dir = pth.join(eval_dir, metrics_dir_name)
    os.makedirs(metrics_dir, exist_ok=True)

    for metric, values in metrics.items():
        np.save(pth.join(metrics_dir, f'{metric}.npy'), values)

    print('Finished collecting image metrics.')


def read_metrics(args, samples_file_name='all_sample_names.txt', metrics_dir_name='metrics', reshape_by_image=True):
    # read all sample names
    eval_dir = os.path.join(args.result_dir, f"{args.exp_name}{args.eval_suffix or ''}", args.label)
    with open(pth.join(eval_dir, samples_file_name)) as f:
        all_sample_names = f.read().splitlines()
        num_samples = len(all_sample_names)

    # load metrics
    metrics_dir = pth.join(eval_dir, metrics_dir_name)
    metrics = {k[:-4]: np.load(pth.join(metrics_dir, k)) for k in os.listdir(metrics_dir) if k.endswith('.npy')}

    # reshape the metrics into (num_images, num_samples_per_image)
    if reshape_by_image:
        # check if the number of samples match the given configuration
        all_image_names = sorted(set(fn.split(pth.sep)[0] for fn in all_sample_names),
                                 key=lambda x: int(x[:x.index('_')]))
        num_images = len(all_image_names)

        has_real_data = getattr(args, 'real_image_dir', None) is not None
        num_samples_per_graph = args.k_subsamples + (1 if args.deterministic else 0)
        num_graphs_per_image = args.real_augment_graph if has_real_data and args.real_augment_graph > 0 else 1
        num_samples_per_image = num_samples_per_graph * num_graphs_per_image

        if num_samples != num_images * num_samples_per_image:
            raise RuntimeError(f'Expected {num_images * num_samples_per_image} samples, but found {num_samples}.')
        if args.num_gen_samples > 0 and args.num_gen_samples != num_images:
            raise RuntimeError(f'Expected {args.num_gen_samples} generated images, but found {len(all_image_names)}.')

        # reshape metrics
        for k, v in metrics.items():
            metrics[k] = v.reshape(num_images, num_samples_per_image)

    else:
        all_image_names = all_sample_names

    return metrics, all_image_names


def rank_metrics(args, metrics, return_raw_scores=False, **kwargs):
    # apply custom arguments
    args = copy.deepcopy(args)
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            raise ValueError(f"Argument '{k}' is not recognized.")

    # check if the number of samples match the given configuration
    num_images = len(next(iter(metrics.values())))
    has_real_data = getattr(args, 'real_image_dir', None) is not None

    num_samples_per_graph = args.k_subsamples + (1 if args.deterministic else 0)
    num_graphs_per_image = args.real_augment_graph if has_real_data and args.real_augment_graph > 0 else 1
    num_samples_per_image = num_samples_per_graph * num_graphs_per_image

    for k, v in metrics.items():
        if v.shape != (num_images, num_samples_per_image):
            raise RuntimeError(f"Expected metrics '{k}' of shape {(num_images, num_samples_per_image)}, "
                               f"but found {tuple(v.shape)}.")

    # compute losses
    losses = np.zeros_like(next(iter(metrics.values())))
    losses += metrics['vgg'] * args.rank_vgg_coeff
    losses += metrics['ds_l1'] * args.rank_l1_coeff
    losses += metrics['ds_lab_l1'] * args.rank_lab_l1_coeff
    losses += metrics['lpips'] * args.rank_lpips_coeff
    lpips, clip = metrics['lpips'], metrics['clip']

    if 'swd' in metrics:
        losses += metrics['swd'] * args.rank_swd_coeff
        swd = metrics['swd']
    else:
        swd = np.zeros_like(losses)

    # truncate losses
    if args.output_num_cand_graphs is not None and args.output_num_cand_graphs < num_graphs_per_image:
        losses = losses[:, :args.output_num_cand_graphs * num_samples_per_graph]
        num_graphs_per_image = args.output_num_cand_graphs
    if args.output_num_cand_samples is not None and args.output_num_cand_samples < num_samples_per_graph:
        losses = losses.reshape(num_images, -1, num_samples_per_graph)[..., :args.output_num_cand_samples].reshape(num_images, -1)
        num_cand_samples = args.output_num_cand_samples
    else:
        num_cand_samples = num_samples_per_graph

    # convert losses to scores
    all_scores = 1 - losses

    # pick deterministic samples
    if args.output_deterministic:
        sample_inds = np.arange(num_graphs_per_image)[None] * num_samples_per_graph
        losses = losses.reshape(num_images, -1, num_cand_samples)[:, :, 0]

    # maximizing rewards (minimizing losses) across deterministic and random samples
    elif args.output_sample_reduction == 'max':
        sample_inds = np.argmin(losses.reshape(num_images, -1, num_cand_samples), axis=2)
        sample_inds += np.arange(num_graphs_per_image) * num_samples_per_graph
        losses = np.take_along_axis(losses, sample_inds, axis=1)

    # no reduction
    else:
        sample_inds = np.arange(num_graphs_per_image)[:, None] * num_samples_per_graph + np.arange(num_cand_samples)
        sample_inds = sample_inds.reshape(1, -1)

    # rank images
    if args.rank_output and losses.shape[1] > 1:
        top_k = min(args.rank_top_k, losses.shape[1]) if args.rank_top_k > 0 else None
        top_inds = np.argsort(losses, axis=1)[:, :top_k]
        losses = np.take_along_axis(losses, top_inds, axis=1)
        sample_inds = np.take_along_axis(sample_inds, top_inds, axis=1)

    # pick top-k samples and compute final scores
    lpips = np.take_along_axis(lpips, sample_inds, axis=1)
    clip = np.take_along_axis(clip, sample_inds, axis=1)
    swd = np.take_along_axis(swd, sample_inds, axis=1)
    scores = 1 - losses

    # compile quantitative results
    results = {}
    if args.rank_output and losses.shape[1] > 1:
        results['top-1'] = {'score': scores[:, 0].mean(), 'lpips': lpips[:, 0].mean(), 'clip': clip[:, 0].mean(), 'swd': swd[:, 0].mean()}
        top_k_tag = f'top-{top_k}' if top_k else 'all'
        results[top_k_tag] = {'score': scores.mean(), 'lpips': lpips.mean(), 'clip': clip.mean(), 'swd': swd.mean()}
    else:
        results['all'] = {'score': scores.mean(), 'lpips': lpips.mean(), 'clip': clip.mean(), 'swd': swd.mean()}

    # return results
    ret = results, scores, sample_inds
    if return_raw_scores:
        ret += (all_scores,)

    return ret


# initialize the worker process for MATch v2 optimization by setting the rank
def init_worker_device(device):
    global match_device
    match_device = device
    print(f'[PID {os.getpid()}] Initialized MATch v2 optimization worker with device: {device}. '
          f'Slacking off for 10 seconds...')

    # delay execution to allow for proper initialization in other workers
    time.sleep(10)


# run MATch v2 optimization for one sample
def match_worker(task_info, convert_kwargs, load_json_kwargs, reward_kwargs, sat_dir, timeout):
    # check if the device is set
    if match_device is None:
        raise RuntimeError(f'MATch v2 optimization worker (PID {os.getpid()}) is not properly initialized. '
                           f'Consider assigning device IDs using a job queue.')

    # set the device ID
    if match_device.startswith('cuda'):
        device_id = match_device.split(':')[-1] if ':' in match_device else '0'
    else:
        device_id = None

    # unpack the task information
    json_path, sbs_path, target_img_path = task_info

    # Initialize the return values
    success, opt_img_path, msg, tb = False, None, '', ''

    try:
        # convert JSON nodes to SBS graph
        convert_sbs(json_path, convert_kwargs, load_json_kwargs, save_sbs_path=sbs_path)

        # prepare command for running MATch v2 optimization
        match_script_file = pth.join(pth.dirname(pth.abspath(__file__)), 'diff_optim_matchv2.py')
        match_cmd = [
            'python', match_script_file, sbs_path, target_img_path,
            '--sat_dir', sat_dir,
            '--vgg_coeff', str(reward_kwargs['vgg']),
            '--vgg_td_level', str(reward_kwargs['vgg_td_level']),
            '--ds_l1_coeff', str(reward_kwargs['ds']),
            '--ds_lab_coeff', str(reward_kwargs['ds_lab']),
            '--lpips_coeff', str(reward_kwargs['lpips']),
        ]
        if device_id:
            match_cmd += ['--device_id', device_id]
        if timeout:
            match_cmd += ['--timeout', str(timeout)]

        # run MATch v2 optimization
        ret = subprocess.run(match_cmd, capture_output=True, text=True)

        # read error messages from the output
        if ret.returncode:
            stderr_lines = ret.stderr.strip().splitlines()
            error_start = stderr_lines.index('Error occurred during optimization:') + 1
            tb_start = stderr_lines.index('Traceback (most recent call last):')
            error_end = stderr_lines.index('End of error message.')

            msg = '\n'.join(stderr_lines[error_start:tb_start])
            tb = '\n'.join(stderr_lines[tb_start:error_end])

    except Exception as e:
        msg, tb = e.args[0], traceback.format_exc()

    # find the optimized image
    result_dir = f'{sbs_path[:-4]}_opt_v2'

    if pth.isdir(result_dir):
        for stage in range(3):
            img_path = pth.join(result_dir, f'stage_{stage}', 'render', 'optimized.jpg')
            if pth.isfile(img_path):
                opt_img_path = img_path
                success = True

    return success, opt_img_path, sbs_path, msg, tb


def run_match_optimization(model_args):
    args = model_args['args']

    # check match profile mode
    if args.match_profile_mode not in ('sample', 'graph'):
        raise ValueError(f'Invalid MATch v2 profile mode: {args.match_profile_mode}')

    # read top-ranking samples
    metrics, image_names = read_metrics(args)

    # run optimization for selected samples in each image, and collect optimized image names
    eval_dir = os.path.join(args.result_dir, f"{args.exp_name}{args.eval_suffix or ''}", args.label)
    optimized_image_names, failed_tasks = [], []
    num_images = len(image_names)

    if num_images:
        # collect optimization tasks for each image from the user-specified profiles
        opt_tasks = [set() for _ in range(num_images)]

        for profile in args.match_profiles:
            try:
                val1, val2 = list(map(int, profile.split('/')))
            except:
                raise ValueError(f'Invalid MATch v2 profile: {profile}')

            # rank samples
            if args.match_profile_mode == 'sample':
                _, _, sample_inds = rank_metrics(
                    args, metrics, return_raw_scores=False, output_num_cand_samples=val2,
                    rank_output=True, rank_top_k=val1)
            else:
                _, _, sample_inds = rank_metrics(
                    args, metrics, return_raw_scores=False, output_num_cand_graphs=val2,
                    rank_output=True, rank_top_k=val1)

            # update optimization tasks
            for i, inds in enumerate(sample_inds):
                opt_tasks[i].update(inds.tolist())

        # sort optimization tasks
        for i, tasks in enumerate(opt_tasks):
            opt_tasks[i] = sorted(tasks)

        # optimization task generator
        def task_generator():
            # iterate over each image
            for image_name, tasks in zip(image_names, opt_tasks):
                target_img_path = pth.join(eval_dir, image_name, 'target.jpg')

                # run optimization for each task
                for ind in tasks:
                    json_path = pth.join(eval_dir, image_name, f'graph_{ind:05d}_quantized.json')
                    sbs_path = pth.join(eval_dir, image_name, f'graph_{ind:05d}.sbs')

                    yield json_path, sbs_path, target_img_path

        # define MATch v2 optimization worker function
        convert_kwargs, load_json_kwargs, _ = get_render_kwargs(model_args)
        reward_kwargs = {
            'vgg': args.rank_vgg_coeff,
            'vgg_td_level': args.rank_vgg_td_level,
            'ds': args.rank_l1_coeff,
            'ds_lab': args.rank_lab_l1_coeff,
            'lpips': args.rank_lpips_coeff
        }
        worker_func = partial(match_worker, convert_kwargs=convert_kwargs, load_json_kwargs=load_json_kwargs,
                              sat_dir=args.sat_dir, reward_kwargs=reward_kwargs, timeout=args.match_timeout)

        # run optimization tasks in parallel
        if args.distributed:
            pool = Pool(len(args.devices))
            pool.map(init_worker_device, args.devices)
            worker_map = pool.imap_unordered(worker_func, task_generator())

        # sequential optimization
        else:
            global match_device
            match_device = args.devices[0]
            worker_map = map(worker_func, task_generator())
            pool = None

        # collect optimization results
        progress_bar = tqdm(desc='MATch v2', total=sum(len(t) for t in opt_tasks))

        for success, opt_img_path, sbs_path, msg, tb in worker_map:
            example_name = pth.relpath(pth.splitext(sbs_path)[0], eval_dir)

            # add optimized image to the list
            if success:
                optimized_image_names.append(pth.relpath(opt_img_path, eval_dir))
                if msg:
                    print(f'[WARNING] Optimization for {example_name} partially succeeded. '
                          f'Keeping results from the last optimization stage. Error message:')

            # record failed task
            else:
                failed_tasks.append(f"{example_name}: {' '.join(msg.strip().splitlines())}")
                print(f'[WARNING] Optimization for {example_name} failed. Error message:')

            # print traceback
            if tb:
                print(tb)

            progress_bar.update()

        progress_bar.close()

        # close the pool
        if pool:
            pool.close()
            pool.join()

    # sort optimized and failed tasks
    optimized_image_names = sorted(sorted(optimized_image_names), key=lambda x: int(x[:x.index('_')]))
    failed_tasks = sorted(sorted(failed_tasks), key=lambda x: int(x[:x.index('_')]))

    # save optimized and failed tasks to file
    print(f'Optimized {len(optimized_image_names) + len(failed_tasks)} images ({len(failed_tasks)} failed).')

    with open(pth.join(eval_dir, 'optimized_image_names.txt'), 'w') as f:
        f.write('\n'.join(optimized_image_names))
    if failed_tasks:
        with open(pth.join(eval_dir, 'failed_optimization.txt'), 'w') as f:
            f.write('\n'.join(failed_tasks))

    # collect metrics for optimized images
    collect_metrics(model_args, samples_file_name='optimized_image_names.txt', metrics_dir_name='optimized_metrics')

    print('Finished running optimization.')


def rank_match_metrics(args, opt_metrics, opt_image_names, metrics, all_image_names, return_raw_scores=False):
    # get parameters from MATch v2 profile
    try:
        profile = args.rank_match_profile
        rank_top_k = int(profile[:profile.find('/')])
        num_cand_samples = int(profile[profile.find('/') + 1:])
        if not 0 < rank_top_k <= num_cand_samples:
            raise ValueError
    except:
        raise ValueError(f'Invalid MATch v2 profile: {profile}')

    # read and rank source metrics
    _, _, sample_inds, all_scores = rank_metrics(
        args, metrics, return_raw_scores=True, output_deterministic=False,
        output_num_cand_samples=num_cand_samples, rank_output=True, rank_top_k=rank_top_k)

    # build a look-up dictionary for optimized samples
    opt_sample_dict = {}
    graph_name_pattern = re.compile(r'^graph_\d{5}')

    for i, opt_name in enumerate(opt_image_names):
        opt_name_segs = opt_name.split(pth.sep)
        sample_name = pth.join(opt_name_segs[0], graph_name_pattern.match(opt_name_segs[1]).group())
        opt_sample_dict[sample_name] = i

    # construct an index array for optimized metrics
    index_arr = np.zeros_like(sample_inds)

    for i, (src_name, inds) in enumerate(zip(all_image_names, sample_inds)):
        for j, ind in enumerate(inds):
            opt_name = pth.join(src_name, f'graph_{ind:05d}')
            index_arr[i, j] = opt_sample_dict.get(opt_name, -1)

    # reindex optimized metrics (fill with source metrics if not found)
    index_mask = index_arr >= 0
    opt_metrics = {k: np.where(index_mask, v[index_arr], np.take_along_axis(metrics[k], sample_inds, axis=1))
                   for k, v in opt_metrics.items()}

    print(f'Found {np.sum(index_mask)} optimized samples out of {index_mask.size}.')

    # rank optimized metrics
    results, opt_scores, opt_inds = rank_metrics(
        args, opt_metrics, return_raw_scores=False, k_subsamples=rank_top_k, deterministic=False, real_image_dir=None,
        output_num_cand_graphs=None, output_num_cand_samples=None, output_deterministic=False,
        output_sample_reduction='none')

    sample_inds = np.take_along_axis(sample_inds, opt_inds, axis=1)

    # helper function for looking up optimized image name
    def lookup_opt_image_name(image_name, ind):
        sample_name = pth.join(image_name, f'graph_{ind:05d}')
        if sample_name in opt_sample_dict:
            return opt_image_names[opt_sample_dict[sample_name]]
        return f'{sample_name}_rendered.jpg'

    ret = results, opt_scores, sample_inds, lookup_opt_image_name
    if return_raw_scores:
        ret += (all_scores,)

    return ret


def compile_results(model_args):
    args = model_args['args']

    # read source metrics
    metrics, all_image_names = read_metrics(args)

    # rank metrics with MATch optimization results
    if args.rank_match_profile is not None:
        opt_metrics, opt_image_names = read_metrics(
            args, samples_file_name='optimized_image_names.txt', metrics_dir_name='optimized_metrics',
            reshape_by_image=False)
        results, scores, sample_inds, lookup_opt_image_name, all_scores = rank_match_metrics(
            args, opt_metrics, opt_image_names, metrics, all_image_names, return_raw_scores=True)

    # rank metrics with prediction results
    else:
        results, scores, sample_inds, all_scores = rank_metrics(args, metrics, return_raw_scores=True)

    # print quantitative results
    print(f"Ranking results from '{args.exp_name}':")
    for k, v in results.items():
        print(f' - {k}: {", ".join(f"{name} = {value:.4f}" for name, value in v.items())}')

    # prepare additional output to generate
    if args.output_num_imgs > 0 or args.output_histogram:
        res_dir = pth.join(args.result_dir, args.output_folder_name)
        os.makedirs(res_dir, exist_ok=True)

    # render ranked output images
    if args.output_num_imgs > 0:
        eval_dir = os.path.join(args.result_dir, f"{args.exp_name}{args.eval_suffix or ''}", args.label)
        has_real_data = getattr(args, 'real_image_dir', None) is not None

        # read source image names
        try:
            with open(pth.join(eval_dir, 'src_graphs.txt')) as f:
                src_image_names = f.read().splitlines()
        except FileNotFoundError:
            prefix_pattern = re.compile(r'^\d+_')
            src_image_names = [prefix_pattern.sub('', fn) for fn in all_image_names]

        # generate comparison grids
        interp_kwargs = dict(mode='bicubic', antialias=True, align_corners=False)
        all_grids = []

        for folder_name, image_name, inds, vals in zip(all_image_names[:args.output_num_imgs], src_image_names, sample_inds, scores):
            # read target image
            if args.high_res_image_dir is not None:
                suffix = '' if has_real_data else '_rendered'
                target = read_image(pth.join(args.high_res_image_dir, f'{image_name}{suffix}.{args.image_ext}'), process=True)
            else:
                target = read_image(pth.join(eval_dir, folder_name, 'target.jpg'), process=True)
            target = F.interpolate(target[None], size=256, **interp_kwargs).squeeze(0).clamp_(0.0, 1.0)

            # read ranked rendered images
            all_imgs = []
            for i in inds:
                try:
                    if args.rank_match_profile is not None:
                        img = read_image(pth.join(eval_dir, lookup_opt_image_name(folder_name, i)), process=True)
                    else:
                        img = read_image(pth.join(eval_dir, folder_name, f'graph_{i:05d}_rendered.jpg'), process=True)
                    img = F.interpolate(img[None], size=256, **interp_kwargs).squeeze(0).clamp_(0.0, 1.0)
                except Exception:
                    img = torch.ones(3, 256, 256)
                all_imgs.append(img)

            # render comparison grid image
            grid_img = gen_comp_grid(target, *all_imgs, val_labels=[f'{v:.3f}' for v in vals])
            all_grids.append(grid_img)

        # save comparison grids
        num_grids_per_col = args.max_img_cols // (scores.shape[1] + 1)
        grid_img = make_grid(all_grids, nrow=num_grids_per_col, padding=4, pad_value=1.0)
        match_label = '_match' if args.rank_match_profile is not None else ''
        write_image(pth.join(res_dir, f'rank{match_label}_{args.exp_name}.jpg'), grid_img, process=True)

    # draw histogram of scores for the first 16 images
    if args.output_histogram:
        res_dir = pth.join(args.result_dir, args.output_folder_name)
        os.makedirs(res_dir, exist_ok=True)

        # number of examples to plot histogram for
        num_examples = min(16, len(all_image_names))
        num_hist_cols = min(4, num_examples)
        num_hist_rows = math.ceil(num_examples / num_hist_cols)

        # histogram parameters
        hist_range, bin_size = (0.0, 1.0), 0.05
        num_bins = int((hist_range[1] - hist_range[0]) / bin_size + 1e-8)

        # plot histograms
        fig, axes = plt.subplots(num_hist_rows, num_hist_cols, figsize=(4 * num_hist_cols, 3 * num_hist_rows))
        axes = axes.flatten() if num_examples > 1 else [axes]

        for i, (ax, img_scores) in enumerate(zip(axes, all_scores[:num_examples])):
            ax.hist(img_scores, bins=num_bins, range=hist_range, color='C0', alpha=0.75)
            ax.set_title(f'Avg: {img_scores.mean():.4g}    Max: {img_scores.max():.4g}', fontsize=12)
            ax.set_xlim(hist_range)
            ax.set_ylim(0, len(img_scores))
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')

        for i in range(num_examples, num_hist_rows * num_hist_cols):
            axes[i].axis('off')

        # save histogram plot
        fig.tight_layout()
        fig.savefig(pth.join(res_dir, f'hist_{args.exp_name}.png'))

    print('Finished compiling results.')

    return results
