# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import os.path as pth
import sys
import json
import argparse

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import get_context, spawn
from torch.distributed import init_process_group, destroy_process_group, reduce, barrier

from .model_def import AssembledParamGenerator
from .sequencer import NodeSequencer, ParamSequencer
from .sequencer.sequences import add_auxiliary_node_types, convert_node_types
from .simple_graph_dataset import SimpleGraphDataset, SimpleGraphIterableDataset
from .utils import (
    load_model_state, save_model_state, load_optim_state, save_optim_state, load_node_types, CosineWarmup,
    collate_data_features, prepare_batch, IterProfiler, tqdm_file
)


# default standard output
default_stdout = sys.stdout


def load_dataset(args, dataset_features, node_sequencer, param_sequencer):
    # prepare dataset options
    is_map_dataset = not (args.data_dir.endswith('.h5') or pth.isfile(pth.join(args.data_dir, 'all_variations.json')))
    dataset_kwargs = {
        'shuffle': not is_map_dataset,
        'node_sequencer': node_sequencer,
        'param_node_sequencer': node_sequencer,
        'param_sequencer': param_sequencer,
        'param_per_node': False,
        'max_num_param_sets': None,
        'mp_rank': args._rank if args.distributed and not is_map_dataset else None,
        'mp_world_size': args._world_size if args.distributed and not is_map_dataset else None,
        **{
            name: getattr(args, name) for name in
            ('use_alpha', 'seed', 'pil_image_loader', 'node_order', 'image_input', 'image_ext', 'data_chunksize')
        },
        **({
            'target_shuffle_queue_size': args.target_shuffle_queue_size,
            'batch_alignment': args.batch_size if args.distributed else None
        } if not is_map_dataset else {})
    }

    # create datasets
    Dataset = SimpleGraphDataset if is_map_dataset else SimpleGraphIterableDataset
    train_dataset = Dataset(data_dir=args.data_dir, data_list=args.train_dataset, feature_list=dataset_features, **dataset_kwargs)
    valdt_dataset = Dataset(data_dir=args.data_dir, data_list=args.val_dataset, feature_list=dataset_features, **dataset_kwargs)

    return train_dataset, valdt_dataset


# convert edge node index to tensor for DP
# return Tensor: [bz, max_edge_seq_len, 2]
def index2tensor(edge_node_inds):
    edge_lens = [len(edge_node_ind) for edge_node_ind in edge_node_inds]
    max_len = max(edge_lens)
    padded_edge_node_inds = [edge_node_ind + [(-10000, -10000)] * (max_len - edge_len) + [(edge_len, edge_len)] for edge_len, edge_node_ind in zip(edge_lens, edge_node_inds)]
    padded_edge_node_inds_tensor = torch.as_tensor(padded_edge_node_inds, dtype=torch.long)
    return padded_edge_node_inds_tensor


def train_param_generator(args):
    # set training device
    device = torch.device(args.devices[args._rank if args.distributed else 0])

    # enable TF32 precision optimization
    if args.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

    # create directories for saved models and logs
    os.makedirs(pth.join(args.model_dir, args.exp_name), exist_ok=True)
    os.makedirs(pth.join(args.log_dir, args.exp_name), exist_ok=True)

    # get node types
    node_types = load_node_types(args.node_type_list)
    node_types = convert_node_types(node_types)
    node_types = add_auxiliary_node_types(node_types, max_num_parents=args.max_num_parents, node_order=args.node_order)

    # create graph structure sequencers
    node_sequencer = NodeSequencer(
        max_num_nodes=args.max_num_nodes, node_types=node_types, use_start_token=False)
    param_sequencer = ParamSequencer(
        max_full_seq_len=args.max_full_seq_len,
        max_num_params=args.max_num_node_params, max_vec_dim=args.max_node_param_vec_dim,
        max_seq_len=args.max_node_param_seq_len, quant_steps=args.node_param_quant_steps,
        node_types=node_types, use_alpha=args.use_alpha)

    # load dataset
    dataset_features = [
        'param_node_type_seq', 'param_node_depth_seq', 'param_node_seq_mask',
        'param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq', 'param_array_elm_idx_seq',
        'param_idx_seq', 'param_seq_mask', 'param_node_inds']
    dataset_features += ['prerendered']

    if args.edge_condition == 'node_edge_gnn':
        dataset_features += ['edge_node_inds']
    elif args.edge_condition is None:
        pass
    else:
        raise RuntimeError(f'Unknown edge conditioning strategy: {args.edge_condition}')

    train_dataset, valdt_dataset = load_dataset(args, dataset_features, node_sequencer, param_sequencer)

    print(f"Size of Training/Validation set{f' [Rank {args._rank}, estimated]' if args.distributed else ''}: "
          f"{len(train_dataset)}, {len(valdt_dataset)}")

    # create data loaders
    is_map_dataset = isinstance(train_dataset, SimpleGraphDataset)
    if is_map_dataset and args.distributed:
        train_sampler = DistributedSampler(train_dataset, seed=args.seed)
        valdt_sampler = DistributedSampler(valdt_dataset, seed=args.seed)
    else:
        train_sampler, valdt_sampler = None, None

    data_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True if is_map_dataset and not args.distributed else None,
        'num_workers': args.num_workers,
        'collate_fn': collate_data_features,
        'pin_memory': True,
        'generator': torch.manual_seed(args.seed),
        'multiprocessing_context': get_context('spawn'),
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **data_loader_kwargs)
    valdt_loader = DataLoader(valdt_dataset, sampler=valdt_sampler, **data_loader_kwargs)

    # read a sample batch
    if args.load_sample_batch and not args.sample_text_prompts:
        sample_batch = next(iter(IterProfiler(train_loader, title='Example batch loading time')))
        sample_images = sample_batch[dataset_features.index('prerendered')]
        torchvision.utils.save_image(sample_images, pth.join(args.model_dir, args.exp_name, 'sample_input.png'))

    # create model
    model = AssembledParamGenerator(args, node_sequencer, param_sequencer, device)
    print(f'Model parameters: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}')
    print(f'Image Encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}')

    # support for multi-gpu training and TorchDynamo
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device]) if args.distributed else model

    # create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule_type == 'cosine_warmup':
        lr_func = CosineWarmup(args.lr_schedule_warmup, args.lr_schedule_annealing, min_ratio=args.lr_schedule_gamma)
        lr_scheduler = LambdaLR(optimizer, lr_func)
    elif args.lr_schedule_type == 'step':
        lr_scheduler = StepLR(optimizer, step_size=args.lr_schedule_step_size, gamma=args.lr_schedule_gamma)
    else:
        raise ValueError(f'Unsupported learning rate scheduler type: {args.lr_schedule_type}')

    # create tensorboard writer
    is_main_rank = not args._rank
    writer = SummaryWriter(log_dir=pth.join(args.log_dir, args.exp_name)) if is_main_rank else None

    # save training configuration
    with open(pth.join(args.model_dir, args.exp_name, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # load model and optimizer states from checkpoint
    start_epoch, global_step = -1, 0

    if args.load_from_epoch >= 0:
        load_from_dir = args.load_from_dir if args.load_from_dir is not None else pth.join(args.model_dir, args.exp_name)
        load_model_state(model=model, state_dict=pth.join(load_from_dir, f'{args.load_from_epoch}_model.pth'), devices=[device])
        start_epoch, global_step = load_optim_state(optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                    state_dict_filename=pth.join(load_from_dir, f'{args.load_from_epoch}_optim.pth'))
        print(f'Model was loaded from epoch {args.load_from_epoch}.')\

    # training loop
    print(f'Initial global step: {global_step}')

    tqdm_kwargs = {'write_to_file': args.tqdm_file, 'update_interval': args.tqdm_print_interval}
    best_loss = np.inf

    for epoch in range(start_epoch + 1, args.epochs):
        print(f'************* [{args.exp_name}] epoch {epoch} *************')

        train_data_iter = enumerate(tqdm_file(train_loader, **tqdm_kwargs) if is_main_rank else train_loader)
        step, train_batch = next(train_data_iter, (-1, None))

        # training/validation schedule loop
        while train_batch is not None:
            # train loop
            model.train()
            for _ in range(args.validation_interval):
                global_step += 1
                optimizer.zero_grad()

                train_batch = prepare_batch(args, train_batch, dataset_features, device)
                loss_seq, param_seq_mask = model(train_batch), train_batch['param_seq_mask'][:, None, 1:]
                loss = loss_seq[param_seq_mask.bool().expand_as(loss_seq)].mean() if args.ignore_padding else loss_seq.mean()

                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

                # update learning rate (cosine annealing with warmup)
                if args.lr_schedule_type == 'cosine_warmup':
                    lr_scheduler.step(global_step)

                # record training loss
                if step % args.record_interval == 0:
                    with torch.no_grad():
                        if args.ignore_padding:
                            loss_sums = (loss_seq * param_seq_mask).sum(dim=(0, 2))
                            loss_counts = param_seq_mask.expand_as(loss_seq).count_nonzero(dim=(0, 2))
                        else:
                            loss_sums = loss_seq.sum(dim=(0, 2))
                            loss_counts = torch.full(loss_sums.shape, param_seq_mask.numel(), device=device)

                    # reduce loss values across processes
                    if args.distributed:
                        barrier()
                        reduce(loss_sums, dst=0)
                        reduce(loss_counts, dst=0)

                    # compute and record loss values
                    if is_main_rank:
                        loss_sums, loss_counts = loss_sums.tolist(), loss_counts.tolist()
                        tot_loss = sum(loss_sums) / sum(loss_counts)
                        id_loss, val_loss = tuple(lv / lc for lv, lc in zip(loss_sums, loss_counts))
                        writer.add_scalars('loss/train', {'tot': loss.item(), 'id': id_loss, 'val': val_loss}, global_step=global_step)

                # fetch the next batch
                step, train_batch = next(train_data_iter, (-1, None))
                if train_batch is None:
                    break

            # validation loop
            model.eval()
            loss_sums = torch.zeros(2, device=device)
            loss_counts = torch.zeros(2, dtype=torch.long, device=device)

            valdt_data_iter = tqdm_file(valdt_loader, **tqdm_kwargs) if is_main_rank else valdt_loader

            with torch.no_grad():
                for valdt_batch in valdt_data_iter:
                    valdt_batch = prepare_batch(args, valdt_batch, dataset_features, device)
                    loss_seq, param_seq_mask = model(valdt_batch), valdt_batch['param_seq_mask'][:, None, 1:]

                    if args.ignore_padding:
                        loss_sums += (loss_seq * param_seq_mask).sum(dim=(0, 2))
                        loss_counts += param_seq_mask.count_nonzero(dim=(0, 2))
                    else:
                        loss_sums += loss.sum(dim=(0, 2))
                        loss_counts += param_seq_mask.numel()

                # reduce loss values across processes
                if args.distributed:
                    barrier()
                    reduce(loss_sums, dst=0)
                    reduce(loss_counts, dst=0)

                # record validation loss
                if is_main_rank:
                    loss_sums, loss_counts = loss_sums.tolist(), loss_counts.tolist()
                    tot_loss = sum(loss_sums) / sum(loss_counts)
                    id_loss, val_loss = tuple(lv / lc for lv, lc in zip(loss_sums, loss_counts))
                    writer.add_scalars('loss/val', {'tot': tot_loss, 'id': id_loss, 'val': val_loss}, global_step=global_step)

                    # save best model checkpoint
                    if tot_loss <= best_loss:
                        best_loss = tot_loss
                        save_model_state(model, pth.join(args.model_dir, args.exp_name, f'best_model.pth'))

        # save model checkpoint at the end of each epoch
        if ((epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs-1) and is_main_rank:
            save_model_state(model, pth.join(args.model_dir, args.exp_name, f'{epoch}_model.pth'))
            save_optim_state(epoch, global_step, optimizer, lr_scheduler, pth.join(args.model_dir, args.exp_name, f'{epoch}_optim.pth'))

        # update learning rate (step decay)
        if args.lr_schedule_type == 'step':
            lr_scheduler.step(epoch)

    # close tensorboard writer
    if writer is not None:
        writer.close()


def launch_distributed(rank, args):
    # initialize distributed training
    port_name = ''.join(chr(c) for c in [77, 65, 83, 84, 69, 82])
    os.environ[f'{port_name}_ADDR'] = 'localhost'
    os.environ[f'{port_name}_PORT'] = str(args.ddp_port)

    init_process_group(backend='nccl', rank=rank, world_size=args._world_size)

    # Disable printing from non-main ranks
    if rank:
        text_io = open(os.devnull, 'w')
        sys.stdout, sys.stderr = text_io, text_io

    # run training
    args._rank = rank
    train_param_generator(args)

    destroy_process_group()


def create_arg_parser():
    p = argparse.ArgumentParser(description='Train a node generator.')

    # general arguments
    p.add_argument('--config', default=None, type=str, help='Path to a config file.')
    p.add_argument('--exp_name', default=None, type=str, help='Name of the experiment.')
    p.add_argument('--data_dir', default=None, type=str, help='Path to training set.')
    p.add_argument('--train_dataset', default=None, type=str, help='Path to training set.')
    p.add_argument('--val_dataset', default=None, type=str, help='Path to training set.')
    p.add_argument('--node_type_list', default=None, type=str, help='Path to a list of different node types.')
    p.add_argument('--devices', default=None, type=str, nargs='+', help='Devices to run on.')
    p.add_argument('--model_dir', default=None, type=str, help='Path to models directory for all experiments.')
    p.add_argument('--log_dir', default=None, type=str, help='Path to logs directory for all experiments.')
    p.add_argument('--use_alpha', default=False, action='store_true', help='If using alpha channel.')
    p.add_argument('--seed', default=414646787, type=int, help='Seed for repeatability.')

    # sequence model arguments
    p.add_argument('--node_order', default='reverse_breadth_first', type=str, help='Node order to use for the node sequence.')
    p.add_argument('--max_num_nodes', default=300, type=int, help='Maximum total number of nodes in any graph.')
    p.add_argument('--max_num_parents', default=10, type=int, help='Maximum number of parent operations for any given operation.')
    p.add_argument('--max_num_node_params', default=128, type=int, help='Maximum number of parameters for any node.')
    p.add_argument('--max_node_param_vec_dim', default=8, type=int, help='Maximum vector dimension for vector-valued node parameters.')
    p.add_argument('--max_node_param_seq_len', default=512, type=int, help='Maximum sequence length for node parameters. Each scalar (element of a) parameter value is a sequence entry + 1 entry for the param_end token.')
    p.add_argument('--max_full_seq_len', default=1250, type=int, help='Maximum sequence length for the full parameters')
    p.add_argument('--node_param_quant_steps', default=32, type=int, help='Number of quantization steps for parameter values.')
    p.add_argument('--hidden_dim', metavar='SIZE', default=256, type=int, nargs='+', help='Dimension of the hidden feature vectors (node encoder, parameter generator).')
    p.add_argument('--num_heads', metavar='NUM', default=8, type=int, nargs='+', help='Number of attention heads (node encoder, parameter generator).')
    p.add_argument('--num_layers', metavar='NUM', default=4, type=int, nargs='+', help='Number of layers hint (node encoder, parameter generator).')
    p.add_argument('--edge_condition', default=None, type=str, choices=('node_edge_gnn'), help='Select strategy to conditioning on edges (set to None to not condition on edges).')
    p.add_argument('--num_gcn', default=2, type=int, help='Number of GCN layers if conditioning on edges.')
    p.add_argument('--cond_type', default='feed_forward', type=str, help='Input conditioning type (feed_forward|cross_attention|merged_cross_attention).')
    p.add_argument('--use_encoder_decoder', default=False, action='store_true', help='Add cross attention from encoder hidden states to sequence generators.')
    p.add_argument('--use_fast_attn', default=False, action='store_true', help='Use fast attention implementation.')
    p.add_argument('--allow_tf32', default=False, action='store_true', help='Allow TF32 precision optimization.')

    # image encoder arguments
    p.add_argument('--image_input', default='render', type=str, nargs='+', help='Type of input image')
    p.add_argument('--image_ext', default='png', type=str, help='Extension type of input image')
    p.add_argument('--pil_image_loader', default=False, type=bool, help='Load images by PIL')
    p.add_argument('--clip_model', default='ViT-B/32', type=str, help='CLIP model name')
    p.add_argument('--normalize_clip', default=False, type=bool, help='Normalize CLIP embedding')
    p.add_argument('--image_encoder_type', default='full', type=str, nargs='+', help='Type of Image Encoder')
    p.add_argument('--embed_type', default=None, type=str, help='Preprocessing of image/text embeddings (project|resize|none)')

    # training arguments
    p.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
    p.add_argument('--batch_size', default=4, type=int, help='Number of graphs in each batch.')
    p.add_argument('--epochs', default=1000, type=int, help='Number of epochs.')
    p.add_argument('--weight_decay', default=0, type=float, help='Weight decay.')
    p.add_argument('--record_interval', default=100, type=int, help='Number of steps between tensorboard records within an epoch.')
    p.add_argument('--validation_interval', default=5000, type=int, help='Number of steps between validations within an epoch.')
    p.add_argument('--checkpoint_interval', default=1, type=int, help='Number of epochs between checkpoints.')
    p.add_argument('--num_workers', default=10, type=int, help='Number of threads (workers) for data loading.')
    p.add_argument('--data_chunksize', default=64, type=int, help='Chunksize for coalesced access to HDF5 dataset.')
    p.add_argument('--target_shuffle_queue_size', default=512, type=int, help='Minimal shuffle queue size for iterable dataset loader workers')
    p.add_argument('--load_sample_batch', default=False, action='store_true', help='Load a sample batch for debugging.')
    p.add_argument('--load_from_epoch', default=-1, type=int, help='The epoch of model being loaded for restarting.')
    p.add_argument('--load_from_dir', default=None, type=str, help='The directory of model being loaded for restarting.')

    # multi-gpu arguments
    p.add_argument('--distributed', default=False, type=bool, help='Enable data parallel.')
    p.add_argument('--ignore_padding', default=False, type=bool, help='ignore Padding tokens when computing losses.')
    p.add_argument('--ddp_port', default=12355, type=int, help='Port for distributed training.')

    # learning rate scheduler
    p.add_argument('--lr_schedule_type', default='step', choices=['step', 'cosine_warmp'], help='Learning rate schedule type (step|cosine_warmup)')
    p.add_argument('--lr_schedule_step_size', default=100, type=int, help='Step size for the "step" scheduler')
    p.add_argument('--lr_schedule_warmup', default=0, type=int, help='Number of warmup steps for the "cosine_warmup" scheduler')
    p.add_argument('--lr_schedule_annealing', default=1000, type=int, help='Number of annealing steps for the "cosine_warmup" scheduler')
    p.add_argument('--lr_schedule_gamma', default=0.1, type=float, help='Gamma for both schedulers')
    p.add_argument('--grad_clip_norm', default=1.0, type=float, help='Gradient norm clipping (0 means disabled)')

    # tqdm file output
    p.add_argument('--tqdm_file', action='store_true', help='Change tqdm print to be more friendly to file logging')
    p.add_argument('--tqdm_print_interval', default=1000, type=int, help='Interval between consecutive tqdm prints in file mode')
    return p


def default_args():
    return create_arg_parser().parse_args(args=[])


if __name__ == '__main__':

    parser = create_arg_parser()

    a = parser.parse_args()
    assert a.config is not None

    if a.config is not None:
        config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        config_path = a.config
        json_config = json.load(open(config_path))

        if any(json_name not in vars(a).keys() for json_name in json_config.keys()):
            unrecognized_args = [json_name for json_name in json_config.keys() if json_name not in vars(a).keys()]
            raise RuntimeError(f'Unrecognized argument(s) in config file: {unrecognized_args}')

        config_parser.set_defaults(**json_config)
        a = config_parser.parse_args()

        config_name = pth.splitext(pth.basename(config_path))[0]
        if not config_name.startswith('train_') or config_name[len('train_'):] != a.exp_name:
            print('************************************************************')
            print('* WARNING: experiment name does not match config filename. *')
            print('************************************************************')

    for k, v in a.__dict__.items():
        print(f'{k}: {v}')
    print('-' * 50)

    # reserved fields for distributed training
    a._world_size = len(a.devices) if a.distributed else None
    a._rank = 0 if a.distributed else None

    # launch on multiple GPUs
    if a.distributed:
        spawn(launch_distributed, args=(a,), nprocs=a._world_size, join=True)

    # launch on a single GPU
    else:
        train_param_generator(args=a)
