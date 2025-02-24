# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from argparse import Namespace
import os
import os.path as pth
import re

from .model.gpt2_config import GPT2Config
from .model.gpt2_image_conditioned import ImageCondEncoder, ImageCondPointerNet, ImageCondGraphConditionalParamEncoder
from .model.clip_image_encoder import CLIPEncoder, CLIPImageEncoder, CLIPImageMultiModalEncoder, get_clip_dim
from .simple_graph import SimpleOrderedNodes
from .utils import load_model_state, unwrap_ddp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn


def get_cond_size(args):
    clip_dim = get_clip_dim(args.clip_model)
    if args.embed_type is None:
        cond_size = 1, clip_dim
    else:
        hidden_dim = args.hidden_dim[0] if isinstance(args.hidden_dim, (tuple, list)) else args.hidden_dim
        cond_size = clip_dim // hidden_dim, hidden_dim
    return cond_size


def split_args(args, num=2):
    # split the arguments associated with the fields below
    fields = ['hidden_dim', 'num_layers', 'num_heads']
    return_args = [Namespace() for _ in range(num)]

    # compatibility with previous versions
    if all(hasattr(args, k) for k in ('node_num_layers', 'edge_num_layers')):
        args.num_layers = (args.node_num_layers, args.edge_num_layers)
    if all(hasattr(args, k) for k in ('node_num_heads', 'edge_num_heads')):
        args.num_heads = (args.node_num_heads, args.edge_num_heads)

    for key in fields:
        val = getattr(args, key)

        # assign each item to a different return argument
        if isinstance(val, (tuple, list)):
            if len(val) != num:
                raise RuntimeError(f"Length of argument '{key}' must be {num}, got {len(val)} instead")
            for i, ra in enumerate(return_args):
                setattr(ra, key, val[i])

        # assign the same value to all return arguments
        else:
            for ra in return_args:
                setattr(ra, key, val)

    return return_args


def get_node_generator(args, node_sequencer, cond_size, is_value_network=False):
    # (+2 to include start/end tokens, in the type count and the sequence length)
    config = GPT2Config(
        seq_vocab_sizes=(len(node_sequencer.node_type_names)+2, node_sequencer.max_seq_len, node_sequencer.max_num_nodes),
        output_seq_dims=(1,) if is_value_network else (len(node_sequencer.node_type_names)+2, node_sequencer.max_num_nodes),
        n_positions=node_sequencer.max_seq_len, # max. sequence length
        n_embd=args.hidden_dim, # dimension of the embeddings and general feature space dimension (must be divisible by n_head)
        n_cond=cond_size[1], # dimension of the CLIP embedding
        n_layer=args.num_layers, # number of attention blocks
        n_head=args.num_heads, # number of attention heads per block
        is_causal=True, # limit attention to previous tokens in the sequence
        separate=False, # separate embeddings for different token types
        cond_type=args.cond_type, # input conditioning method
        use_fast_attn=args.use_fast_attn # use memory-efficient attention
    )

    model = ImageCondEncoder(config)
    return model


def get_edge_generator(args, slot_sequencer, edge_sequencer, cond_size, is_value_network=False):
    # split encoder and decoder configurations
    enc_args, dec_args = split_args(args)

    # (+2 to include start/end tokens, in the type count and the sequence length)
    slot_enc_config = GPT2Config(
        seq_vocab_sizes=(
            len(slot_sequencer.node_type_names)+2, # slot_node_type_seq
            slot_sequencer.max_num_nodes, # slot_node_idx_seq
            slot_sequencer.max_num_nodes, # slot_node_depth_seq
            slot_sequencer.max_seq_len, # slot_idx_seq
            slot_sequencer.max_num_output_slots+slot_sequencer.max_num_parents+1), # slot_id_seq (+1 since the values at the padding and the start/stop tokens is max_idx+1)
        output_seq_dims=(dec_args.hidden_dim,), # slot embeddings
        n_positions=slot_sequencer.max_seq_len, # max. sequence length
        n_embd=enc_args.hidden_dim, # dimension of the embeddings and general feature space dimension (must be divisible by n_head)
        n_cond=cond_size[1], # dimension of the CLIP embedding
        n_layer=enc_args.num_layers, # number of attention blocks
        n_head=enc_args.num_heads, # number of attention heads per block
        is_causal=False, # limit attention to previous tokens in the sequence
        cond_type=args.cond_type, # input conditioning method
        use_fast_attn=args.use_fast_attn # use memory-efficient attention
    )

    # Use addtional positions for merged cross attention
    gen_n_positions = edge_sequencer.max_seq_len
    if args.cond_type == 'merged_cross_attention':
        gen_n_positions = max(slot_sequencer.max_seq_len + cond_size[0], gen_n_positions)

    # (+2 to include start/end tokens, in the type count and the sequence length)
    pointer_gen_config = GPT2Config(
        seq_vocab_sizes=(
            None, # edge_seq ('None' means the sequence is already embedded - the indices in this sequence will be used as pointers into the node embeddings)
            edge_sequencer.max_seq_len, # edge_idx_seq
            3), # edge_elm_seq
        output_seq_dims=(1,) if is_value_network else (dec_args.hidden_dim,), # featvec to compare to slot embeddings
        n_positions=gen_n_positions, # max. sequence length
        n_embd=dec_args.hidden_dim, # dimension of the embeddings and general feature space dimension (must be divisible by n_head)
        n_cond=cond_size[1], # dimension of the CLIP embedding
        n_layer=dec_args.num_layers, # number of attention blocks
        n_head=dec_args.num_heads, # number of attention heads per block
        is_causal=True, # limit attention to previous tokens in the sequence
        cond_type=args.cond_type, # input conditioning method
        is_decoder=args.use_encoder_decoder, # add cross-attention to node embeddings
        use_fast_attn=args.use_fast_attn # use memory-efficient attention
    )

    model = ImageCondPointerNet(
        node_enc_config=slot_enc_config,
        pointer_gen_config=pointer_gen_config,
        use_inner_product=not is_value_network,
        check_semantic_validity=args.check_semantic_validity)

    return model


def get_full_param_generator(args, node_sequencer, param_sequencer, cond_size, is_value_network=False):
    # split encoder and decoder configurations
    enc_args, dec_args = split_args(args)

    # (+2 to include start/end tokens, in the type count and the sequence length)
    node_enc_config = GPT2Config(
        seq_vocab_sizes=(len(node_sequencer.node_type_names) + 2, node_sequencer.max_num_nodes),
        # node_type_seq, node_depth_seq (no positional encoding)
        output_seq_dims=(dec_args.hidden_dim,),  # per-node features to be used as condition by the parameter generator
        n_positions=node_sequencer.max_seq_len,  # max. sequence length
        n_embd=enc_args.hidden_dim,  # dimension of the embeddings and general feature space dimension (must be divisible by n_head)
        n_cond=cond_size[1],  # dimension of the CLIP embedding
        n_layer=enc_args.num_layers,  # number of attention blocks
        n_head=enc_args.num_heads,  # number of attention heads per block
        is_causal=False,  # limit attention to previous tokens in the sequence
        separate=False,  # separate embeddings for different token types
        cond_type=args.cond_type, # input conditioning method
        use_fast_attn=args.use_fast_attn # use memory-efficient attention
    )

    # Use addtional positions for merged cross attention
    gen_n_positions = param_sequencer.max_full_seq_len
    if args.cond_type == 'merged_cross_attention':
        gen_n_positions = max(node_sequencer.max_seq_len + cond_size[0], gen_n_positions)

    # (+2 to include start/end tokens, in the type count and the sequence length)
    # (+1 to include node separte token)
    param_gen_config = GPT2Config(
        seq_vocab_sizes=(
            None,  # node embedding sequence
            param_sequencer.max_num_id_tokens,  # param_id_seq
            param_sequencer.max_seq_len,  # param_token_idx_seq
            param_sequencer.quant_steps,  # param_val_seq
            param_sequencer.max_vec_dim,  # param_vector_elm_idx_seq
            param_sequencer.max_seq_len,  # param_array_elm_idx_seq
            param_sequencer.max_num_params),  # param_idx_seq
        output_seq_dims=(1,) if is_value_network else (param_sequencer.max_num_id_tokens, param_sequencer.quant_steps),
        # param_id_seq, param_val_seq
        n_positions=gen_n_positions,  # max. full sequence length
        n_embd=dec_args.hidden_dim,  # dimension of the embeddings and general feature space dimension (must be divisible by n_head)
        n_cond=cond_size[1],  # dimension of the CLIP embedding
        n_layer=dec_args.num_layers,  # number of attention blocks
        n_head=dec_args.num_heads,  # number of attention heads per block
        is_decoder=args.use_encoder_decoder, # add cross-attention to node embeddings
        is_causal=True,  # limit attention to previous tokens in the sequence
        separate=False,  # separate embeddings for different token types
        conditional=True,
        cond_vocab_size=None,
        # vocabulary size for the given condition token (None if the feature is given directly) - this is None since it is a joint conditioning on node type and graph category
        cond_type=args.cond_type, # input conditioning method
        use_fast_attn=args.use_fast_attn # use memory-efficient attention
    )

    if args.edge_condition == 'node_edge_gnn':
        model = ImageCondGraphConditionalParamEncoder(node_enc_config=node_enc_config, generator_config=param_gen_config,
                                                      node_seq_stop_token=node_sequencer.stop_token, n_gcn=args.num_gcn)
    else:
        raise NotImplemented(f'Unknown edge conditional model: {args.edge_condition}')
    return model


def get_image_encoder(args, device, bypass_clip=False):
    clip_model = getattr(args, 'clip_model', 'ViT-B/32')
    normalize_clip = getattr(args, 'normalize_clip', False)
    multi_modal_input = getattr(args, 'sample_text_prompts', False)
    embed_type = getattr(args, 'embed_type', None)
    hidden_dim = args.hidden_dim[0] if isinstance(args.hidden_dim, (tuple, list)) else args.hidden_dim

    # create the image encoder model
    if args.image_encoder_type == 'clip':
        if multi_modal_input:
            model = CLIPImageMultiModalEncoder(hidden_dim, clip_model, device, embed_type=embed_type, normalize=normalize_clip, bypass_clip=bypass_clip)
        else:
            model = CLIPImageEncoder(hidden_dim, clip_model, device, embed_type=embed_type, normalize=normalize_clip, bypass_clip=bypass_clip)
    else:
        raise RuntimeError(f'Unknown Image encoder type: {args.image_encoder_type}')
    return model


class AssembledGeneratorBase(nn.Module):
    """Assembled image-conditioned procedural material generation model.
    """
    def __init__(self, args, device, bypass_clip=False, is_value_network=False):
        # test if the given configuration is valid
        embed_type = getattr(args, 'embed_type', None)
        cond_type = getattr(args, 'cond_type', 'feed_forward')

        # feed-forward conditioning requires a 1D CLIP embedding
        if embed_type == 'project_resize' and cond_type == 'feed_forward':
            raise RuntimeError(f"Embedding type 'project_resize' is not compatible with conditioning method 'feed_forward'")

        # Transformer hidden size must be consistent for projection-based CLIP embeddings
        if embed_type and isinstance(args.hidden_dim, (tuple, list)) and max(args.hidden_dim) != min(args.hidden_dim):
            raise RuntimeError(f'Inconsistent hidden dimensions for projection-based CLIP embedding: {args.hidden_dim}')

        # for merged cross-attention, the CLIP embedding dimension must match the Transformer hidden size(s)
        if embed_type is None and cond_type == 'merged_cross_attention':
            clip_dim = get_cond_size(args)[1]
            if isinstance(args.hidden_dim, (tuple, list)):
                mismatch_error = any(hd != clip_dim for hd in args.hidden_dim)
            else:
                mismatch_error = args.hidden_dim != clip_dim
            if mismatch_error:
                raise RuntimeError(f"Cannot apply 'merged_cross_attention' with mismatched CLIP embedding and Transformer hidden size "
                                   f"({clip_dim} != {args.hidden_dim})")

        super().__init__()

        self.encoder = get_image_encoder(args, device, bypass_clip=bypass_clip).to(device)
        self.decoder = None

        self.is_value_network = is_value_network

    def compile(self):
        self.encoder = torch.compile(self.encoder, dynamic=True)
        self.decoder = torch.compile(self.decoder, dynamic=True)
        return self


class AssembledNodeGenerator(AssembledGeneratorBase):
    """Assembled node generator model with text/image encoder and conditional node sequence generator.
    """
    def __init__(self, args, node_sequencer, device, **kwargs):
        super().__init__(args, device, **kwargs)

        self.decoder = get_node_generator(
            args, node_sequencer, get_cond_size(args), is_value_network=self.is_value_network).to(device)
        self.node_sequencer = node_sequencer

    def forward(self, data, return_logits=False, return_log_probs=False):
        # prepare inputs
        sequences = tuple(data[k] for k in ('node_type_seq', 'node_idx_seq', 'node_depth_seq'))
        labels = tuple(data[k] for k in ('node_type_seq', 'node_depth_seq'))

        # prepare logits masks
        semantic_masks, action_masks = None, None

        if any(k.endswith('_semantic_mask') for k in data):
            semantic_masks = tuple(data[k] for k in ('node_type_semantic_mask', 'node_depth_semantic_mask'))
        if any(k.endswith('_action_mask') for k in data):
            action_masks = tuple(data[k] for k in ('node_type_action_mask', 'node_depth_action_mask'))

        # forward pass
        image_embedding = self.encoder(data['prerendered'])
        outputs = self.decoder(
            sequences=sequences,
            cond=image_embedding,
            attention_mask=data['node_seq_mask'],
            **({
                'labels': labels,
                'output_log_probs': return_log_probs,
                'semantic_masks': semantic_masks,
                'action_masks': action_masks
            } if not self.is_value_network else {})
        )

        return outputs if not self.is_value_network and return_logits else outputs[0]

    def generate(self, data, return_sequences=False, **node_sequencer_kwargs):
        with torch.no_grad():
            images_embedding = self.encoder(data['prerendered'])
            ordered_nodes, all_seqs = self.node_sequencer.generate_nodes(self.decoder, images_embedding, **node_sequencer_kwargs)

        return ordered_nodes, *((all_seqs,) if return_sequences else ())


class AssembledEdgeGenerator(AssembledGeneratorBase):
    """Assembled edge generator model with text/image encoder and conditional edge sequence generator.
    """
    def __init__(self, args, slot_sequencer, edge_sequencer, device, **kwargs):
        super().__init__(args, device, **kwargs)

        self.decoder = get_edge_generator(
            args, slot_sequencer, edge_sequencer, get_cond_size(args), is_value_network=self.is_value_network).to(device)
        self.slot_sequencer = slot_sequencer
        self.edge_sequencer = edge_sequencer

    def forward(self, data, return_logits=False, return_log_probs=False):
        # prepare inputs
        node_sequences = tuple(data[k] for k in ('slot_node_type_seq', 'slot_node_idx_seq', 'slot_node_depth_seq', 'slot_idx_seq', 'slot_id_seq'))
        edge_sequences = tuple(data[k] for k in ('edge_seq', 'edge_idx_seq', 'edge_elm_seq'))

        # prepare logits masks
        semantic_masks, action_masks = None, None

        if any(k.endswith('_semantic_mask') for k in data):
            semantic_masks = data['edge_semantic_mask']
        if any(k.endswith('_action_mask') for k in data):
            action_masks = data['edge_action_mask']

        # forward pass
        image_embedding = self.encoder(data['prerendered'])
        outputs = self.decoder(
            node_sequences=node_sequences,
            edge_sequences=edge_sequences,
            cond=image_embedding,
            node_attention_mask=data['slot_seq_mask'],
            edge_attention_mask=data['edge_seq_mask'],
            **({
                'labels': data['edge_seq'],
                'output_log_probs': return_log_probs,
                'semantic_masks': semantic_masks,
                'action_masks': action_masks
            } if not self.is_value_network else {})
        )

        return outputs if not self.is_value_network and return_logits else outputs[0]

    def generate(self, data, ordered_nodes, return_sequences=False, **edge_sequencer_kwargs):
        with torch.no_grad():
            images_embedding = self.encoder(data['prerendered'])
            ordered_nodes, all_seqs = self.edge_sequencer.generate_edges(
                self.decoder, images_embedding, ordered_nodes, self.slot_sequencer, **edge_sequencer_kwargs)

        return ordered_nodes, *((all_seqs,) if return_sequences else ())


class AssembledParamGenerator(AssembledGeneratorBase):
    """Assembled parameter generator model with text/image encoder and conditional parameter sequence generator.
    """
    def __init__(self, args, node_sequencer, param_sequencer, device, **kwargs):
        super().__init__(args, device, **kwargs)

        decoder = get_full_param_generator(
            args, node_sequencer, param_sequencer, get_cond_size(args), is_value_network=self.is_value_network)
        self.decoder = decoder.to(device)
        self.node_sequencer = node_sequencer
        self.param_sequencer = param_sequencer

        self.edge_cond_type = args.edge_condition

    def compile(self):
        self.encoder = torch.compile(self.encoder, dynamic=True)

        # compile the decoder model in parts if it uses graph convolution
        if isinstance(self.decoder, ImageCondGraphConditionalParamEncoder):
            self.decoder = self.decoder.compile()
        else:
            self.decoder = torch.compile(self.decoder, dynamic=True)

        return self

    def forward(self, data, return_logits=False, return_log_probs=False):
        # prepare inputs
        node_sequences = tuple(data[k] for k in ('param_node_type_seq', 'param_node_depth_seq'))
        gen_sequences = tuple(data[k] for k in ('param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq', 'param_array_elm_idx_seq', 'param_idx_seq'))
        labels = tuple(data[k] for k in ('param_id_seq', 'param_val_seq'))

        # prepare logits masks
        semantic_masks, action_masks = None, None

        if any(k.endswith('_semantic_mask') for k in data):
            semantic_masks = tuple(data[k] for k in ('param_id_semantic_mask', 'param_val_semantic_mask'))
        if any(k.endswith('_action_mask') for k in data):
            action_masks = tuple(data[k] for k in ('param_id_action_mask', 'param_val_action_mask'))

        # forward pass
        image_embedding = self.encoder(data['prerendered'])
        outputs = self.decoder(
            node_sequences=node_sequences,
            gen_sequences=gen_sequences,
            edge_node_inds=data['edge_node_inds'],
            param_node_inds=data['param_node_inds'],
            cond=image_embedding,
            node_attention_mask=data['param_node_seq_mask'],
            gen_attention_mask=data['param_seq_mask'],
            **({
                'labels': labels,
                'output_log_probs': return_log_probs,
                'semantic_masks': semantic_masks,
                'action_masks': action_masks
            } if not self.is_value_network else {})
        )

        return outputs if not self.is_value_network and return_logits else outputs[0]

    def generate(self, data, ordered_nodes, return_sequences=False, **param_sequencer_kwargs):
        with torch.no_grad():
            images_embedding = self.encoder(data['prerendered'])
            ordered_nodes, all_seqs = self.param_sequencer.generate_params(
                self.decoder, images_embedding, ordered_nodes, self.node_sequencer, self.edge_cond_type, **param_sequencer_kwargs)

        return ordered_nodes, *((all_seqs,) if return_sequences else ())


class CondMatFormer(nn.Module):
    """Collection of node, edge, and parameter generator models, assembled into a single model.
    The model is used for inference and some components can be ablated for partial generation.
    """
    GENERATOR_TYPES = ('node', 'edge', 'param')

    def __init__(self, model_args, node_sequencer=None, slot_sequencer=None, edge_sequencer=None,
                 param_node_sequencer=None, param_sequencer=None, bypass_clip=False, is_value_network=False,
                 graph_reformer=None, device='cpu', distributed=False):
        super().__init__()

        # at least some parts of the model must be enabled
        node_args, edge_args, param_args = [model_args.get(gen_type) for gen_type in self.GENERATOR_TYPES]

        if all(args is None for args in (node_args, edge_args, param_args)):
            raise ValueError("At least one of 'node', 'edge', or 'param' must be enabled")
        if (node_args is None) != (edge_args is None):
            raise ValueError("'node' and 'edge' must be enabled together")

        # create the CLIP encoder
        clip_args = node_args if node_args is not None else param_args
        clip_model_name = getattr(clip_args, 'clip_model', 'ViT-B/32')

        image_encoder_type = getattr(clip_args, 'image_encoder_type', 'clip')
        if image_encoder_type != 'clip':
            raise RuntimeError(f"Unknown image encoder type '{image_encoder_type}'")

        self.clip_encoder = CLIPEncoder(model_name=clip_model_name, device=device) if not bypass_clip else nn.Identity()

        # create the models (bypassing CLIP encoders)
        self.node_generator, self.edge_generator, self.param_generator = [None] * 3
        self.is_value_network = is_value_network
        common_kwargs = {'device': device, 'bypass_clip': True, 'is_value_network': is_value_network}

        if node_args is not None:
            self.node_generator = AssembledNodeGenerator(node_args, node_sequencer, **common_kwargs)
            self.edge_generator = AssembledEdgeGenerator(edge_args, slot_sequencer, edge_sequencer, **common_kwargs)
        if param_args is not None:
            self.param_generator = AssembledParamGenerator(param_args, param_node_sequencer, param_sequencer, **common_kwargs)

        # distributed data parallel
        if distributed:
            self.distribute_model(device)

        self.generation_kwargs = {}
        self.graph_reformer = None

        if not self.is_value_network:
            # prepare keyword arguments for generation
            make_kwargs = lambda args, custom_fields=[]: {
                k: getattr(args, k) for k in ('temperature', 'prob_k', 'nucleus_top_p', 'semantic_validate', *custom_fields)
                if getattr(args, k, None) is not None}

            if node_args is not None:
                self.generation_kwargs.update({
                    'node': {
                        **make_kwargs(node_args, custom_fields=['max_gen_nodes']),
                        'node_order': node_args.node_order,
                    },
                    'edge': make_kwargs(edge_args)
                })
            if param_args is not None:
                self.generation_kwargs['param'] = make_kwargs(param_args)

            # graph reformer
            self.graph_reformer = graph_reformer

    def has_generator(self, gen_type):
        if gen_type not in self.GENERATOR_TYPES:
            raise RuntimeError(f"Unknown generator type '{gen_type}'")
        return getattr(self, f'{gen_type}_generator') is not None

    def get_generator(self, gen_type, unwrap=False):
        if gen_type not in self.GENERATOR_TYPES:
            raise RuntimeError(f"Unknown generator type '{gen_type}'")
        gen = getattr(self, f'{gen_type}_generator')
        return unwrap_ddp(gen) if unwrap else gen

    def set_trainable(self, trainable):
        # set the trainable state for all generators
        if isinstance(trainable, bool):
            self.requires_grad_(trainable)

        # set the trainable state for specific generators
        elif isinstance(trainable, dict):
            for gen_type, flag in trainable.items():
                if self.has_generator(gen_type):
                    self.get_generator(gen_type).requires_grad_(flag)

        else:
            raise ValueError(f"Argument 'trainable' must be a boolean or a dictionary. "
                             f"Got '{type(trainable).__name__}' instead.")

    def distribute_model(self, device):
        for gen_type in self.GENERATOR_TYPES:
            if self.has_generator(gen_type):
                gen_key = f'{gen_type}_generator'
                setattr(self, gen_key, DDP(getattr(self, gen_key), device_ids=[device]))

    def load_model_state(self, model_args, device, exclude_lm_head=False):
        # load the model state for each generator if applicable
        for gen_type in self.GENERATOR_TYPES:
            if self.has_generator(gen_type) and model_args.get(gen_type) is not None:
                generator, args = self.get_generator(gen_type), model_args[gen_type]
                model_step = args.model_step

                # load a specific model state
                if model_step == 'best' or model_step.isdigit():
                    state_file_name = f'{model_step}_model.pth'

                # load the last model state
                elif model_step == 'last':
                    state_pattern = re.compile(r'(\d+)_model.pth')
                    all_state_files = [f for f in os.listdir(pth.join(args.model_dir, args.exp_name))
                                      if state_pattern.fullmatch(f)]
                    if not all_state_files:
                        raise RuntimeError('No model state file found.')
                    state_file_name = sorted(all_state_files, key=lambda f: int(state_pattern.match(f).group(1)))[-1]

                state_dict_file = pth.join(args.model_dir, args.exp_name, state_file_name)
                load_model_state(generator, state_dict_file, device, exclude_lm_head=exclude_lm_head)

    def load_from_checkpoint(self, checkpoint_file, device, exclude_lm_head=False):
        # read checkpoint file
        if isinstance(checkpoint_file, str):
            state_dict = torch.load(checkpoint_file, map_location=device)
        else:
            state_dict = checkpoint_file

        # load the model state for each generator if applicable
        loaded_gen_types = []

        for gen_type in self.GENERATOR_TYPES:
            if self.has_generator(gen_type):
                match_key = f'{gen_type}_generator.'
                generator_state_dict = {k[len(match_key):]: v for k, v in state_dict.items() if k.startswith(match_key)}
                if generator_state_dict:
                    load_model_state(self.get_generator(gen_type), generator_state_dict, None, exclude_lm_head=exclude_lm_head)
                    loaded_gen_types.append(gen_type)

        print(f"Loaded model state for generators '{', '.join(loaded_gen_types)}' from '{checkpoint_file}'")

    def forward(self, data, gen_types=None, bypass_clip=False, **kwargs):
        # compute CLIP embedding and replace the input
        if not bypass_clip and not isinstance(self.clip_encoder, nn.Identity):
            clip_embedding = self.clip_encoder(data['prerendered']).float()
            data = data.copy()
            data['prerendered'] = clip_embedding

        # prepare forward pass options
        fwd_kwargs = {k: kwargs.copy() for k in self.GENERATOR_TYPES
                      if self.has_generator(k) and (gen_types is None or k in gen_types)}

        # forward pass for each generator if applicable
        outputs = {'clip_embedding': data['prerendered']}
        for gen_type, kwargs in fwd_kwargs.items():
            outputs[gen_type] = self.get_generator(gen_type)(data, **kwargs)

        return outputs

    def generate(self, data, ordered_nodes=None, bypass_clip=False, return_sequences=False, deterministic=False, **kwargs):
        # does not work with value networks
        if self.is_value_network:
            raise RuntimeError('Cannot generate parameters with a value network')

        # compute CLIP embedding and replace the input
        if not bypass_clip and not isinstance(self.clip_encoder, nn.Identity):
            with torch.no_grad():
                clip_embedding = self.clip_encoder(data['prerendered']).float()
                data = data.copy()
                data['prerendered'] = clip_embedding

        # prepare generation options
        gen_kwargs = {k: {**self.generation_kwargs.get(k, {}), **kwargs}
                      for k in self.GENERATOR_TYPES if self.has_generator(k)}
        for k, v in gen_kwargs.items():
            v.update({
                'return_sequences': return_sequences if isinstance(return_sequences, bool) else k in return_sequences,
                'deterministic': deterministic if isinstance(deterministic, bool) else k in deterministic
            })

        # generate nodes and edges
        outputs = {'clip_embedding': data['prerendered']}

        if self.has_generator('node'):
            with torch.no_grad():
                ordered_nodes, *outputs['node'] = self.get_generator('node', unwrap=True).generate(data, **gen_kwargs['node'])
                ordered_nodes, *outputs['edge'] = self.get_generator('edge', unwrap=True).generate(data, ordered_nodes, **gen_kwargs['edge'])

            # reform graph structure (e.g., prune isolated nodes)
            if self.graph_reformer is not None:
                ordered_nodes = self.graph_reformer(ordered_nodes)

        # use ground truth graph structures if provided
        elif ordered_nodes is None:
            ordered_nodes = [SimpleOrderedNodes(n, nd) for n, nd in zip(data['nodes'], data['node_depths'])]

        # generate parameters
        if self.has_generator('param'):
            with torch.no_grad():
                ordered_nodes, *outputs['param'] = self.get_generator('param', unwrap=True).generate(data, ordered_nodes, **gen_kwargs['param'])

        return ordered_nodes, outputs

    def get_log_probs(self, data, gen_types=None, bypass_clip=False):
        # does not work with value networks
        if self.is_value_network:
            raise RuntimeError('Cannot calculate log probabilities with a value network')

        # run forward evaluation
        outputs = self.forward(data, gen_types=gen_types, bypass_clip=bypass_clip, return_logits=True, return_log_probs=True)
        outputs = tuple({k: v[i] for k, v in outputs.items() if k in self.GENERATOR_TYPES}
                        for i in (2, 3, 4))

        # apply sequence masks
        masks = {k: data[f'{k}_seq_mask'][:, 1:] for k in outputs[0]}
        for res in outputs:
            for k, v in res.items():
                res[k] = v * masks[k]

        return outputs

    def get_values(self, data, gen_types=None, bypass_clip=False):
        # does not work with policy networks
        if not self.is_value_network:
            raise RuntimeError('Cannot calculate values with a policy network')

        # run forward evaluation
        outputs = self.forward(data, gen_types=gen_types, bypass_clip=bypass_clip)
        outputs = {k: (v[0] if isinstance(v, (list, tuple)) else v)[:, :-1, 0]
                   for k, v in outputs.items() if k in self.GENERATOR_TYPES}

        # apply sequence masks
        masks = {k: data[f'{k}_seq_mask'][:, 1:] for k in outputs}
        for k, v in outputs.items():
            outputs[k] = v * masks[k]

        return outputs
