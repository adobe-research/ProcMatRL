# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import sys
import copy
import math
from itertools import chain
import torch
import warnings

from tqdm import tqdm

from .edge_sequencer import EdgeSequencer
from ..utils import logits_regularize, nucleus_filtering, stack_tensor_lists
from ..simple_graph import SimpleGraph
from ..simple_graph.convert_simple_graph_parameters import get_param_limits


# Paul's solution
class ParamSequencer:
    def __init__(self, max_full_seq_len, max_num_params, max_vec_dim, max_seq_len, quant_steps, node_types, use_alpha):
        self.max_full_seq_len = max_full_seq_len
        self.max_num_params = max_num_params
        self.max_vec_dim = max_vec_dim
        self.max_seq_len = max_seq_len
        self.quant_steps = quant_steps

        self.node_types = node_types
        self.node_type_names = [node_type['name'] for node_type in self.node_types]
        if len(self.node_type_names) != len(self.node_types):
            raise RuntimeError('Number of node type names and node types does not match.')
        self._legacy_flattened = getattr(self.node_types, '_legacy_flattened', True)

        self.use_alpha = use_alpha

        self.start_token = self.max_num_params
        self.stop_token = self.max_num_params + 1
        self.node_sep_token = self.max_num_params + 2
        self.max_num_id_tokens = self.max_num_params + 3

        self.node_start_tokens = [self.start_token, self.node_sep_token]  # a short note for decoding

        print('In parameter sequencer:')
        print(f'Start Token: {self.start_token}')
        print(f'Stop Token: {self.stop_token}')
        print(f'Node Sep Token: {self.node_sep_token}')

    def get_sequences(self, nodes, add_param_stop_token=True):
        N = self.max_full_seq_len
        param_id_seq = torch.full((N,), self.stop_token, dtype=torch.long)
        param_token_idx_seq = torch.zeros(N, dtype=torch.long)
        param_val_seq = torch.zeros(N, dtype=torch.long)
        param_vector_elm_idx_seq = torch.zeros(N, dtype=torch.long)
        param_array_elm_idx_seq = torch.zeros(N, dtype=torch.long)
        param_idx_seq = torch.zeros(N, dtype=torch.long)
        param_node_inds = torch.full((N,), len(nodes), dtype=torch.long)

        # first token
        param_id_seq[0] = self.start_token
        param_token_idx_seq[0] = 0
        param_val_seq[0] = 0
        param_vector_elm_idx_seq[0] = 0
        param_array_elm_idx_seq[0] = 0
        param_idx_seq[0] = 0
        param_node_inds[0] = 0
        seq_idx = 1

        for node_idx, node in enumerate(nodes):
            node_param_id_seq, node_param_token_idx_seq, node_param_val_seq, node_param_vector_elm_idx_seq, node_param_array_elm_idx_seq, node_param_idx_seq = self.get_sequences_for_one_node(node)
            node_param_seq_len = node_param_id_seq.shape[0]
            # if seq_idx + node_param_seq_len >= N:
            #     raise RuntimeError(f'Too many parameters serialized {seq_idx} + {node_param_seq_len} > {N}')

            # append parameter sequence of current node to the main sequence
            if seq_idx + node_param_seq_len <= N:
                param_id_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_id_seq
                param_token_idx_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_token_idx_seq
                param_val_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_val_seq
                param_vector_elm_idx_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_vector_elm_idx_seq
                param_array_elm_idx_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_array_elm_idx_seq
                param_idx_seq[seq_idx: seq_idx + node_param_seq_len] = node_param_idx_seq
                param_node_inds[seq_idx: seq_idx + node_param_seq_len] = node_idx

            seq_idx += node_param_seq_len

            # attach a node separate token
            if seq_idx < N:
                param_id_seq[seq_idx] = self.node_sep_token
                param_token_idx_seq[seq_idx] = (node_param_token_idx_seq[-1] if len(node_param_token_idx_seq) > 0 else 0) + 1
                param_val_seq[seq_idx] = 0
                param_vector_elm_idx_seq[seq_idx] = 0
                param_array_elm_idx_seq[seq_idx] = 0
                param_idx_seq[seq_idx] = 0
                param_node_inds[seq_idx] = node_idx + 1  # we attach the next node's embedding

            seq_idx += 1

        # last token
        if add_param_stop_token:
            if seq_idx < N:
                param_id_seq[seq_idx] = self.stop_token
                param_token_idx_seq[seq_idx] = 0
                param_val_seq[seq_idx] = 0
                param_vector_elm_idx_seq[seq_idx] = 0
                param_array_elm_idx_seq[seq_idx] = 0
                param_idx_seq[seq_idx] = 0
                param_node_inds[seq_idx] = len(nodes)

            seq_idx += 1

        param_seq_len = seq_idx
        if param_seq_len > N:
            raise RuntimeError(f'Too many parameters serialized {param_seq_len} > {N}.')

        # sequence mask
        param_seq_mask = torch.zeros(N, dtype=torch.int32)
        param_seq_mask[:param_seq_len] = 1

        param_seqs = param_id_seq, param_token_idx_seq, param_val_seq, param_vector_elm_idx_seq, param_array_elm_idx_seq, param_idx_seq, param_seq_mask, param_node_inds
        return tuple(seq[:param_seq_len] for seq in param_seqs)

    def get_sequences_for_one_node(self, node):
        param_types = self.node_types[self.node_type_names.index(node.type)]['parameters']
        param_type_names = [param_type['name'] for param_type in param_types]
        if len(param_type_names) > self.max_num_params:  # +1
            raise RuntimeError(f'Too many parameter types for a node of type {node.type}.')

        # the id is the index in the full list of possible parameters of a given node type
        N = self.max_seq_len
        param_id_seq = torch.ones(N, dtype=torch.long) * self.stop_token
        param_token_idx_seq = torch.zeros(N, dtype=torch.long)
        param_val_seq = torch.zeros(N, dtype=torch.long)
        param_vector_elm_idx_seq = torch.zeros(N, dtype=torch.long)
        param_array_elm_idx_seq = torch.zeros(N, dtype=torch.long)
        param_idx_seq = torch.zeros(N, dtype=torch.long)

        seq_idx = 0
        for param_idx, (param_name, param_val) in enumerate(zip(node.param_names, node.param_vals)):
            param_dtype = SimpleGraph.get_param_dtype(node_type=node.type,
                                                      param_type_info=param_types[param_type_names.index(param_name)],
                                                      use_alpha=self.use_alpha,
                                                      legacy_flattened=self._legacy_flattened)
            param_tensor_rank = SimpleGraph.get_param_tensor_rank(param_dtype=param_dtype)

            if param_tensor_rank == 'scalar':
                if isinstance(param_val, list) and node.type == 'F.levels' and param_name in ['in_low', 'in_high']:
                    warnings.warn('Detected abnormal Levels node parameters due to compatibility.')
                    param_val = (param_val[0] + param_val[1] + param_val[2]) / 3.0
                elm_vals = [param_val]
                vector_elm_inds = [0]
                array_elm_inds = [0]
            elif param_tensor_rank == 'vector':
                elm_vals = param_val
                vector_elm_inds = list(range(len(param_val)))
                array_elm_inds = [0] * len(param_val)
            elif param_tensor_rank == 'array':
                elm_vals = list(chain(*param_val))
                vector_elm_inds = list(chain(*(range(len(array_elm_val)) for array_elm_val in param_val)))
                array_elm_inds = list(chain(
                    *([array_elm_ind] * len(array_elm_val) for array_elm_ind, array_elm_val in enumerate(param_val))))
            else:
                raise RuntimeError(f'Unknown parameter tensor rank: {param_tensor_rank}.')

            for elm_val, array_elm_idx, vector_elm_idx in zip(elm_vals, array_elm_inds, vector_elm_inds):
                # if seq_idx + 1 >= N:
                #     raise RuntimeError(f'Too many parameters in node {node.type}.')

                if elm_val < 0 or elm_val >= self.quant_steps:
                    raise RuntimeError(
                        f'Value for parameter {param_name} is out of bounds: {elm_val} is not in [0, {self.quant_steps - 1}]')

                if seq_idx < N:
                    param_id_seq[seq_idx] = param_type_names.index(param_name)
                    param_token_idx_seq[seq_idx] = seq_idx + 1  # shift by 1
                    param_val_seq[seq_idx] = elm_val
                    param_vector_elm_idx_seq[seq_idx] = vector_elm_idx
                    param_array_elm_idx_seq[seq_idx] = array_elm_idx
                    param_idx_seq[seq_idx] = param_idx

                seq_idx += 1

        param_seq_len = seq_idx
        if param_seq_len > N:
            raise RuntimeError(f"Too many parameters in node '{node.type}', sequence length is {param_seq_len} > {N}.")

        # trim to valid
        param_id_seq = param_id_seq[:seq_idx]
        param_token_idx_seq = param_token_idx_seq[:seq_idx]
        param_val_seq = param_val_seq[:seq_idx]
        param_vector_elm_idx_seq = param_vector_elm_idx_seq[:seq_idx]
        param_array_elm_idx_seq = param_array_elm_idx_seq[:seq_idx]
        param_idx_seq = param_idx_seq[:seq_idx]

        return param_id_seq, param_token_idx_seq, param_val_seq, param_vector_elm_idx_seq, param_array_elm_idx_seq, param_idx_seq

    def get_params(self, nodes, param_id_seq, param_val_seq, precise=True):
        if param_id_seq[0] != self.start_token:
            raise RuntimeError('Param id sequence does not start with a start token.')
        else:  # remove start token
            param_id_seq = param_id_seq[1:]
            param_val_seq = param_val_seq[1:]

        if param_id_seq[-1] != self.stop_token:
            if precise:
                raise RuntimeError('Parameter sequence does not end with an end token.')
            else:
                print('Parameter sequence does not end with an end token.')

        # split sequence into subsequences that each represent a single node (split after each node_sep_token)
        split_indices = (param_id_seq == self.node_sep_token).nonzero(as_tuple=True)[0] + 1

        ## special case: cut-off sequence
        if len(split_indices) < len(nodes):
            node_param_id_seq = torch.tensor_split(param_id_seq[:-1], split_indices)
            node_param_val_seq = torch.tensor_split(param_val_seq[:-1], split_indices)
        else:
            node_param_id_seq = torch.tensor_split(param_id_seq, split_indices)[:-1]
            node_param_val_seq = torch.tensor_split(param_val_seq, split_indices)[:-1]

        if len(nodes) != len(node_param_id_seq) or len(nodes) != len(node_param_val_seq):
            if precise:
                raise RuntimeError('The number of nodes does not match the number of split parameter sequences')
            else:
                print('Cannot parse enough nodes from parameter sequences.')

        for node, param_ids, param_vals in zip(nodes, node_param_id_seq, node_param_val_seq):
            self.get_params_for_one_node(node, param_ids, param_vals)

        return nodes

    def get_params_for_one_node(self, node, param_id_seq, param_val_seq):
        # remove stop tokens
        # param_id_seq, param_val_seq = sequences_to_matrix(sequences=(param_id_seq, param_val_seq),
        #                                                   start_token=None, stop_token=self.node_sep_token,
        #                                                   num_cols=1)
        param_id_seq = param_id_seq[:-1]
        param_val_seq = param_val_seq[:-1]

        if len(param_id_seq) == 0:
            return node

        param_types = self.node_types[self.node_type_names.index(node.type)]['parameters']

        # split sequence into subsequences that each represent a single parameter (split where param_id[i] != param_id[i+1])
        # split_inds = torch.nonzero(param_id_seq[1:] != param_id_seq[:-1], as_tuple=True)[0]
        # param_ids = split_at_indices(tensor=param_id_seq, before_split_inds=split_inds, dim=0)
        # param_vals = split_at_indices(tensor=param_val_seq, before_split_inds=split_inds, dim=0)
        split_inds = torch.nonzero(param_id_seq[1:] != param_id_seq[:-1], as_tuple=True)[0] + 1
        param_ids = torch.tensor_split(param_id_seq, split_inds)
        param_vals = torch.tensor_split(param_val_seq, split_inds)
        param_ids = [param_id[0].item() for param_id in param_ids]
        param_vals = [param_val.cpu().tolist() for param_val in param_vals]

        # add parameters to node and remove any existing parameters
        node.param_names = []
        node.param_vals = []
        for param_id, param_val in zip(param_ids, param_vals):
            if param_id >= len(param_types):
                break
            param_type = param_types[param_id]
            param_name = param_type['name']
            param_dtype = SimpleGraph.get_param_dtype(node_type=node.type, param_type_info=param_type, use_alpha=self.use_alpha, legacy_flattened=self._legacy_flattened)

            param_vector_dim = SimpleGraph.get_param_vector_dim(param_dtype=param_dtype)
            param_tensor_rank = SimpleGraph.get_param_tensor_rank(param_dtype=param_dtype)
            node.param_names.append(param_name)
            if param_tensor_rank == 'scalar':
                # take first value of generated parameters
                node.param_vals.append(0)
                if len(param_val) > 0:
                    node.param_vals[-1] = param_val[0]
            elif param_tensor_rank == 'vector':
                # take first param_vector_dim values of generated parameters (fill with zeros if there are too few)
                node.param_vals.append([0] * param_vector_dim)
                gen_size = min(len(param_val), param_vector_dim)
                node.param_vals[-1][:gen_size] = param_val[:gen_size]
            elif param_tensor_rank == 'array':
                # determine array length from the number of complete vectors generated for the parameter (minimum is 1),
                # for each array element proceed as in the vector case
                param_array_len = max(1, int(math.floor(len(param_val) / param_vector_dim)))
                param_array = []
                for array_idx in range(param_array_len):
                    offset = array_idx * param_vector_dim
                    gen_size = min(len(param_val) - offset, param_vector_dim)
                    if gen_size > 0:
                        vector = [0] * param_vector_dim
                        vector[:gen_size] = param_val[offset:offset + gen_size]
                        param_array.append(vector)
                node.param_vals.append(param_array)

        return node

    def generate_params(self, param_decoder, cond, ordered_nodes, node_sequencer, edge_cond_type,
                        temperature=1, prob_k=0, nucleus_top_p=None, deterministic=False, semantic_validate=False,
                        progress_bar=False):
        # assert use_alpha == self.use_alpha, 'use_alpha is different from the use_alpha in the sequencer'
        if not semantic_validate:
            raise RuntimeError('Semantic validation is required for full parameter generation.')

        # check if batch sizes match
        batch_size, device = len(cond), cond.device
        if len(ordered_nodes) != batch_size:
            raise RuntimeError('Number of conditions does not match number of partial graphs.')

        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(10000, recursion_limit))
        ordered_nodes = copy.deepcopy(ordered_nodes)  # copy to avoid modifying the original nodes
        sys.setrecursionlimit(recursion_limit)

        # get node and edge sequences for the current batch
        node_sequences = [node_sequencer.get_sequences(on) for on in ordered_nodes]
        node_sequences = list(zip(*node_sequences))
        node_type_seq, node_depth_seq, node_seq_mask \
            = [stack_tensor_lists(node_sequences[i]).to(device) for i in (0, 2, 3)]

        edge_node_inds = [EdgeSequencer.get_edge_node_inds(on) for on in ordered_nodes]

        # precompute node embeddings
        if edge_cond_type == 'node_edge_gnn':
            node_embed = param_decoder.compute_node_embeddings(
                node_sequences=(node_type_seq, node_depth_seq),
                edge_node_inds=edge_node_inds,
                cond=cond,
                node_attention_mask=node_seq_mask)
        elif edge_cond_type is None:
            node_embed = param_decoder.compute_node_embeddings(
                node_sequences=(node_type_seq, node_depth_seq),
                cond=cond,
                node_attention_mask=node_seq_mask)
        else:
            raise RuntimeError(f'Unknown edge conditioning strategy: {edge_cond_type}')

        # initialize
        param_id_seq_t = torch.full((batch_size, self.max_full_seq_len), self.stop_token, dtype=torch.long, device=device)
        param_token_idx_seq_t, param_val_seq_t, param_vector_elm_idx_seq_t, param_array_elm_idx_seq_t, param_idx_seq_t, param_node_inds_t \
            = [torch.zeros_like(param_id_seq_t) for _ in range(6)]
        param_seq_mask_t = torch.zeros_like(param_id_seq_t, dtype=torch.int32)

        # parameter sequence semantic masks
        param_id_semantic_mask_t = torch.zeros(batch_size, self.max_full_seq_len - 1, self.max_num_id_tokens, dtype=torch.bool, device=device)
        param_val_semantic_mask_t = torch.zeros(batch_size, self.max_full_seq_len - 1, self.quant_steps, dtype=torch.bool, device=device)

        param_id_stop_mask = torch.ones(self.max_num_id_tokens, dtype=torch.bool, device=device)
        param_id_stop_mask[self.stop_token] = False
        param_val_stop_mask = torch.ones(self.quant_steps, dtype=torch.bool, device=device)
        param_val_stop_mask[0] = False

        # parameter sequence action masks
        param_id_action_mask_t, param_val_action_mask_t = None, None

        if prob_k > 0 or nucleus_top_p is not None:
            param_id_action_mask_t = torch.zeros_like(param_id_semantic_mask_t)
            param_val_action_mask_t = torch.zeros_like(param_val_semantic_mask_t)

        # first token
        param_id_seq_t[:, 0] = self.start_token
        param_seq_mask_t[:, 0] = 1

        # helper arrays
        param_tensor_rank = [None] * batch_size
        param_vector_dim = [None] * batch_size
        param_elm_idx = [None] * batch_size

        cur_param_id_list = [self.start_token] * batch_size     # the last column of param_id_seq
        cur_param_token_idx_list = [0] * batch_size             # the last column of param_token_idx_seq
        cur_param_idx_list = [0] * batch_size                   # the last column of param_idx_seq
        cur_param_node_inds_list = [0] * batch_size             # the last column of param_node_inds

        step_args = cur_param_id_list, cur_param_token_idx_list, cur_param_idx_list, cur_param_node_inds_list
        validator = Validator(ordered_nodes, self)

        # each step of parameter generation
        def gen_step(param_idx, cur_param_id_list, cur_param_token_idx_list, cur_param_idx_list, cur_param_node_inds_list):

            # update validator state
            validator.update(cur_param_node_inds_list)

            # evaluate the decoder model to get the next token logits
            param_id_seq = param_id_seq_t[:, :param_idx+1]
            param_token_idx_seq = param_token_idx_seq_t[:, :param_idx+1]
            param_val_seq = param_val_seq_t[:, :param_idx+1]
            param_vector_elm_idx_seq = param_vector_elm_idx_seq_t[:, :param_idx+1]
            param_array_elm_idx_seq = param_array_elm_idx_seq_t[:, :param_idx+1]
            param_idx_seq = param_idx_seq_t[:, :param_idx+1]
            param_node_inds = param_node_inds_t[:, :param_idx+1]
            param_seq_mask = param_seq_mask_t[:, :param_idx+1]

            model_params = {
                'node_sequences': node_embed,
                'gen_sequences': (param_id_seq, param_token_idx_seq, param_val_seq, param_vector_elm_idx_seq, param_array_elm_idx_seq, param_idx_seq),
                'param_node_inds': param_node_inds,
                'cond': cond,
                'node_attention_mask': node_seq_mask,
                'gen_attention_mask': param_seq_mask}

            if edge_cond_type == 'node_edge_gnn':
                model_params['edge_node_inds'] = edge_node_inds
            elif edge_cond_type is None:
                raise NotImplementedError('Non-edge conditioned version is not implemented.')
            else:
                raise RuntimeError(f'Unknown edge conditioning strategy: {edge_cond_type}')

            all_logits = param_decoder(**model_params)[0]
            param_id_logits, param_val_logits = [logits[:, -1] for logits in all_logits]

            # semantic validity
            param_id_mask = param_id_semantic_mask_t[:, param_idx]
            param_val_mask = param_val_semantic_mask_t[:, param_idx]

            # sample parameter ids
            invalid_param_id_mask = validator.gen_invalid_id_mask(
                cur_param_id_list, param_tensor_rank, param_vector_dim, param_elm_idx, cur_param_token_idx_list, device)
            param_id_mask.copy_(invalid_param_id_mask)
            param_id_logits[param_id_mask] = -1e9

            if nucleus_top_p is not None:
                param_id_logits, param_id_mask_nf = nucleus_filtering(logits=param_id_logits, top_p=nucleus_top_p, return_mask=True)
                param_id_action_mask_t[:, param_idx] |= param_id_mask_nf
            if prob_k > 0:
                param_id_logits, param_id_mask_tk = logits_regularize(logits=param_id_logits, temperature=temperature, top_k=prob_k, return_mask=True)
                param_id_action_mask_t[:, param_idx] |= param_id_mask_tk

            if deterministic:
                next_param_id = torch.argmax(param_id_logits, dim=-1)
            else:
                param_id_probs = torch.softmax(param_id_logits, dim=-1)
                next_param_id = torch.multinomial(param_id_probs, num_samples=1).squeeze(1)

            # sample parameter values
            next_param_id_list = next_param_id.tolist()

            invalid_param_val_mask = validator.gen_invalid_val_mask(
                next_param_id_list, cur_param_id_list, param_vector_dim, param_elm_idx, device)
            param_val_mask.copy_(invalid_param_val_mask)
            param_val_logits[param_val_mask] = -1e9

            if nucleus_top_p is not None:
                param_val_logits, param_val_mask_nf = nucleus_filtering(logits=param_val_logits, top_p=nucleus_top_p, return_mask=True)
                param_val_action_mask_t[:, param_idx] |= param_val_mask_nf
            if prob_k > 0:
                param_val_logits, param_val_mask_tk = logits_regularize(logits=param_val_logits, temperature=temperature, top_k=prob_k, return_mask=True)
                param_val_action_mask_t[:, param_idx] |= param_val_mask_tk

            if deterministic:
                next_param_val = torch.argmax(param_val_logits, dim=-1)
            else:
                param_val_probs = torch.softmax(param_val_logits, dim=-1)
                next_param_val = torch.multinomial(param_val_probs, num_samples=1).squeeze(1)

            # determine the next token for other auxiliary sequences
            next_param_token_idx, next_param_vector_elm_idx, next_param_array_elm_idx, next_param_idx, next_param_node_inds, next_param_seq_mask \
                = [[0] * batch_size for _ in range(6)]
            n_finished = 0

            for b, (cur_id, next_id, cur_node_idx) in enumerate(zip(cur_param_id_list, next_param_id_list, cur_param_node_inds_list)):

                # predicted stop token
                if next_id == self.stop_token:
                    next_param_token_idx[b] = 0
                    next_param_vector_elm_idx[b] = 0
                    next_param_array_elm_idx[b] = 0
                    next_param_idx[b] = 0
                    next_param_node_inds[b] = len(ordered_nodes[b])
                    n_finished += 1

                # predicted new node
                elif next_id == self.node_sep_token:
                    param_tensor_rank[b] = None
                    param_vector_dim[b] = None
                    param_elm_idx[b] = None

                    next_param_token_idx[b] = (0 if cur_id in self.node_start_tokens else cur_param_token_idx_list[b]) + 1
                    next_param_vector_elm_idx[b] = 0
                    next_param_array_elm_idx[b] = 0
                    next_param_idx[b] = 0
                    next_param_node_inds[b] = cur_node_idx + 1  # move to next node

                # predicted new parameter
                elif cur_id != next_id:
                    # new parameter is starting
                    node = ordered_nodes[b][cur_node_idx]
                    param_types = self.node_types[self.node_type_names.index(node.type)]['parameters']
                    param_dtype = SimpleGraph.get_param_dtype(node_type=node.type, param_type_info=param_types[next_id], use_alpha=self.use_alpha, legacy_flattened=self._legacy_flattened)

                    param_vector_dim[b] = SimpleGraph.get_param_vector_dim(param_dtype=param_dtype)
                    param_tensor_rank[b] = SimpleGraph.get_param_tensor_rank(param_dtype=param_dtype)
                    param_elm_idx[b] = 0

                    next_param_token_idx[b] = (0 if cur_id in self.node_start_tokens else cur_param_token_idx_list[b]) + 1
                    next_param_vector_elm_idx[b] = 0
                    next_param_array_elm_idx[b] = 0
                    next_param_idx[b] = 0 if cur_id in self.node_start_tokens else cur_param_idx_list[b] + 1
                    next_param_node_inds[b] = cur_node_idx

                # current parameter is continuing
                else:
                    param_elm_idx[b] += 1

                    next_param_token_idx[b] = cur_param_token_idx_list[b] + 1
                    next_param_vector_elm_idx[b] = param_elm_idx[b] % param_vector_dim[b]
                    next_param_array_elm_idx[b] = int(math.floor(param_elm_idx[b] / param_vector_dim[b]))
                    next_param_idx[b] = cur_param_idx_list[b]
                    next_param_node_inds[b] = cur_node_idx

                next_param_seq_mask[b] = 0 if cur_id == next_id == self.stop_token else 1

            # update the sequences
            param_id_seq_t[:, param_idx+1] = next_param_id
            param_token_idx_seq_t[:, param_idx+1] = torch.tensor(next_param_token_idx, dtype=torch.long, device=device)
            param_val_seq_t[:, param_idx+1] = next_param_val
            param_vector_elm_idx_seq_t[:, param_idx+1] = torch.tensor(next_param_vector_elm_idx, dtype=torch.long, device=device)
            param_array_elm_idx_seq_t[:, param_idx+1] = torch.tensor(next_param_array_elm_idx, dtype=torch.long, device=device)
            param_idx_seq_t[:, param_idx+1] = torch.tensor(next_param_idx, dtype=torch.long, device=device)
            param_node_inds_t[:, param_idx+1] = torch.tensor(next_param_node_inds, dtype=torch.long, device=device)
            param_seq_mask_t[:, param_idx+1] = torch.tensor(next_param_seq_mask, dtype=torch.int32, device=device)

            return n_finished, next_param_id_list, next_param_token_idx, next_param_idx, next_param_node_inds

        # generate parameters
        param_iter = range(self.max_full_seq_len - 1)
        if progress_bar:
            param_iter = tqdm(param_iter, desc='Generating nodes')

        for param_idx in param_iter:
            n_finished, *step_args = gen_step(param_idx, *step_args)

            # exit if all sequences are finished
            if n_finished == batch_size:
                break

        seq_len = param_idx + 2

        # enforce the stop token at the last position
        param_id_seq_t[:, -1] = self.stop_token
        param_val_seq_t[:, -1] = 0
        param_id_semantic_mask_t[:, -1] = param_id_stop_mask
        param_val_semantic_mask_t[:, -1] = param_val_stop_mask

        if param_id_action_mask_t is not None:
            param_id_action_mask_t[:, -1] = param_id_stop_mask
            param_val_action_mask_t[:, -1] = param_val_stop_mask

        # get parameters for each node
        for b in range(batch_size):
            ordered_nodes[b] = self.get_params(
                ordered_nodes[b], param_id_seq_t[b, :seq_len].cpu(), param_val_seq_t[b, :seq_len].cpu(), precise=False)

        # dictionary of all sequences
        # masks are inverted to indicate valid positions
        trunc_seq = lambda seq_t: seq_t[:, :seq_len].contiguous()
        trunc_mask = lambda mask_t: ~mask_t[:, :seq_len-1]

        all_seqs = {
            'param_node_type_seq': node_type_seq,
            'param_node_depth_seq': node_depth_seq,
            'param_node_seq_mask': node_seq_mask,
            'edge_node_inds': edge_node_inds,
            'param_id_seq': trunc_seq(param_id_seq_t),
            'param_token_idx_seq': trunc_seq(param_token_idx_seq_t),
            'param_val_seq': trunc_seq(param_val_seq_t),
            'param_vector_elm_idx_seq': trunc_seq(param_vector_elm_idx_seq_t),
            'param_array_elm_idx_seq': trunc_seq(param_array_elm_idx_seq_t),
            'param_idx_seq': trunc_seq(param_idx_seq_t),
            'param_node_inds': trunc_seq(param_node_inds_t),
            'param_seq_mask': trunc_seq(param_seq_mask_t),
            'param_id_semantic_mask': trunc_mask(param_id_semantic_mask_t),
            'param_val_semantic_mask': trunc_mask(param_val_semantic_mask_t)
        }

        if param_id_action_mask_t is not None:
            all_seqs.update({
                'param_id_action_mask': trunc_mask(param_id_action_mask_t),
                'param_val_action_mask': trunc_mask(param_val_action_mask_t)
            })

        return ordered_nodes, all_seqs

    def generate_params_beam(self, model, node_samples, node_depth_samples, graph_cond, node_sequencer, edge_cond_type,
                             max_graph_batch_size, use_alpha, devices, has_existing_params=False, existing_node_num=None,
                             beam_width=1, temperature=1, prob_k=5, length_penalty=0, return_k=1):

        raise NotImplementedError('Beam Search is not implemented.')


class Validator:
    def __init__(self, ordered_nodes, param_sequencer):
        self.ordered_nodes = ordered_nodes
        self.node_types = param_sequencer.node_types
        self.node_type_names = param_sequencer.node_type_names
        self.max_num_id_tokens = param_sequencer.max_num_id_tokens
        self.max_seq_len = param_sequencer.max_seq_len
        self.quant_steps = param_sequencer.quant_steps
        self.start_token = param_sequencer.start_token
        self.stop_token = param_sequencer.stop_token
        self.node_sep_token = param_sequencer.node_sep_token
        self.node_start_tokens = param_sequencer.node_start_tokens
        self.use_alpha = param_sequencer.use_alpha
        self._legacy_flattened = param_sequencer._legacy_flattened

        # status for each batch
        self.batch_size = len(ordered_nodes)
        self.param_id_max = [None] * self.batch_size
        self.param_val_max = [None] * self.batch_size
        self.node_index = [None] * self.batch_size
        self.eos = [False] * self.batch_size

    def is_finished(self):
        return sum(self.eos) == self.batch_size

    def update(self, param_node_inds):
        for b in range(len(param_node_inds)):
            self.update_node(b, param_node_inds[b])

    def update_node(self, b, node_idx):
        # check if we should move to the next node or stop
        if self.node_index[b] == node_idx:
            return
        if len(self.ordered_nodes[b]) == node_idx:
            self.param_id_max[b] = None
            self.param_val_max[b] = None
            self.eos[b] = True
            return

        # get node type and parameter types
        node = self.ordered_nodes[b][node_idx]
        node_type = self.node_types[self.node_type_names.index(node.type)]
        param_types = node_type['parameters']
        param_id_max = len(node_type['parameters'])

        # load parameter value limits pertaining to the current node
        self.param_id_max[b] = param_id_max
        self.param_val_max[b] = [None] * param_id_max
        for param_id in range(param_id_max):
            param_dtype = SimpleGraph.get_param_dtype(node_type=node.type,
                                                      param_type_info=param_types[param_id],
                                                      use_alpha=self.use_alpha,
                                                      legacy_flattened=self._legacy_flattened)
            if param_dtype == 'STRING':
                param_info = param_types[param_id]['dtypes'][param_dtype]
                self.param_val_max[b][param_id] = len(list(param_info['value_freq'].keys())) - 1
            elif param_dtype.startswith('INTEGER'):  # No INTEGER*_ARRAY type
                val_min, val_max = get_param_limits(node_func=node_type['func'],
                                                    param_name=param_types[param_id]['name'],
                                                    param_stats=param_types[param_id]['dtypes'],
                                                    use_alpha=self.use_alpha)
                if isinstance(val_min, int) and isinstance(val_max, int):
                    max_limit = int(val_max - val_min)
                else:
                    max_limit = [int(val_max_ - val_min_) for val_min_, val_max_ in zip(val_min, val_max)]
                self.param_val_max[b][param_id] = max_limit
            elif param_dtype.startswith('BOOLEAN'):
                self.param_val_max[b][param_id] = 1

        self.node_index[b] = node_idx

    @staticmethod
    def continue_sample_param(cur_param_id, param_tensor_rank, param_vector_dim, param_elm_idx):
        if param_tensor_rank == 'scalar':
            return cur_param_id + 1
        elif param_tensor_rank == 'vector':
            if param_elm_idx + 1 == param_vector_dim:
                return cur_param_id + 1
            else:
                return -1
        elif param_tensor_rank == 'array':
            if (param_elm_idx + 1) % param_vector_dim == 0:
                return cur_param_id
            else:
                return -1
        else:
            raise RuntimeError(f'Unknown Tensor rank: {param_tensor_rank} for parameter id {cur_param_id}')

    def gen_invalid_id_mask(self, param_id_seq, param_tensor_rank, param_vector_dim, param_elm_idx, param_token_idx_seq, device):
        # make all invalid
        batch_size = len(param_id_seq)
        invalid_param_id_mask = torch.ones(batch_size, self.max_num_id_tokens, dtype=torch.bool, device=device)

        # set feasible range for each parameter
        for b, cur_id in enumerate(param_id_seq):

            # end of sequence, force predicting stop token
            if self.eos[b]:
                invalid_param_id_mask[b, self.stop_token] = False

            # start of a new node, predicting the next parameter id or node_sep_token
            elif cur_id in self.node_start_tokens:
                # cur_id == start_token or node_seq_token
                invalid_param_id_mask[b, :self.param_id_max[b]] = False
                invalid_param_id_mask[b, self.node_sep_token] = False

            # exhausted all parameter tokens for the current node (separator token also counts),
            # force predicting separator token
            elif param_token_idx_seq[b] >= self.max_seq_len - 2:
                invalid_param_id_mask[b, self.node_sep_token] = False

            # continue generating ids for the current node
            else:
                next_id = self.continue_sample_param(cur_id, param_tensor_rank[b], param_vector_dim[b], param_elm_idx[b])
                if next_id == -1:  # we should continue with the current parameter
                    invalid_param_id_mask[b, cur_id] = False
                else:  # keep sampled index monotonic increasing
                    invalid_param_id_mask[b, next_id:self.param_id_max[b]] = False
                    invalid_param_id_mask[b, self.node_sep_token] = False

        return invalid_param_id_mask

    def gen_invalid_val_mask(self, next_param_id, param_id_seq, param_vector_dim, param_elm_idx, device):
        # make all invalid
        batch_size = len(next_param_id)
        invalid_param_val_mask = torch.ones(batch_size, self.quant_steps, dtype=torch.bool, device=device)

        # set feasible range for each parameter
        for b, next_id in enumerate(next_param_id):

            # end of sequence, force predicting zero
            if self.eos[b]:
                invalid_param_val_mask[b, 0] = False

            # start of a new node, force predicting zero
            elif next_id in self.node_start_tokens:
                invalid_param_val_mask[b, 0] = False

            # param index out of bound, force predicting zero
            elif next_id >= self.param_id_max[b]:
                invalid_param_val_mask[b, 0] = False

            # continue generating values for the current node
            else:
                max_limits = self.param_val_max[b][next_id]
                if max_limits is None:
                    invalid_param_val_mask[b] = False
                else:
                    if isinstance(max_limits, int):
                        limit = max_limits
                    else:  # a list
                        cur_id = param_id_seq[b]
                        if next_id != cur_id:  # start a new parameter
                            limit = max_limits[0]
                        else:  # next value in current parameter
                            limit = max_limits[(param_elm_idx[b] + 1) % param_vector_dim[b]]
                    # should not generate values greater then the limit
                    if limit < self.quant_steps - 1:
                        invalid_param_val_mask[b, :limit+1] = False

        return invalid_param_val_mask


class GTParamSequence:
    def __init__(self, node_samples, param_sequencer, device, add_param_stop_token=True):
        param_id_seq = []
        param_token_idx_seq = []
        param_val_seq = []
        param_vector_elm_idx_seq = []
        param_array_elm_idx_seq = []
        param_idx_seq = []
        param_seq_mask = []
        param_node_inds = []

        for nodes in node_samples:
            seq_data = param_sequencer.get_sequences(nodes=nodes, add_param_stop_token=add_param_stop_token)
            param_id_seq.append(seq_data[0])
            param_token_idx_seq.append(seq_data[1])
            param_val_seq.append(seq_data[2])
            param_vector_elm_idx_seq.append(seq_data[3])
            param_array_elm_idx_seq.append(seq_data[4])
            param_idx_seq.append(seq_data[5])
            param_seq_mask.append(seq_data[6])
            param_node_inds.append(seq_data[7])

        self.param_id_seq = torch.stack(param_id_seq, dim=0).to(device)
        self.param_token_idx_seq = torch.stack(param_token_idx_seq, dim=0).to(device)
        self.param_val_seq = torch.stack(param_val_seq, dim=0).to(device)
        self.param_vector_elm_idx_seq = torch.stack(param_vector_elm_idx_seq, dim=0).to(device)
        self.param_array_elm_idx_seq = torch.stack(param_array_elm_idx_seq, dim=0).to(device)
        self.param_idx_seq = torch.stack(param_idx_seq, dim=0).to(device)
        self.param_seq_mask = torch.stack(param_seq_mask, dim=0).to(device)
        self.param_node_inds = torch.stack(param_node_inds, dim=0).to(device)
