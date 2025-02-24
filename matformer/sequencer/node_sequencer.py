# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch

from tqdm import tqdm

from .sequences import sequences_to_matrix
from ..simple_graph import SimpleNode, SimpleOrderedNodes
from ..utils import logits_regularize, nucleus_filtering


class NodeSequencer():
    def __init__(self, max_num_nodes, node_types, use_start_token=True, exclude_node_type_inds=None, max_num_slots=None):
        self.max_num_nodes = max_num_nodes

        self.node_types = node_types
        self.node_type_names = [node_type['name'] for node_type in self.node_types]
        if len(self.node_type_names) != len(self.node_types):
            raise RuntimeError('Number of node type names and node types does not match.')

        self.use_start_token = use_start_token

        self.stop_token = len(self.node_type_names)+1
        if self.use_start_token:
            self.start_token = len(self.node_type_names)
            self.max_seq_len = self.max_num_nodes+2
        else:
            self.start_token = None
            self.max_seq_len = self.max_num_nodes+1

        print('Created tokens in the node sequencer.')
        print(f'Start Token: {self.start_token}')
        print(f'Stop Token: {self.stop_token}')
        print(f'# of excluded node type indices: {len(exclude_node_type_inds) if exclude_node_type_inds is not None else 0}/{len(self.node_type_names)}')

        # Exclude node types
        self.exclude_node_type_inds = exclude_node_type_inds

        # Constraints on the total number of slots
        self.max_num_slots = max_num_slots
        self.node_type_slots = [*(len(node_type['output_names']) + len(node_type['input_names'])
                                for node_type in self.node_types), 0, 0]

        # Avoid generating the same output node multiple times
        self.node_type_is_output = [*(name.startswith('output_') for name in self.node_type_names), False, False]

    def get_nodes(self, node_type_seq, node_depth_seq):
        node_types_inds, node_depths = sequences_to_matrix(sequences=(node_type_seq, node_depth_seq), start_token=self.start_token, stop_token=self.stop_token, num_cols=1)
        node_types_inds = node_types_inds.view(-1).tolist()
        node_depths = node_depths.view(-1)

        nodes = []
        node_type_counts = {}
        for node_type_idx in node_types_inds:
            node_type_name = self.node_type_names[node_type_idx]
            if node_type_name not in node_type_counts:
                node_type_counts[node_type_name] = 1
            else:
                node_type_counts[node_type_name] += 1
            nodes.append(SimpleNode(name=f'{node_type_name}_{node_type_counts[node_type_name]-1}', type=node_type_name))

        return nodes, node_depths

    def get_sequences(self, ordered_nodes, add_stop_token=True):

        if len(ordered_nodes) > self.max_num_nodes:
            raise RuntimeError(f'Too many nodes.')

        node_type_seq = torch.full((self.max_seq_len,), self.stop_token, dtype=torch.long)
        node_idx_seq = torch.zeros_like(node_type_seq)
        node_depth_seq = torch.zeros_like(node_type_seq)

        seq_idx = 0

        # node sequence start
        if self.use_start_token:
            node_type_seq[0] = self.start_token
            node_idx_seq[0] = 0
            node_depth_seq[0] = 0
            seq_idx += 1

        for node, node_depth in ordered_nodes.items():
            node_type_seq[seq_idx] = self.node_type_names.index(node.type)
            node_idx_seq[seq_idx] = seq_idx
            node_depth_seq[seq_idx] = node_depth
            seq_idx += 1

        if add_stop_token:
            # node sequence end
            node_type_seq[seq_idx] = self.stop_token
            node_idx_seq[seq_idx] = seq_idx
            node_depth_seq[seq_idx] = 0
            seq_idx += 1

        node_seq_len = seq_idx

        # node sequence mask
        node_seq_mask = torch.zeros_like(node_type_seq, dtype=torch.int32)
        node_seq_mask[:node_seq_len] = 1

        return tuple(seq[:node_seq_len] for seq in (node_type_seq, node_idx_seq, node_depth_seq, node_seq_mask))

    def generate_nodes(self, node_decoder, cond, deterministic=False, max_gen_nodes=None, temperature=1, prob_k=0,
                       nucleus_top_p=None, node_order=None, semantic_validate=False, progress_bar=False):
        # check node order option
        reverse_bfs = ['reverse_breadth_first', 'reverse_breadth_first_no_auxiliary_nodes']
        reverse_bfs_flipped = ['reverse_breadth_first_flipped', 'reverse_breadth_first_flipped_no_auxiliary_nodes']

        if node_order is not None and node_order not in reverse_bfs + reverse_bfs_flipped:
            raise RuntimeError(f'Invalid node order: {node_order}')
        if semantic_validate and node_order is None:
            raise RuntimeError('Semantic validation requires node order.')

        # must generate from a start token
        if not self.use_start_token:
            raise RuntimeError('Cannot generate nodes using sequences without start token.')

        ordered_nodes = []
        batch_size, device = len(cond), cond.device

        # node sequence containers
        max_seq_len = max_gen_nodes or self.max_seq_len
        node_type_seq_t = torch.full((batch_size, max_seq_len), self.stop_token, dtype=torch.long, device=device)
        node_idx_seq_t = torch.zeros_like(node_type_seq_t)
        node_depth_seq_t = torch.zeros_like(node_type_seq_t)
        node_seq_mask_t = torch.zeros_like(node_type_seq_t, dtype=torch.int32)

        node_type_seq_t[:, 0] = self.start_token
        node_idx_seq_t[:, 0] = 0
        node_depth_seq_t[:, 0] = 0
        node_seq_mask_t[:, 0] = 1

        # node sequence semantic masks
        node_type_semantic_mask_t = torch.zeros(batch_size, max_seq_len - 1, len(self.node_type_names) + 2, dtype=torch.bool, device=device)
        node_depth_semantic_mask_t = torch.zeros(batch_size, max_seq_len - 1, self.max_num_nodes, dtype=torch.bool, device=device)

        node_type_stop_mask = torch.ones(len(self.node_type_names) + 2, dtype=torch.bool, device=device)
        node_type_stop_mask[self.stop_token] = False
        node_depth_stop_mask = torch.ones(self.max_num_nodes, dtype=torch.bool, device=device)
        node_depth_stop_mask[0] = False

        # node sequence action masks
        node_type_action_mask_t, node_depth_action_mask_t = None, None

        if nucleus_top_p is not None:
            node_type_action_mask_t = torch.zeros_like(node_type_semantic_mask_t)
            node_depth_action_mask_t = torch.zeros_like(node_depth_semantic_mask_t)

        # excluded node types
        exclude_node_type_inds = None

        if self.exclude_node_type_inds:
            exclude_node_type_inds = torch.tensor(self.exclude_node_type_inds, dtype=torch.long, device=device)

        # number of slots for each node type
        max_num_slots, node_type_slots = self.max_num_slots, None
        num_slots = torch.zeros(batch_size, dtype=torch.long, device=device)
        node_type_slots = torch.tensor(self.node_type_slots, dtype=torch.long, device=device)

        # output nodes should only be generated once
        node_type_is_output = torch.tensor(self.node_type_is_output, dtype=torch.bool, device=device)
        output_node_generated = torch.zeros(batch_size, len(self.node_type_names) + 2, dtype=torch.bool, device=device)

        # helper arrays
        depth_indices = torch.arange(self.max_num_nodes, dtype=torch.long, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        seq_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # each step of node generation
        def gen_step(node_idx, seq_finished, num_slots):

            # obtain next token logits from the model
            node_type_seq = node_type_seq_t[:, :node_idx+1]
            node_idx_seq = node_idx_seq_t[:, :node_idx+1]
            node_depth_seq = node_depth_seq_t[:, :node_idx+1]
            node_seq_mask = node_seq_mask_t[:, :node_idx+1]

            all_logits = node_decoder(
                sequences=(node_type_seq, node_idx_seq, node_depth_seq),
                cond=cond,
                attention_mask=node_seq_mask)[0]
            node_type_logits, node_depth_logits = [logits[:, -1] for logits in all_logits]

            # semantic validity
            node_type_mask = node_type_semantic_mask_t[:, node_idx]
            node_depth_mask = node_depth_semantic_mask_t[:, node_idx]

            # mask out the start token and already generated output nodes
            node_type_mask[:, self.start_token] = True
            node_type_mask |= output_node_generated

            # mask out excluded node types
            if exclude_node_type_inds is not None:
                node_type_mask[:, exclude_node_type_inds] = True

            # enforce additional constraints on the number of slots
            if max_num_slots is not None:
                node_type_mask |= num_slots[:, None] + node_type_slots > max_num_slots

            # extra semantic validation to ensure node depth order
            if semantic_validate and node_idx > 0:
                # next node depth should be either +0 or +1 or -1 (if reversed)
                last_depth = node_depth_seq[:, -1:]
                # increased numbering
                if node_order in reverse_bfs:
                    node_depth_mask |= (depth_indices < last_depth) | (depth_indices > last_depth + 1)
                # decreased numbering
                else:
                    node_depth_mask |= (depth_indices > last_depth) | (depth_indices < last_depth - 1)

                # if using reverse order, we make sure a stop token is predicted
                # only when the node depth has decreased to zero.
                if node_order in reverse_bfs_flipped:
                    node_type_mask[:, self.stop_token] |= last_depth.squeeze(1) != 0

            # sample the next node type id token
            node_type_mask[seq_finished] = node_type_stop_mask
            node_type_logits[node_type_mask] = -1e9

            if nucleus_top_p is not None:
                node_type_logits, node_type_mask_nf = nucleus_filtering(logits=node_type_logits, top_p=nucleus_top_p, return_mask=True)
                node_type_action_mask_t[:, node_idx] = node_type_mask_nf
            if prob_k > 0:
                node_type_logits, node_type_mask_tk = logits_regularize(logits=node_type_logits, temperature=temperature, top_k=prob_k, return_mask=True)
                node_type_action_mask_t[:, node_idx] |= node_type_mask_tk

            if deterministic:
                next_type = torch.argmax(node_type_logits, dim=-1)
            else:
                next_type = torch.multinomial(node_type_logits.softmax(dim=-1), num_samples=1).squeeze(-1)

            # sample the next node depth token
            node_depth_mask[seq_finished | (next_type == self.stop_token)] = node_depth_stop_mask
            node_depth_logits[node_depth_mask] = -1e9

            if nucleus_top_p is not None:
                node_depth_logits, node_depth_mask_nf = nucleus_filtering(logits=node_depth_logits, top_p=nucleus_top_p, return_mask=True)
                node_depth_action_mask_t[:, node_idx] = node_depth_mask_nf
            if prob_k > 0:
                node_depth_logits, node_depth_mask_tk = logits_regularize(logits=node_depth_logits, temperature=temperature, top_k=prob_k, return_mask=True)
                node_depth_action_mask_t[:, node_idx] |= node_depth_mask_tk

            if deterministic:
                next_depth = torch.argmax(node_depth_logits, dim=-1)
            else:
                next_depth = torch.multinomial(node_depth_logits.softmax(dim=-1), num_samples=1).squeeze(-1)

            # update the node sequence
            next_type_masked = torch.where(seq_finished, self.stop_token, next_type)
            node_type_seq_t[:, node_idx+1] = next_type_masked
            node_idx_seq_t[:, node_idx+1] = torch.where(seq_finished, 0, node_idx+1)
            node_depth_seq_t[:, node_idx+1] = torch.where(seq_finished, 0, next_depth)
            node_seq_mask_t[:, node_idx+1] = ~seq_finished

            seq_finished |= next_type == self.stop_token

            # update the number of slots and generated output nodes
            num_slots += node_type_slots[next_type_masked]
            output_node_generated[batch_indices, next_type_masked] |= node_type_is_output[next_type_masked]

            return seq_finished, num_slots

        # iteratively generate nodes
        node_iter = range(max_seq_len - 1)
        if progress_bar:
            node_iter = tqdm(node_iter, desc='Generating nodes')

        for node_idx in node_iter:
            seq_finished, num_slots = gen_step(node_idx, seq_finished, num_slots)

            # stop if all sequences are finished
            if seq_finished.all().item():
                break

        seq_len = node_idx + 2

        # enforce the stop token at the last position
        node_type_seq_t[:, -1] = self.stop_token
        node_depth_seq_t[:, -1] = 0
        node_type_semantic_mask_t[:, -1] = node_type_stop_mask
        node_depth_semantic_mask_t[:, -1] = node_depth_stop_mask

        if node_type_action_mask_t is not None:
            node_type_action_mask_t[:, -1] = node_type_stop_mask
            node_depth_action_mask_t[:, -1] = node_depth_stop_mask

        # convert sequences to nodes
        for b in range(batch_size):
            nodes, node_depths = self.get_nodes(node_type_seq=node_type_seq_t[b, :seq_len].cpu(), node_depth_seq=node_depth_seq_t[b, :seq_len].cpu())
            ordered_nodes.append(SimpleOrderedNodes(nodes, node_depths))

        # dictionary of all sequences
        # masks are inverted to indicate valid positions
        trunc_seq = lambda seq_t: seq_t[:, :seq_len].contiguous()
        trunc_mask = lambda mask_t: mask_t[:, :seq_len-1].contiguous().logical_not_()

        all_seqs = {
            'node_type_seq': trunc_seq(node_type_seq_t),
            'node_idx_seq': trunc_seq(node_idx_seq_t),
            'node_depth_seq': trunc_seq(node_depth_seq_t),
            'node_seq_mask': trunc_seq(node_seq_mask_t),
            'node_type_semantic_mask': trunc_mask(node_type_semantic_mask_t),
            'node_depth_semantic_mask': trunc_mask(node_depth_semantic_mask_t)
        }

        if node_type_action_mask_t is not None:
            all_seqs.update({
                'node_type_action_mask': trunc_mask(node_type_action_mask_t),
                'node_depth_action_mask': trunc_mask(node_depth_action_mask_t)
            })

        return ordered_nodes, all_seqs
