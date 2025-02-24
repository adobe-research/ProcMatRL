# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import warnings
import copy
from itertools import chain
from collections import Counter

from tqdm import tqdm
import torch
import numpy as np
import networkx as nx

from .sequences import sequences_to_matrix
from ..utils import logits_regularize, nucleus_filtering, stack_tensor_lists

class SlotSequencer():
    def __init__(self, max_num_nodes, node_types, max_num_slots, max_num_output_slots, max_num_parents):
        self.max_num_nodes = max_num_nodes
        self.node_types = node_types
        self.node_type_names = [node_type['name'] for node_type in self.node_types]
        if len(self.node_type_names) != len(self.node_types):
            raise RuntimeError('Number of node type names and node types does not match.')

        self.max_num_slots = max_num_slots
        self.max_num_output_slots = max_num_output_slots
        self.max_num_parents = max_num_parents

        # print('WARNINIG: comment in below once re-trained with correct max_num_parents and max_num_output_slots')
        # # correct: max_num_output_slots=14, max_num_parents=21 (with aux. nodes)
        for node_type in node_types:
            if len(node_type['input_names']) > self.max_num_parents:
                raise RuntimeError(f'Too many input slots for node type {node_type["name"]}: {len(node_type["input_names"])} > {self.max_num_parents}.')
            if len(node_type['output_names']) > self.max_num_output_slots:
                raise RuntimeError(f'Too many output slots for node type {node_type["name"]}: {len(node_type["output_names"])} > {self.max_num_output_slots}.')

        self.start_token = len(self.node_type_names)
        self.stop_token = len(self.node_type_names) + 1

        print('Created tokens in the slot sequencer.')
        print(f'Start Token: {self.start_token}')
        print(f'Stop Token: {self.stop_token}')

        self.max_seq_len = self.max_num_slots + 2

    def get_sequences(self, ordered_nodes):

        if len(ordered_nodes) > self.max_num_nodes:
            raise RuntimeError('Too many nodes.')

        slot_node_type_seq = torch.full((self.max_num_slots+2,), self.stop_token, dtype=torch.long)
        slot_node_idx_seq = torch.zeros_like(slot_node_type_seq)
        slot_node_depth_seq = torch.zeros_like(slot_node_type_seq)
        slot_id_seq = torch.zeros_like(slot_node_type_seq)
        slot_idx_seq = torch.zeros_like(slot_node_type_seq)

        # always have the start token as first element and the stop token as second element
        # (otherwise the encoding is different for different sequences, since the node index of the stop token changes)

        slot_idx = 0

        # start token
        slot_node_type_seq[slot_idx] = self.start_token
        slot_node_idx_seq[slot_idx] = 0
        slot_node_depth_seq[slot_idx] = 0
        slot_id_seq[slot_idx] = self.max_num_output_slots + self.max_num_parents
        slot_idx_seq[slot_idx] = slot_idx
        slot_idx += 1

        # stop token
        slot_node_type_seq[slot_idx] = self.stop_token
        slot_node_idx_seq[slot_idx] = 0
        slot_node_depth_seq[slot_idx] = 0
        slot_id_seq[slot_idx] = self.max_num_output_slots + self.max_num_parents
        slot_idx_seq[slot_idx] = slot_idx
        slot_idx += 1

        for node_idx, (node, node_depth) in enumerate(ordered_nodes.items()):
            node_type_idx = self.node_type_names.index(node.type)
            node_type = self.node_types[node_type_idx]
            node_slot_count = len(node_type['output_names']) + len(node_type['input_names'])

            # was slot_idx + node_slot_count >= self.max_num_slots + 2
            if slot_idx + node_slot_count > self.max_num_slots + 2:  # +2 because of the start and stop tokens
                raise RuntimeError('Too many slots.')

            # per-node sequences
            slot_node_type_seq[slot_idx:slot_idx+node_slot_count] = node_type_idx
            slot_node_idx_seq[slot_idx:slot_idx+node_slot_count] = node_idx
            slot_node_depth_seq[slot_idx:slot_idx+node_slot_count] = node_depth

            # per-slot sequences
            for output_slot_idx in range(len(node_type['output_names'])):
                slot_id_seq[slot_idx] = output_slot_idx
                slot_idx_seq[slot_idx] = slot_idx
                slot_idx += 1
            for input_slot_idx in range(len(node_type['input_names'])):
                slot_id_seq[slot_idx] = self.max_num_output_slots + input_slot_idx
                slot_idx_seq[slot_idx] = slot_idx
                slot_idx += 1

        slot_seq_len = slot_idx

        slot_seq_mask = torch.zeros_like(slot_node_type_seq, dtype=torch.int32)
        slot_seq_mask[:slot_seq_len] = 1

        return tuple(seq[:slot_seq_len] for seq in (slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_id_seq, slot_idx_seq, slot_seq_mask))

    def get_type(self, type_idx, slot_id):
        if slot_id >= self.max_num_output_slots + self.max_num_parents or type_idx >= len(self.node_type_names):
            node_type, slot_name, is_input_slot = None, None, None
        else:
            node_type = self.node_types[type_idx]
            is_input_slot = slot_id >= self.max_num_output_slots
            if is_input_slot:
                if isinstance(node_type['input_names'], list):
                    slot_name = node_type['input_names'][slot_id-self.max_num_output_slots]
                else:
                    slot_name = list(node_type['input_names'].items())[slot_id-self.max_num_output_slots][0]
            else:
                slot_name = list(node_type['output_names'].items())[slot_id][0]

        return node_type, slot_name, is_input_slot


class EdgeSequencer():
    def __init__(self, max_num_edges):

        self.max_num_edges = max_num_edges

        self.max_seq_len = self.max_num_edges * 2 + 2

    @staticmethod
    def get_edge_node_inds(ordered_nodes):
        node_map = {node.name: i for i, node in enumerate(ordered_nodes)}
        if len(node_map) != len(ordered_nodes):
            raise RuntimeError('Node name mapping might contain nodes with duplicate names')
        edge_node_inds = []
        for node_idx, node in enumerate(ordered_nodes):
            for parent, _ in node.parents:
                if parent is not None:
                    parent_idx = node_map[parent.name]
                    edge_node_inds.append([parent_idx, node_idx])

        return edge_node_inds

    def get_edges(self, nodes, slot_node_type_sequence, slot_node_idx_sequence, slot_id_sequence, edge_sequence, slot_sequencer):

        if any(slot_id > slot_sequencer.max_num_output_slots+slot_sequencer.max_num_parents for slot_id in slot_id_sequence):
            raise RuntimeError('Invalid slot id.')
        if any(slot_node_idx > 0 and slot_node_idx >= slot_sequencer.max_num_nodes for slot_node_idx in slot_node_idx_sequence):
            raise RuntimeError('Invalid slot node index.')

        stop_token_indices = (slot_node_type_sequence[2:] == slot_sequencer.stop_token).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() == 0:
            warnings.warn('Slot node type sequence does not contain a stop token.')
            slot_node_type_last_index = len(slot_node_type_sequence)
        else:
            slot_node_type_last_index = stop_token_indices[0] + 2

        start_token_index = 0  # first index in slot sequence is start token
        stop_token_index = 1  # second index in slot sequence is stop token

        edges = sequences_to_matrix(sequences=(edge_sequence,), start_token=start_token_index, stop_token=stop_token_index, num_cols=2)[0]
        edges = edges.tolist()

        # add edges to nodes (and remove any existing edges)
        for node in nodes:
            node.parents = []
            node.children = set()
        warning_strings = []
        for edge in edges:

            if (edge[0] in (start_token_index, stop_token_index) or edge[0] >= slot_node_type_last_index
                or edge[1] == stop_token_index or edge[1] >= slot_node_type_last_index):
                warning_strings.append('WARNING: semantic validity: skipping edge with slot indices that are out of range.')
                continue

            if edge[1] == start_token_index:
                warning_strings.append('WARNING: semantic validity: skipping edge without valid output slot options.')
                continue

            from_node_idx = slot_node_idx_sequence[edge[0]].item()
            to_node_idx = slot_node_idx_sequence[edge[1]].item()
            from_node_slot_id = slot_id_sequence[edge[0]].item()
            to_node_slot_id = slot_id_sequence[edge[1]].item()

            if any(idx < 0 or idx >= len(nodes) for idx in [from_node_idx, to_node_idx]):
                warning_strings.append('WARNING: semantic validity: skipping edge with node indices that are out of range.')
                continue

            from_node = nodes[from_node_idx]
            to_node = nodes[to_node_idx]

            if from_node_slot_id < slot_sequencer.max_num_output_slots or from_node_slot_id >= slot_sequencer.max_num_output_slots+slot_sequencer.max_num_parents:
                warning_strings.append('WARNING: semantic validity: skipping edge that does not start from an input slot.')
                continue
            if to_node_slot_id >= slot_sequencer.max_num_output_slots:
                warning_strings.append('WARNING: semantic validity: skipping edge that does not end in an output slot.')
                continue

            from_node_input_slot_idx = from_node_slot_id - slot_sequencer.max_num_output_slots
            to_node_output_slot_idx = to_node_slot_id

            if from_node_input_slot_idx >= len(slot_sequencer.node_types[slot_sequencer.node_type_names.index(from_node.type)]['input_names']):
                warning_strings.append('WARNING: semantic validity: skipping edge that starts from an input slot index that is out of bounds for the given node type.')
                continue
            if to_node_output_slot_idx >= len(slot_sequencer.node_types[slot_sequencer.node_type_names.index(to_node.type)]['output_names']):
                warning_strings.append('WARNING: semantic validity: skipping edge that starts from an output slot index that is out of bounds for the given node type.')
                continue

            if len(from_node.parents) >= from_node_input_slot_idx+1 and from_node.parents[from_node_input_slot_idx][0] is not None:
                warning_strings.append('WARNING: semantic validity: skipping duplicate edge for the same input slot.')
                continue

            # _, to_slot_name, _ = slot_sequencer.get_type(
            #     type_idx=slot_sequencer.node_type_names.index(to_node.type),
            #     slot_id=to_node_slot_id)

            # if to_node.type.startswith('output_') and to_slot_name == 'output' and from_node.type != 'output_root':
            #     warning_strings.append('WARNING: semantic validity: skipping edge between the "output" slot of an output node and the input slot of a node that is not the output root.')
            #     continue

            # edge goes to_node (parent) -> from_node (child), since edges are created in reverse order
            from_node_descendants = list(chain(*([[from_node]] + from_node.get_descendants(num_descendants=10))))
            if to_node in from_node_descendants:
                warning_strings.append('WARNING: semantic validity: skipping an edge that would create a cycle.')
                continue

            if len(from_node.parents) < from_node_input_slot_idx+1:
                pad_count = (from_node_input_slot_idx+1 - len(from_node.parents))
                from_node.parents = from_node.parents + [(None, None)] * pad_count

            from_node.parents[from_node_input_slot_idx] = (to_node, to_node_output_slot_idx)

            to_node.children.add(from_node)

        warning_count = dict(Counter(warning_strings))
        for warning_str, count in warning_count.items():
            print(f'{warning_str} (x {count})')

        return nodes

    def get_sequences(self, ordered_nodes, slot_sequencer, add_edge_stop_token=True):
        if len(ordered_nodes) > slot_sequencer.max_num_nodes:
            raise RuntimeError(f'Too many nodes.')

        slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_id_seq, slot_idx_seq, slot_seq_mask \
            = slot_sequencer.get_sequences(ordered_nodes)

        # map from node to node index
        node_idx_map = {node.name: node_idx for node_idx, node in enumerate(ordered_nodes)}
        if len(node_idx_map) != len(ordered_nodes):
            raise RuntimeError('Duplicate node names exist.')

        # map from node index and slot id to slot index
        slot_node_type_list = slot_node_type_seq.tolist()
        slot_idx_list = slot_idx_seq.tolist()
        slot_node_idx_list = slot_node_idx_seq.tolist()
        slot_id_list = slot_id_seq.tolist()

        slot_idx_map = {(slot_node_idx, slot_id): slot_idx
                        for slot_node_type, slot_idx, slot_node_idx, slot_id in zip(slot_node_type_list, slot_idx_list, slot_node_idx_list, slot_id_list)
                        if slot_node_type not in [slot_sequencer.start_token, slot_sequencer.stop_token]}

        # edge sequence start
        start_token_index = 0  # first index in slot sequence is start token
        stop_token_index = 1  # second index in slot sequence is stop token

        edge_seq = torch.full((self.max_seq_len,), stop_token_index, dtype=torch.long)
        edge_idx_seq = torch.zeros_like(edge_seq)
        edge_elm_seq = torch.full_like(edge_seq, 2)

        edge_seq[0] = start_token_index

        # edge sequence (starting from the output root node, iterate through nodes in breadth-first order and through input slots of the node)
        seq_idx = 1
        for node in ordered_nodes:
            node_idx = node_idx_map[node.name]

            for input_slot_idx, (parent_node, parent_output_slot_id) in enumerate(node.parents):
                # do not create edges to parent_end nodes, these are not necessary
                if parent_node is not None and parent_node.type != 'parent_end':
                    parent_node_idx = node_idx_map[parent_node.name]
                    input_slot_id = slot_sequencer.max_num_output_slots + input_slot_idx

                    if seq_idx + 2 >= self.max_seq_len:
                        raise RuntimeError(f'Too many edges.')

                    edge_seq[seq_idx] = slot_idx_map[(node_idx, input_slot_id)]
                    edge_seq[seq_idx + 1] = slot_idx_map[(parent_node_idx, parent_output_slot_id)]
                    edge_idx_seq[seq_idx] = seq_idx
                    edge_idx_seq[seq_idx + 1] = seq_idx + 1
                    edge_elm_seq[seq_idx] = 0
                    edge_elm_seq[seq_idx + 1] = 1

                    seq_idx += 2

        # edge sequence end
        if add_edge_stop_token:
            edge_seq[seq_idx] = stop_token_index
            edge_idx_seq[seq_idx] = seq_idx
            edge_elm_seq[seq_idx] = 2
            seq_idx += 1

        edge_seq_mask = torch.zeros_like(edge_seq, dtype=torch.int32)
        edge_seq_mask[:seq_idx] = 1

        slot_seqs = slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_id_seq, slot_idx_seq, slot_seq_mask
        edge_seqs = edge_seq, edge_idx_seq, edge_elm_seq, edge_seq_mask
        edge_seqs = tuple(seq[:seq_idx] for seq in edge_seqs)
        return *slot_seqs, *edge_seqs

    def generate_edges(self, edge_decoder, cond, ordered_nodes, slot_sequencer, deterministic=False, temperature=1, prob_k=0,
                       nucleus_top_p=None, semantic_validate=False, progress_bar=False):
        ordered_nodes = copy.deepcopy(ordered_nodes) # copy to avoid modifying the original nodes

        # check if batch sizes match
        batch_size, device = len(cond), cond.device
        if len(ordered_nodes) != batch_size:
            raise RuntimeError('Number of conditions does not match number of partial graphs.')

        # get slot sequences from input partial graphs
        slot_sequences = [slot_sequencer.get_sequences(on) for on in ordered_nodes]
        slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_id_seq, slot_idx_seq, slot_seq_mask \
            = [stack_tensor_lists(seqs, pad_value=slot_sequencer.stop_token if not i else 0).to(device)
               for i, seqs in enumerate(zip(*slot_sequences))]

        # compute node slot embeddings
        slot_embed_seq = edge_decoder.compute_node_embeddings(
            node_sequences=(slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_idx_seq, slot_id_seq),
            cond=cond, node_attention_mask=slot_seq_mask)

        # start of the edge sequences
        start_token_index = 0 # first index in slot sequence is start token
        stop_token_index = 1 # second index in slot sequence is stop token

        edge_seq_t = torch.full((batch_size, self.max_seq_len), stop_token_index, dtype=torch.long, device=device)
        edge_idx_seq_t = torch.zeros_like(edge_seq_t)
        edge_elm_seq_t = torch.full_like(edge_seq_t, 2)
        edge_seq_mask_t = torch.zeros_like(edge_seq_t, dtype=torch.int32)

        edge_seq_t[:, 0] = start_token_index
        edge_seq_mask_t[:, 0] = 1

        # edge sequence semantic mask
        num_slots = slot_node_type_seq.shape[-1]
        edge_semantic_mask_t = torch.zeros(batch_size, self.max_seq_len - 1, num_slots, dtype=torch.bool, device=device)

        edge_stop_mask = torch.ones(num_slots, dtype=torch.bool, device=device)
        edge_stop_mask[stop_token_index] = False

        # edge sequence action mask
        edge_action_mask_t = torch.zeros_like(edge_semantic_mask_t) if nucleus_top_p is not None else None

        # for semantic validity: get invalid slots for edge start and end
        # edges cannot start at padded slots
        invalid_start_slots = slot_node_type_seq >= len(slot_sequencer.node_types)
        # edges cannot start at output slots
        invalid_start_slots |= slot_id_seq < slot_sequencer.max_num_output_slots
        # edges cannot start at padding of the slots sequence
        invalid_start_slots |= slot_id_seq >= slot_sequencer.max_num_output_slots + slot_sequencer.max_num_parents
        # a stop token can always be generated as a start slot
        invalid_start_slots[:, stop_token_index] = False

        # edges cannot end at padded slots
        invalid_end_slots = slot_node_type_seq >= len(slot_sequencer.node_types)
        # edges cannot end at input slots or start/end tokens
        invalid_end_slots |= slot_id_seq >= slot_sequencer.max_num_output_slots

        # for semantic validity: get masks for various special slots
        output_node_output_slot_mask = np.zeros(slot_node_type_seq.shape, dtype=bool) # output slots of output_* nodes
        child_end_output_slot_mask = np.zeros_like(output_node_output_slot_mask) # output slots connecting to child_end nodes
        parent_end_output_slot_mask = np.zeros_like(output_node_output_slot_mask) # output slots of parent_end nodes

        slot_node_type_list = slot_node_type_seq.tolist()
        slot_id_list = slot_id_seq.tolist()

        for b in range(batch_size):
            for slot_idx, (node_type_idx, slot_id) in enumerate(zip(slot_node_type_list[b], slot_id_list[b])):
                node_type, slot_name, is_input_slot = slot_sequencer.get_type(type_idx=node_type_idx, slot_id=slot_id)

                if node_type is not None:
                    if node_type['name'].startswith('output_') and node_type['name'] != 'output_root' and slot_name == 'output' and not is_input_slot:
                        output_node_output_slot_mask[b, slot_idx] = True

                    if slot_name == 'child_end' and not is_input_slot:
                        child_end_output_slot_mask[b, slot_idx] = True
                        raise RuntimeError('Can\'t reach here')

                    if node_type['name'] == 'parent_end' and slot_name =='output' and not is_input_slot:
                        parent_end_output_slot_mask[b, slot_idx] = True

        output_node_output_slot_mask = torch.from_numpy(output_node_output_slot_mask).to(device, non_blocking=True)
        child_end_output_slot_mask = torch.from_numpy(child_end_output_slot_mask).to(device, non_blocking=True)
        parent_end_output_slot_mask = torch.from_numpy(parent_end_output_slot_mask).to(device, non_blocking=True)

        # extra cyclic validity
        if semantic_validate:
            validator = CyclicValidator(slot_node_idx_seq.tolist())
            for b in range(batch_size):
                validator.add_graph(len(ordered_nodes[b]))

        # helper arrays
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        seq_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # each step of edge generation
        def gen_step(edge_idx, seq_finished):

            # obtain next token logits from the model
            edge_seq = edge_seq_t[:, :edge_idx+1]
            edge_idx_seq = edge_idx_seq_t[:, :edge_idx+1]
            edge_elm_seq = edge_elm_seq_t[:, :edge_idx+1]
            edge_seq_mask = edge_seq_mask_t[:, :edge_idx+1]

            edge_logits = edge_decoder(
                node_sequences=slot_embed_seq,
                edge_sequences=(edge_seq, edge_idx_seq, edge_elm_seq),
                cond=cond,
                edge_attention_mask=edge_seq_mask,
                node_attention_mask=slot_seq_mask)[0]
            edge_logits = edge_logits[:, -1]

            # semantic validity
            edge_mask = edge_semantic_mask_t[:, edge_idx]

            if edge_idx % 2 == 0: # currently generating edge start slot
                edge_mask.copy_(invalid_start_slots)

            else: # currently generating edge end slot
                edge_mask.copy_(invalid_end_slots)

                # get corresponding start slot indices
                start_slot_indices = torch.where(seq_finished, -1, edge_seq[:, -1]).tolist()

                # semantic validity that depends on the choice of start slot (skip finished sequences)
                for b, start_slot_idx in enumerate(start_slot_indices):
                    if start_slot_idx < 0:
                        continue

                    # look up the node type and slot name of the start slot
                    start_node_type, start_slot_name, _ = slot_sequencer.get_type(
                        type_idx=slot_node_type_list[b][start_slot_idx],
                        slot_id=slot_id_list[b][start_slot_idx])

                    # 'parent_end' input slots can only connect to the output slot of 'parent_end' nodes
                    # and no other input slots can connect to the output slot of 'parent_end' nodes
                    if start_slot_name == 'parent_end':
                        edge_mask[b] |= ~parent_end_output_slot_mask[b]
                        raise RuntimeError('Can\'t reach here')
                    else:
                        edge_mask[b] |= parent_end_output_slot_mask[b]

                    # input slots of 'child_end' nodes can only connect to 'child_end' output slots
                    # and no other input slots can connect to 'child_end' output slots
                    if start_node_type['name'] == 'child_end':
                        edge_mask[b] |= ~child_end_output_slot_mask[b]
                        raise RuntimeError('Can\'t reach here')
                    else:
                        edge_mask[b] |= child_end_output_slot_mask[b]

                    # input slots of 'output_root' nodes can only connect to 'output' slot of 'output_*' nodes
                    # and no other input slots can connect to the 'output' slot of output_* nodes
                    if start_node_type['name'] == 'output_root':
                        edge_mask[b] |= ~output_node_output_slot_mask[b]
                    else:
                        edge_mask[b] |= output_node_output_slot_mask[b]

                    # set slots that would create cycles to -1e9
                    if semantic_validate:
                        reachable_nodes = torch.tensor(validator.get_reachable_nodes(b, start_slot_idx), device=device)
                        cyclic_slot_mask = (slot_node_idx_seq[b] == reachable_nodes.unsqueeze(dim=1)).any(dim=0)
                        edge_mask[b] |= cyclic_slot_mask

            # sequence finished, only the stop token is valid
            edge_mask[seq_finished] = edge_stop_mask

            # mask out invalid actions
            edge_logits[edge_mask] = -1e9

            # sample the next edge token
            if nucleus_top_p is not None:
                edge_logits, edge_action_mask_nf = nucleus_filtering(logits=edge_logits, top_p=nucleus_top_p, return_mask=True)
                edge_action_mask_t[:, edge_idx] = edge_action_mask_nf
            if prob_k > 0:
                edge_logits, edge_action_mask_tk = logits_regularize(logits=edge_logits, temperature=temperature, top_k=prob_k, return_mask=True)
                edge_action_mask_t[:, edge_idx] |= edge_action_mask_tk

            if deterministic:
                next_edge = torch.argmax(edge_logits, dim=-1)
            else:
                edge_probs = torch.softmax(edge_logits, dim=-1)
                next_edge = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)

            # update the edge sequence
            edge_seq_t[:, edge_idx+1] = torch.where(seq_finished, stop_token_index, next_edge)
            edge_idx_seq_t[:, edge_idx+1] = torch.where(seq_finished, 0, edge_idx + 1)
            edge_elm_seq_t[:, edge_idx+1] = torch.where(seq_finished, 2, edge_idx % 2)
            edge_seq_mask_t[:, edge_idx+1] = ~seq_finished

            seq_finished |= next_edge == stop_token_index

            # semantic validity: mark input slots that already have an edge as invalid (an input slot can only have a single edge)
            if edge_idx % 2 == 0: # currently generating edge start slot
                invalid_start_slots[batch_indices, next_edge] = True

            # update DAG when a full edge is generated
            if semantic_validate and edge_idx % 2 == 1:
                last_edges = torch.where(seq_finished.unsqueeze(1), -1, edge_seq_t[:, edge_idx:edge_idx+2]).tolist()

                for b, (from_slot_idx, to_slot_idx) in enumerate(last_edges):
                    if from_slot_idx >= 0 and to_slot_idx >= 0:
                        validator.add_edge(b, from_slot_idx, to_slot_idx)

            return seq_finished

        # iteratively expand the edge sequences
        edge_iter = range(self.max_seq_len - 1)
        if progress_bar:
            edge_iter = tqdm(edge_iter, desc='Generating nodes')

        for edge_idx in edge_iter:
            seq_finished = gen_step(edge_idx, seq_finished)

            # stop if all sequences are finished
            if seq_finished.all().item():
                break

        seq_len = edge_idx + 2

        # enforce the stop token at the last position
        edge_seq_t[:, -1] = stop_token_index
        edge_semantic_mask_t[:, -1] = edge_stop_mask

        if edge_action_mask_t is not None:
            edge_action_mask_t[:, -1] = edge_stop_mask

        # convert sequences to edges and update the partial graphs
        for b in range(batch_size):
            ordered_nodes[b] = self.get_edges(
                nodes=ordered_nodes[b],
                slot_node_type_sequence=slot_sequences[b][0],
                slot_node_idx_sequence=slot_sequences[b][1],
                slot_id_sequence=slot_sequences[b][3],
                edge_sequence=edge_seq_t[b, :seq_len].cpu(),
                slot_sequencer=slot_sequencer)
            # train_dataset.remove_auxiliary_tokens(graph)

        # dictionary of all sequences
        # masks are inverted to indicate valid positions
        trunc_seq = lambda seq_t: seq_t[:, :seq_len].contiguous()
        trunc_mask = lambda mask_t: mask_t[:, :seq_len-1].contiguous().logical_not_()

        all_seqs = {
            'slot_node_type_seq': slot_node_type_seq,
            'slot_node_idx_seq': slot_node_idx_seq,
            'slot_node_depth_seq': slot_node_depth_seq,
            'slot_id_seq': slot_id_seq,
            'slot_idx_seq': slot_idx_seq,
            'slot_seq_mask': slot_seq_mask,
            'edge_seq': trunc_seq(edge_seq_t),
            'edge_idx_seq': trunc_seq(edge_idx_seq_t),
            'edge_elm_seq': trunc_seq(edge_elm_seq_t),
            'edge_seq_mask': trunc_seq(edge_seq_mask_t),
            'edge_semantic_mask': trunc_mask(edge_semantic_mask_t),
        }

        if edge_action_mask_t is not None:
            all_seqs['edge_action_mask'] = trunc_mask(edge_action_mask_t)

        return ordered_nodes, all_seqs


class CyclicValidator:
    def __init__(self, slot_node_idx):
        self.slot_node_idx_map = slot_node_idx
        self.graphs = []

    def add_graph(self, n_nodes):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))
        self.graphs.append(graph)

    def add_edge(self, graph_idx, from_slot_idx, to_slot_idx, validate=False):
        from_node_idx = self.slot_node_idx_map[graph_idx][from_slot_idx]
        to_node_idx = self.slot_node_idx_map[graph_idx][to_slot_idx]
        self.graphs[graph_idx].add_edge(from_node_idx, to_node_idx)

        # check cycle
        if validate:
            try:
                cycles = nx.find_cycle(self.graphs[graph_idx], orientation='original')
            except nx.exception.NetworkXNoCycle:
                return
            raise RuntimeError(f'Cycles detected in the edge generation! Number of cycles: {cycles}.')

    def get_reachable_nodes(self, graph_idx, start_slot_idx):
        start_node_idx = self.slot_node_idx_map[graph_idx][start_slot_idx]
        reachable = list(nx.ancestors(self.graphs[graph_idx], start_node_idx)) + [start_node_idx]
        return reachable
