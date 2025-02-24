# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import math
import copy
import os.path as pth
from collections import OrderedDict

from ..diffsbs.io_json import get_node_type_signature, match_node_type_signature
from ..simple_graph import SimpleNode


def sequence_stop_idx(sequence, stop_token):
    stop_token_inds = (sequence == stop_token).nonzero(as_tuple=True)[0]
    if stop_token_inds.numel() == 0:
        raise RuntimeError(f'Sequence does not contain a stop token.')
    stop_token_idx = stop_token_inds[0].item()

    return stop_token_idx


def sequences_to_matrix(sequences, start_token, stop_token, num_cols):
    # decode a matrix with a known number of columns stored in a sequence with start/stop tokens

    if start_token is not None:
        start_token_idx = 0
        if sequences[0][start_token_idx] != start_token:
            raise RuntimeError(f'Sequence does not start with a start token.')
    else:
        start_token_idx = -1
    stop_token_idx = sequence_stop_idx(sequence=sequences[0], stop_token=stop_token)
    # stop_token_inds = (sequences[0] == stop_token).nonzero(as_tuple=True)[0]
    # if stop_token_inds.numel() == 0:
    #     raise RuntimeError(f'Sequence does not contain a stop token.')
    # stop_token_idx = stop_token_inds[0]
    content_len = (stop_token_idx-1) - start_token_idx
    num_rows = content_len // num_cols

    return [seq[start_token_idx+1:(start_token_idx+1)+(num_rows*num_cols)].view(num_rows, num_cols) for seq in sequences]


def add_auxiliary_tokens(graph, node_order, node_types, node_type_names, revert_to_v0, sorted_by_name, fixed_output_order):
    # add output root node
    if node_order == 'reverse_breadth_first':
        add_output_root_node(graph=graph, revert_to_v0=revert_to_v0, sorted_by_name=sorted_by_name, fixed_output_order=fixed_output_order)
        # parent end nodes still useful because otherwise there is no indication in the sequence
        # when switching to the parents of a different node
        add_parent_end_nodes(graph=graph)
        add_empty_input_nodes(graph=graph)
    elif node_order == 'topological_random':
        pass
    elif node_order == 'breadth_first_random':
        add_child_end_nodes(graph=graph, node_types=node_types, node_type_names=node_type_names)
    elif node_order == 'reverse_breadth_first_flipped':
        add_output_root_node(graph=graph, revert_to_v0=revert_to_v0, sorted_by_name=sorted_by_name, fixed_output_order=fixed_output_order)
        add_parent_end_nodes(graph=graph)
        add_empty_input_nodes(graph=graph)
    elif node_order.endswith('no_auxiliary_nodes'):
        pass
    else:
        raise RuntimeError(f'Unknown node order: {node_order}')


def remove_auxiliary_tokens(graph):
    removed_nodes = []
    removed_nodes.extend(remove_empty_output_nodes(graph=graph))
    removed_nodes.extend(remove_empty_input_nodes(graph=graph))
    removed_nodes.extend(remove_child_end_nodes(graph=graph))
    removed_nodes.extend(remove_parent_end_nodes(graph=graph))
    removed_nodes.extend(remove_output_root_node(graph=graph))
    return removed_nodes


def add_output_root_node(graph, revert_to_v0, sorted_by_name, fixed_output_order):
    leaf_nodes = graph.get_leafs()  # leaf nodes should only consist of the special node_output_* nodes
    if len(leaf_nodes) == 0:
        raise RuntimeError('No leaf nodes found.')
    if fixed_output_order:
        fixed_order_output_nodes = sort_output_nodes_to_fixed_order(leaf_nodes)
    else:
        fixed_order_output_nodes = None
    if fixed_order_output_nodes is None:
        if sorted_by_name:
            leaf_nodes.sort(key=lambda node: node.name)
        else:
            leaf_nodes.sort(key=lambda node: node.type)
    else:
        leaf_nodes = fixed_order_output_nodes
    # deprecated code (v0) is problematic, we don't need to add parent node here!
    if revert_to_v0:
        # add end token to the parents of the output root
        leaf_end_node = SimpleNode(name='output_root_parent_end', type='parent_end')
        leaf_nodes.append(leaf_end_node)
        graph.add_node(leaf_end_node)
    output_root_node = SimpleNode(name='output_root', type='output_root')
    graph.add_node(output_root_node)
    for leaf_node_idx, leaf_node in enumerate(leaf_nodes):
        graph.add_connection(
            parent=leaf_node, parent_output_slot=0,
            child=output_root_node, child_input_slot=leaf_node_idx)


def remove_output_root_node(graph):
    removed_nodes = []
    # find and validate output root node
    output_root_nodes = [node for node in graph.nodes if node.type == 'output_root']

    # remove output root node(s) from graph
    for output_root_node in output_root_nodes:
        removed_nodes.append(graph.remove_node(output_root_node))
    return removed_nodes


def add_parent_end_nodes(graph):
    parent_end_nodes = []
    for node in graph.nodes:
        if len(node.parents) == 0 or node.parents[-1][0] is None or node.parents[-1][0].type != 'parent_end':
            parent_end_node = SimpleNode(name=f'{node.name}_parent_end', type='parent_end')
            parent_end_nodes.append(parent_end_node)
            graph.add_connection(
                parent=parent_end_node, parent_output_slot=0,
                child=node, child_input_slot=len(node.parents))
    graph.add_nodes(parent_end_nodes)


def remove_parent_end_nodes(graph):
    removed_nodes = []
    parent_end_nodes = [node for node in graph.nodes if node.type == 'parent_end']
    for node in parent_end_nodes:
        removed_nodes.append(graph.remove_node(node))
    return removed_nodes


def add_child_end_nodes(graph, node_types, node_type_names):
    child_end_nodes = []
    for node in graph.nodes:
        if node.type in ['child_end', 'empty_output']:
            continue
        if len(node.children) == 0 or 'child_end' not in [child.type for child in node.children]:
            child_end_node = SimpleNode(name=f'{node.name}_child_end', type='child_end')
            child_end_nodes.append(child_end_node)
            output_slot_idx = list(node_types[node_type_names.index(node.type)]['output_names']).index('child_end')
            graph.add_connection(
                parent=node, parent_output_slot=output_slot_idx,
                child=child_end_node, child_input_slot=0)
    graph.add_nodes(child_end_nodes)


def remove_child_end_nodes(graph):
    removed_nodes = []
    parent_end_nodes = [node for node in graph.nodes if node.type == 'child_end']
    for node in parent_end_nodes:
        removed_nodes.append(graph.remove_node(node))
    return removed_nodes


def add_empty_input_nodes(graph):
    empty_input_nodes = []
    for node in graph.nodes:
        for parent_idx, (parent, _) in enumerate(node.parents):
            if parent is None:
                empty_input_node = SimpleNode(name=f'{node.name}_empty_input_{parent_idx}', type='empty_input')
                empty_input_nodes.append(empty_input_node)
                graph.add_connection(
                    parent=empty_input_node, parent_output_slot=0,
                    child=node, child_input_slot=parent_idx)
    graph.add_nodes(empty_input_nodes)


def remove_empty_input_nodes(graph):
    removed_nodes = []
    empty_input_nodes = [node for node in graph.nodes if node.type == 'empty_input']
    for node in empty_input_nodes:
        removed_nodes.append(graph.remove_node(node))
    return removed_nodes


def add_empty_output_nodes(graph, node_types, node_type_names):
    empty_output_nodes = []
    for node in graph.nodes:
        if node.type in ['child_end', 'empty_output']:
            continue
        num_outputs = node_types[node_type_names.index(node.type)]['output_names'].index('child_end')-1
        child_slot_inds = [slot_idx for _, slot_idx in node.get_child_slots()]
        for output_idx in range(num_outputs):
            if output_idx not in child_slot_inds:
                empty_output_node = SimpleNode(name=f'{node.name}_empty_output_{output_idx}', type='empty_output')
                empty_output_nodes.append(empty_output_node)
                graph.add_connection(
                    parent=empty_output_node, parent_output_slot=output_idx,
                    child=node, child_input_slot=0)
    graph.add_nodes(empty_output_nodes)


def remove_empty_output_nodes(graph):
    removed_nodes = []
    empty_output_nodes = [node for node in graph.nodes if node.type == 'empty_output']
    for node in empty_output_nodes:
        removed_nodes.append(graph.remove_node(node))
    return removed_nodes

# def add_node_param_end_tokens(graph):
#     for node in graph.nodes:
#         if len(node.param_names) == 0 or node.param_names[-1] != 'param_end':
#             node.param_names.append('param_end')
#             node.param_vals.append(0)

# def remove_node_param_end_tokens(graph):
#     for node in graph.nodes:
#         node.param_vals = [param_val for param_idx, param_val in enumerate(node.param_vals) if node.param_names[param_idx] != 'param_end']
#         node.param_names = [param_name for param_idx, param_name in enumerate(node.param_names) if node.param_names[param_idx] != 'param_end']


# must be applied to converted node types
def add_auxiliary_node_types(node_types, max_num_parents, node_order):

    node_types = copy.deepcopy(node_types)
    
    if node_order == 'reverse_breadth_first':
        use_parent_end = True
        use_empty_input = True
        use_output_root = True
        use_child_end = False
        use_empty_output = False
    elif node_order == 'topological_random':
        use_parent_end = False
        use_empty_input = False
        use_output_root = False
        use_child_end = False
        use_empty_output = False
    elif node_order == 'breadth_first_random':
        use_parent_end = False
        use_empty_input = False
        use_output_root = False
        use_child_end = True
        use_empty_output = False
    elif node_order == 'reverse_breadth_first_flipped':
        use_parent_end = True
        use_empty_input = True
        use_output_root = True
        use_child_end = False
        use_empty_output = False
    # simply skip adding auxiliary nodes
    elif node_order.endswith('no_auxiliary_nodes'):
        return node_types
    else:
        raise RuntimeError(f'Unknown node order: {node_order}')

    # add auxiliary node types
    print('Added auxiliary node types: ')
    reserved_type_names = ['parent_end', 'child_end', 'output_root', 'empty_input', 'empty_output']
    if any(reserved_type_name in node_types for reserved_type_name in reserved_type_names):
        raise RuntimeError(f'A node type has one of the reserved names {reserved_type_names}.')
    if use_parent_end:
        print(f'Parent End: {len(node_types)}')
        node_types.append({'name': 'parent_end', 'input_names': [], 'output_names': OrderedDict([('output', {})]), 'def_graph_name': None, 'def_path': None, 'parameters': {}})
    if use_child_end:
        raise NotImplementedError
        node_types.append({'name': 'child_end', 'input_names': OrderedDict([('input', {})]), 'output_names': OrderedDict(), 'def_graph_name': None, 'def_path': None, 'parameters': {}})
    if use_output_root:
        print(f'Output Root: {len(node_types)}')
        node_types.append({'name': 'output_root', 'input_names': [f'input_{i}' for i in range(max_num_parents)], 'output_names': OrderedDict(), 'def_graph_name': None, 'def_path': None, 'parameters': {}})
    if use_empty_input:
        print(f'Empty Input: {len(node_types)}')
        node_types.append({'name': 'empty_input', 'input_names': [], 'output_names': OrderedDict([('output', {})]), 'def_graph_name': None, 'def_path': None, 'parameters': {}})
    if use_empty_output:
        raise NotImplementedError
        node_types.append({'name': 'empty_output', 'input_names': OrderedDict([('input', {})]), 'output_names': OrderedDict(), 'def_graph_name': None, 'def_path': None, 'parameters': {}})

    # add auxiliary node type parameters
    # not needed since parameter sequences have their own stop token
    # reserved_param_names = ['param_end']
    # for node_type in node_types:
    #     if any(reserved_param_name in node_type['parameters'] for reserved_param_name in reserved_param_names):
    #         raise RuntimeError(f'A node parameter has one of the reserved names {reserved_param_names}.')
    #     node_type['parameters'].insert(0, {'name': 'param_end', 'dtypes': {'FLOAT1': {'max_quantized': 0}}}) # always add param_end to the beginning of the list, so that it always has a consistent index across node types

    # add output slot that connects an output node to the output root
    if use_output_root:
        for node_type in node_types:
            if node_type['name'].startswith('output_'):
                if len(node_type['output_names']) == 0:
                    node_type['output_names']['output'] = {}
                elif not (len(node_type['output_names']) == 1 and list(node_type['output_names'])[0] == 'output'):
                    raise RuntimeError('Found output node type with non-zero output slots.')

    # this should be safer, but do not use this as it would change the behaviour compared to the previous training runs
    # # add input slot for the parent_end node
    # if use_child_end:
    #     reserved_input_names = ['parent_end']
    #     for node_type in node_types:
    #         if any(reserved_input_name in node_type['input_names'] for reserved_input_name in reserved_input_names):
    #             raise RuntimeError(f'An input slot has one of the reserved names {reserved_input_names}.')
    #         node_type['input_names']['parent_end'] = {}

    # add output slot for the child_end node
    if use_child_end:
        reserved_output_names = ['child_end']
        for node_type in node_types:
            if any(reserved_output_name in node_type['output_names'] for reserved_output_name in reserved_output_names):
                raise RuntimeError(f'An output slot has one of the reserved names {reserved_output_names}.')
            node_type['output_names']['child_end'] = {}

    return node_types

def remove_auxiliary_node_types(node_types):

    node_types = copy.deepcopy(node_types)

    # remove output slot for the child_end node
    for node_type in node_types:
        if 'child_end' in node_type['output_names']:
            del node_type['output_names']['child_end']

    # remove input slot for the parent_end node
    # this should be safer, but do not use this as it would change the behaviour compared to the previous training runs
    # for node_type in node_types:
    #     if 'parent_end' in node_type['input_names']:
    #         del node_type['input_names']['parent_end']

    # remove output slot that connects an output node to the output root
    for node_type in node_types:
        if node_type['name'].startswith('output_'):
            node_type['output_names'] = OrderedDict()

    # remove auxiliary node type parameters
    # not needed since parameter sequences have their own stop token
    # for node_type in node_types:
    #     del node_type['parameters']['param_end'] # not needed since parameter sequences have their own stop token

    # remove auxiliary node types
    if 'parent_end' in node_types:
        del node_types['parent_end']
    if 'child_end' in node_types:
        del node_types['child_end']
    if 'output_root' in node_types:
        del node_types['output_root']
    if 'empty_input' in node_types:
        del node_types['empty_input']
    if 'empty_output' in node_types:
        del node_types['empty_output']

    return node_types


# adapter class for flattening node types
class NodeTypeAdapter:
    def __init__(self, node_types, sort_node_type=True):
        # set compatibility flag
        self._legacy_flattened = getattr(node_types, '_legacy_flattened', False)

        # flatten node types
        node_types = copy.deepcopy(node_types)
        node_types_flattened, node_type_indices = [], []

        node_type_keys = list(node_types.keys())
        node_type_keys = sorted(node_type_keys) if sort_node_type else node_type_keys

        ind = 0
        for key in node_type_keys:
            val = node_types[key]
            node_type_indices.append(ind)

            if isinstance(val, list):
                node_types_flattened.extend([{'name': f'{key}:{i}', **nt} for i, nt in enumerate(val)])
                ind += len(val)
            elif isinstance(val, dict):
                node_types_flattened.append({'name': key, **val})
                ind += 1
            else:
                raise TypeError(f'Invalid node type entry: expect list or dict, but got {type(val).__name__}.')

        if ind != len(node_types_flattened):
            raise RuntimeError('Inconsistent numbers of node types.')
        node_type_indices.append(ind)

        # flatten node parameters
        for node_type in node_types_flattened:
            node_type['parameters'] = [{'name': type_name, 'dtypes': type_info} for type_name, type_info in node_type['parameters'].items()]
            node_type['parameters'] = sorted(node_type['parameters'], key=lambda x: x['name'])

        self.node_types = node_types_flattened
        self.node_type_keys = node_type_keys
        self.node_type_key_dict = {key: i for i, key in enumerate(node_type_keys)}
        self.node_type_indices = node_type_indices

    def __iter__(self):
        return iter(self.node_types)

    def __getitem__(self, index):
        return self.node_types[index]

    def __len__(self):
        return len(self.node_types)

    @staticmethod
    def _signature_to_dict(signature):
        sd = signature.copy()
        for key in ('input_names', 'output_names'):
            sd[key] = dict(sorted(sd[key].items(), key=lambda x: x[0]))
        sd_params = {name: data['type'] for name, data in sd['parameters'].items()}
        sd['parameters'] = dict(sorted(sd_params.items(), key=lambda x: x[0]))
        return sd

    @staticmethod
    def _node_type_to_dict(node_type):
        sd = node_type.copy()
        for key in ('input_names', 'output_names'):
            slots = sd[key]
            for name, data in slots.items():
                all_dtypes = list(data['types'].keys())
                slots[name] = all_dtypes[0] if len(all_dtypes) == 1 else all_dtypes
            sd[key] = dict(sorted(slots.items(), key=lambda x: x[0]))
        sd_params = sd['parameters']
        for name, data in sd_params.items():
            all_dtypes = list(data.keys())
            sd_params[name] = all_dtypes[0] if len(all_dtypes) == 1 else all_dtypes
        sd['parameters'] = dict(sorted(sd_params.items(), key=lambda x: x[0]))
        return sd

    # find disambiguated node type given a json node input
    def query(self, json_node, **sig_kwargs):
        # look up node type by name
        node_type_name = json_node['func']
        if self._legacy_flattened and json_node.get('def_path') is not None:
            node_type_name = f"{pth.basename(json_node['def_path'])}:{json_node['func']}"

        i = self.node_type_key_dict.get(node_type_name, -1)
        if i < 0:
            raise RuntimeError(f"Node type '{node_type_name}' is not found.")

        # legacy case: the input node type dictionary is already flattened
        if self._legacy_flattened:
            return self.node_types[i]

        # gather node type candidates (unflatten node parameters)
        ind, next_ind = self.node_type_indices[i:i+2]
        node_type_cands = copy.deepcopy(self.node_types[ind:next_ind])
        for nt in node_type_cands:
            nt['parameters'] = {p['name']: p['dtypes'] for p in nt['parameters']}

        # match node type signature
        signature = get_node_type_signature(json_node, **sig_kwargs)
        ignored_params = ['format', 'pixelsize', 'pixelratio']
        ind_offset = next((i for i, nt in enumerate(node_type_cands)
                           if match_node_type_signature(signature, nt, ignored_params=ignored_params)), -1)
        if ind_offset < 0:
            # import json
            # print('Source signature:')
            # print(json.dumps(self._signature_to_dict(signature), indent=4))
            # print(f'Candidate node types ({len(node_type_cands)} in total):')
            # print(json.dumps([self._node_type_to_dict(nt) for nt in node_type_cands], indent=4))
            raise RuntimeError(f"No matching signature found for input node '{json_node['name']}' of type '{json_node['func']}'")

        return self.node_types[ind + ind_offset]

    # add a new node type (must be unique)
    def append(self, node_type):
        node_type_name = node_type['name']
        if node_type_name in self.node_type_keys:
            raise RuntimeError(f"Node type '{node_type_name}' already exists.")

        self.node_types.append(node_type)
        self.node_type_keys.append(node_type_name)
        self.node_type_key_dict[node_type_name] = len(self.node_type_keys) - 1
        self.node_type_indices.append(len(self.node_types))

    # unflatten the node type dictionary
    def unflatten(self):
        node_types_flattened, node_type_indices = self.node_types, self.node_type_indices
        node_types = OrderedDict()
        for i, key in enumerate(self.node_type_keys):
            ind, next_ind = node_type_indices[i:i+2]
            node_type = copy.deepcopy(node_types_flattened[ind:next_ind])

            # unflatten node parameters
            for nt in node_type:
                nt['parameters'] = {p['name']: p['dtypes'] for p in nt['parameters']}
                del nt['name']

            node_types[key] = node_type[0] if len(node_type) == 1 else node_type

        node_types._legacy_flattened = self._legacy_flattened

        return node_types


# convert to have a fixed ordering for both node types and paramter types in each node
def convert_node_types(node_types, sort_node_types=True):
    return NodeTypeAdapter(node_types, sort_node_type=sort_node_types)


def unconvert_node_types(node_type_adapter):
    return node_type_adapter.unflatten()


def sort_output_nodes_to_fixed_order(output_nodes):
    fixed_order_output_types = {'output_baseColor': 0,
                                'output_normal': 1,
                                'output_roughness': 2,
                                'output_metallic': 3}
    sorted_output_nodes = [None]*len(fixed_order_output_types)
    # sort to a fixed order by types
    for output_node in output_nodes:
        output_node_type = output_node.type
        output_node_type = output_node_type[:output_node_type.find(':')] if ':' in output_node_type else output_node_type
        if output_node_type in fixed_order_output_types:
            idx = fixed_order_output_types[output_node_type]
        else:
            print(f'Detected a non-output leaf node. {output_node_type}')
            return None
        if sorted_output_nodes[idx] is None:
            sorted_output_nodes[idx] = output_node
        else:
            print(f'Detected two same output nodes of the same type: {output_node_type}')
            return None
    # remove non-existing nodes
    sorted_output_nodes_ = []
    for output_node in sorted_output_nodes:
        if output_node is not None:
            sorted_output_nodes_.append(output_node)

    return sorted_output_nodes_
