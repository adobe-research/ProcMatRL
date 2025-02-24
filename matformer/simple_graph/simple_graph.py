# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os.path as pth
import itertools
from collections import deque
from .sbs_param_type import param_type_name_to_idx, SBSParamType
from ..utils import unique, vis_simple_graph


class SimpleNode:
    def __init__(self, name, type, param_names=None, param_vals=None, parents=None, children=None):
        self.name = name
        self.type = type
        self.parents = parents if parents is not None else [] # order of the parent is important, it defines the input slots. Each parent is a tuple (parent_node, parent_out_idx)
        self.children = children if children is not None else set()
        self.param_names = param_names if param_names is not None else []
        self.param_vals = param_vals if param_vals is not None else []

        assert len(self.param_names) == len(self.param_vals)

        # these unique indices are computed during generation
        self.idx = None

    def __str__(self):
        return f'<{self.type} node named {self.name}>'

    def __repr__(self):
        return f'<{self.type} node named {self.name}>'

    def get_ancestors(self, num_ancestors=None):
        if num_ancestors is None:
            num_ancestors = 999
        nodes = [self]
        ancestors = []
        visited_set = set(nodes)
        for _ in range(num_ancestors):
            parents = list(itertools.chain.from_iterable([node.parents for node, _ in nodes]))
            # remove parents that have already been visited due to cycles (effectively breaking the cycles)
            parents = [parent for parent in parents if parent not in visited_set]
            if len(parents) == 0:
                break
            # if any(child in visited_set for child in children):
            #     raise RuntimeError('Found edge cycle while searching for descendants.')
            ancestors.append(parents)
            visited_set.update(parents)

            nodes = parents

        return ancestors

    def get_descendants(self, num_descendants=None):
        if num_descendants is None:
            num_descendants = 999
        nodes = [self]
        descendants = []
        visited_set = set(nodes)
        for _ in range(num_descendants):
            children = list(itertools.chain.from_iterable([node.children for node in nodes]))
            # remove childs that have already been visited due to cycles (effectively breaking the cycles)
            children = [child for child in children if child not in visited_set]
            if len(children) == 0:
                break
            # if any(child in visited_set for child in children):
            #     raise RuntimeError('Found edge cycle while searching for descendants.')
            descendants.append(children)
            visited_set.update(children)

            nodes = children

        return descendants

    def get_max_dist_to_output(self):
        max_depth = 0
        for child in self.children:
            child_depth = child.get_max_dist_to_output()
            if child_depth+1 > max_depth:
                max_depth = child_depth+1
        return max_depth

    def get_child_slots(self):
        child_slots = []
        for child in self.children:
            child_slots.extend([(child, out_idx) for child_parent, out_idx in child.parents if child_parent == self])
        return child_slots


class SimpleOrderedNodes:
    def __init__(self, nodes=None, depths=None):
        # check input validity
        if depths is not None:
            if nodes is None:
                raise RuntimeError('Nodes must be provided when depths are provided.')
            if len(nodes) != len(depths):
                raise RuntimeError('The number of nodes and depths are not equal.')

        self.nodes = nodes if nodes is not None else []
        self.depths = depths if depths is not None else []

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __iter__(self):
        return iter(self.nodes)

    def get(self, idx):
        return self.nodes[idx], self.depths[idx]

    def items(self):
        return zip(self.nodes, self.depths)

    def to_graph(self):
        return SimpleGraph(nodes=self.nodes)


pruned_graphs_counter = 0


class SimpleGraph:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes is not None else []

    def get_depth(self):
        # brute force for now
        return max(node.get_max_dist_to_output() for node in self.nodes)

    def get_leafs(self):
        return [node for node in self.nodes if len(node.children) == 0]

    def add_node(self, node):
        node_names = [node.name for node in self.nodes]
        if node.name in node_names:
            raise RuntimeError('A node with the given name already exists.')
        self.nodes.append(node)

    def add_nodes(self, nodes):
        node_names = set(node.name for node in self.nodes)
        new_node_names = set(node.name for node in nodes)
        if len(node_names.intersection(new_node_names)) != 0:
            raise RuntimeError('A node with the given name already exists.')
        self.nodes.extend(nodes)

    def remove_node(self, node):
        if node not in self.nodes:
            raise RuntimeError('Given node is not in the graph.')

        # remove references to the node from its parents
        for parent, _ in node.parents:
            if parent is not None:
                if node in parent.children:
                    parent.children.remove(node)

        # remove references to the node from its children
        for child in node.children:
            if node in [n for n, _ in child.parents]:
                child.parents = [(None, None) if n == node else (n, idx) for n, idx in child.parents] # child may have multiple edges to the node
                # remove padded None parents
                while len(child.parents) > 0 and child.parents[-1][0] == None:
                    del child.parents[-1]

        # remove node from graph
        self.nodes.remove(node)

        return node

    @staticmethod
    def add_connection(parent, parent_output_slot, child, child_input_slot):
        if child_input_slot >= len(child.parents):
            child.parents.extend([(None, None)]*(1 + len(child.parents) - child_input_slot))
        child.parents[child_input_slot] = (parent, parent_output_slot)
        parent.children.add(child)

    def get_node_by_name(self, name):
        matching_node = [node for node in self.nodes if node.name == name]
        if len(matching_node) > 1:
            raise RuntimeError(f'Duplicate node name found: {name}')
        elif len(matching_node) == 0:
            raise RuntimeError(f'Node not found for name: {name}')
        return matching_node[0]

    def get_nodes_by_names(self, names):
        matching_nodes = []
        for name in names:
            matching_nodes.append(self.get_node_by_name(name))
        return matching_nodes

    @staticmethod
    def get_param_dtype(node_type, param_type_info, use_alpha, legacy_flattened=False):
        param_name = param_type_info['name']

        # helper function for matching node type name (considering disambiguation suffixes)
        def name_match(name, options):
            return (name in options
                    or any(name.startswith(s + ':') for s in options)
                    or any(name.endswith('.sbs:' + s) for s in options))    # for compatibility

        if node_type == 'F.gradient_map' and param_name == 'anchors':
            dtype = 'FLOAT5_ARRAY' if use_alpha else 'FLOAT4_ARRAY'
            # assert dtype in param_type_info['dtypes']
        elif node_type == 'F.levels' and param_name in ['out_low', 'out_high', 'in_mid'] + ([] if legacy_flattened else ['in_low', 'in_high']):
            dtype = 'FLOAT4' if use_alpha else 'FLOAT3'
            # assert dtype in param_type_info['dtypes']
        elif node_type == 'F.curve' and param_name == 'anchors':
            dtype = 'FLOAT6_ARRAY'
        elif name_match(node_type, ['F.uniform_color', 'F.make_it_tile_patch']) and param_name in ['rgba', 'background_color']:
            dtype = 'FLOAT4'
        elif name_match(node_type, ['F.quantize']) and param_name == 'quantize_number':
            dtype = 'INTEGER4'
        elif name_match(node_type, ['tile_generator']) and param_name == 'interstice':
            dtype = 'FLOAT4'
        elif name_match(node_type, ['st_sand']) and param_name == 'Waves_Distortion':
            dtype = 'FLOAT1'
        elif name_match(node_type, ['window_generator']) and param_name == 'window_brace_offset':
            dtype = 'FLOAT2'
        elif name_match(node_type, ['perforated_swirl_filter']) and param_name == 'use_scale_input':
            dtype = 'FLOAT1'
        elif name_match(node_type, ['GT_BasicParameters']) and param_name == 'normal_format':
            dtype = 'INTEGER1'
        elif name_match(node_type, ['quilt_filter']) and param_name == 'selective_material_mask':
            dtype = 'INTEGER1'
        else:
            if len(param_type_info['dtypes']) != 1:
                raise RuntimeError(f'Expected parameter {param_name} of node type {node_type} to have a single data type, but found multiple data types: {list(param_type_info["dtypes"].keys())}.')
            dtype = list(param_type_info['dtypes'].keys())[0]

        return dtype

    @staticmethod
    def get_param_tensor_rank(param_dtype):
        if '_ARRAY' in param_dtype:
            return 'array'
        elif SimpleGraph.get_param_vector_dim(param_dtype) == 1:
            return 'scalar'
        else:
            return 'vector'

    @staticmethod
    def get_param_vector_dim(param_dtype):
        for dtype_prefix in ['FLOAT', 'INTEGER', 'BOOLEAN']:
            if param_dtype.startswith(dtype_prefix):
                if '_' in param_dtype:
                    vector_dim_str = param_dtype[len(dtype_prefix):param_dtype.find('_')]
                else:
                    vector_dim_str = param_dtype[len(dtype_prefix):]
                return 1 if len(vector_dim_str) == 0 else int(vector_dim_str)

        if param_dtype == 'STRING':
            return 1
        else:
            raise RuntimeError(f'Unknown parameter data type: {param_dtype}.')

    # @staticmethod
    # def get_param_array_len(param_dtype):
    #     return 1 if '_ARRAY' not in param_dtype else int(param_dtype[param_dtype.rfind('_ARRAY')+len('_ARRAY'):])

    @staticmethod
    def load_json_legacy(json_nodes, legacy_node_names=False):
        # legacy
        if isinstance(json_nodes, dict):
            json_nodes = list(json_nodes.values())

        # create nodes
        nodes = {}
        output_parent_node = {}
        output_parent_json_node = {}
        for json_node in json_nodes:
            if legacy_node_names:
                node = SimpleNode(name=json_node['name'], type=json_node['func'])
            else:
                node = SimpleNode(name=json_node['name'], type=get_node_type_name(json_node))
            nodes[node.name] = node
            for output_info in json_node['outputs']:
                output_name = output_info[1]
                if output_name in output_parent_node:
                    raise RuntimeError(f'Duplicate node output name: {output_name}.')
                output_parent_node[output_name] = node
                output_parent_json_node[output_name] = json_node

            # create node parameters
            for json_param in json_node['params']:
                param_name = json_param[0]
                param_val = json_param[1]
                node.param_names.append(param_name)
                node.param_vals.append(param_val)

        # create edges between nodes
        for json_node in json_nodes:
            node = nodes[json_node['name']]
            for node_input in json_node['inputs']:

                # get the parent node based on the parent output in the graph
                parent_output_name = node_input[1]
                if parent_output_name is None:
                    # unconnected input
                    node.parents.append((None, None))
                else:
                    if parent_output_name not in output_parent_node:
                        raise RuntimeError(f'No parent operation for node output {parent_output_name}.')
                    parent_node = output_parent_node[parent_output_name]
                    parent_json_node = output_parent_json_node[parent_output_name]

                    # parent_node_name = list(output_parent[op_output_graph.predecessors(parent_output_name))
                    # elif len(parent_node_name) == 0:
                    #     raise RuntimeError(f'No parent operation for node output {parent_output_name}.')
                    # parent_node_name = parent_node_name[0]
                    # parent_node = nodes[parent_node_name]

                    # get the slot in the parent node based on the parent output name
                    # parent_json_node = json_nodes[parent_node_name]
                    parent_output_output_names = [output[1] for output in parent_json_node['outputs']]
                    if parent_output_name not in parent_output_output_names:
                        raise RuntimeError(f'Bad connection from child {node.name} to parent {parent_node.name}: the child is connected to a non-existent output slot of the parent.')
                    parent_output_slot = parent_output_output_names.index(parent_output_name)

                    # create edge between parent and current node
                    node.parents.append((parent_node, parent_output_slot))
                    parent_node.children.add(node)

        nodes = [node for _, node in nodes.items()]

        # add nodes to a graph
        graph = SimpleGraph(nodes=nodes)

        return graph

    @staticmethod
    def load_json(json_nodes, node_types, node_type_names, legacy_node_names=False, legacy_json_loading=False):

        if legacy_json_loading:
            return SimpleGraph.load_json_legacy(json_nodes=json_nodes, legacy_node_names=legacy_node_names)

        if len(node_type_names) != len(node_types):
            raise RuntimeError('Number of node type names and node types does not match.')

        # create nodes
        nodes = {}
        output_parent_node = {}
        for json_node in json_nodes:

            # create a node
            if legacy_node_names:
                node = SimpleNode(name=json_node['name'], type=json_node['func'])
            else:
                node = SimpleNode(name=json_node['name'], type=get_node_type_name(json_node, node_types=node_types))
            nodes[node.name] = node

            # update the map of output names to parent nodes
            node_type_info = node_types[node_type_names.index(node.type)]
            node_type_output_names = list(node_type_info['output_names'].keys())
            for output_info in json_node['outputs']:
                output_slot_name = output_info[0]
                output_name = output_info[1]
                if output_name in output_parent_node:
                    raise RuntimeError(f'Duplicate node output name: {output_name}.')
                if output_slot_name not in node_type_output_names:
                    raise RuntimeError(f'Unexpected output slot name: {output_slot_name} for node type {node.type}.')
                output_slot_idx = node_type_output_names.index(output_slot_name)
                output_parent_node[output_name] = (node, output_slot_idx)

            # create node parameters
            for json_param in json_node['params']:
                param_name = json_param[0]
                param_val = json_param[1]
                node.param_names.append(param_name)
                node.param_vals.append(param_val)

        # create edges between nodes
        for json_node in json_nodes:
            node = nodes[json_node['name']]
            node_type_info = node_types[node_type_names.index(node.type)]
            node_type_input_names = list(node_type_info['input_names'].keys())
            for node_input in json_node['inputs']:

                # special case: 'roughness' input in the 'st_stylized_look_filter.sbs:st_stylized_look_filter' should not be used
                # (st_stylized_look_filter is defined in 423 different .sbs files. Usually it seems to be the same, but some of them have a 'roughness' input that is not present in others,
                # so just ignore this input and don't use it) - same special case also in update_node_types() of node_types.py
                if node.type == 'st_stylized_look_filter.sbs:st_stylized_look_filter' and node_input[0] == 'roughness':
                    continue

                parent_output_name = node_input[1]

                # get the parent node based on the parent output in the graph
                if parent_output_name is not None:
                    if parent_output_name not in output_parent_node:
                        raise RuntimeError(f'No parent operation for node output {parent_output_name}.')
                    parent_node, parent_output_slot_idx = output_parent_node[parent_output_name]

                    # create edge between parent and current node
                    input_slot_name = node_input[0]
                    if input_slot_name not in node_type_input_names:
                        raise RuntimeError(f'Unexpected input slot name: {input_slot_name} for node type {node.type}.')
                    input_slot_idx = node_type_input_names.index(input_slot_name)
                    if len(node.parents) < input_slot_idx+1:
                        node.parents = node.parents + [(None, None)] * (input_slot_idx+1 - len(node.parents))
                    node.parents[input_slot_idx] = (parent_node, parent_output_slot_idx)
                    parent_node.children.add(node)

        nodes = [node for _, node in nodes.items()]

        # add nodes to a graph
        graph = SimpleGraph(nodes=nodes)

        return graph

    # node_types must be the converted node types
    def save_json(self, node_types, use_alpha):
        json_nodes = []
        node_type_names = [node_type['name'] for node_type in node_types]
        for node in self.nodes:

            if node.type not in node_type_names:
                raise RuntimeError(f'Node type stats not found for node type {node.type}.')

            node_type_info = node_types[node_type_names.index(node.type)]

            # inputs
            # when loading the json graph in SBSGraph,
            # only supported nodes have inputs, so the input slot names will always be determined by the known list of inputs of a supported node
            inputs = []
            for input_slot_idx, (parent, parent_out_idx) in enumerate(node.parents):
                if isinstance(node_type_info['input_names'], list): # legacy
                    input_slot_name = node_type_info['input_names'][input_slot_idx]
                    input_slot_dtype = SBSParamType.ENTRY_VARIANT.value
                else:
                    # raise RuntimeError('Still need to test below!')
                    input_slot_name, input_slot_info = list(node_type_info['input_names'].items())[input_slot_idx]
                    if len(input_slot_info['types']) == 0:
                        raise RuntimeError(f'Node type {node.type} input slot {input_slot_name} has no types.')
                    # pick data type that was observed most frequently for the given input slot
                    input_slot_dtype = param_type_name_to_idx(sorted(list(input_slot_info['types'].items()), key=lambda x: x[1], reverse=True)[0][0])

                # input_slot_name = list(node_type_info['input_names'])[input_slot_idx]
                if parent is None:
                    input_name = None
                else:
                    input_name = f'output_{parent.name}_{parent_out_idx}'
                inputs.append([input_slot_name, input_name, input_slot_dtype])

            # outputs
            # when loading the json graph in SBSGraph,
            # for unsupported nodes, the output slot name given here is used.
            # for supported nodes, the output slot name will be determined by the known list of outputs of a supported node
            outputs = []
            if len(node.children) > 0:
                child_slots = [slot_idx for _, slot_idx in node.get_child_slots()]
                for output_slot_idx in range(max(child_slots)+1):
                    if isinstance(node_type_info['output_names'], list): # legacy
                        output_slot_name = node_type_info['output_names'][output_slot_idx]
                        output_slot_dtype = SBSParamType.ENTRY_VARIANT.value
                    else:
                        output_slot_name, output_slot_info = list(node_type_info['output_names'].items())[output_slot_idx]
                        if len(output_slot_info['types']) == 0:
                            raise RuntimeError(f'Node type {node.type} output slot {output_slot_name} has no types.')
                        # pick data type that was observed most frequently for the given output slot
                        output_slot_dtype = param_type_name_to_idx(sorted(list(output_slot_info['types'].items()), key=lambda x: x[1], reverse=True)[0][0])
                    if output_slot_idx > len(node_type_info['output_names']):
                        raise RuntimeError(f'Too many outputs predicted for node type {node.type}.')
                    # output_slot_name = node_type_info['output_names'][output_slot_idx]
                    output_name = f'output_{node.name}_{output_slot_idx}'
                    outputs.append([output_slot_name, output_name, output_slot_dtype])

            # parameters
            params = []
            param_type_names = [node_type['name'] for node_type in node_type_info['parameters']]
            if node.param_names is not None:
                for param_name, param_val in zip(node.param_names, node.param_vals):
                    param_dtype = self.get_param_dtype(node_type=node.type, param_type_info=node_type_info['parameters'][param_type_names.index(param_name)],
                                                       use_alpha=use_alpha, legacy_flattened=getattr(node_types, '_legacy_flattened', True))
                    if param_dtype.endswith('_ARRAY'):
                        param_dtype = f'{param_dtype}_{len(param_val)}'
                    params.append([param_name, param_val, param_dtype, None])

            json_node = {
                'name': node.name,
                'func': node_type_info['func'],
                'def_graph_name': node_type_info['def_graph_name'],
                'def_path': node_type_info['def_path'],
                'inputs': inputs,
                'outputs': outputs,
                'params': params}
            json_nodes.append(json_node)

        return json_nodes

    def is_connected(self, node):
        if node not in self.nodes:
            return False

        for parent, _ in node.parents:
            if parent is not None:
                return True

        for child in node.children:
            if node in [n for n, _ in child.parents]:
                return True

        return False

    def get_output_nodes(self):
        output_root = [node for node in self.nodes if node.type == 'output_root']
        if len(output_root) > 0:
            raise RuntimeError('Auxiliary nodes should be remove before this operation.')
        else:
            material_outputs = ['output_baseColor', 'output_normal', 'output_roughness', 'output_metallic']
            output_nodes = [node for node in self.nodes if any(node.type.startswith(output_name) for output_name in material_outputs)]
            if len(output_nodes) == 0:
                raise RuntimeError('No output nodes was detected.')
            return output_nodes

    def prune(self, out_dir=None):
        global pruned_graphs_counter
        output_nodes = self.get_output_nodes()

        node_queue = deque(output_nodes)
        visited_nodes = set(output_nodes)
        reachable_nodes = []
        while len(node_queue) > 0:
            node = node_queue.popleft()
            # if no empty input node, it's possible to have a None input
            if node is None:
                continue
            reachable_nodes.append(node)
            univisited_parent_nodes = [parent_node for parent_node in unique([n for n, _ in node.parents])
                                       if parent_node not in visited_nodes]

            node_queue.extend(univisited_parent_nodes)
            visited_nodes.update(univisited_parent_nodes)

        if len(reachable_nodes) != len(self.nodes):
            unreachable_nodes = [node for node in self.nodes if node not in reachable_nodes]

            if out_dir is not None:
                vis_simple_graph(graph=self, filename=pth.join(out_dir, f'{pruned_graphs_counter}_before.pdf'),
                                 include_parent_end=True, colorized_nodes=unreachable_nodes)

            for node in unreachable_nodes:
                self.remove_node(node)

            if out_dir is not None:
                vis_simple_graph(graph=self, filename=pth.join(out_dir, f'{pruned_graphs_counter}_after.pdf'), include_parent_end=True)
                pruned_graphs_counter += 1


def get_node_type_name(json_node, node_types=None, legacy_flattened=None):
    # TODO: only json_node['func'] for all supported nodes (where json_node['func'] starts with F.)?
    from ..sequencer import NodeTypeAdapter
    if isinstance(node_types, NodeTypeAdapter):
        return node_types.query(json_node, skip_seeds=True)['name']

    # detect which version of node type dictionary it is
    if legacy_flattened is None:
        legacy_flattened = getattr(node_types, '_legacy_flattened', True) if isinstance(node_types, dict) else True

    if legacy_flattened and json_node['def_path'] is not None:
        node_type_name = f'{pth.basename(json_node["def_path"])}:{json_node["func"]}'
    else:
        node_type_name = json_node['func']
    return node_type_name


# verify if two graphs are equal
def is_node_equal(node_x, node_y):
    # if node_x.name != node_y.name:
    #     print(f'The node names are not equal: {node_x.name}/{node_y.name}')
    #     return False

    if node_x.type != node_y.type:
        print(f'The node types are not equal: {node_x.type}/{node_y.type} (node names: {node_x.name}/{node_y.name})')
        return False

    for param_name_x, param_val_x, param_name_y, param_val_y in zip(node_x.param_names, node_x.param_vals, node_y.param_names, node_y.param_vals):
        if param_name_x != param_name_y:
            print(f'The param names are not equal: {param_name_x}/{param_name_y}')
            return False
        if param_val_x != param_val_y:
            print(f'The param values of {param_name_x} are not equal: {param_val_x}/{param_val_y}')
            return False

    return True


def get_edge_list(nodes):
    index = {}
    for i, node in enumerate(nodes):
        index[node] = i

    index[None] = None

    parents = []
    for i, node in enumerate(nodes):
        for j, parent in enumerate(node.parents):
            parents.append((i, j, index[parent[0]], parent[1]))

    children = []
    for i, node in enumerate(nodes):
        child_idx = sorted([index[child] for child in node.children])
        children.append((i, *child_idx))

    return parents, children


def is_edge_equal(nodes_x, nodes_y):
    parents_x, children_x = get_edge_list(nodes_x)
    parents_y, children_y = get_edge_list(nodes_y)

    if len(parents_x) != len(parents_y):
        print('The number of parent connections are not equal')
        return False
    for parent_x, parent_y in zip(parents_x, parents_y):
        if parent_x != parent_y:
            print('Parent connection is not equal')
            return False

    if len(children_x) != len(children_y):
        print('The number of child connections are not equal')
        return False
    for child_x, child_y in zip(children_x, children_y):
        if child_x != child_y:
            print('Child connection is not equal')
            return False

    return True


def is_graph_equal(nodes_x, nodes_y):
    if len(nodes_x) != len(nodes_y):
        print('The number of node is not equal.')
        return False

    # whether two nodes are equal
    for node_x, node_y in zip(nodes_x, nodes_y):
        if not is_node_equal(node_x, node_y):
            return False

    return is_edge_equal(nodes_x, nodes_y)