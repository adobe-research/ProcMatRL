# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import glob
import os
import sys
import copy
import random
from collections import deque
import numbers
import subprocess
import platform
import tempfile
import signal

import numpy as np
import torch
import torchvision
from ordered_set import OrderedSet # pip install ordered-set
import networkx as nx

from .sbs_param_type import SBSParamType, param_val_to_type, param_type_idx_to_name
from . import sbs_graph_nodes
from .sbs_graph_nodes import SBSNode, SBSNodeDefinition, SBSOutputNode, SBSInputNode, SBSUnsupportedNode, SBSNodeParameter, SBSNodeInput, SBSNodeOutput
from .sbs_function_graph import SBSFunctionGraph, SBSFunctionInput, SBSFunctionUnsupported
from .sbs_resource import SBSBitmapResource, SBSResourceDefinition
from . import io_sbs # import load_sbs_graph, save_sbs_graph, legacy_generator_output_name_map, legacy_generator_map
from .io_json import load_json_graph, save_json_graph, create_json_node, get_node_type_signature, match_node_type_signature
from .sbs_utils import read_image, write_image, MissingDependencyError, InvalidNodeError, UnsupportedNodeExpansionError, randomize_tensor, resolve_dependency_path, missing_dependencies, center_normal_map

class SBSGraphOutput:
    '''
    An output slot of a graph.
    '''
    def __init__(self, name, usage=None, group='', parent=None):
        self.name = name
        self.usage = usage
        self.group = group
        self.parent = parent

    def get_dtype(self):
        if self.parent is None:
            # graph output is not connected, this output is undefined
            return None
        else:
            return self.parent.get_dtype()

class SBSGraphParameterPreset:
    '''
    A preset for graph parameters.
    '''
    def __init__(self, name, params=None):
        self.name = name
        self.params = params if params is not None else []

    def add_param(self, param):
        if param.name in [gp.name for gp in self.params]:
            raise RuntimeError('A parameter with the given name already exists in this graph, remove it first.')
        self.params.append(param)

class SBSGraph(torch.nn.Module):
    '''
    A node graph.
    '''
    def __init__(self, graph_name, use_alpha=False, res=None):
        super().__init__()

        # basic information: graph name and resolution
        self.name = graph_name

        self.res = [11, 11] if res is None else res  # exponent power of two, Default in substance designer: [2048, 2048]

        # all nodes in the graph
        self.nodes = torch.nn.ModuleList()

        # list of graph inputs, parameters and outputs
        self.inputs = [] # connectable inputs of the graph
        self.params = [] # parameters of the graph
        self.outputs = []

        self.param_presets = [] # parameter presets (not available anymore once graph as been conditioned)

        self.resources = [] # a list of resources of the graph (like bitmaps)

        # alpha channel switch (default: off)
        self.use_alpha = use_alpha

        # set in get_active_nodes
        self.active_unsupported_gens = None
        self.active_unsupported_gen_outputs = None # active outputs of active unsupported generator nodes (an active node may have both active and inactive outputs)
        self.active_node_seq = None
        self.active_output_nodes = None

        self.device = torch.device('cpu')

        self.node_output_hooks = {}

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            for node in self.nodes:
                node.device = device
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def get_node_by_name(self, name, error_on_miss=False):
        for node in self.nodes:
            if node.name == name:
                return node
        if error_on_miss:
            raise RuntimeError(f'A node with name {name} was not found in graph {self.name}.')
        else:
            return None

    def get_input_by_name(self, name, error_on_miss=False):
        for gi in self.inputs:
            if gi.name == name:
                return gi
        if error_on_miss:
            raise RuntimeError(f'An input with name {name} was not found in graph {self.name}.')
        else:
            return None

    def get_param_by_name(self, name, error_on_miss=False):
        for gp in self.params:
            if gp.name == name:
                return gp
        if error_on_miss:
            raise RuntimeError(f'A parameter with name {name} was not found in graph {self.name}.')
        else:
            return None

    def get_output_by_name(self, name, error_on_miss=False):
        for go in self.outputs:
            if go.name == name:
                return go
        if error_on_miss:
            raise RuntimeError(f'An output with name {name} was not found in graph {self.name}.')
        else:
            return None

    def get_node_output_by_uname(self, uname, error_on_miss=False):
        for node in self.nodes:
            for node_output in node.outputs:
                if node_output.uname() == uname:
                    return node_output
        if error_on_miss:
            raise RuntimeError(f'A node output with variable name {uname} was not found graph {self.name}.')
        else:
            return None

    def gen_unique_node_name(self, node_type):
        '''
        Generate a unique variable name for a node in the graph.
        '''

        prefix = f'{node_type}_'
        existing_inds = [int(node.name[len(prefix):]) for node in self.nodes if node.name.startswith(prefix) and node.name[len(prefix):].isdigit()]
        if len(existing_inds) == 0:
            idx = 1
        else:
            idx = max(existing_inds)+1

        return f'{prefix}{idx}'

    def make_node(self, node_type, node_name=None, node_func=None, graph_input=None, graph_output=None, node_def=None):

        # create unique name
        if node_name is None:
            node_name = self.gen_unique_node_name(node_type)

        if node_type == 'Unsupported':
            node = SBSUnsupportedNode(
                name=node_name, node_func=node_func,
                output_res=self.res, use_alpha=self.use_alpha, definition=node_def)
        elif node_type == 'Input':
            if graph_input is None:
                raise RuntimeError('Need reference to graph input parameter for input node.')
            node = SBSInputNode(name=node_name, graph_input=graph_input)
        elif node_type == 'Output':
            if graph_output is None:
                raise RuntimeError('Need reference to graph output for output node.')
            node = SBSOutputNode(name=node_name, graph_output=graph_output)
        else:
            node_class = getattr(sbs_graph_nodes, f'SBS{node_type}Node')
            node = node_class(name=node_name, output_res=self.res, use_alpha=self.use_alpha)

        return node

    def create_node(self, node_type, node_name=None, node_func=None, graph_input=None, graph_output=None, node_def=None):

        node = self.make_node(node_type=node_type, node_name=node_name, node_func=node_func, graph_input=graph_input, graph_output=graph_output, node_def=node_def)

        self.add_node(node)

        return node

    def create_output(self, name, usage=None, group=''):

        graph_output = SBSGraphOutput(name=name, usage=usage, group=group)

        self.add_output(graph_output=graph_output)

        return graph_output

    def add_input(self, graph_input):
        if graph_input.name in [gi.name for gi in self.inputs]:
            raise RuntimeError('An input with the given name already exists in this graph, remove it first.')
        self.inputs.append(graph_input)

    def add_param(self, graph_param):
        if graph_param.name in [gp.name for gp in self.params]:
            raise RuntimeError('A parameter with the given name already exists in this graph, remove it first.')
        self.params.append(graph_param)

    def add_output(self, graph_output):
        if graph_output.name in [n.name for n in self.outputs]:
            raise RuntimeError('An output with the given name already exists in this graph, remove it first.')
        self.outputs.append(graph_output)

    def remove_output(self, output):
        index = [oi for oi, o in enumerate(self.outputs) if o == output]
        if len(index) == 0:
            raise RuntimeError('The given output is not in this graph.')
        elif len(index) > 1:
            raise RuntimeError('This output has duplicates in the graph.')
        index = index[0]
        del self.outputs[index]

    def add_param_preset(self, param_preset):
        if param_preset.name in [preset.name for preset in self.param_presets]:
            raise RuntimeError('A parameter preset with the given name already exists in this graph, remove it first.')
        self.param_presets.append(param_preset)

    def remove_param_preset(self, param_preset):
        index = [pi for pi, p in enumerate(self.param_presets) if p == param_preset]
        if len(index) == 0:
            raise RuntimeError('The given parameter preset is not in this graph.')
        elif len(index) > 1:
            raise RuntimeError('This parameter preset has duplicates in the graph.')
        index = index[0]
        del self.param_presets[index]

    def get_param_preset_by_name(self, name, error_on_miss=False):
        for preset in self.param_presets:
            if preset.name == name:
                return preset
        if error_on_miss:
            raise RuntimeError(f'A parameter preset with name {name} was not found in this graph.')
        else:
            return None

    def apply_param_preset(self, preset_name):
        param_preset = self.get_param_preset_by_name(name=preset_name, error_on_miss=True)
        for preset_param in param_preset.params:
            graph_param = self.get_param_by_name(name=preset_param.name)
            if graph_param is None:
                print (f'WARNING: when applying a preset: the parameter {preset_param.name} was not found in the graph.')
                # raise RuntimeError(f'Could not apply preset, the parameter {preset_param.name} was not found in the graph.')
            else:
                graph_param.val = preset_param.val
                graph_param.dtype = preset_param.dtype

    def add_node(self, node):
        if node.name in [n.name for n in self.nodes]:
            raise RuntimeError('A node with the given name already exists in this graph, remove it first.')
        self.nodes.append(node)

        # move the node to the current device
        node.to(self.device)

    def remove_node(self, node):
        index = [ni for ni, n in enumerate(self.nodes) if n == node]
        if len(index) == 0:
            raise RuntimeError('The given node is not in this graph.')
        elif len(index) > 1:
            raise RuntimeError('This node has duplicates in the graph.')
        index = index[0]
        del self.nodes[index]

        # remove all connections of the removed node
        # (so that the parent node outputs don't have dangling children)
        for node_input in list(node.get_connected_inputs()):
            node_input.disconnect()

        # remove all connections that reference the removed node
        for node_output in node.outputs:
            node_output.disconnect()

    def get_function_nodes(self):
        fnodes = []
        for node in self.nodes:
            for source_param in node.source_params:
                if isinstance(source_param.val, SBSFunctionGraph):
                    for fnode in source_param.val.nodes:
                        fnodes.append((source_param.val, fnode))
        return fnodes

    def get_unused_source_params(self):
        unused_source_params = []
        for node in self.nodes:
            unused_source_params.extend(node.get_unused_source_params())
        return unused_source_params

    @staticmethod
    def load_sbs(
            graph_name, filename, resource_dirs, use_alpha=False, res=None,
            output_usages=None, prune_inactive_nodes=True, expand_unsupported_nodes=True, expand_unsupported_fnodes=True, skip_unsupported_expansions=False, force_opengl_normals=True,
            remove_levels_node_in_normals_branch=False, remove_passthrough_nodes=False, validate=True, allow_unsupported_nodes=False, condition_active_node_params=True,
            use_abs_paths=True, explicit_default_params=False, skip_unsupported_params=False, param_preset=None, clamp_params=True, default_output_format=0,
            remove_useless_invert_nodes=False, add_levels_nodes=False):

        # set default output format: 0 is L8; 1 is L16; 2 is HDR 16F; 3 is HDR 32F
        SBSNode.default_output_format = default_output_format

        # load the graph
        graph = io_sbs.load_sbs_graph(
            graph_name=graph_name, filename=filename, resource_dirs=resource_dirs, use_alpha=use_alpha, res=res,
            use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params, skip_unsupported_params=skip_unsupported_params)

        # preprocess the graph
        graph.preprocess(
            output_usages=output_usages, filename=filename, resource_dirs=resource_dirs, prune_inactive_nodes=prune_inactive_nodes,
            expand_unsupported_nodes=expand_unsupported_nodes, expand_unsupported_fnodes=expand_unsupported_fnodes, skip_unsupported_expansions=skip_unsupported_expansions,
            force_opengl_normals=force_opengl_normals, remove_levels_node_in_normals_branch=remove_levels_node_in_normals_branch, remove_passthrough_nodes=remove_passthrough_nodes, validate=validate,
            allow_unsupported_nodes=allow_unsupported_nodes, condition_active_node_params=condition_active_node_params,
            use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params, param_preset=param_preset,
            clamp_params=clamp_params, remove_useless_invert_nodes=remove_useless_invert_nodes, add_levels_nodes=add_levels_nodes)

        return graph

    def save_sbs(self, filename=None, resolve_resource_dirs=None, package_dependencies_dir=None, use_networkx=True):
        return io_sbs.save_sbs_graph(graph=self, filename=filename, resolve_resource_dirs=resolve_resource_dirs,
                                     package_dependencies_dir=package_dependencies_dir, use_networkx=use_networkx)

    @staticmethod
    def load_json(
            graph_name, filename=None, json_data=None, resource_dirs=None, use_alpha=False, res=None,
            output_usages=None, prune_inactive_nodes=True, expand_unsupported_nodes=True, expand_unsupported_fnodes=True, skip_unsupported_expansions=False, force_opengl_normals=True,
            remove_levels_node_in_normals_branch=True, remove_passthrough_nodes=True, validate=True, allow_unsupported_nodes=False, condition_active_node_params=True,
            use_abs_paths=True, explicit_default_params=False, param_preset=None, clamp_params=True, default_output_format=0, remove_useless_invert_nodes=False,
            add_levels_nodes=False):

        if resource_dirs is None:
            resource_dirs = {}

        # set default output format: 0 is L8; 1 is L16; 2 is HDR 16F; 3 is HDR 32F
        SBSNode.default_output_format = default_output_format

        # load the graph
        graph = load_json_graph(graph_name=graph_name, filename=filename, json_data=json_data, use_alpha=use_alpha, res=res, use_abs_paths=use_abs_paths)

        # preprocess the graph
        graph.preprocess(
            output_usages=output_usages, filename=filename, resource_dirs=resource_dirs, prune_inactive_nodes=prune_inactive_nodes,
            expand_unsupported_nodes=expand_unsupported_nodes, expand_unsupported_fnodes=expand_unsupported_fnodes, skip_unsupported_expansions=skip_unsupported_expansions,
            force_opengl_normals=force_opengl_normals, remove_levels_node_in_normals_branch=remove_levels_node_in_normals_branch, remove_passthrough_nodes=remove_passthrough_nodes, validate=validate,
            allow_unsupported_nodes=allow_unsupported_nodes, condition_active_node_params=condition_active_node_params,
            use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params, param_preset=param_preset,
            clamp_params=clamp_params, remove_useless_invert_nodes=remove_useless_invert_nodes, add_levels_nodes=add_levels_nodes)

        return graph

    def save_json(self, filename=None, ignore_default_params=False, **json_kwargs):
        return save_json_graph(graph=self, filename=filename, ignore_default_params=ignore_default_params, **json_kwargs)

    def preprocess(
            self, output_usages=None, filename=None, resource_dirs=None, prune_inactive_nodes=True,
            expand_unsupported_nodes=True, expand_unsupported_fnodes=True, skip_unsupported_expansions=False, force_opengl_normals=True,
            remove_levels_node_in_normals_branch=False, remove_passthrough_nodes=False, validate=True,
            allow_unsupported_nodes=False, condition_active_node_params=True, use_abs_paths=True,
            explicit_default_params=False, param_preset=None, clamp_params=True, remove_useless_invert_nodes=False,
            add_levels_nodes=False):

        if param_preset is not None:
            if isinstance(param_preset, str):
                self.apply_param_preset(preset_name=param_preset)
            elif isinstance(param_preset, int):
                self.apply_param_preset(preset_name=self.param_presets[param_preset].name)
            else:
                raise RuntimeError('Unknown format for parameter preset.')

        if prune_inactive_nodes:
            # prune nodes that are not used before expanding nodes, so that we don't need to skip graphs
            # due to unsupported nodes that are not actually used
            self.get_active_nodes(prune=True, output_usages=output_usages, differentiable_only=False)

        if expand_unsupported_nodes:
            self.expand_unsupported_nodes(
                filename=filename, resource_dirs=resource_dirs, allow_unsupported_nodes=allow_unsupported_nodes, skip_unsupported_expansions=skip_unsupported_expansions,
                use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params)
            # prune again after expansion to remove introduced inactive nodes
            if prune_inactive_nodes:
                self.get_active_nodes(prune=True, output_usages=output_usages, differentiable_only=False)
        if expand_unsupported_fnodes:
            self.expand_unsupported_fnodes(filename=filename, resource_dirs=resource_dirs, use_abs_paths=use_abs_paths)
        if remove_levels_node_in_normals_branch:
            self.remove_levels_node_in_normals_branch()
        if add_levels_nodes:
            self.add_levels_node(usage='baseColor')
            self.add_levels_node(usage='roughness')
        if remove_passthrough_nodes:
            self.remove_passthrough_nodes()
        self.get_active_nodes(prune=prune_inactive_nodes, output_usages=output_usages, differentiable_only=False)
        if validate:
            self.validate(raise_exception=True, allow_unsupported_nodes=allow_unsupported_nodes, output_usages=output_usages, filename=filename, resource_dirs=resource_dirs)
        if condition_active_node_params:
            self.condition_active_node_params(clamp_params)
        if force_opengl_normals:
            self.force_opengl_normals()
        if remove_useless_invert_nodes:
            self.remove_useless_invert_nodes()

        # only use differentiably nodes as active node sequence
        # self.get_active_nodes(prune=False, output_usages=output_usages, differentiable_only=True)

    def enable_partial_differentiation(self, output_usages=None, max_allowed_partial_diff_depth=np.inf):
        # find differentiable node sequences
        self.get_partial_active_nodes(output_usages=output_usages, max_depth=max_allowed_partial_diff_depth)
        self.use_partial_differentiation = True
        # # detach a subgraph that computes unsupport nodes
        # self.active_subgraph_of_unsupported_gens = self.subgraph_of_unsupported_nodes()

    def estimate_num_of_split_graphs(self):
        tot = 1
        for node in self.active_node_seq:
            if node.type in ['Switch', 'MultiSwitch']:
                connected_inputs = node.get_connected_inputs()
                tot *= len(connected_inputs)
            elif node.type == 'Unsupported' and node.func in ['material_switch']:
                tot *= 2  # split to two materials
        return tot

    def estimate_min_split_graphs(self):
        tot = 0
        for node in self.active_node_seq:
            if node.type in ['Switch', 'MultiSwitch']:
                connected_inputs = node.get_connected_inputs()
                tot = max(len(connected_inputs), tot)
            elif node.type == 'Unsupported' and node.func in ['material_switch']:
                tot = max(2, tot)  # split to two materials
        return tot

    def refactor_switch_nodes(self, max_num_graphs=256, switch_to_other=False):
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(10000, recursion_limit))
        est = self.estimate_num_of_split_graphs()

        # remove switch nodes
        graph_without_switch_nodes = copy.deepcopy(self)
        self.remove_switch_nodes(graph_without_switch_nodes, switch_to_other)
        all_graphs = [graph_without_switch_nodes]

        # sample other branches
        if est <= max_num_graphs:
            split_graphs = self.refactor_switch_nodes_recursively(self)
        else:
            min_split_graphs = self.estimate_min_split_graphs()  # ensure each branch is covered
            split_graphs = self.resample_switch_nodes_randomly(self, max(min_split_graphs, max_num_graphs))
        all_graphs.extend(split_graphs)

        sys.setrecursionlimit(recursion_limit)
        return all_graphs

    @staticmethod
    def refactor_target_switch_node(graph, node_name, input_id, switch_to_other=False):
        for node in graph.active_node_seq:
            if node.name == node_name:
                if node.type in ['Switch', 'MultiSwitch']:
                    # reconnect input nodes directly to the output of switch nodes
                    parent = node.inputs[input_id].parent
                    parent_output = node.inputs[input_id].parent_output
                    if parent is None or parent_output is None:
                        # it's trick here that we automatically use the other slot
                        if switch_to_other:
                            another_input_id = 1 - input_id
                            parent = node.inputs[another_input_id].parent
                            parent_output = node.inputs[another_input_id].parent_output
                            print(f'Try to switch to another slot {another_input_id} of node {node_name}')
                        if parent is None or parent_output is None:
                            raise RuntimeError(f'Slot {input_id} of node {node_name} is not connected.')

                    connected_child_inputs = node.get_connected_child_inputs()
                    for child_input in connected_child_inputs:
                        child_input.connect(parent_output=parent_output)

                elif node.type == 'Unsupported' and node.func in ['material_switch']:
                    def find_input_slots():
                        for node_input_ in node.inputs:
                            if node_input_.name == input_slot_name:
                                return node_input_
                        raise RuntimeError('Cannot find input slots by names.')

                    # find all output slots and corresponding input slots, draw a direct connection
                    for outputs in node.outputs:
                        if len(outputs.children) > 0:
                            input_slot_name = f'material{input_id}_{outputs.name}'
                            node_input = find_input_slots()
                            parent_output = node_input.parent_output
                            if not SBSGraph.is_valid_input_connection(graph, node_input) or parent_output is None:
                                raise NotImplementedError('Unconnected material switch input is not supported yet. '
                                                          'Should create a default node here.')

                            connected_child_inputs = [child_input for child_input in outputs.children]
                            for child_input in connected_child_inputs:
                                child_input.connect(parent_output=parent_output)

                # remove this switch node
                for k, active_node in enumerate(graph.active_node_seq):
                    if active_node.name == node.name:
                        graph.active_node_seq.pop(k)
                        break
                graph.remove_node(node)
                return

        raise RuntimeError(f'Cannot find {node_name}')

    @staticmethod
    def is_valid_input_connection(graph, node_input):
        return node_input.parent is not None and (node_input.parent in graph.active_node_seq or node_input.parent in graph.active_unsupported_gens)

    @staticmethod
    def refactor_switch_nodes_recursively(graph):
        split_graphs = []

        for node in graph.active_node_seq:
            if node.type in ['Switch', 'MultiSwitch']:
                for i, node_input in enumerate(node.inputs):
                    if SBSGraph.is_valid_input_connection(graph, node_input):
                        split_graph = copy.deepcopy(graph)
                        SBSGraph.refactor_target_switch_node(split_graph, node.name, i)
                        split_graphs.append(split_graph)
                break

            elif node.type == 'Unsupported' and node.func in ['material_switch']:
                for input_id in [1, 2]:
                    split_graph = copy.deepcopy(graph)
                    SBSGraph.refactor_target_switch_node(split_graph, node.name, input_id)
                    split_graphs.append(split_graph)
                break

        if len(split_graphs) == 0:
            return [graph]
        else:
            all_recursively_split_graphs = []
            for split_graph in split_graphs:
                recursively_split_graphs = SBSGraph.refactor_switch_nodes_recursively(split_graph)
                all_recursively_split_graphs.extend(recursively_split_graphs)
            return all_recursively_split_graphs

    @staticmethod
    def resample_switch_nodes_randomly(graph, n_samples):
        split_graphs = []
        for i in range(n_samples):
            split_graph = copy.deepcopy(graph)
            SBSGraph.refactor_switch_nodes_randomly(split_graph, i)
            split_graphs.append(split_graph)
        return split_graphs

    @staticmethod
    def refactor_switch_nodes_randomly(graph, branch_index=-1):
        switch_node = None
        valid_input_index = []
        for node in graph.active_node_seq:
            if node.type in ['Switch', 'MultiSwitch']:
                for i, node_input in enumerate(node.inputs):
                    if SBSGraph.is_valid_input_connection(graph, node_input):
                        valid_input_index.append(i)
                switch_node = node
                break
            elif node.type == 'Unsupported' and node.func in ['material_switch']:
                valid_input_index = [1, 2]
                switch_node = node
                break

        if switch_node is not None:
            if 0 <= branch_index < len(valid_input_index):
                sampled_index = valid_input_index[branch_index]
            else:
                sampled_index = random.sample(valid_input_index, 1)[0]
            # Can be optimized.
            SBSGraph.refactor_target_switch_node(graph, switch_node.name, sampled_index)
            return SBSGraph.refactor_switch_nodes_randomly(graph, branch_index)
        else:
            return graph

    @staticmethod
    def remove_switch_nodes(graph, switch_to_other=False):
        index = None
        switch_node = None
        for node in graph.active_node_seq:
            if node.type == 'Switch':
                switch = node.get_param_by_name('flag')
                index = 0 if switch.val else 1
                switch_node = node
                break
            elif node.type == 'MultiSwitch':
                input_selection = node.get_param_by_name('input_selection')
                index = input_selection.val - 1
                if index == -1:
                    index = 0
                switch_node = node
                break
            elif node.type == 'Unsupported' and node.func in ['material_switch']:
                switch = node.get_param_by_name('Switch')
                index = 2 if switch.val else 1
                switch_node = node
                break

        if index is not None:
            SBSGraph.refactor_target_switch_node(graph, switch_node.name, index, switch_to_other=switch_to_other)
            return SBSGraph.remove_switch_nodes(graph, switch_to_other=switch_to_other)
        else:
            return graph

    def expand_unsupported_nodes(self, filename, resource_dirs, allow_unsupported_nodes=False, skip_unsupported_expansions=False, use_abs_paths=True, explicit_default_params=False):
        # expand all unsupported non-atomic non-generator nodes

        unsupported_nodes = [node for node in self.nodes if node.type == 'Unsupported' and node.definition() is not None and len(node.get_connected_inputs()) > 0]
        max_expansions = 1000
        num_expansions = 0
        skipped_nodes = []
        while len(unsupported_nodes) > 0:
            if num_expansions+1 > max_expansions:
                raise RuntimeError('Could not remove the unsupported nodes with the given node expansion budget.')
            unsupported_node = unsupported_nodes[0]
            try:
                self.expand_node(
                    ex_node=unsupported_node, filename=filename, resource_dirs=resource_dirs, allow_unsupported_nodes=allow_unsupported_nodes,
                    use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params)
            except UnsupportedNodeExpansionError as err:
                if skip_unsupported_expansions:
                    # print(f'Skipping expansion of node {unsupported_node.name} (function: {unsupported_node.func}):\n  {str(err)}')
                    skipped_nodes.append(unsupported_node)
                else:
                    raise err
            unsupported_nodes = [node for node in self.nodes if node.type == 'Unsupported' and node.definition() is not None and len(node.get_connected_inputs()) > 0 and node not in skipped_nodes]
            num_expansions += 1

    def expand_unsupported_fnodes(self, filename, resource_dirs, use_abs_paths=True):
        # expand all unsupported function nodes

        unsupported_fnodes = [fnode for fnode in self.get_function_nodes() if isinstance(fnode[1], SBSFunctionUnsupported)]
        max_expansions = 1000
        num_expansions = 0
        skipped_nodes = []
        while len(unsupported_fnodes) > 0:
            if num_expansions+1 > max_expansions:
                raise RuntimeError('Could not remove the unsupported nodes with the given node expansion budget.')
            fgraph = unsupported_fnodes[0][0]
            unsupported_fnode = unsupported_fnodes[0][1]
            # try:
            # cannot handle unsupported function graph nodes at the moment, since functiong graphs are not saved and always need to be executed to get the output values before saving to sbs
            fgraph.expand_node(ex_node=unsupported_fnode, filename=filename, resource_dirs=resource_dirs, use_abs_paths=use_abs_paths)
            # except UnsupportedNodeExpansionError as err:
            #     if skip_unsupported_expansions:
            #         print(f'Skipping expansion of a function graph node with type: {unsupported_node.type}:\n'+str(err))
            #         skipped_nodes.append(unsupported_node)
            #     else:
            #         raise err
            unsupported_fnodes = [fnode for fnode in self.get_function_nodes() if isinstance(fnode[1], SBSFunctionUnsupported) and fnode not in skipped_nodes]
            num_expansions += 1

    # there is no check for cyclic dependencies, so this function will not return if there are cyclic dependencies
    def expand_node(self, ex_node, filename, resource_dirs, allow_unsupported_nodes=False, use_abs_paths=True, explicit_default_params=False):
        '''
        Expand a given node with its definition.
        The node is replaced by the graph that defines it.
        '''
        if ex_node not in self.nodes:
            raise RuntimeError('The given node is not in the graph.')

        # print(f'expanding node {ex_node.name} (function: {ex_node.func})')

        if ex_node.definition() is None:
            raise RuntimeError('Cannot expand an atomic node.')

        expanded_node_dep_path = resolve_dependency_path(path=ex_node.definition().path, source_filename=filename, resource_dirs=resource_dirs)

        try:
            if os.path.splitext(expanded_node_dep_path)[1] == '.sbsar':
                raise NotImplementedError('Cannot expand a node that is defined by a compiled substance graph (a file ending in .sbsar).')

            # parse graph that defines the node type
            node_def_graph = SBSGraph.load_sbs(
                graph_name=ex_node.definition().graph, filename=expanded_node_dep_path, resource_dirs=resource_dirs, use_alpha=self.use_alpha,
                prune_inactive_nodes=False, expand_unsupported_nodes=True, expand_unsupported_fnodes=True, force_opengl_normals=False,
                remove_levels_node_in_normals_branch=False, remove_passthrough_nodes=True, validate=True,
                allow_unsupported_nodes=allow_unsupported_nodes, condition_active_node_params=False,
                use_abs_paths=use_abs_paths, explicit_default_params=explicit_default_params)
        except (NotImplementedError, InvalidNodeError) as err:
            raise UnsupportedNodeExpansionError(f'Error while loading the definition graph of node {ex_node.name} (function: {ex_node.func}):\n  {str(err)}') from err

        # replace input nodes in the parameter function graphs of each node in the node definition graph
        # with the function graph of parameters of the node (or a constant value)
        # skip input nodes that start with $, these refer to global parameters that  will only be resolved when the graph is run
        for ndg_node in node_def_graph.nodes:
            for ndg_param in ndg_node.source_params:
                if isinstance(ndg_param.val, SBSFunctionGraph):
                    for ndg_fnode in list(ndg_param.val.nodes): # copy the node map since we will be mutating it
                        if isinstance(ndg_fnode, SBSFunctionInput) and not ndg_fnode.data.startswith('$'):

                            # find corresponding parameter of the expanded node
                            ex_param = None
                            for p in ex_node.source_params:
                                if p.name == ndg_fnode.data:
                                    ex_param = p

                            if ex_param is None:
                                # parameter is not defined in the expanded node, use default value from the definition graph
                                # node_def_graph_input_dict = {p.name: p.val for p in node_def_graph.params}
                                def_graph_param = node_def_graph.get_param_by_name(name=ndg_fnode.data)
                                if def_graph_param is None:
                                    if ndg_fnode.data == 'Bricks_Pits_Slider':
                                        # this float1 parameter is undefined in some substance source materials like
                                        # Sbs/Terracotta/bricks_iron_combined_light/bricks_iron_combined_light.sbs
                                        # (probably all graphs that use Sbs/common_dependencies/light_buff_smooth_iron_spot.sbs, where it is defined)
                                        # use 0 for it for now
                                        ex_param_val = 0.0
                                    elif ndg_node.name == 'Normal_1' and node_def_graph.name == 'Bitmap2Material_2_Light' and ndg_fnode.data == '':
                                        # this float1 parameter is undefined
                                        ex_param_val = 0.0
                                    elif ndg_node.name == 'BlurHQ_1' and node_def_graph.name == 'mg_edge_notch' and ndg_fnode.data == 'radius':
                                        # this float1 parameter is undefined
                                        ex_param_val = 0.0
                                    else:
                                        raise RuntimeError(f'Could not find default value for parameter {ndg_fnode.data} in graph {node_def_graph.name}.')
                                else:
                                    ex_param_val = def_graph_param.val # node_def_graph_input_dict[ndg_fnode.data]
                            else:
                                # parameter is defined in the expanded node
                                ex_param_val = ex_param.val

                            ndg_param.val.expand_input_node(ex_node=ndg_fnode, new_val=ex_param_val)

        # update names of nodes in the definition graph and add them to the current graph
        for ndg_node in node_def_graph.nodes:

            # update name of the node (this also updates the variable names of the node outputs)
            new_name = self.gen_unique_node_name(ndg_node.type)
            ndg_node.name = new_name
            if ndg_node.type not in ['Input', 'Output']:
                self.add_node(ndg_node)

        # change all connections pointing to the expanded node
        # to point to the corresponding node output in the definition graph instead
        outer_nodes_connected_to_ndg = set()
        for ex_node_output in list(ex_node.outputs):

            # find the corresponding connection in the node definition graph
            def_graph_output = node_def_graph.get_output_by_name(ex_node_output.name)
            if def_graph_output is None or def_graph_output.parent is None or len(def_graph_output.parent.inputs) != 1:
                raise RuntimeError(f'The output {ex_node_output.name} of node {ex_node.name} was not found in its definition.')
            def_graph_conn = def_graph_output.parent.inputs[0]

            for node_input in set(ex_node_output.children): # a node output can have multiple connections to the same child node
                node_input.connect(parent_output=def_graph_conn.parent_output)
                if node_input.node is None:
                    raise RuntimeError('Found a node input that is not attached to a node.')
                outer_nodes_connected_to_ndg.add(node_input.node)

        # change all connections pointing to an input node in the ndg
        # to point to the corresponding node output in the current graph instead
        for ndg_node in list(node_def_graph.nodes) + list(outer_nodes_connected_to_ndg):
            for ndg_input in list(ndg_node.get_connected_inputs()): # copy the list of node inputs since I will be mutating it
                if ndg_input.parent.type == 'Input' and ndg_input.parent in node_def_graph.nodes:

                    # find the corresponding connection of the expanded node
                    input_param_name = ndg_input.parent.graph_input.name
                    ex_input = ex_node.get_input_by_name(input_param_name)
                    if ex_input is not None:
                        # this input of the expanded node is connected,
                        # change the connection of the definition node to point to the target of the connection in the expanded node
                        ndg_input.connect(parent_output=ex_input.parent_output)

        # remove all connections of the expanded node (need to do it after the loop above)
        for ex_node_input in list(ex_node.get_connected_inputs()): # copy the list of connections since I will be mutating it
            ex_node_input.disconnect()

        # remove all output and input nodes from the node definition graph
        # (this also removes any references to the output nodes)
        for ndg_node in list(node_def_graph.nodes):
            if ndg_node.type in ['Input', 'Output']:
                node_def_graph.remove_node(ndg_node)

        # remove the expanded node
        self.remove_node(ex_node)

    # get a list of unique dependencies
    def dependencies(self):
        deps = set()
        for node in self.nodes:
            if node.definition() is not None:
                deps.add(node.definition())
            for source_param in node.source_params:
                if isinstance(source_param.val, SBSFunctionGraph):
                    deps.update(source_param.val.dependencies())

        # if resources are present add a dependency to the sbs document itself (denoted by '?himself), this is how the resources can be accessed
        if len(self.resources) > 0 and '?himself' not in [dep.path for dep in deps]:
            deps.add(SBSResourceDefinition(resource='', path='?himself'))

        deps = list(deps)

        return deps

    # def update_node_definition_paths(self, old_subpath, new_subpath):
    #     for node in self.nodes:
    #         if node.definition() is not None and node.definition().path is not None:
    #             if old_subpath in node.definition().path:
    #                 node.definition().path.replace(old_subpath, new_subpath)
    #         for source_param in node.source_params:
    #             if isinstance(source_param.val, SBSFunctionGraph):
    #                 source_param.val.update_node_definition_paths(old_subpath=old_subpath, new_subpath=new_subpath)

    def validate(self, raise_exception=False, allow_unsupported_generators=True, allow_unsupported_nodes=False, output_usages=None, filename=None, resource_dirs=None):
        all_errors = []

        # check for invalid nodes
        # Invalid nodes cannot be fully represented.
        # Important information is lost when loading them
        # (like the function graph that defines them or a resource like an image that they depend on),
        # so saving an sbs file when these nodes are present gives and invalid file.
        invalid_node_errors = []
        for node in self.nodes:
            if node.func in ['fxmaps', 'pixelprocessor', 'bitmap', 'svg']:
                invalid_node_errors.append(f'The graph contains an invalid node: {node.name} (type: {node.func})')
        all_errors += invalid_node_errors

        # check for missing dependencies
        missing_dependency_errors = []
        missing_deps = missing_dependencies(deps=self.dependencies(), source_filename=filename, resource_dirs=resource_dirs)
        for missing_dep in missing_deps:
            missing_dependency_errors.append(f'The graph is missing a dependency: {missing_dep.path}')
        all_errors += missing_dependency_errors

        # check for unsupported nodes
        # Unsupported nodes can be represented, but cannot be executed differentiably.
        # Loading and saving works, but the graph can only be executed non-differentiably with the SAT by calling run_sat(...).
        unsupported_node_errors = []
        if not allow_unsupported_nodes:
            for node in self.nodes:
                if allow_unsupported_generators:
                    if node.type == 'Unsupported' and len(node.get_connected_inputs()) > 0:
                        unsupported_node_errors.append(f'The graph contains an unsupported non-generator node: {node.name} (type: {node.func}).')
                else:
                    if node.type == 'Unsupported':
                        unsupported_node_errors.append(f'The graph contains an unsupported node: {node.name} (type: {node.func}).')
            all_errors += unsupported_node_errors

        # check for output nodes with missing parents
        missing_output_parent_errors = []
        for output in self.outputs:
            if output.group == 'Material' and (output_usages is None or output.usage in output_usages):
                if output.parent is None:
                    missing_output_parent_errors.append(f'The graph contains an output node without parent: {output.name} (usage: {output.usage}).')
        all_errors += missing_output_parent_errors

        if raise_exception:
            if len(all_errors) > 0:
                if len(invalid_node_errors) > 0:
                    raise InvalidNodeError(invalid_node_errors[0])
                if len(missing_dependency_errors) > 0:
                    raise MissingDependencyError(missing_dependency_errors[0])
                elif len(unsupported_node_errors) > 0:
                    raise NotImplementedError(unsupported_node_errors[0])
                elif len(missing_output_parent_errors) > 0:
                    raise NotImplementedError(missing_output_parent_errors[0])
                else:
                    # this should not happen
                    raise RuntimeError('Unknown error.')

        return len(all_errors) == 0, invalid_node_errors, missing_dependency_errors, unsupported_node_errors, missing_output_parent_errors

    def add_render_node(self, shape_type, env_map_filename):

        shape_options = ['Sphere', 'Plane', 'Cylinder']

        if shape_type not in shape_options:
            raise RuntimeError(f'Unrecognized shape type: {shape_type}')

        # output_channels = self.run_sat(
        #     graph_filename=graph_filename, output_name=os.path.splitext(output_filename)[0], sat_dir=sat_dir,
        #     use_output_defaults=False, output_usages=None, return_output_channels=True, write_output_channels=False, randomize_generators=False, seed=None,
        #     image_format=os.path.splitext(output_filename)[1][1:])

        # add environment map resource
        env_map_resource_name = f'env_map_{os.path.splitext(os.path.basename(env_map_filename))[0]}'
        env_map_resource = SBSBitmapResource(name=env_map_resource_name)
        # env_map_resource.get_param_by_name('filepath').val = os.path.realpath(os.path.abspath(env_map_filename))
        env_map_resource.get_param_by_name('filepath').val = env_map_filename
        env_map_resource.get_param_by_name('format').val = os.path.splitext(env_map_filename)[1][1:]
        self.resources.append(env_map_resource)

        # add bitmap node that loads the environment map
        bitmap_node = self.create_node(node_type='Unsupported', node_name=self.gen_unique_node_name('bitmap'), node_func='bitmap', node_def=None)
        bitmap_node.add_param(SBSNodeParameter(name='bitmapresourcepath', val=env_map_resource_name, dtype=SBSParamType.STRING, name_xml='bitmapresourcepath'))
        # bitmap_node.add_param(SBSNodeParameter(name='format', val=SBSParamType.ENTRY_VARIANT, dtype=SBSParamType.INTEGER1, name_xml='bitmapresourcepath'))
        bitmap_node.get_param_by_name('format').val = SBSParamType.ENTRY_VARIANT.value
        bitmap_node.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='output'))

        # add PBR_render node
        render_node = self.create_node(
            node_type='Unsupported', node_name=self.gen_unique_node_name('PBR_render'), node_func='PBR_render',
            node_def=SBSNodeDefinition(graph='PBR_render', path='sbs://pbr_render.sbs'))
        render_node.add_input(SBSNodeInput(name='basecolor', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='basecolor'))
        render_node.add_input(SBSNodeInput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='normal'))
        render_node.add_input(SBSNodeInput(name='emissive', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='emissive'))
        render_node.add_input(SBSNodeInput(name='roughness', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='roughness'))
        render_node.add_input(SBSNodeInput(name='metallic', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='metallic'))
        render_node.add_input(SBSNodeInput(name='specularlevel', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='specularlevel'))
        render_node.add_input(SBSNodeInput(name='height', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='height'))
        render_node.add_input(SBSNodeInput(name='ambient_occlusion', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='ambient_occlusion'))
        render_node.add_input(SBSNodeInput(name='opacity', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='opacity'))
        render_node.add_input(SBSNodeInput(name='anisotropyLevel', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='anisotropyLevel'))
        render_node.add_input(SBSNodeInput(name='anisotropyAngle', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='anisotropyAngle'))
        render_node.add_input(SBSNodeInput(name='lens_dirt_map', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='lens_dirt_map'))
        render_node.add_input(SBSNodeInput(name='lens_aperture_map', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='lens_aperture_map'))
        render_node.add_input(SBSNodeInput(name='background_input', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='background_input'))
        render_node.add_input(SBSNodeInput(name='environment_map', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='environment_map'))

        render_node.add_param(SBSNodeParameter(name='shape', val=shape_options.index(shape_type), dtype=SBSParamType.INTEGER1, name_xml='shape'))
        render_node.add_param(SBSNodeParameter(name='normal_format', val=1, dtype=SBSParamType.INTEGER1, name_xml='normal_format')) # set OpenGL normal format
        render_node.add_param(SBSNodeParameter(name='camera_exposure', val=0.5, dtype=SBSParamType.FLOAT1, name_xml='camera_exposure')) # increase exposure a bit to make images brighter
        render_node.add_param(SBSNodeParameter(name='displacement_intensity', val=0.02, dtype=SBSParamType.FLOAT1, name_xml='displacement_intensity')) # increase exposure a bit to make images brighter

        render_node.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='output'))
        render_node.add_output(SBSNodeOutput(name='raw_irradiance', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='raw_irradiance'))
        render_node.add_output(SBSNodeOutput(name='raw_specular', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='raw_specular'))
        render_node.add_output(SBSNodeOutput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='normal'))
        render_node.add_output(SBSNodeOutput(name='uv', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='uv'))

        render_node.get_input_by_name('environment_map').connect(bitmap_node.get_output_by_name('output'))

        # add additional output and output node
        render_output_usage = 'rendered'
        render_output = self.create_output(name='render_output', usage=render_output_usage, group='render_output')
        render_output_node = self.create_node(node_type='Output', node_name=self.gen_unique_node_name('render_output'), graph_output=render_output)

        render_node.get_output_by_name('output').connect(render_output_node.get_input_by_name('input'))

        # connect existing outputs to PBR_render node
        # pbr_input_compatible_usages = {
        #     'basecolor': ['baseColor', 'diffuse'],
        #     'normal': ['normal'],
        #     'emissive': ['emissive'],
        #     'roughness': ['roughness'],
        #     'metallic': ['metallic'],
        #     'specularlevel': ['specularLevel', 'specular', 'glossiness'],
        #     'height': ['height', 'displacement'],
        #     'ambient_occlusion': ['ambientOcclusion'],
        #     'opacity': ['opacity'],
        #     'anisotropyLevel': ['anisotropyLevel'],
        #     'anisotropyAngle': ['anisotropyAngle']}
        pbr_input_compatible_usages = {
            'basecolor': ['baseColor', 'diffuse'],
            'normal': ['normal'],
            'roughness': ['roughness'],
            'metallic': ['metallic'],
            'height': ['height', 'displacement']}
        outputs_by_usage = {}
        for output in self.outputs:
            outputs_by_usage[output.usage] = output
        for pbr_input_name, compatible_usages in pbr_input_compatible_usages.items():
            for compatible_usage in compatible_usages:
                if compatible_usage in outputs_by_usage and outputs_by_usage[compatible_usage].parent is not None:
                    render_node.get_input_by_name(pbr_input_name).connect(outputs_by_usage[compatible_usage].parent.inputs[0].parent_output)
                    break

        return render_node, render_output_node, bitmap_node, render_output, env_map_resource

    def render_on_shape(self, output_filename, shape_type, env_map_filename, graph_filename, sat_dir, resource_dirs):

        render_node, render_output_node, bitmap_node, render_output, env_map_resource = self.add_render_node(
            shape_type, env_map_filename)

        # # TEMP: save graph with auxiliary nodes
        # self.save_sbs(filename=f'{os.path.splitext(output_filename)[0]}_temp_graph.sbs')

        # render with SAT, using only the PBR_render node output as output usage
        output_channels = self.run_sat(
            graph_filename=graph_filename, output_name=os.path.splitext(output_filename)[0], sat_dir=sat_dir, resource_dirs=resource_dirs,
            use_output_defaults=False, output_usages=[render_output.usage], return_output_channels=True, write_output_channels=False, randomize_generators=False, seed=None,
            image_format=os.path.splitext(output_filename)[1][1:])
        write_image(filename=output_filename, img=output_channels[render_output.usage].cpu().squeeze(dim=0), process=True)

        # remove all nodes and resources that were added for rendering
        self.remove_node(render_output_node)
        self.remove_node(render_node)
        self.remove_node(bitmap_node)
        self.remove_output(render_output)
        del self.resources[self.resources.index(env_map_resource)]

    def decompose_graph_from_outputs(self, output_usages=None, max_depth=np.inf):
        output_nodes = [output.parent for output in self.outputs if output.group == 'Material' and (output_usages is None or output.usage in output_usages)]
        # remove unconnected output nodes or output nodes that are directly connected to a valueprocessor (these produce non-image outputs, which can't be handled)
        output_nodes = [output_node for output_node in output_nodes if output_node.inputs[0].parent is not None and output_node.inputs[0].parent.func != 'valueprocessor']
        output_nodes = list(OrderedSet(output_nodes))
        if any(node is None for node in output_nodes):
            raise NotImplementedError('Output nodes without parent are not supported.')
        queue = deque(output_nodes)
        queue_depth = deque([0] * len(output_nodes))
        reachable_nodes = OrderedSet(node for node in output_nodes)

        # Run backward BFS to find unsupported nodes
        reachable_node_outputs = OrderedSet()
        while queue:
            node = queue.popleft()
            depth = queue_depth.popleft()
            for node_input in node.get_connected_inputs():
                reachable_node_outputs.add(node_input.parent_output)
                if node_input.parent.type == 'Unsupported':
                    reachable_nodes.add(node_input.parent)
                else:
                    if node_input.parent not in reachable_nodes and depth < max_depth:
                        queue.append(node_input.parent)
                        queue_depth.append(depth + 1)
                        reachable_nodes.add(node_input.parent)

        return reachable_nodes, reachable_node_outputs, output_nodes

    def reachable_from_outputs(self, output_usages=None):
        output_nodes = [output.parent for output in self.outputs if output.group == 'Material' and (output_usages is None or output.usage in output_usages)]
        output_nodes = list(OrderedSet(output_nodes))
        if any(node is None for node in output_nodes):
            raise NotImplementedError('Output nodes without parent are not supported.')
        queue = deque(output_nodes)
        reachable_nodes = OrderedSet(node for node in output_nodes)

        # Run backward BFS
        reachable_node_outputs = OrderedSet()
        while queue:
            node = queue.popleft()
            for node_input in node.get_connected_inputs():
                reachable_node_outputs.add(node_input.parent_output)
                if node_input.parent not in reachable_nodes:
                    queue.append(node_input.parent)
                    reachable_nodes.add(node_input.parent)

        return reachable_nodes, reachable_node_outputs, output_nodes


    def get_active_nodes(self, prune=False, output_usages=None, differentiable_only=True, max_depth=np.inf):
        '''
        Run data flow analysis to find active nodes and determine a topologically ordered sequence of active nodes.
        '''
        if differentiable_only:
            # Calculate reachable nodes and unsupported nodes from output
            reachable_nodes, reachable_node_outputs, self.active_output_nodes = self.decompose_graph_from_outputs(output_usages=output_usages, max_depth=max_depth)

            # initialize active nodes as reachable nodes that connected to only unreachable parent outputs
            # (note that a node can have a mix of reachable and unreachable outputs,
            # thus just checking that one of parent nodes is reachable is not enough to ensure that a given node can be reached.)
            active_init_nodes = OrderedSet() # reachable_unsupported_nodes
            for node in reachable_nodes:
                if all((node_input.parent_output is None or node_input.parent_output not in reachable_node_outputs) for node_input in node.inputs):
                    active_init_nodes.append(node)
        else:
            # Calculate reachable nodes from output
            reachable_nodes, reachable_node_outputs, self.active_output_nodes = self.reachable_from_outputs(output_usages=output_usages)

            # Initialize the set of active nodes with zero indegrees
            active_init_nodes = OrderedSet()
            for node in reachable_nodes:
                if len(node.get_connected_inputs()) == 0: # any node without input connections can act as a generator
                    active_init_nodes.add(node)

        # Determine other active nodes and count indegrees
        queue = deque(active_init_nodes)
        active_names = {node.name for node in active_init_nodes}
        indegrees = {name: 0 for name in active_names}

        # Run forward BFS (is this necessary apart from getting the indegree?)
        while queue:
            node = queue.popleft()
            for node_output in node.outputs:
                if node_output in reachable_node_outputs:
                    for child_node in OrderedSet(child_node_input.node for child_node_input in node_output.children):
                        if child_node in reachable_nodes:
                            if child_node.name not in active_names:
                                queue.append(child_node)
                                active_names.add(child_node.name)
                                indegrees[child_node.name] = 1
                            else:
                                indegrees[child_node.name] += 1

        # Check if all optimizable outputs are covered
        for node in self.active_output_nodes:
            if node.name not in active_names:
                raise RuntimeError('An optimizable output is not connected')

        # Run topology sorting to compute node sequence.
        # Does not contain input nodes, output nodes,
        # or non-differentiable generator nodes
        # (or any non-differentiable nodes if differentiable_only is set).
        queue = deque(active_init_nodes)
        self.active_unsupported_gens = []
        self.active_unsupported_gen_outputs = []
        self.active_node_seq = []
        while queue:
            node = queue.popleft()
            if differentiable_only:
                is_gen_node = node.type == 'Unsupported'
            else:
                is_gen_node = node.type == 'Unsupported' and len(node.get_connected_inputs()) == 0
            if is_gen_node:
                self.active_unsupported_gens.append(node)
                for output in node.outputs:
                    if output in reachable_node_outputs:
                        self.active_unsupported_gen_outputs.append(output)
            elif node.type not in ['Input', 'Output']:
                self.active_node_seq.append(node)
            for node_output in node.outputs:
                if node_output in reachable_node_outputs:
                    child_nodes = OrderedSet(child_node_input.node for child_node_input in node_output.children)
                    for child_node in child_nodes:
                        if child_node.name in active_names:
                            indegrees[child_node.name] -= 1
                            if not indegrees[child_node.name]:
                                queue.append(child_node)
        self.active_unsupported_gens = list(OrderedSet(self.active_unsupported_gens))
        self.active_unsupported_gen_outputs = list(OrderedSet(self.active_unsupported_gen_outputs))

        # optionally prune all nodes that are not in the node seq (nodes that are not needed when running the graph)
        if prune:
            active_nodes = self.active_node_seq + self.active_unsupported_gens + self.active_output_nodes
            for node in list(self.nodes):
                if node not in active_nodes:
                    self.remove_node(node)

            if output_usages is not None:
                outputs = []
                for output in self.outputs:
                    if output.group == 'Material' and output.usage in output_usages:
                        outputs.append(output)
                self.outputs = outputs

    def gen_subgraph_of_nondifferentiable_nodes(self):

        # deepcopy current graph
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(10000, recursion_limit))
        subgraph = copy.deepcopy(self)
        sys.setrecursionlimit(recursion_limit)

        # fetch names of active unsupported generator nodes
        unsupported_node_names = OrderedSet()
        for unsupported_node in self.active_unsupported_gens:
            unsupported_node_names.add(unsupported_node.name)

        # fetch unique names of active unsupported generator node outputs (not all outputs of unsupported generator nodes are active)
        unsupported_node_output_names = []
        for node_output in self.active_unsupported_gen_outputs:
            unsupported_node_output_names.append(node_output.uname())

        # find unsupported node in the just-copied graph
        unsupported_nodes = []
        for node in list(subgraph.nodes):
            if node.name in unsupported_node_names:
                unsupported_nodes.append(node)

        if len(unsupported_nodes) != len(self.active_unsupported_gens):
            raise RuntimeError(f'Can\'t find enough unsupported node: {len(unsupported_nodes)}/{len(self.active_unsupported_gens)})')

        # find a sequence of node that computes unsupported nodes
        queue = deque(unsupported_nodes)
        reachable_nodes = set(unsupported_nodes)
        while queue:
            node = queue.popleft()
            for node_input in node.get_connected_inputs():
                if node_input.parent not in reachable_nodes:
                    queue.append(node_input.parent)
                    reachable_nodes.add(node_input.parent)

        # delete all the irrelevant nodes
        for node in list(subgraph.nodes):
            if node not in reachable_nodes:
                subgraph.remove_node(node)

        # delete all output nodes
        for graph_output in subgraph.outputs:
            subgraph.remove_output(graph_output)

        # create output nodes for all output slots of the unsupported nodes
        for unsupported_node in unsupported_nodes:
            for node_output in unsupported_node.outputs:
                if node_output.uname() in unsupported_node_output_names:
                    graph_output = subgraph.create_output(name=node_output.uname(), group='input')
                    graph_output_node = subgraph.create_node(node_type='Output', node_name=node_output.uname(), graph_output=graph_output)
                    graph_output_node.get_input_by_name('input').connect(node_output)

        return subgraph

    def get_partial_active_nodes(self, output_usages=None, max_depth=np.inf):
        # Calculate reachable nodes and unsupported nodes from output
        reachable_nodes, reachable_node_outputs, self.active_output_nodes = self.decompose_graph_from_outputs(output_usages=output_usages, max_depth=max_depth)

        # initialize active nodes as reachable nodes that connected to only unreachable parent outputs
        # (note that a node can have a mix of reachable and unreachable outputs,
        # thus just checking that one of parent nodes is reachable is not enough to ensure that a given node can be reached.)
        active_init_nodes = OrderedSet() # reachable_unsupported_nodes
        for node in reachable_nodes:
            if all((node_input.parent_output is None or node_input.parent_output not in reachable_node_outputs) for node_input in node.inputs):
                active_init_nodes.append(node)

        # Determine other active nodes and count indegrees
        queue = deque(active_init_nodes)
        active_names = {node.name for node in active_init_nodes}
        indegrees = {name: 0 for name in active_names}

        # Run forward BFS (is this necessary apart from getting the indegree?)
        while queue:
            node = queue.popleft()
            for node_output in node.outputs:
                if node_output in reachable_node_outputs:
                    for child_node in set(child_node_input.node for child_node_input in node_output.children):
                        if child_node in reachable_nodes:
                            if child_node.name not in active_names:
                                queue.append(child_node)
                                active_names.add(child_node.name)
                                indegrees[child_node.name] = 1
                            else:
                                indegrees[child_node.name] += 1

        # Check if all optimizable outputs are covered
        for node in self.active_output_nodes:
            if node.name not in active_names:
                raise RuntimeError('An optimizable output is not connected')

        # Run topology sorting to compute node sequence
        # (does not contain unsupported nodes, input nodes, or output nodes)
        queue = deque(active_init_nodes)
        self.active_unsupported_gens = []
        self.active_unsupported_gen_outputs = []
        self.active_node_seq = []
        while queue:
            node = queue.popleft()
            if node.type == 'Unsupported':
                self.active_unsupported_gens.append(node)
                for output in node.outputs:
                    if output in reachable_node_outputs:
                        self.active_unsupported_gen_outputs.append(output)
            elif node.type not in ['Input', 'Output']:
                self.active_node_seq.append(node)
            for node_output in node.outputs:
                if node_output in reachable_node_outputs:
                    for child_node in set(child_node_input.node for child_node_input in node_output.children):
                        if child_node.name in active_names:
                            indegrees[child_node.name] -= 1
                            if not indegrees[child_node.name]:
                                queue.append(child_node)

        self.active_unsupported_gens = list(set(self.active_unsupported_gens))
        self.active_unsupported_gen_outputs = list(set(self.active_unsupported_gen_outputs))

    # run graph with substance automation toolkit
    def run_sat(
        self, graph_filename, output_name, sat_dir, resource_dirs,
        use_output_defaults=True, output_usages=None, return_output_channels=True, write_output_channels=True, save_sbsar=False,
        randomize_generators=False, generators_only=True, seed=None, image_format='png', center_normals=False,
        engine='sse2', use_networkx=True, timeout=None):

        # exit early if the graph has no output
        if not self.outputs:
            print('WARNING: The graph has no output. Skipping SAT rendering.')
            return {} if return_output_channels else None

        sat_dir = os.path.realpath(os.path.abspath(sat_dir)) # absolute path and fully resolve symlinks, since that is how dependencies are are also given and the SAT gets confused

        if output_usages is None:
            output_usages = ['baseColor', 'normal', 'roughness', 'metallic']

        # validate all dependencies (even unused ones)
        # the cooker needs all dependencies to be valid
        missing_deps = missing_dependencies(deps=self.dependencies(), source_filename=graph_filename, resource_dirs=resource_dirs)
        if missing_deps:
            err_str = 'Missing dependencies:\n'
            for missing_dep in missing_deps:
                err_str += f'{missing_dep.path}\n'
            raise MissingDependencyError(err_str)

        # create rng for generator seeds
        if randomize_generators:
            self.randomize_node_seeds(seed=seed, generators_only=generators_only)

        # save graph to .sbs
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        output_graph_filename = f'{output_name}_temp_for_sat_rendering.sbs'
        if save_sbsar:
            cooked_output_graph_filename = f'{output_name}.sbsar'
        else:
            cooked_output_graph_filename = f'{output_name}_temp_for_sat_rendering.sbsar'

        resolve_resource_dirs = copy.deepcopy(resource_dirs)
        if 'sbs' in resolve_resource_dirs:
            del resolve_resource_dirs['sbs'] # do not resolve 'sbs://...' dependencies
        self.save_sbs(filename=output_graph_filename, resolve_resource_dirs=resolve_resource_dirs, use_networkx=use_networkx)

        # run SAT cooker and render
        try:
            # render one variation of the input images
            command_cooker = (
                f'"{os.path.join(sat_dir, "sbscooker")}" '
                f'--inputs "{output_graph_filename}" '
                f'--alias "sbs://{os.path.join(sat_dir, "resources", "packages")}" '
                f'--output-path {{inputPath}}')
            # command_cooker = [
            #     os.path.join(sat_dir, 'sbscooker'),
            #     '--inputs', output_graph_filename,
            #     '--alias', f'sbs://{os.path.join(sat_dir, "resources", "packages")}',
            #     '--output-path', '{inputPath}'
            # ]
            cp = subprocess.run(command_cooker, shell=True, capture_output=True, text=True)
            if cp.returncode != 0:
                raise RuntimeError(f'Error while running sbs cooker:\n{cp.stderr}')

            command_render = (
                f'"{os.path.join(sat_dir, "sbsrender")}" render '
                f'--inputs "{cooked_output_graph_filename}" '
                f'--input-graph "{self.name}" ')
            # command_render += (
            #     f'--input-graph-output-usage {output_usages[0]} ' if len(output_usages) == 1 else '')
            command_render += (
                f'--output-format "{image_format}" '
                f'--output-path "{os.path.dirname(output_name)}" '
                f'--output-name "{os.path.basename(output_name)}_{{outputUsages}}" '
            )

            if engine is not None:
                command_render += f'--engine {engine} '
            command_render += '--cpu-count 1 '

            max_retry = 10
            for ri in reversed(range(max_retry)):
                # a preferred solution where the timed-out sbsrender process is killed along with its child processes
                if platform.system() in ('Linux', 'Darwin'):
                    proc_kwargs = {'start_new_session': True}
                else:
                    proc_kwargs = {'preexec_fn': os.setsid}
                with subprocess.Popen(command_render, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **proc_kwargs) as p:
                    try:
                        _, p_stderr = p.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired as e:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                        raise RuntimeError(f'sbsrender timed out after {timeout} second(s).') from e
                    p_returncode = p.poll()
                if not p_returncode:
                    break
                if p_stderr.find('[ERROR]') >= 0:
                    raise RuntimeError(f'Error while running sbs render:\n{p_stderr}')
                elif not ri:
                    raise RuntimeError(f'Failed to run sbs render after {max_retry} attempt(s).')

        except Exception as e:
            sbs_exception = e
        else:
            sbs_exception = None

        finally:
            # clean up intermediate sbs and sbsar files
            if not save_sbsar:
                if platform.system() in ('Linux', 'Darwin'):
                    command_clean = f'rm "{output_graph_filename}" "{cooked_output_graph_filename}"'
                elif platform.system() == 'Windows':
                    output_graph_filename_win = output_graph_filename.replace('/', '\\')
                    cooked_output_graph_filename_win = cooked_output_graph_filename.replace('/', '\\')
                    command_clean = f'del "{output_graph_filename_win}" "{cooked_output_graph_filename_win}"'
                else:
                    raise RuntimeError('Unsupported operating system')
                subprocess.run(command_clean, shell=True)

            # re-raise the captured exception (if any) after cleaning up
            if sbs_exception is not None:
                raise sbs_exception

        # remove output channel images that are not needed
        rendered_output_usages = list(set(output.usage for output in self.outputs))
        for usage in rendered_output_usages:
            if usage not in output_usages:
                channel_filename = f'{output_name}_{usage}.{image_format}'
                try:
                    os.remove(channel_filename)
                except FileNotFoundError:
                    print(f'Cannot find output material maps {channel_filename}')

        # create defaults for output usages that were not rendered by the graph
        # and return output channels
        target_res = [2**self.res[0], 2**self.res[1]]
        output_channels = {}
        image_exception = None

        for usage in output_usages:
            channel_filename = f'{output_name}_{usage}.{image_format}'
            img, img_src = None, None
            if return_output_channels or write_output_channels:
                try:
                    img_src = read_image(filename=channel_filename, process=True, use_alpha=self.use_alpha).unsqueeze(dim=0).to(device=self.device)
                    img = img_src
                    if img.shape[-2] != target_res[0] or img.shape[-1] != target_res[1]:
                        img = torchvision.transforms.functional.resize(img, size=target_res, antialias=True)
                    if usage == 'normal':
                        img = img.expand(-1, 3, -1, -1) if img.shape[-3] == 1 else img
                        img = center_normal_map(img) if center_normals else img
                except FileNotFoundError:
                    print(f'WARNING: output {usage} is missing, using default value.')
                    img = self.default_output_image(usage=usage)
                except Exception as e:
                    print(f'ERROR: unable to process output material maps {channel_filename}. {type(e).__name__}: {str(e)}')
                    image_exception = e if image_exception is None else image_exception

            # record output channel image
            if return_output_channels and img is not None:
                output_channels[usage] = img

            # update output channel image or delete it
            if write_output_channels:
                if img is not None and img is not img_src:
                    write_image(filename=channel_filename, img=img.cpu().squeeze(dim=0), process=True)
            elif os.path.exists(channel_filename):
                os.remove(channel_filename)

        # re-raise the captured exception (if any) after cleaning up
        if image_exception is not None:
            raise image_exception

        if return_output_channels:
            return output_channels

    @staticmethod
    def run_sat_on_sbs(graph_name, filename, output_name, sat_dir, res=(9, 9), output_usages=None, image_format='png', engine='sse2', timeout=None):
        if output_usages is None:
            output_usages = ['baseColor', 'normal', 'roughness', 'metallic']

        cooked_graph_filename = f'{os.path.splitext(filename)[0]}.sbsar'
        cooked_graph_filename_tmp = f'{output_name}_temp_for_sat_rendering.sbsar'
        if platform.system() == 'Windows':
            cooked_graph_filename = cooked_graph_filename.replace('/', '\\')
            cooked_graph_filename_tmp = cooked_graph_filename_tmp.replace('/', '\\')

        command_cooker = (
            f'"{os.path.join(sat_dir, "sbscooker")}" '
            f'--inputs "{filename}" '
            f'--alias "sbs://{os.path.join(sat_dir, "resources", "packages")}" '
            f'--output-path {{inputPath}}')
        try:
            completed_process = subprocess.run(command_cooker, shell=True, capture_output=True, text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f'sbscooker timed out after {timeout} second(s).')
        if completed_process.returncode != 0:
            raise RuntimeError(f'Error while running sbs cooker:\n{completed_process.stderr}')

        if platform.system() == 'Linux':
            command_move = f'mv "{cooked_graph_filename}" "{cooked_graph_filename_tmp}" 1> /dev/null '
        elif platform.system() == 'Windows':
            command_move = f'move "{cooked_graph_filename}" "{cooked_graph_filename_tmp}" 1> nul '
        else:
            raise RuntimeError('Unsupported operating system')
        subprocess.run(command_move, shell=True, check=True)

        command_render = (
            f'"{os.path.join(sat_dir, "sbsrender")}" render '
            f'--inputs "{cooked_graph_filename_tmp}" '
            f'--input-graph "{graph_name}" '
            f'--output-format "{image_format}" '
            f'--output-path "{os.path.dirname(output_name)}" '
            f'--output-name "{os.path.basename(output_name)}_{{outputUsages}}" '
            f'--set-value "$outputsize@{res[0]},{res[1]}" ')

        if engine is not None:
            command_render += f'--engine {engine} '

        if platform.system() == 'Linux':
            command_render += '1> /dev/null  2>&1'
        elif platform.system() == 'Windows':
            command_render += '1> nul  2>&1'
        else:
            raise RuntimeError('Unsupported operating system')

        max_retry = 10
        for ri in reversed(range(max_retry)):
            try:
                completed_process = subprocess.run(command_render, shell=True, capture_output=True, text=True, check=False, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'sbsrender timed out after {timeout} second(s).')
            if not completed_process.returncode:
                break
            if completed_process.stderr.find('[ERROR]') >= 0:
                raise RuntimeError(f'Error while running sbs render:\n{completed_process.stderr}')
            elif not ri:
                raise RuntimeError(f'Failed to run sbs render after {max_retry} attempt(s).')

        if platform.system() == 'Linux':
            command_clean = f'rm "{cooked_graph_filename_tmp}" '
        elif platform.system() == 'Windows':
            command_clean = f'del "{cooked_graph_filename_tmp}"'
        else:
            raise RuntimeError('Unsupported operating system')
        subprocess.run(command_clean, shell=True, check=True)

        output_maps = glob.glob(f'{output_name}*')
        prefix = os.path.basename(output_name)
        for output_map in output_maps:
            usage = os.path.splitext(os.path.basename(output_map))[0][len(prefix) + 1:]
            if usage not in output_usages:
                os.remove(output_map)

    def randomize_node_seeds(self, seed=None, rng=None, generators_only=False):

        if rng is None:
            # if seed is None:
            #     raise RuntimeError('Need to provide either an RNG or a seed.')
            rng = np.random.default_rng(seed=seed)

        seeds = ['randomseed', 'seed']
        for node in self.nodes:
            if generators_only and len(node.get_connected_inputs()) > 0:
                continue
            for seed in seeds:
                randomseed_param = node.get_param_by_name(seed)
                if randomseed_param is not None:
                    # we decrease random range
                    randomseed_param.val = rng.integers(1000).item()

    # def gen_input_graph(self):
    #     '''
    #     Generate an sbs file that outputs the unsupported generator nodes outputs
    #     '''

    #     # increase recursion limit to at least 10k before doing the deepcopy of the graph,
    #     # otherwise we might exceed the recursion depth
    #     recursion_limit = sys.getrecursionlimit()
    #     sys.setrecursionlimit(max(10000, recursion_limit))
    #     input_graph = copy.deepcopy(self)
    #     sys.setrecursionlimit(recursion_limit)

    #     # remove all graph outputs
    #     for graph_output in list(input_graph.outputs):
    #         input_graph.remove_output(graph_output)

    #     # create graph outputs and output nodes for all output slots of the generator nodes and remove all non-generator nodes
    #     for node in list(input_graph.nodes): # may mutate the list of nodes

    #         if node.type == 'Unsupported' and len(node.get_connected_inputs()) == 0:
    #             # create output nodes for all output slots of the generator nodes
    #             for node_output in node.outputs:
    #                 graph_output = input_graph.create_output(name=node_output.uname(), group='input')
    #                 graph_output_node = input_graph.create_node(node_type='Output', node_name=node_output.uname(), graph_output=graph_output)
    #                 graph_output_node.get_input_by_name('input').connect(node_output)

    #         else:
    #             # remove all non-generator nodes
    #             input_graph.remove_node(node)

    #     return input_graph

    def gen_input_images(self, sat_dir, resource_dirs, temp_dir=None, image_variations=1, randomize=False, seed=None, image_format='png', engine='sse2', device=torch.device('cpu'), use_networkx=True, ignore_image_input=False, timeout=None):
        '''
        Generate the outputs of unsupported generator nodes and return them as a dictionary of images
        '''

        # # check if any input images need to be created
        # # (input images are only created for unsupported generator nodes)
        # if not any(node.type=='Unsupported' and len(node.get_connected_inputs()) == 0 for node in self.nodes):
        #     return {}

        # input_graph = self.subgraph_of_unsupported_nodes() if self.use_partial_differentiation else self.gen_input_graph()
        input_graph = self.gen_subgraph_of_nondifferentiable_nodes()

        if len(input_graph.outputs) == 0:
            return {}

        # create temporary directoary for the input graph
        with tempfile.TemporaryDirectory(dir=temp_dir, prefix='sbs_input_images') as input_graph_dir:

            input_graph_filename = os.path.join(input_graph_dir, 'graph.sbs')

            # validate all dependencies (even unused ones)
            # the cooker needs all dependencies to be valid
            missing_deps = missing_dependencies(deps=input_graph.dependencies(), source_filename=input_graph_filename, resource_dirs=resource_dirs)
            if len(missing_deps) > 0:
                err_str = 'Missing dependencies:'
                for missing_dep in missing_deps:
                    err_str += f'{missing_dep.path}\n'
                raise MissingDependencyError(err_str)

            # create rng for random input image variations
            rng = None
            if randomize:
                rng = np.random.default_rng(seed=seed)

            # generate random variations of the input images
            cooked_input_graph_filename = os.path.splitext(input_graph_filename)[0] + '.sbsar'
            for variation_index in range(image_variations):
                variation_dir = os.path.join(
                    os.path.splitext(input_graph_filename)[0],
                    f'input_images_{variation_index}')
                os.makedirs(variation_dir, mode=0o775, exist_ok=True)

                # traverse all  nodes and set the 'randomseed' parameter to the variation index
                if randomize:
                    input_graph.randomize_node_seeds(rng=rng, generators_only=False)
                    # for node in input_graph.nodes:
                    #     randomseed_param = node.get_param_by_name('randomseed')
                    #     if randomseed_param is not None:
                    #         randomseed_param.val = str(rng.integers(10000000).item())

                # save input graph for the current variation
                resolve_resource_dirs = copy.deepcopy(resource_dirs)
                if 'sbs' in resolve_resource_dirs:
                    del resolve_resource_dirs['sbs'] # do not resolve 'sbs://...' dependencies
                input_graph.save_sbs(filename=input_graph_filename, resolve_resource_dirs=resolve_resource_dirs, use_networkx=use_networkx)

                # render one variation of the input images
                command_cooker = (
                    f'"{os.path.join(sat_dir, "sbscooker")}" '
                    f'--inputs "{input_graph_filename}" '
                    f'--alias "sbs://{os.path.join(sat_dir, "resources", "packages")}" '
                    f'--output-path {{inputPath}}')
                completed_process = subprocess.run(command_cooker, shell=True, capture_output=True, text=True, check=False)
                if completed_process.returncode != 0:
                    raise RuntimeError(f'Error while running sbs cooker:\n{completed_process.stderr}')

                command_render = (
                    f'"{os.path.join(sat_dir, "sbsrender")}" render '
                    f'--inputs "{cooked_input_graph_filename}" '
                    f'--input-graph "{self.name}" '
                    f'--output-format "{image_format}" '
                    f'--output-path "{variation_dir}" '
                    f'--output-name "{{outputNodeName}}" '
                )

                if engine is not None:
                    command_render += f'--engine {engine} '

                if platform.system() == 'Linux':
                    command_render += '1> /dev/null 2>&1'
                elif platform.system() == 'Windows':
                    command_render += '1> nul 2>&1'
                else:
                    raise RuntimeError('Unsupported operating system')

                max_retry = 10
                for ri in reversed(range(max_retry)):
                    try:
                        completed_process = subprocess.run(command_render, shell=True, capture_output=True, text=True, check=False, timeout=timeout)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f'sbsrender timed out after {timeout} second(s).')
                    if not completed_process.returncode:
                        break
                    if completed_process.stderr.find('[ERROR]') >= 0:
                        raise RuntimeError(f'Error while running sbs render:\n{completed_process.stderr}')
                    elif not ri:
                        raise RuntimeError(f'Failed to run sbs render after {max_retry} attempt(s).')

            # clean up
            if platform.system() == 'Linux':
                command_clean = f'rm "{cooked_input_graph_filename}"'
            elif platform.system() == 'Windows':
                cooked_input_graph_filename_win = cooked_input_graph_filename.replace('/', '\\')
                command_clean = f'del "{cooked_input_graph_filename_win}"'
            else:
                raise RuntimeError('Unsupported operating system')
            subprocess.run(command_clean, shell=True, check=True)

            input_image_names = {}

            for node in self.active_unsupported_gens:
                for node_output in node.outputs:
                    if node_output in self.active_unsupported_gen_outputs:
                        input_image_names[node_output.uname()] = f'{node_output.uname()}.{image_format}'

            variation_indices = random.choices(range(image_variations), k=len(input_image_names))
            input_image_dir = os.path.splitext(input_graph_filename)[0]

            if len(self.inputs) > 0 or any(inp.dtype in [1, 2] for inp in self.params):
                # the graph uses an input image (needs to be given manually)
                if ignore_image_input:
                    print('Image inputs are not unsupported. The input images generation simply ignore this input. ')
                else:
                    raise NotImplementedError('Generating an dictionary of input images is not supported for graphs with image inputs (these would need to be given manually).')

            input_image_dict = {}
            for i, (param_name, input_image_filename) in enumerate(input_image_names.items()):
                input_image = torch.from_numpy(read_image(os.path.join(
                    input_image_dir, f'input_images_{variation_indices[i]}', input_image_filename)))

                if len(input_image.shape) == 3:
                    input_image = input_image.permute(2,0,1)
                    if self.use_alpha and input_image.shape[0] == 3:
                        input_image = torch.cat([
                            input_image,
                            torch.ones(1, input_image.shape[1], input_image.shape[2], device=torch.device('cpu'))
                            ], dim=0)
                    elif not self.use_alpha and input_image.shape[0] == 4:
                        input_image = input_image[:3, :, :]
                    input_image = input_image.unsqueeze(0).to(device)
                else:
                    input_image = input_image.unsqueeze(0).unsqueeze(0).to(device)

                input_image_dict[param_name] = input_image

        return input_image_dict

    def save_networkx(self):
        nxgraph = nx.DiGraph()
        # nodes
        for node in self.nodes:
            nxgraph.add_node(node.name)
        # edges
        for node in self.nodes:
            for node_input in node.inputs:
                if node_input.parent is not None:
                    nxgraph.add_edge(node_input.parent.name, node.name)
        return nxgraph

    def condition_active_node_params(self, clamp_params=True):
        if self.active_node_seq is None:
            raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

        # delete all presets, they will no longer be valid
        for param_preset in list(self.param_presets): # copy the list since it will be mutated
            self.remove_param_preset(param_preset=param_preset)

        input_dict = {p.name: p.val for p in self.params}
        for node in self.active_unsupported_gens:
            node.condition_params(source_params=node.source_params, input_dict=input_dict, clamp_params=clamp_params)
        for node in self.active_node_seq:
            node.condition_params(source_params=node.source_params, input_dict=input_dict, clamp_params=clamp_params)

    # add auxiliary color <-> grayscale conversion nodes to ensure that node signatures match for connected nodes
    def update_node_dtypes(self, harmonize_signatures=False, error_on_unmatched=False):

        # if self.active_node_seq is None or self.active_unsupported_gens is None:
        #     raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

        variable_dtypes = {}

        processed_nodes = set()
        node_queue = deque(node for node in self.nodes if len(node.get_connected_inputs()) == 0)

        # # initialize variable data types with the output of the active unsupported generator nodes
        # for node in self.active_unsupported_gens:
        #     node_signatures = node.signatures()
        #     if len(node_signatures) == 0:
        #         raise RuntimeError(f'Node {node.name} ({node.func}) does not have any signatures.')
        #     elif len(node_signatures) >= 2:
        #         print(f'WARNING: Node {node.name} ({node.func}) does not have any inputs and more than one signature, greedily using the last signature.')
        #     output_signature = node_signatures[-1][1]
        #     for output_name, output_dtype in output_signature.items():
        #         variable_dtypes[node.get_output_by_name(output_name).uname()] = output_dtype

        # go through all other active nodes in sequence
        while len(node_queue) > 0:
            node = node_queue.popleft()
            processed_nodes.add(node)

            target_input_signature = {}
            for node_input in node.inputs:
                target_input_signature[node_input.name] = variable_dtypes[node_input.parent_output.uname()] if node_input.parent_output is not None else None
            candidate_signatures = node.matching_signatures(signature=(target_input_signature, {}))
            if len(candidate_signatures) == 0:
                # no matching input signature found

                if error_on_unmatched:
                    err_str = f'Signature Error: Node {node.name} ({node.func}) does not have a signature matching the given inputs:\n'
                    err_str += f'  {target_input_signature}\n'
                    err_str += 'Expected one of the following input signatures:\n'
                    for sig in node.signatures():
                        err_str += f'  {sig[0]}\n'
                    raise RuntimeError(err_str)

                if not harmonize_signatures:
                    # use the first signature, even though it does not match
                    matched_signature = node.signatures()[0]
                else:
                    # introduce auxiliary color <-> grayscale conversion nodes to match one of the input signatures

                    # choose a signature based on number and type of conversions needed
                    node_signatures = node.signatures()
                    if len(node_signatures) == 0:
                        raise RuntimeError(f'Node {node.name} ({node.func}) has zero signatures.')
                    best_sig = None
                    best_sig_conversion = None
                    best_sig_conversion_cost = None
                    for node_signature in node_signatures:
                        sig_conversion_cost = 0
                        sig_conversion = {}
                        for input_name in target_input_signature:
                            if target_input_signature[input_name] is None:
                                pass # input is not connected, no conversion cost
                            elif target_input_signature[input_name] == node_signature[0][input_name]:
                                pass # matching types, no conversion cost
                            elif target_input_signature[input_name] in [SBSParamType.ENTRY_GRAYSCALE.value, SBSParamType.ENTRY_COLOR.value, SBSParamType.ENTRY_VARIANT.value] and node_signature[0][input_name] == SBSParamType.ENTRY_VARIANT.value:
                                pass # input is either color or grayscale and node accepts both color or grayscale, no conversion cost
                            elif target_input_signature[input_name] == SBSParamType.ENTRY_GRAYSCALE.value and node_signature[0][input_name] == SBSParamType.ENTRY_COLOR.value:
                                sig_conversion_cost += 1 # GRAYSCALE -> COLOR conversion cost = 1
                                sig_conversion[input_name] = 'g2c'
                            elif target_input_signature[input_name] == SBSParamType.ENTRY_COLOR.value and node_signature[0][input_name] == SBSParamType.ENTRY_GRAYSCALE.value:
                                sig_conversion_cost += 2 # COLOR -> GRAYSCALE conversion cost = 2 (more expensive since information may be lost)
                                sig_conversion[input_name] = 'c2g'
                            else:
                                raise RuntimeError(
                                    f'Unexpected data types when comparing input {input_name} to signature of node {node.name} ({node.func}):\n'
                                    f'input: {target_input_signature[input_name]} - node signature: {node_signature[0][input_name]}')
                        if best_sig_conversion_cost is None or sig_conversion_cost < best_sig_conversion_cost:
                            best_sig = node_signature
                            best_sig_conversion = sig_conversion
                            best_sig_conversion_cost = sig_conversion_cost

                    # add auxiliary color <-> grayscale conversion nodes
                    for input_name, input_conversion in best_sig_conversion.items():
                        node_input = node.get_input_by_name(input_name)
                        if input_conversion == 'g2c':
                            # grayscale -> color conversion is done with a Gradient Map node with default parameters
                            assert node_input.parent_output is not None
                            aux_node = self.create_node(node_type='GradientMap')
                            aux_node.get_input_by_name('img_in').connect(parent_output=node_input.parent_output) # connect auxiliary node to parent of the current input
                            node_input.connect(parent_output=aux_node.get_output_by_name('')) # parent of current input to the auxiliary node
                        elif input_conversion == 'c2g':
                            # color -> grayscale conversion is done with a Grayscale Conversion node with default parameters
                            assert node_input.parent_output is not None
                            aux_node = self.create_node(node_type='C2G')
                            aux_node.get_input_by_name('img_in').connect(parent_output=node_input.parent_output) # connect auxiliary node to parent of the current input
                            node_input.connect(parent_output=aux_node.get_output_by_name('')) # parent of current input to the auxiliary node
                        else:
                            raise RuntimeError(f'Unknown conversion type: {input_conversion}')

                    matched_signature = best_sig
            else:
                # matching signature found (first one should be the default signature)
                matched_signature = candidate_signatures[0]

            # update variable dtypes with the outputs of the matched node signature
            for output_name, output_dtype in matched_signature[1].items():
                variable_dtypes[node.get_output_by_name(output_name).uname()] = output_dtype

            for node_input in node.inputs:
                node_input.dtype = matched_signature[0][node_input.name]
            for node_output in node.outputs:
                node_output.dtype = matched_signature[1][node_output.name]

            # add unprocessed child nodes to the queue if the dtypes of all inputs have already been determined (to maintain a topological order)
            unprocessed_child_nodes = set(node.get_child_nodes()) - processed_nodes
            for child_node in unprocessed_child_nodes:
                if all(child_node_input.parent_output.uname() in variable_dtypes for child_node_input in child_node.inputs if child_node_input.parent_output is not None):
                    node_queue.append(child_node)

        # update the active nodes and make another signature discovery pass to account for auxiliary nodes that may have been added
        if harmonize_signatures:
            self.update_node_dtypes(harmonize_signatures=False, error_on_unmatched=True)

    def force_opengl_normals(self):
        relevant_node_param_names = {
            'Normal': 'normal_format',
            'Curvature': 'normal_format',
            'HeightToNormal': 'normal_format',
            'NormalToHeight': 'normal_format',
            'CurvatureSmooth': 'normal_format',
            'NormalColor': 'normal_format',
            'VectorWarp': 'vector_format'}
        for node in self.nodes:
            if node.type in relevant_node_param_names:
                node.get_param_by_name(relevant_node_param_names[node.type]).val = 'gl'

    def force_directx_normals(self):
        relevant_node_param_names = {
            'Normal': 'normal_format',
            'Curvature': 'normal_format',
            'HeightToNormal': 'normal_format',
            'NormalToHeight': 'normal_format',
            'CurvatureSmooth': 'normal_format',
            'NormalColor': 'normal_format',
            'VectorWarp': 'vector_format'}
        for node in self.nodes:
            if node.type in relevant_node_param_names:
                node.get_param_by_name(relevant_node_param_names[node.type]).val = 'dx'

    def remove_passthrough_nodes(self):
        for node in list(self.nodes): # may mutate the list of nodes
            if node.type == 'Passthrough':

                # change connection of all children to point to the parent of the Passthrough node
                # (removes the connection if there is no parent)
                for child_input in node.get_connected_child_inputs():
                    child_input.connect(parent_output=node.inputs[0].parent_output)

                # remove connection from Passthrough node to its parent
                node.inputs[0].disconnect()

    def remove_useless_invert_nodes(self):
        useless_invert_node = None
        idx = -1
        for k, node in enumerate(self.active_node_seq):
            if node.type == 'Invert':
                switch_param = node.get_param_by_name('invert_switch')
                # if it is a useless node
                if not switch_param.val:
                    useless_invert_node, idx = node, k
                    break

        if useless_invert_node is not None:
            assert len(useless_invert_node.inputs) == 1
            for child_input in useless_invert_node.get_connected_child_inputs():
                child_input.connect(parent_output=useless_invert_node.inputs[0].parent_output)
            # remove this useless invert node
            self.active_node_seq.pop(idx)
            self.remove_node(useless_invert_node)
            # remove other nodes recursively for safety because self.active_node_seq is mutable
            self.remove_useless_invert_nodes()

    def remove_levels_node_in_normals_branch(self):
        # normals branch are the nodes that contribute only to the normal output
        reachable_from_other_outputs, _, _ = self.reachable_from_outputs(output_usages=['baseColor', 'metallic', 'roughness'])
        # reachable_from_normal_output, _, _ = self.reachable_from_outputs(output_usages=['normal'])

        levels_nodes = []

        output_nodes = [output.parent for output in self.outputs if output.group == 'Material' and output.usage == 'normal']
        output_nodes = list(set(output_nodes))
        if any(node is None for node in output_nodes):
            raise NotImplementedError('Output nodes without parent are not supported.')

        # queue = deque(output_nodes)
        # reachable_nodes = {node for node in output_nodes}
        # # Run backward BFS
        # while queue:
        #     node = queue.popleft()
        #     for node_input in node.get_connected_inputs():
        #         parent = node_input.parent
        #         if parent.type == 'Levels':
        #             levels_nodes.append(parent)
        #         if parent not in reachable_nodes:
        #             queue.append(parent)
        #             reachable_nodes.add(parent)

        # for output_node in output_nodes:
        #     assert len(output_node.inputs) == 1
        #     parent = output_node.inputs[0].parent
        #     if parent is not None and parent.type == 'Levels':
        #         levels_nodes.append(parent)

        # it may not be a complete terminate criteria
        def terminate(node):
            if node.type == 'Normal' or (node.type == 'Unsupported' and node.func != 'normal_intensity'):
                return True
            else:
                return False

        queue = deque(output_nodes)
        reachable_nodes = {node for node in output_nodes}
        # Run backward BFS
        while queue:
            node = queue.popleft()
            for node_input in node.get_connected_inputs():
                parent = node_input.parent
                if parent.type == 'Levels':
                    levels_nodes.append(parent)
                if parent not in reachable_nodes and not terminate(parent):
                    queue.append(parent)
                    reachable_nodes.add(parent)

        levels_nodes = list(set(levels_nodes))
        # remove the levels nodes
        for levels_node in levels_nodes:
            assert len(levels_node.inputs) == 1

            # remove the levels node if it has a parent and it is not reachable from any other relevant output
            if (levels_node.inputs[0].parent is not None) and (levels_node not in reachable_from_other_outputs):
                # connect all children to parent
                parent_output = levels_node.inputs[0].parent_output
                for child_input in levels_node.get_connected_child_inputs():
                    child_input.connect(parent_output=parent_output)
                # remove this level node
                self.remove_node(levels_node)

    def add_levels_node(self, usage):
        output_node = [output.parent for output in self.outputs if output.group == 'Material' and output.usage == usage]
        output_node = list(set(output_node))
        if len(output_node) == 0:
            print(f'Cannot find output node for {usage} map. '
                  f'TODO: it is better to create one for optimization.')
            return
        if len(output_node) > 1:
            raise RuntimeError(f'{usage} output node is not unique. ')
        output_node = output_node[0]
        if output_node is None:
            raise NotImplementedError('Output nodes without parent are not supported.')

        output_node_input = output_node.get_connected_inputs()
        if len(output_node_input) != 1:
            raise RuntimeError('An output node has more than one input.')
        output_node_input = output_node_input[0]

        if output_node_input.parent.type == 'Levels':
            levels_node = output_node_input.parent
            if usage == 'baseColor':
                # expand default values to float4
                def expand_to_float4(param):
                    if isinstance(param.val, torch.Tensor) and param.val.size() == torch.Size([]):
                        val = param.val.item()
                        param.val = [val, val, val, 1.0]
                    elif isinstance(param.val, (int, float)):
                        param.val = [param.val, param.val, param.val, 1.0]
                expand_to_float4(levels_node.get_param_by_name('in_low'))
                expand_to_float4(levels_node.get_param_by_name('in_mid'))
                expand_to_float4(levels_node.get_param_by_name('in_high'))
                expand_to_float4(levels_node.get_param_by_name('out_low'))
                expand_to_float4(levels_node.get_param_by_name('out_high'))
        else:
            prev_node_output = output_node_input.parent_output
            levels_node = self.create_node(node_type='Levels')
            if 'node_pos' in output_node_input.parent.user_data and 'node_pos' in output_node.user_data:
                node_pos = [(p0 + p1) / 2 for p0, p1 in zip(output_node_input.parent.user_data['node_pos'], output_node.user_data['node_pos'])]
                levels_node.user_data['node_pos'] = node_pos
            # set parameters in a color Levels node as float4
            if usage == 'baseColor':
                levels_node.get_param_by_name('in_low').val = [0.0, 0.0, 0.0, 1.0]
                levels_node.get_param_by_name('in_mid').val = [0.5, 0.5, 0.5, 1.0]
                levels_node.get_param_by_name('in_high').val = [1.0, 1.0, 1.0, 1.0]
                levels_node.get_param_by_name('out_low').val = [0.0, 0.0, 0.0, 1.0]
                levels_node.get_param_by_name('out_high').val = [1.0, 1.0, 1.0, 1.0]

            # connect output node
            output_node_input.disconnect()
            levels_node.get_output_by_name('').connect(output_node_input)
            # connect input node
            prev_node_output.disconnect()
            levels_node.get_input_by_name('img_in').connect(prev_node_output)

        # adding an additional HSL node is  helpful for fast convergence
        if usage == 'baseColor':
            self.add_hsl_node_before_levels_node(levels_node)

    def add_hsl_node_before_levels_node(self, levels_node):
        if levels_node.type != 'Levels':
            raise RuntimeError('This function is designed to add HSL node before a Levels Node.')
        levels_node_input = levels_node.get_input_by_name('img_in')
        if levels_node_input.parent.type == 'HSL':
            return
        else:
            prev_node_output = levels_node_input.parent_output
            hsl_node = self.create_node(node_type='HSL')
            if 'node_pos' in levels_node_input.parent.user_data and 'node_pos' in levels_node.user_data:
                node_pos = [(p0 + p1) / 2 for p0, p1 in
                            zip(levels_node_input.parent.user_data['node_pos'], levels_node.user_data['node_pos'])]
                hsl_node.user_data['node_pos'] = node_pos
            # connect output node
            levels_node_input.disconnect()
            hsl_node.get_output_by_name('').connect(levels_node_input)
            # connect input node
            prev_node_output.disconnect()
            hsl_node.get_input_by_name('img_in').connect(prev_node_output)

    def get_node_param_values(self, copy_values=False, exclude_trainable=False, output_usages=None):
        if output_usages is None:
            node_set = self.nodes
        else:
            node_set, _, _ = self.reachable_from_outputs(output_usages=output_usages)

        with torch.no_grad():
            node_params = {}
            for node in node_set:
                node_params[node.name] = {}
                for param in node.params:
                    if exclude_trainable and param.trainable:
                        continue # leave unchanged

                    param_val = param.val
                    if copy_values:
                        if isinstance(param_val, torch.Tensor):
                            param_val = param_val.clone().detach()
                        elif isinstance(param_val, list):
                            param_val = copy.deepcopy(param_val)
                        elif isinstance(param_val, (numbers.Number, str)):
                            pass
                        else:
                            raise RuntimeError('Unexpected parameter type.')
                    node_params[node.name][param.name] = param_val

        return node_params

    def set_node_param_values(self, node_params, copy_values=False, exclude_trainable=False, output_usages=None):
        node_set = None
        if output_usages is not None:
            node_set, _, _ = self.reachable_from_outputs(output_usages=output_usages)

        with torch.no_grad():
            for node_name, params in node_params.items():
                node = self.get_node_by_name(node_name)
                if output_usages is not None and node not in node_set:
                    continue

                for param_name, param_val in params.items():
                    param = node.get_param_by_name(param_name)

                    # set parameters of Unsupported nodes that are not present because they are at their defaults
                    if node.type == 'Unsupported' and param is None:
                        node.add_param(SBSNodeParameter(name_xml=param_name, val=param_val, name=param_name, dtype=param_val_to_type(param_val)))
                        continue

                    if exclude_trainable and param.trainable:
                        continue # leave unchanged

                    if copy_values:
                        if isinstance(param_val, torch.Tensor):
                            if param_val.shape == param.val.shape:
                                param.val.copy_(param_val)
                            else:
                                param.val = param_val.clone().detach()
                        elif isinstance(param_val, list):
                            param.val = copy.deepcopy(param_val)
                        elif isinstance(param_val, (numbers.Number, str)):
                            param.val = param_val
                        else:
                            raise RuntimeError('Unexpected parameter type.')
                    else:
                        param.val = param_val

                # check that the parameters of all nodes are present in the given parameter dictionary
                # except for Unsupported nodes, remove missing parameters for these
                for param in list(node.params): # may mutate the node params
                    if exclude_trainable and param.trainable:
                        continue # leave unchanged
                    if param.name not in params:
                        if node.type == 'Unsupported':
                            # remove parameter of unsupported node that is not present in the given node parameters
                            node.remove_param(param)
                        else:
                            raise RuntimeError(f'The parameter {param.name} of node {node.name} missing in the given parameter dictionary.')

    def set_node_params_from_stats(self, node_type_stats, mode, seed=None, exclude_unsupported=False, exclude_trainable=False, output_usages=None):
        # exclude the following base parameters in all nodes
        # these parameters control the image resolution, etc. and do not need to be changed
        # (all base parameters except randomseed)
        params_use_defaults = ['outputsize', 'format', 'pixelsize', 'pixelratio', 'tiling', 'use_alpha']

        rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(100000000))

        if output_usages is None:
            node_set = self.nodes
        else:
            node_set, _, _ = self.reachable_from_outputs(output_usages=output_usages)

        for node in node_set: # TOOD: inactive nodes may not have been captured in the node types, iterate over all active_unsupported_gens and active_node_seq? (active_output_nodes) are not needed since they have no parameters

            # skip output nodes, these don't have any parameters
            if node.type == 'Output':
                continue
            if exclude_unsupported and node.type == 'Unsupported':
                continue

            if node.definition() is None or node.definition().path is None:
                node_type_name = node.func
            else:
                node_type_name = f'{os.path.basename(node.definition().path)}:{node.func}'

            if node_type_name not in node_type_stats:
                raise RuntimeError(f'Unrecognized node type: {node_type_name}')

            stats = node_type_stats[node_type_name]
            node_param_stats = stats['parameters']

            # special cases for nodes
            if node.type == 'Switch':
                node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                default_node = node_class(name='default', output_res=self.res, use_alpha=self.use_alpha)
                for param in node.params:
                    if param.name not in ['flag']:
                        param.val = default_node.get_param_by_name(param.name).val

                only_input_2_is_connected = node.get_input_by_name('img_1').parent is None and node.get_input_by_name('img_2').parent is not None
                node.get_param_by_name('flag').val = False if only_input_2_is_connected else True

            elif node.type == 'MultiSwitch':
                node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                default_node = node_class(name='default', output_res=self.res, use_alpha=self.use_alpha)
                for param in node.params:
                    if param.name not in ['input_selection', 'input_number']:
                        param.val = default_node.get_param_by_name(param.name).val

                first_connected_input_idx = None
                last_connected_input_idx = None
                for i in range(len(node.inputs)):
                    if node.get_input_by_name(f'input_{i+1}').parent is not None:
                        if first_connected_input_idx is None:
                            first_connected_input_idx = i
                        last_connected_input_idx = i
                if first_connected_input_idx is None:
                    # no inputs connected, set to defaults
                    node.get_param_by_name('input_selection').val = 1
                    node.get_param_by_name('input_number').val = 2
                else:
                    node.get_param_by_name('input_selection').val = first_connected_input_idx+1
                    node.get_param_by_name('input_number').val = last_connected_input_idx+1

            elif node.type == 'GradientMap':
                node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                default_node = node_class(name='default', output_res=self.res, use_alpha=self.use_alpha)
                for param in node.params:
                    if param.name not in ['mode', 'anchors']:
                        param.val = default_node.get_param_by_name(param.name).val

                node.get_param_by_name('mode').val = 'color'
                # node.get_param_by_name('num_anchors').val = 3 # stats have a mode at 2, but use 3 to get more degrees of freedom

                param_val= torch.as_tensor(self.param_val_from_stats(param_stats=node_param_stats['anchors'], mode=mode, rng=rng, param_dtype='FLOAT4_ARRAY_3')[0])
                # param_val = torch.as_tensor(node.get_default_anchors(), device=node.get_param_by_name('anchors').val.device)
                if node.get_param_by_name('anchors').val.shape == param_val.shape:
                     # update node anchors (they already have the correct shape)
                    with torch.no_grad():
                        node.get_param_by_name('anchors').val.copy_(param_val)
                else:
                    # replace node anchors (they have a different shape now)
                    node.get_param_by_name('anchors').val = param_val

            elif node.type == 'Curve':
                node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                default_node = node_class(name='default', output_res=self.res, use_alpha=self.use_alpha)
                for param in node.params:
                    if param.name not in ['anchors']:
                        param.val = default_node.get_param_by_name(param.name).val

                # node.get_param_by_name('num_anchors').val = 3 # stats have a mode at 2, but use 3 to get more degrees of freedom

                param_val = torch.as_tensor(self.param_val_from_stats(param_stats=node_param_stats['anchors'], mode=mode, rng=rng, param_dtype='FLOAT6_ARRAY_3')[0])
                # param_val = torch.as_tensor(node.get_default_anchors(), device=node.get_param_by_name('anchors').val.device)
                if node.get_param_by_name('anchors').val.shape == param_val.shape:
                    # update node anchors (they already have the correct shape)
                    with torch.no_grad():
                        node.get_param_by_name('anchors').val.copy_(param_val)
                else:
                    # replace node anchors (they have a different shape now)
                    node.get_param_by_name('anchors').val = param_val

            else:
                if node.type == 'Unsupported':
                    default_node = SBSUnsupportedNode(name='default', node_func=node.func, output_res=self.res, use_alpha=self.use_alpha, definition=node.definition())
                elif node.type == 'Input':
                    default_node = SBSInputNode(name='default', graph_input=None, output_res=self.res, use_alpha=self.use_alpha)
                else:
                    node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                    default_node = node_class(name='default', output_res=self.res, use_alpha=self.use_alpha)

                for param in list(node.params): # may mutate the node parameters
                    if exclude_trainable and param.trainable: #or (node_func, param.name) in exclude_node_params:
                        continue # leave unchanged

                    # special cases for node parameters
                    if node.type in ['Normal', 'Curvature', 'HeightToNormal', 'NormalToHeight', 'CurvatureSmooth', 'NormalColor', 'VectorWarp'] and param.name in ['normal_format', 'vector_format']:
                        pass # leave unchanged
                    elif node.type == 'Distance' and param.name == 'combine' and not node.use_alpha:
                        # combine only works with use_alpha = True
                        param.val = False
                    elif param.name in params_use_defaults or param.name not in node_param_stats:
                        if node.type == 'Unsupported' and param.name not in [p.name for p in default_node.params]:
                            # unsupported node and not a base parameter -> remove the parameter to effectively use its default
                            node.remove_param(param)
                        else:
                            param.val = default_node.get_param_by_name(param.name).val
                    else:
                        param.val, _ = self.param_val_from_stats(param_stats=node_param_stats[param.name], mode=mode, rng=rng)

                # set parameters of Unsupported nodes that are not present because they are at their defaults
                if node.type == 'Unsupported':
                    node_param_names = [p.name for p in node.params]
                    for param_name, param_stats in node_param_stats.items():
                        if param_name not in node_param_names:
                            param_val, param_dtype = self.param_val_from_stats(param_stats=param_stats, mode=mode, rng=rng)
                            node.add_param(SBSNodeParameter(name_xml=param_name, val=param_val, name=param_name, dtype=SBSParamType[param_dtype].value))

    @staticmethod
    def param_val_from_stats(param_stats, mode, rng, param_dtype=None):

        # choose data type
        if param_dtype is None:
            dtypes, dtype_freqs = zip(*[(dtype, dtype_stats['count']) for dtype, dtype_stats in param_stats.items()])
            if mode == 'pick_mode':
                idx = np.argmax(np.array(dtype_freqs))
            elif mode in ['randomize', 'randomize_uniform']:
                idx = rng.choice(range(len(dtypes)), p=np.array(dtype_freqs, dtype=np.float32)/sum(dtype_freqs))
            else:
                raise RuntimeError(f'Unknown parameter selection mode {mode}')
            param_dtype = dtypes[idx]
        else:
            if param_dtype not in param_stats:
                raise RuntimeError(f'Could not find requested data type {param_dtype} in parameter stats.')

        param_stats = param_stats[param_dtype]
        param_type = param_stats['type']

        if param_type in ['INTEGER1', 'BOOLEAN', 'STRING']:
            if len(param_stats['value_freq']) == 0:
                raise RuntimeError('Non-float parameter with empty value frequency list.')
            # use value frequency as distribution for all strings and booleans, or for scalar integers with <= 50 different observed values

            vals, freqs = zip(*param_stats['value_freq'].items())
            if mode == 'pick_mode':
                idx = np.argmax(np.array(freqs))
            elif mode == 'randomize':
                idx = rng.choice(range(len(vals)), p=np.array(freqs, dtype=np.float32)/sum(freqs))
            elif mode == 'randomize_uniform':
                idx = rng.choice(range(len(vals)))
            else:
                raise RuntimeError(f'Unknown parameter selection mode {mode}')
            if param_type == 'BOOLEAN':
                param_val = False if vals[idx] in ['False', 'false', '0'] else bool(vals[idx])
            elif param_type == 'INTEGER1':
                param_val = int(vals[idx])
            else:
                param_val = vals[idx]
        else:
            # otherwise use normal distribution fitted to the observed values, and clamped to the min and max observed values
            if mode == 'pick_mode':
                param_val = param_stats['mean']
            elif mode == 'randomize':
                param_val = rng.normal(loc=param_stats['mean'], scale=param_stats['std'])
                param_val = np.clip(param_val, a_min=param_stats['min'], a_max=param_stats['max'])
            elif mode == 'randomize_uniform':
                param_val = rng.uniform(low=param_stats['min'], high=param_stats['max'])
            else:
                raise RuntimeError(f'Unknown parameter selection mode {mode}')

            if isinstance(param_val, np.ndarray):
                param_val = param_val.tolist()

            # convert to intergers if necessary
            if param_type.startswith('INTEGER'):
                if isinstance(param_val, list):
                    param_val = [round(v) for v in param_val]
                else:
                    param_val = round(param_val)

        return param_val, param_dtype

    def set_node_params_to_defaults(self, exclude_trainable=False):
        # set all parameters to their default values
        for node in self.nodes:
            if node.type in ['Input', 'Output', 'MultiSwitch', 'Switch']:
                # input and output nodes only have base parameters, don't change them
                # (creating them would require assigning a graph input/output)
                # also ignore some other types of nodes that only control the graph structure (like MultiSwitch nodes)
                continue
            elif node.type == 'Unsupported':
                # remove all parameters for unsupported nodes
                for param in list(node.params): # may mutate the node parameters
                    node.remove_param(param=param)
            else:
                # create a new node of the given type to get the default values of all its parameters
                node_class = getattr(sbs_graph_nodes, f'SBS{node.type}Node')
                default_node = node_class(name=node.name, output_res=self.res, use_alpha=self.use_alpha)
                for param in node.params:
                    if exclude_trainable and param.trainable:
                        continue
                    default_val = default_node.get_param_by_name(name=param.name).val
                    if isinstance(default_val, torch.Tensor):
                        param.val = default_val.clone().detach() # deepcopy does not detach tensors
                    else:
                        param.val = copy.deepcopy(default_val)

    def randomize_trainable_node_params(self, dist='uniform', seed=None, offset_max=None, scale_offset_max=None, offset_std=None, std=None):
        if self.active_node_seq is None:
            raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)

        with torch.no_grad():
            for node in self.active_node_seq:
                if node.type == 'Output':
                    continue
                for param in node.params:
                    if param.trainable:
                        # all trainable parameters are float tensors and their range should always be [0, 1]
                        randomize_tensor(val=param.val, rng=rng, dist=dist, offset_max=offset_max, scale_offset_max=scale_offset_max, offset_std=offset_std, std=std, val_min=0.0, val_max=1.0)

    class ParameterSampler:
        def __init__(self, node_type_stats, np_rng, torch_rng, valid_count=100,
                     cont_scale_stat=0.25, cont_scale_uniform=0.2,
                     discrete_scale_stat=0.25, discrete_scale_uniform=0.5,
                     unsupported_only=True, sample_unsupported_nodes_prob=0.9,
                     trainable_rescale_factor=1.5):

            self.node_type_stats = node_type_stats
            self.np_rng = np_rng
            self.torch_rng = torch_rng
            self.valid_count = valid_count
            self.cont_scale_stat = cont_scale_stat
            self.cont_scale_uniform = cont_scale_uniform
            self.discrete_scale_stat = discrete_scale_stat
            self.discrete_scale_uniform = discrete_scale_uniform
            self.unsupported_only = unsupported_only
            self.sample_unsupported_nodes = random.random() < sample_unsupported_nodes_prob
            self.trainable_scale_stat = self.cont_scale_stat * trainable_rescale_factor

            self.skipped_node_types = ['Input', 'Output', 'Switch', 'MultiSwitch']
            self.base = ['outputsize', 'format', 'pixelsize', 'pixelratio', 'use_alpha', 'tiling', 'randomseed']
            self.base += ['tile_mode', 'seed']  # alias of tiling and randomseed
            self.normal_format = ['normal_format', 'vector_format']
            self.custom_excluded = ['cropping', 'max_intensity']
            self.excluded_params = self.base + self.normal_format + self.custom_excluded

            self.error_log = []

        def is_excluded_param(self, param_name, param_dtype):
            if param_name in self.excluded_params:
                return True

            # Empirically prevent sampling different patterns but the rule may have problems
            # some parameters can be excluded by this rule.
            lowercase = param_name.lower()
            if param_dtype.startswith('INTEGER') and ('pattern' in lowercase) \
                    and ('tile' not in lowercase) and ('x_amount' not in lowercase) \
                    and ('y_amount' not in lowercase) and ('amount' not in lowercase):
                return True

            if param_name.find('normal_format') != -1:
                return True

            return False

        @staticmethod
        def get_node_type_name(node, with_graph_name=False):
            if node.definition() is None or node.definition().path is None:
                node_type_name, is_sub_func = node.func, False
            else:
                def_path = node.definition().path
                node_type_name = f'{os.path.basename(def_path)}:{node.func}' if with_graph_name else node.func
                is_sub_func = not def_path.startswith('sbs://')
            return node_type_name, is_sub_func

        def get_node_param_stat(self, node, skip_seeds=False):
            node_type_name, _ = self.get_node_type_name(node)
            if node_type_name not in self.node_type_stats:
                self.error_log.append(f'Failed to find statistics for node {node_type_name}')
                return {}
            else:
                node_type_stats = self.node_type_stats[node_type_name]
                if isinstance(node_type_stats, list):
                    node_type_signature = get_node_type_signature(create_json_node(node), skip_seeds=skip_seeds)
                    node_type_stats = next((nt for nt in node_type_stats if match_node_type_signature(node_type_signature, nt)), {})
                    if not node_type_stats:
                        self.error_log.append(f'Failed to find statistics for node {node_type_name}')
                        return {}

                return node_type_stats['parameters']

        def sample(self, node):
            if node.type == 'Unsupported' and not self.sample_unsupported_nodes:
                return

            if node.type in self.skipped_node_types or (node.type == 'Unsupported' and node.func == 'material_switch'):
                return

            node_type_name_wg, is_sub_func = self.get_node_type_name(node, with_graph_name=True)
            node_param_stats = self.get_node_param_stat(node)
            seeds = self.np_rng.integers(low=0, high=100000000, size=len(node.params)).tolist()

            for param, seed in zip(node.params, seeds):
                param_dtype = param_type_idx_to_name(param.dtype)
                if self.is_excluded_param(param.name, param_dtype):
                    continue

                if param.name in node_param_stats:
                    param_stats = node_param_stats[param.name]
                else:
                    self.error_log.append(f'Failed to find statistics for {param.name} of {node.func}')
                    param_stats = {}

                if param.trainable:
                    self.perturb_trainable(param, param_dtype, param_stats=param_stats)

                elif (not self.unsupported_only or node.type == 'Unsupported') and any(param_dtype.startswith(k) for k in ['FLOAT', 'INTEGER']):
                    self.perturb(param, param_dtype, param_stats=param_stats, is_sub_func=is_sub_func, seed=seed)

                    # it's a weird bug!
                    if node_type_name_wg == 'pattern_tile_generator.sbs:tile_generator' and param.name == 'scale' and param.val == 0:
                        param.val = 0.01

        def perturb_trainable(self, param, param_dtype, param_stats):
            if not isinstance(param.val, torch.Tensor):
                raise RuntimeError('A trainable parameter should be a torch tensor (nn.Parameter)')

            if (param_dtype in param_stats) and (param_stats[param_dtype]['count'] >= self.valid_count):
                std = torch.as_tensor(param_stats[param_dtype]['std'], device=param.val.device) * self.trainable_scale_stat
                min_ = torch.as_tensor(param_stats[param_dtype]['min'], device=param.val.device)
                max_ = torch.as_tensor(param_stats[param_dtype]['max'], device=param.val.device)
                param_val = torch.normal(mean=param.val, std=std, generator=self.torch_rng)
                param_val = torch.clamp(param_val, min=min_, max=max_)
            else:
                # TODO: default min and max value is set as (0.0, 1.0) but it's not always correct
                val_min = torch.as_tensor(param.val_min if param.val_min is not None else 0.0, device=param.val.device)
                val_min = torch.minimum(val_min, param.val)
                val_max = torch.as_tensor(param.val_max if param.val_max is not None else 1.0, device=param.val.device)
                val_max = torch.maximum(val_max, param.val)
                min_ = torch.clamp(param.val * (1.0 - self.cont_scale_uniform), min=val_min, max=val_max)
                max_ = torch.clamp(param.val * (1.0 + self.cont_scale_uniform), min=val_min, max=val_max)
                min_, max_ = torch.minimum(min_, max_), torch.maximum(min_, max_)
                param_val = torch.rand(size=param.val.shape, generator=self.torch_rng, device=param.val.device)
                param_val = param_val * (max_ - min_) + min_

            with torch.no_grad():
                param.val.copy_(param_val)

        def perturb(self, param, param_dtype, param_stats, is_sub_func, seed=None):
            scale_stat = self.cont_scale_stat if param_dtype.startswith('FLOAT') else self.discrete_scale_stat
            scale_uniform = self.cont_scale_uniform if param_dtype.startswith('FLOAT') else self.discrete_scale_uniform
            rng = np.random.default_rng(seed=seed)

            # sample parameter value from dataset distribution if there are sufficient observations
            # force uniform sampling if the node is a user-defined sub-function
            if (not is_sub_func) and (param_dtype in param_stats) and (param_stats[param_dtype]['count'] >= self.valid_count):
                default = np.asarray(param.val)
                stats = param_stats[param_dtype]
                
                # ranged parameter
                if param.val_range is None:
                    std = np.asarray(stats['std']) * scale_stat
                    min_, max_ = np.asarray(stats['min']), np.asarray(stats['max'])
                    param_val = rng.normal(loc=default, scale=std)
                    if param.clamped:
                        param_val = np.clip(param_val, a_min=min_, a_max=max_)
                # categorical parameter
                else:
                    value_freq = {v: freq for v, freq in stats['value_freq'].items() if int(v) in param.val_range}
                    value_freq_total = max(sum(value_freq.values()), 1)
                    value_probs = [value_freq.get(str(v), 0) / value_freq_total for v in param.val_range]
                    param_val = rng.choice(param.val_range, p=value_probs)

            # conservative uniform sampling
            else:
                default = np.asarray(param.val)

                # ranged parameter
                if param.val_range is None:
                    val_min = np.asarray(param.val_min if param.val_min is not None else 0.0)
                    val_min = np.minimum(val_min, default) if not param.clamped else val_min
                    val_max = np.asarray(param.val_max if param.val_max is not None else 1.0)
                    val_max = np.maximum(val_max, default * 2 - val_min) if not param.clamped else val_max
                    min_ = np.clip(default * (1.0 - scale_uniform), a_min=val_min, a_max=val_max)
                    max_ = np.clip(default * (1.0 + scale_uniform), a_min=val_min, a_max=val_max)
                    # default can be negative; in such case, we need to swap min and max values
                    min_, max_ = np.minimum(min_, max_), np.maximum(min_, max_)
                    param_val = rng.uniform(low=min_, high=max_)
                # categorical parameter
                else:
                    rand_int, rand_prob = rng.choice(param.val_range), rng.random()
                    param_val = rand_int if rand_prob < scale_uniform else default

            if isinstance(param_val, (np.ndarray, np.integer, np.floating)):
                param_val = param_val.tolist()

            if param_dtype.startswith('INTEGER'):
                if isinstance(param_val, list):
                    param_val = [int(round(v)) for v in param_val]
                elif isinstance(param_val, float):
                    param_val = int(round(param_val))
                elif not isinstance(param_val, int):
                    raise RuntimeError(f'Unknown type of sampled parameters: {type(param_val)}')

            param.val = param_val

    @staticmethod
    def default_sample_args():
        return {'valid_count': 100,
                'cont_scale_stat': 0.06,
                'cont_scale_uniform': 0.2,
                'discrete_scale_stat': 0.06,
                'discrete_scale_uniform': 1.0,
                'unsupported_only': True,
                "sample_unsupported_nodes_prob": 0.9,
                "trainable_rescale_factor": 1.8}

    def randomize_node_params_heuristically(self, node_type_stats, sample_args=None, seed=0):
        if self.active_node_seq is None:
            raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

        np_rng = np.random.default_rng(seed=seed)
        torch_rng = torch.Generator(device=self.device)
        torch_rng.manual_seed(seed)

        if sample_args is None:
            sample_args = self.default_sample_args()
        sampler = self.ParameterSampler(node_type_stats, np_rng, torch_rng,
                                        valid_count=sample_args['valid_count'],
                                        cont_scale_stat=sample_args['cont_scale_stat'],
                                        cont_scale_uniform=sample_args['cont_scale_uniform'],
                                        discrete_scale_stat=sample_args['discrete_scale_stat'],
                                        discrete_scale_uniform=sample_args['discrete_scale_uniform'],
                                        unsupported_only=sample_args['unsupported_only'],
                                        sample_unsupported_nodes_prob=sample_args['sample_unsupported_nodes_prob'],
                                        trainable_rescale_factor=sample_args['trainable_rescale_factor'])

        for node in self.active_node_seq:
            sampler.sample(node)
        for node in self.active_unsupported_gens:
            sampler.sample(node)

        return sampler.error_log

    def clamp_trainable_node_params(self):
        with torch.no_grad():
            for node in self.active_node_seq:
                for param in node.params:
                    if param.trainable and (param.val_min is not None or param.val_max is not None):
                        param.val.clamp_(min=param.val_min, max=param.val_max)

    # hook functions must accept to named arguments: node and val
    def register_node_output_hook(self, output_uname, func):
        # make sure a node output with the given variable name exists in the graph
        self.get_node_output_by_uname(uname=output_uname, error_on_miss=True)
        # add hook
        self.node_output_hooks[output_uname] = func

    def clear_node_output_hooks(self):
        self.node_output_hooks = {}

    def default_output_image(self, usage, device=None):
        if device is None:
            device = self.device

        if usage == 'metallic':
            output_image = torch.zeros(1, 1, 2**self.res[0], 2**self.res[1], device=self.device)
        elif usage == 'roughness':
            output_image = torch.ones(1, 1, 2**self.res[0], 2**self.res[1], device=self.device)
        elif usage == 'baseColor':
            output_image = torch.ones(1, 4 if self.use_alpha else 3, 2**self.res[0], 2**self.res[1], device=self.device)
        elif usage == 'normal':
            output_image = torch.ones(1, 4 if self.use_alpha else 3, 2**self.res[0], 2**self.res[1], device=self.device)
            output_image[:2,:,:] = output_image[:2,:,:] / 2.0
        elif usage == 'height':
            output_image = torch.zeros(1, 1, 2**self.res[0], 2**self.res[1], device=self.device)
        else:
            raise RuntimeError(f'Default image not supported for usage: {usage}')

        return output_image

    def forward(self, input_image_dict, save_node_outputs=False, use_output_defaults=True, output_usages=None, validate_signatures=False, slient=False):
        '''
        Make a forward pass through the graph (using the given input images as initial variables).
        '''

        if self.active_node_seq is None:
            raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

        if any(node.type == 'Unsupported' for node in self.active_node_seq):
            raise RuntimeError('Cannot run a node sequence that contains unsupported nodes.')

        # initialize variables (non-parameter node outputs and inputs) used in the substance graph
        variables = copy.copy(input_image_dict)

        # apply output hooks to the outputs of the unsupported generator nodes
        if len(self.node_output_hooks) > 0:
            for uname in variables:
                if uname in self.node_output_hooks:
                    node = self.get_node_output_by_uname(uname=uname, error_on_miss=True).node
                    self.node_output_hooks[uname](node=node, val=variables[uname])

        # update intermediate outputs with the initial variables (generated input images)
        all_node_outputs = None
        if save_node_outputs:
            all_node_outputs = []
            for node in self.nodes:
                if node.type == 'Unsupported' and len(node.get_connected_inputs()) == 0:
                    for node_output in node.outputs:
                        # to get an output index for the node output, I would also need to store the node outputs in a list (not a map), or also store an explicit index
                        all_node_outputs.append((node.name, node_output.uname(), variables[node_output.uname()].clone().detach().cpu()))

        # forward through the node sequence
        for node in self.active_node_seq:

            # get input variables for the node
            node_inputs = {}
            for node_input_slot in node.inputs:
                if node_input_slot.parent is None:
                    node_inputs[node_input_slot.name] = None
                else:
                    node_inputs[node_input_slot.name] = variables[node_input_slot.parent_output.uname()]

            # run node
            node_outputs = node(node_inputs, validate_signatures=validate_signatures)

            # save outputs of the node as variables
            for output_index, node_output_slot in enumerate(node.outputs):
                variables[node_output_slot.uname()] = node_outputs[output_index]
                # apply output hook
                if len(self.node_output_hooks) > 0 and node_output_slot.uname() in self.node_output_hooks:
                    self.node_output_hooks[node_output_slot.uname()](node=node, val=variables[node_output_slot.uname()])
                # update intermediate outputs
                if save_node_outputs:
                    all_node_outputs.append((node.name, node_output_slot.uname(), node_outputs[output_index].clone().detach().cpu()))

        # get graph outputs
        outputs = {}
        for output in self.outputs:
            output_variable_name = output.parent.get_variable_name()
            if output_variable_name is not None and output_variable_name in variables:
                outputs[output.usage] = variables[output_variable_name]

        # optionally use defaults for output usages that are not in the graph
        if use_output_defaults:
            if output_usages is None:
                output_usages = ['baseColor', 'normal', 'metallic', 'roughness']
            for usage in output_usages:
                if usage not in outputs:
                    if not slient:
                        print(f'Output {usage} is missing, using default value.')
                    outputs[usage] = self.default_output_image(usage=usage)

        # scale outputs to target resolution if necessary (nodes may define their own resolutions, so outputs may have different resolutions)
        res_h = 1 << self.res[0]
        res_w = 1 << self.res[1]
        for output_name, output in outputs.items():
            if output.shape[2] != res_h or output.shape[3] != res_w:
                outputs[output_name] = torch.nn.functional.interpolate(output, size=[res_h, res_w], mode='bilinear', align_corners=False)

        if save_node_outputs:
            return outputs, all_node_outputs
        else:
            return outputs