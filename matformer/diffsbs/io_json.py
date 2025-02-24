# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import json
import pickle
from collections import OrderedDict
from copy import deepcopy

import torch

from .sbs_param_type import param_val_to_type, param_type_idx_to_name, param_type_name_to_idx, SBSParamType
from .sbs_utils import resolve_dependency_path


def load_json_graph(graph_name, filename=None, json_data=None, use_alpha=False, res=None, use_abs_paths=True):
    from .sbs_graph import SBSGraph
    from .sbs_graph_nodes import SBSNodeOutput, SBSNodeInput, SBSNodeParameter, SBSNodeDefinition

    # load json graph
    if json_data is None:
        if filename is None:
            raise RuntimeError('Need to provide either filename and/or json data.')
        with open(filename, 'r') as f:
            json_nodes = json.load(f)
    else:
        json_nodes = json_data

    # validate json graph
    json_node_names = [json_node['name'] for json_node in json_nodes]
    if len(json_node_names) != len(set(json_node_names)):
        raise RuntimeError('Node names need to be unique')

    # create graph
    graph = SBSGraph(graph_name=graph_name, use_alpha=use_alpha, res=res)

    # create graph outputs
    graph_outputs = {}
    for json_node in json_nodes:
        if json_node['func'] in [None, 'output'] or json_node['func'].startswith('output_'):
            # output_name = f'Output_{len(graph.outputs)}'
            output_name = json_node['name']
            if json_node['func'].startswith('output_'):
                output_usage = json_node['func'][len('output_'):]
            else:
                output_usage = output_name
            graph_output = graph.create_output(name=output_name, usage=output_usage, group='Material')
            graph_outputs[json_node['name']] = graph_output

    from . import io_sbs
    node_func_to_type = {f'F.{nf}': nt for (nt, nf) in io_sbs.type_dict.values()}

    # create nodes and create a map from output names to node outputs
    nodes = {}
    node_outputs = {}
    for json_node in json_nodes:

        # get info about the node type, gathered from the dataset (so some info is available even for unsupported nodes)
        # and also add dependencies if needed
        node_def = None
        if json_node['func'] not in [None, 'output'] and not json_node['func'].startswith('output_'):

            node_def_path = json_node['def_path']
            node_def_graph_name = json_node['def_graph_name']

            # create node definition with existing dependency or create a new one
            if node_def_path is not None:
                # make relative dependency paths absolute, to make them independent of the filename
                # also resolve symlinks since the sbs renderer/cooker seem to have difficulties with symlinks
                if use_abs_paths:
                    if node_def_path is not None and '://' not in node_def_path:
                        node_def_path = os.path.realpath(os.path.abspath(resolve_dependency_path(path=node_def_path, source_filename=filename, resource_dirs={})))
                node_def = SBSNodeDefinition(graph=node_def_graph_name, path=node_def_path)

        node_type = None
        graph_output = None
        if json_node['func'] in [None, 'output'] or json_node['func'].startswith('output_'):
            # output node
            node_type = 'Output'
            graph_output = graph_outputs[json_node['name']]
        elif json_node['func'] not in node_func_to_type:
            # unsupported node
            node_type = 'Unsupported'
        else:
            # supported node
            node_type = node_func_to_type[json_node['func']]

        node = graph.create_node(node_type=node_type, node_name=json_node['name'], node_func=json_node['func'], graph_output=graph_output, node_def=node_def)
        # node.definition = node_def
        nodes[json_node['name']] = node

        # Distance nodes may need an alpha channel when combine is true (and the img_source input is connected)
        # set combine to False to be safe (should have no effect if the img_source input is not connected)
        if node.type == 'Distance' and not graph.use_alpha:
            node.get_param_by_name('combine').val = False

        # add node outputs for unsupported nodes
        # (node outputs for supported nodes have already been generated)
        # and save for all node outputs: (node, node output)
        for json_output in json_node['outputs']:
            output_name = json_output[0] # node_output_names[oi]
            output_uname = json_output[1]

            if node.type == 'Unsupported':
                output_dtype = json_output[2] # SBSParamType.ENTRY_COLOR.value # might not be correct for all node types, ideally we should create a list of node output types for each output parameter
                node.add_output(SBSNodeOutput(name=output_name, dtype=output_dtype, name_xml=output_name))

            # save (node, node output)
            node_outputs[output_uname] = node.get_output_by_name(output_name)

        # add node parameters
        for json_param in json_node['params']:

            param_name = json_param[0]
            param_val = json_param[1]
            param_dtype = json_param[2]
            # param_isdefault = json_param[3]

            param = node.get_param_by_name(name=param_name)

            if node.type == 'Unsupported':
                if param is None:
                    node.add_param(SBSNodeParameter(name=param_name, val=param_val, dtype=param_type_name_to_idx(param_dtype), name_xml=param_name))
                else:
                    if param_type_name_to_idx(param_dtype) != param.dtype:
                        raise RuntimeError('Data types do not match.')
                    param.val = param_val
            else:
                if param is None:
                    raise RuntimeError(f'Could not find parameter {param_name} in supported node {node.name} (type: {node.type}).')
                param.val = param_val

    # add node inputs for unuspported nodes and create connections
    # (node inputs for supported nodes have already been generated)
    for json_node in json_nodes:
        node = nodes[json_node['name']]

        # for ii, json_input in enumerate(json_node['inputs']):
        for json_input in json_node['inputs']:
            input_name = json_input[0]
            input_parent_output = json_input[1]

            # generate node inputs for unsupported nodes
            # node inputs for supported nodes have already been generated for the known parameters
            if node.type == 'Unsupported':
                input_name = json_input[0]
                input_dtype = json_input[2]
                node.add_input(node_input=SBSNodeInput(name=input_name, dtype=input_dtype, name_xml=input_name))

            if input_parent_output is None:
                # no connection
                continue

            if input_parent_output not in node_outputs:
                raise RuntimeError(f'Dangling connection found in graph {graph_name}.')

            # need to select node input by index, since generated graphs don't have names for the input slots
            node.get_input_by_name(input_name).connect(parent_output=node_outputs[input_parent_output])

    return graph


def create_json_node(node, ignore_default_params=False):
    from .sbs_graph_nodes import SBSNode

    def_path = node.definition().path if node.definition() is not None else None

    json_node = {
        'name': node.name,
        'func': node.func,
        'def_graph_name': node.definition().graph if node.definition() is not None else None,
        'def_path': def_path,
        'inputs': [],
        'outputs': [],
        'params': []}

    for node_input in node.inputs:
        if node_input.parent is None:
            json_node['inputs'].append([node_input.name, None, node_input.dtype])
        else:
            json_node['inputs'].append([node_input.name, f'output_{node_input.parent_output.uname()}', node_input.dtype])

    for node_output in node.outputs:
        json_node['outputs'].append([node_output.name, f'output_{node_output.uname()}', node_output.dtype])

    if node.type != 'Unsupported':
        default_params = {p.name: p for p in SBSNode.get_default_params(node_type=node.type, res=node.res, use_alpha=node.use_alpha)}
    else:
        default_params = None
    for param in node.params:

        # special case: for some reason the three substance graphs that use st_sand as subgraph use an additional parameter 'Sand_Color'
        # for the st_sand node that is not defined as paramter in the subgraph. Not sure why this does not cause an error when loading the sbs file in Substance Designer.
        # TODO: in load_sbs, check that paramters that are defined for an unsupported non-atomic node actually exist in the node (using load_sbs_graph_signature)
        if node.func == 'st_sand' and param.name == 'Sand_Color':
            print(f'WARNING: skipping invalid parameter {param.name} of node with function {node.func}')
            continue

        param_val = param.val.tolist() if isinstance(param.val, torch.Tensor) else param.val
        if node.type != 'Unsupported':
            default_val = default_params[param.name].val.tolist() if isinstance(param.val, torch.Tensor) else default_params[param.name].val
            is_default = bool(param_val == default_val)
            if ignore_default_params and is_default:
                continue
        else:
            is_default = False
        # need cast to bool since sometimes one or both of the values are numpy classes (like float64), which results in the numpy bool_ class instead of bool and bool_ can't be JSON serialized
        json_node['params'].append([param.name, param_val, param_type_idx_to_name(param_val_to_type(param_val)), is_default])

    return json_node


def save_json_graph(graph, filename=None, ignore_default_params=False, **json_kwargs):
    if graph.active_node_seq is None:
        raise RuntimeError('Active nodes have not been identified yet, call get_active_nodes first.')

    # generate edge list
    json_nodes = []
    for node in graph.active_node_seq:
        json_nodes.append(create_json_node(node, ignore_default_params=ignore_default_params))

    # also add all unsupported generator nodes
    # input slots are also written to retain complete signatures
    for node in graph.active_unsupported_gens:
        json_nodes.append(create_json_node(node))

    # also add all reachable output nodes
    for node in graph.active_output_nodes:

        def_path = node.definition().path if node.definition() is not None else None

        json_node = {
            'name': node.name,
            'func': f'output_{node.graph_output.usage}',
            'def_graph_name': node.definition().graph if node.definition() is not None else None,
            'def_path': def_path,
            'inputs': [],
            'outputs': [],
            'params': []}
        json_nodes.append(json_node)
        json_node['inputs'].append(['input', f'output_{node.get_variable_name()}', node.get_dtype()])

    if filename is not None:
        save_json_file(obj=json_nodes, filename=filename, **json_kwargs)

    return json_nodes


def load_json_file(filename, decompress=False):
    if decompress:
        with open(filename, 'rb') as f:
            json_data = pickle.load(f)
    else:
        with open(filename, 'r') as f:
            json_data = json.load(f)

    return json_data


def save_json_file(obj, filename, round_float=None, compress=False, **json_kwargs):
    # helper function to round floating-point values
    def round_float_value(obj):
        if isinstance(obj, float):
            return round(obj, round_float)
        elif isinstance(obj, dict):
            return {k: round_float_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [round_float_value(v) for v in obj]
        else:
            return obj

    # round floats if necessary
    if round_float is not None:
        if not isinstance(round_float, int) or round_float < 0:
            raise ValueError(f'Invalid rounding level for floats: {round_float}')
        obj = round_float_value(obj)
    if compress:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'w') as f:
            json.dump(obj, f, **json_kwargs)


def get_node_type_signature(json_node, skip_seeds=False, quantized=False):
    # initialize an empty signature
    signature = {
        'func': json_node['func'],
        'def_graph_name': json_node['def_graph_name'],
        'def_path': json_node['def_path'],
        'input_names': OrderedDict(),
        'output_names': OrderedDict(),
        'parameters': {},
    }

    # node type inputs and outputs
    sig_inputs = signature['input_names']
    sig_outputs = signature['output_names']
    for json_input in json_node['inputs']:
        # special case: 'roughness' input in the 'st_stylized_look_filter.sbs:st_stylized_look_filter' should not be used
        # (st_stylized_look_filter is defined in 423 different .sbs files. Usually it seems to be the same, but some of them have a 'roughness' input that is not present in others,
        # so just ignore this input and don't use it)
        # if node_type_name == 'st_stylized_look_filter.sbs:st_stylized_look_filter' and json_input[0] == 'roughness':
        #     continue
        if json_input[2] is None:
            print('WARNING: None found for node input dtype!')
            input_dtype = SBSParamType.ENTRY_VARIANT.name
        else:
            input_dtype = param_type_idx_to_name(json_input[2])
        sig_inputs[json_input[0]] = input_dtype

    for json_output in json_node['outputs']:
        if json_output[0] not in sig_outputs:
            sig_outputs[json_output[0]] = {'types': {}}
        if json_output[2] is None:
            print('WARNING: None found for node output dtype!')
            output_dtype = SBSParamType.ENTRY_VARIANT.name
        else:
            output_dtype = param_type_idx_to_name(json_output[2])
        sig_outputs[json_output[0]] = output_dtype

    # node type parameters
    sig_params = signature['parameters']
    for param in json_node['params']:
        param_name = param[0]
        if param_name in ['randomseed', 'seed'] and skip_seeds:
            continue
        param_val = param[1]
        param_dtype = param[2]
        if quantized:
            param_dtype = param_type_idx_to_name(param_val_to_type(val=param_val))

        # validate parameter
        if param_val is None:
            raise RuntimeError('Unset parameter found.')
        if not any(param_dtype.startswith(prefix) for prefix in ['FLOAT', 'INTEGER', 'BOOLEAN', 'STRING']):
            raise RuntimeError(f'Unrecognized parameter type: {param_dtype}')

        # add parameter stats if necessary
        sig_params[param_name] = {'type': param_dtype, 'val': param_val}

    return signature


def match_node_type_signature(signature, node_type, ignored_params=[]):
    # check function name and definition graph name
    if (node_type['func'] != signature['func'] or
        node_type['def_graph_name'] != signature['def_graph_name']):
        return False

    # check definition path
    if node_type['def_path'] == signature['def_path']:
        return True

    # check whether the function signatures match
    # first check slot and parameter names
    for item in ['input_names', 'output_names']:
        if node_type[item].keys() != signature[item].keys():
            return False

    # check input and output slot types (exclude output node types)
    is_output_node = node_type['func'] in ('output_baseColor', 'output_normal', 'output_roughness', 'output_metallic')
    for item in ['input_names', 'output_names']:
        for slot_name, slot_dtype in signature[item].items():
            if not is_output_node and slot_dtype not in node_type[item][slot_name]['types']:
                return False

    # check parameter names
    ignored_params = set(ignored_params)
    if node_type['parameters'].keys() - ignored_params != signature['parameters'].keys() - ignored_params:
        return False

    # check parameter types
    nt_params = node_type['parameters']
    for param_name, param_info in signature['parameters'].items():
        param_dtype = param_info['type']
        if '_ARRAY_' in param_dtype:
            param_array_item_dtype = param_dtype[:param_dtype.find('_ARRAY_')]
            if not any(k.startswith(param_array_item_dtype) for k in nt_params[param_name].keys()):
                return False
        elif param_dtype not in node_type['parameters'][param_name]:
            return False

    # found an exact match
    return True


def get_json_diff(json_nodes):
    json_diff = {'__type__': 'diff'}

    # for each node, only keep params that are different from the default
    for i, jn in enumerate(json_nodes):
        for j, jp in enumerate(jn['params']):
            if not jp[3]:
                json_diff.setdefault(i, {})[j] = jp[1]

    return json_diff


def apply_json_diff(json_nodes_init, json_diff):
    # check input type
    if not isinstance(json_diff, dict) or json_diff.get('__type__') != 'diff':
        raise ValueError('Invalid input type for json_diff.')

    # apply the difference between two json files of the same graph
    json_nodes = deepcopy(json_nodes_init)
    json_diff = deepcopy(json_diff)
    del json_diff['__type__']

    for i, node_diff in json_diff.items():
        json_params = json_nodes[int(i)]['params']
        for j, param_val in node_diff.items():
            json_params[int(j)][1] = param_val
            json_params[int(j)][3] = False

    return json_nodes
