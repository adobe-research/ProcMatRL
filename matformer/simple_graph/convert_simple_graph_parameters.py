# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import numpy as np

from .simple_graph import get_node_type_name
from ..diffsbs.io_json import get_node_type_signature, match_node_type_signature


def unconvert_simple_graph_parameters(json_nodes, node_types, step_count, use_alpha):

    for json_node in json_nodes:
        node_type_info = node_types[get_node_type_name(json_node, node_types=node_types)]
        # node_type_info = node_types[json_node['func']]

        # disambiguation
        if isinstance(node_type_info, list):
            graph_node_signature = get_node_type_signature(json_node, skip_seeds=True)
            node_type_info = next((nt for nt in node_type_info if match_node_type_signature(graph_node_signature, nt)), None)
            if node_type_info is None:
                raise RuntimeError(f'Node type signature not found: {graph_node_signature}')

        for json_param in json_node['params']:

            param_name = json_param[0]
            param_val = json_param[1]
            param_dtype = json_param[2]
            is_default = json_param[3]

            param_stats = node_type_info['parameters'][param_name]
            # param_dtype = get_param_dtype(node_type=json_node['func'], param_name=param_name, param_stats=param_stats)

            param_val_min, param_val_max = get_param_limits(node_func=json_node['func'], param_name=param_name,
                                                            param_stats=param_stats, use_alpha=use_alpha)
            param_val = dequantize_param_val(val=param_val, dtype=param_dtype, step_count=step_count,
                                             val_min=param_val_min, val_max=param_val_max, param_stats=param_stats)

            json_param[0] = param_name
            json_param[1] = param_val
            json_param[2] = param_dtype
            json_param[3] = is_default

    return json_nodes


def dequantize_param_val(val, dtype, step_count, val_min, val_max, param_stats, delta=0.5):
    val_min = np.array(val_min)
    val_max = np.array(val_max)

    if dtype.startswith('FLOAT'):
        # +delta to center the value in the range of float values that are quantized to the given integer
        val = (((np.array(val).astype('float64') + delta) / step_count) * (val_max-val_min) + val_min).tolist()

    elif dtype.startswith(('INTEGER', 'BOOLEAN')):
        val_max = val_max.astype('int64')
        val_min = val_min.astype('int64')
        if (val_max-val_min).max() < step_count:
            val = np.array(val) + val_min  # no need to quantize, integer with fewer possible values than step_count
        else:
            # +delta to center the value in the range of float values that are quantized to the given integer
            val = (((np.array(val).astype('float64') + delta) / step_count) * (val_max-val_min) + val_min)
        if dtype.startswith('BOOLEAN'):
            val = val.astype('bool').tolist()
        else:
            val = val.astype('int64').tolist()

    elif dtype == 'STRING':
        choices = sorted(list(param_stats[dtype]['value_freq'].keys()))
        if val >= len(choices):
            raise RuntimeError(f'Choice parameter index {val}, exceeds the number of choices: {choices}.')
        val = choices[val]

    else:
        raise RuntimeError(f'Unexpected data type: {dtype}.')

    return val


def get_param_limits(node_func, param_name, param_stats, use_alpha):
    if node_func == 'F.gradient_map' and param_name == 'anchors':

        # convert min/max to single vector
        n_dim = 5 if use_alpha else 4
        val_min = np.full([n_dim], np.inf)
        val_max = np.full([n_dim], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype.startswith('FLOAT2_ARRAY'):
                if use_alpha:
                    expanded_min = np.array(param_dtype_stats['min'])[:, [0, 1, 1, 1, 1]]
                    expanded_min[:, -1] = 1  # manually set as 1
                    expanded_max = np.array(param_dtype_stats['max'])[:, [0, 1, 1, 1, 1]]
                    expanded_max[:, -1] = 1  # manually set as 1
                    val_min = np.minimum(val_min, expanded_min.min(axis=0))
                    val_max = np.maximum(val_max, expanded_max.max(axis=0))
                else:
                    val_min = np.minimum(val_min, np.array(param_dtype_stats['min'])[:, [0, 1, 1, 1]].min(axis=0))
                    val_max = np.maximum(val_max, np.array(param_dtype_stats['max'])[:, [0, 1, 1, 1]].max(axis=0))
            elif param_dtype.startswith('FLOAT4_ARRAY'):
                assert not use_alpha
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']).min(axis=0))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['min']).max(axis=0))
            elif param_dtype.startswith('FLOAT5_ARRAY'):
                assert use_alpha
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']).min(axis=0))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['min']).max(axis=0))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'F.curve' and param_name == 'anchors':

        # convert min/max to single vector
        val_min = np.full([6], np.inf)
        val_max = np.full([6], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype.startswith('FLOAT6_ARRAY'):
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']).min(axis=0))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']).max(axis=0))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'F.levels' and param_name in ['out_low', 'out_high', 'in_mid', 'in_low', 'in_high']:

        # convert min/max to single vector
        n_dim = 4 if use_alpha else 3
        val_min = np.full([n_dim], np.inf)
        val_max = np.full([n_dim], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'FLOAT1':
                expanded_min = np.array(param_dtype_stats['min']).repeat(n_dim)
                expanded_max = np.array(param_dtype_stats['max']).repeat(n_dim)
                if use_alpha:
                    expanded_min[-1], expanded_max[-1] = 1, 1
                val_min = np.minimum(val_min, expanded_min)
                val_max = np.maximum(val_max, expanded_max)
            elif param_dtype == 'FLOAT3':
                assert not use_alpha
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            elif param_dtype == 'FLOAT4':
                assert use_alpha
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func in ['F.uniform_color', 'F.make_it_tile_patch'] and param_name in ['rgba', 'background_color']:

        # convert min/max to single vector
        val_min = np.full([4], np.inf)
        val_max = np.full([4], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'FLOAT1':
                val_min = np.minimum(val_min, np.concatenate([np.array(param_dtype_stats['min']).repeat(3), [1.0]]))
                val_max = np.maximum(val_max, np.concatenate([np.array(param_dtype_stats['max']).repeat(3), [1.0]]))
            elif param_dtype == 'FLOAT3':
                val_min = np.minimum(val_min, np.concatenate([np.array(param_dtype_stats['min']), [1.0]]))
                val_max = np.maximum(val_max, np.concatenate([np.array(param_dtype_stats['max']), [1.0]]))
            elif param_dtype == 'FLOAT4':
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'tile_generator' and param_name == 'interstice':
        val_min = np.full([4], np.inf)
        val_max = np.full([4], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'FLOAT2':  # (gap x/y, rand factor x/y) (?)
                # warnings.warn("This functionality is deprecated. The new dataset should not contain this inconsistency.")
                raise RuntimeError("This functionality is deprecated. The new dataset should not contain this inconsistency.")
                val_min = np.minimum(val_min, np.tile(param_dtype_stats['min'], 2))
                val_max = np.maximum(val_max, np.tile(param_dtype_stats['max'], 2))
            elif param_dtype == 'FLOAT4':  # (gap x, rand factor x, gap y, rand factor y)
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'F.quantize' and param_name == 'quantize_number':
        val_min = np.full([4], np.inf)
        val_max = np.full([4], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'INTEGER1':
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']).repeat(4))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']).repeat(4))
            elif param_dtype == 'INTEGER3':
                val_min = np.minimum(val_min, np.concatenate([np.array(param_dtype_stats['min']), [1]]))
                val_max = np.maximum(val_max, np.concatenate([np.array(param_dtype_stats['max']), [1]]))
            elif param_dtype == 'INTEGER4':
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'st_sand' and param_name == 'Waves_Distortion':
        val_min = np.inf
        val_max = -np.inf
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'INTEGER1':
                val_min = min(val_min, float(param_dtype_stats['min']))
                val_max = max(val_max, float(param_dtype_stats['max']))
            elif param_dtype == 'FLOAT1':
                val_min = min(val_min, param_dtype_stats['min'])
                val_max = max(val_max, param_dtype_stats['max'])
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')

    elif node_func == 'window_generator' and param_name == 'window_brace_offset':
        val_min = np.full([2], np.inf)
        val_max = np.full([2], -np.inf)
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'FLOAT1':
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']).repeat(2))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']).repeat(2))
            elif param_dtype == 'FLOAT2':
                val_min = np.minimum(val_min, np.array(param_dtype_stats['min']))
                val_max = np.maximum(val_max, np.array(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')
        val_min = val_min.tolist()
        val_max = val_max.tolist()

    elif node_func == 'perforated_swirl_filter' and param_name == 'use_scale_input':
        val_min = np.inf
        val_max = -np.inf
        for param_dtype, param_dtype_stats in param_stats.items():
            if param_dtype == 'FLOAT1':
                val_min = min(val_min, param_dtype_stats['min'])
                val_max = max(val_max, param_dtype_stats['max'])
            elif param_dtype == 'BOOLEAN':
                val_min = min(val_min, float(param_dtype_stats['min']))
                val_max = max(val_max, float(param_dtype_stats['max']))
            else:
                raise RuntimeError(f'Unexpected data type for parameter {param_name} of node type {node_func}: {param_dtype}.')

    elif node_func == 'GT_BasicParameters' and param_name == 'normal_format':
        assert 'INTEGER1' in param_stats.keys() and 'BOOLEAN' in param_stats.keys()
        val_min, val_max = 0, 1

    elif node_func == 'quilt_filter' and param_name == 'selective_material_mask':
        assert 'INTEGER1' in param_stats.keys() and 'BOOLEAN' in param_stats.keys()
        val_min, val_max = 0, 1

    else:
        if len(param_stats) != 1:
            raise RuntimeError(f'Expected parameter {param_name} of node type {node_func} to have a single data type, but found multiple data types: {list(param_stats.keys())}.')
        param_dtype_stats = list(param_stats.values())[0]
        val_min = param_dtype_stats['min']
        val_max = param_dtype_stats['max']

    return val_min, val_max
