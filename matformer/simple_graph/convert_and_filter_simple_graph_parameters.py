# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import numpy as np

from .simple_graph import get_node_type_name
from .convert_simple_graph_parameters import get_param_limits, dequantize_param_val
from ..diffsbs.io_json import get_node_type_signature, match_node_type_signature


def exempt_integer_parameters(node_type_name, param_name):
    if node_type_name == 'anisotropic_noise.sbs:anisotropic_noise' and param_name == 'Y_Amount':
        return True
    elif node_type_name == 'noise_anisotropic_noise.sbs:anisotropic_noise' and param_name == 'Y_Amount':
        return True
    elif node_type_name == 'pattern_scratches_generator.sbs:scratches_generator' and param_name == 'spline_number':
        return True
    elif node_type_name == 'noise_cells_1.sbs:cells_1' and param_name == 'scale':
        return True
    elif node_type_name == 'waveform_1.sbs:waveform_1' and param_name == 'Samples':
        return True
    elif node_type_name == 'perforated_swirl_filter.sbs:perforated_swirl_filter' and param_name == 'pattern_amount':
        return True
    elif node_type_name == 'scratches_generator.sbs:scratches_generator' and param_name == 'scratches_amount':
        return True
    elif node_type_name == 'pattern_scratches_generator.sbs:scratches_generator_normal' and param_name == 'spline_number':
        return True
    elif node_type_name == 'EZSplatter.sbs:EZSplatter' and param_name == 'Number':
        return True
    elif node_type_name == 'mg_mask_builder.sbs:mg_mask_builder_2' and param_name == 'scratches_amount':
        return True
    elif node_type_name == 'crop.sbs:crop_grayscale' and param_name == 'input_size':
        return True
    elif node_type_name == 'crop.sbs:multi_crop_grayscale' and param_name == 'input_size':
        return True
    elif node_type_name == 'spin_brushed_filter.sbs:spin_brushed_filter' and param_name == 'scratch_y_amount':
        return True

    return False


def clamp_val_max(val_min, val_max, step_count):
    val_min = np.asarray(val_min, dtype=np.int64)
    val_max = np.asarray(val_max, dtype=np.int64)

    val_max_allowed = val_min + step_count - 1
    val_max_clamped = np.minimum(val_max, val_max_allowed)

    return val_max_clamped


def unconvert_clamped_simple_graph_parameters(json_nodes, node_types, step_count, use_alpha):
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

            param_val_min, param_val_max = get_param_limits(node_func=json_node['func'], param_name=param_name,
                                                            param_stats=param_stats, use_alpha=use_alpha)

            if param_dtype.startswith('INTEGER') and not exempt_integer_parameters(get_node_type_name(json_node), param_name):
                param_val_max = clamp_val_max(param_val_min, param_val_max, step_count)

            param_val = dequantize_param_val(val=param_val, dtype=param_dtype, step_count=step_count,
                                             val_min=param_val_min, val_max=param_val_max, param_stats=param_stats)

            json_param[0] = param_name
            json_param[1] = param_val
            json_param[2] = param_dtype
            json_param[3] = is_default

    return json_nodes
