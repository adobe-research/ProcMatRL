# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import time
import copy
import random
import inspect
import platform

# Beichen Li: ignore user warnings from imageio
import warnings
warnings.filterwarnings('ignore', module='imageio', category=UserWarning)

import imageio
import torch
import numpy as np
import networkx

from skimage.transform import resize

# for substance node implementation
def input_check(tensor_args=None, profile=True):
    '''
    Decorator that checks if the input is a 4D pytorch tensor
        num_tensor_input: number of tensor input that requires checking
    '''
    def decorator(func):

        # get indices of tensor arguments
        func_arg_names = list(inspect.signature(func).parameters)
        tensor_arg_inds = [func_arg_names.index(tensor_arg) for tensor_arg in tensor_args]
        shared_arg_names = ['outputsize', 'format', 'pixelsize', 'pixelratio', 'tile_mode', 'seed', 'device']

        def wrapper(*args, **kwargs):

            # delete shared arguments that are not used by the function
            for arg_name in list(kwargs.keys()):
                if arg_name in shared_arg_names and arg_name not in func_arg_names:
                    # print(f'Warning: argument {arg_name} is ignored by function {func.__name__}')
                    del kwargs[arg_name]

            if tensor_args is not None:
                for ti in range(len(tensor_args)):
                    arg = None
                    if tensor_arg_inds[ti] < len(args):
                        arg = args[tensor_arg_inds[ti]]
                    elif tensor_args[ti] in kwargs:
                        arg = kwargs[tensor_args[ti]]
                    assert arg is None or (torch.is_tensor(arg) and len(arg.shape) == 4), f'Input tensor {tensor_args[ti]} needs to be a 4D PyTorch tensor.'
            if profile:
                t_start = time.time()
                retvals = func(*args, **kwargs)
                t_end = time.time()
                if t_end - t_start >= 0.3:
                    print("[PROFILE] {:16s}: {:.3f}s".format(func.__name__, t_end - t_start))
            else:
                retvals = func(*args, **kwargs)
            return retvals

        # extend wrapper parameters with shared parameters
        # (add them programatically to the wrapper so it does not clutter the code - also it would take too long to add them to all functions)
        func_sig = inspect.signature(func)
        func_params = func_sig.parameters
        wrapper_params_list = list(inspect.signature(func).parameters.values())
        if 'outputsize' not in func_params:
            wrapper_params_list.append(inspect.Parameter('outputsize', inspect.Parameter.KEYWORD_ONLY, default=[9, 9]))
        if 'format' not in func_params:
            wrapper_params_list.append(inspect.Parameter('format', inspect.Parameter.KEYWORD_ONLY, default=0))
        if 'pixelsize' not in func_params:
            wrapper_params_list.append(inspect.Parameter('pixelsize', inspect.Parameter.KEYWORD_ONLY, default=[1, 1]))
        if 'pixelratio' not in func_params:
            wrapper_params_list.append(inspect.Parameter('pixelratio', inspect.Parameter.KEYWORD_ONLY, default=0))
        if 'tile_mode' not in func_params:
            wrapper_params_list.append(inspect.Parameter('tile_mode', inspect.Parameter.KEYWORD_ONLY, default=3))
        if 'seed' not in func_params:
            wrapper_params_list.append(inspect.Parameter('seed', inspect.Parameter.KEYWORD_ONLY, default=0))
        if 'device' not in func_params:
            wrapper_params_list.append(inspect.Parameter('device', inspect.Parameter.KEYWORD_ONLY, default=torch.device('cpu')))
        wrapper.__signature__ = func_sig.replace(parameters=wrapper_params_list)

        return wrapper
    return decorator

@input_check(tensor_args=['img_in'])
def roll_row(img_in, n):
	# return torch.cat((img_in[:,:,n:,:], img_in[:,:,:n,:]), 2)
    return img_in.roll(-n, 2)

@input_check(tensor_args=['img_in'])
def roll_col(img_in, n):
	# return torch.cat((img_in[:,:,:,n:], img_in[:,:,:,:n]), 3)
    return img_in.roll(-n, 3)

@input_check(tensor_args=['img_in'])
def normalize(img_in):
    return img_in / torch.sqrt((img_in ** 2).sum(1, keepdim=True))

def color_input_check(tensor_input, err_var_name, with_alpha=None):
    if tensor_input.shape[1] == 1:
        tensor_input = tensor_input.expand(-1, 3, -1, -1)
    assert tensor_input.shape[1] in [3, 4], '%s should be a color image' % err_var_name
    if with_alpha is not None:
        if with_alpha and tensor_input.shape[1] == 3:
            # add alpha channel
            tensor_input = torch.cat([tensor_input, torch.ones(tensor_input.shape[0], 1, tensor_input.shape[2], tensor_input.shape[3], device=tensor_input.device, dtype=tensor_input.dtype)], dim=1)
        elif not with_alpha and tensor_input.shape[1] == 4:
            # remove alpha channel
            tensor_input = tensor_input[:, :3, :, :]
    return tensor_input

def grayscale_input_check(tensor_input, err_var_name):
    if tensor_input.shape[1] in [3, 4]:
        tensor_input = tensor_input[:, :3].mean(dim=1, keepdim=True) # convert to grayscale
    assert tensor_input.shape[1] == 1, '%s should be a grayscale image' % err_var_name
    return tensor_input

def get_shape_from_any(tensors, default_shape):
    for tensor in tensors:
        if tensor is not None and tensor.ndim == 4:
            return list(tensor.shape)
    return default_shape

# as compatiable tensor
def as_comp_tensor(x, dtype=None, device=None):
    if device is not None and isinstance(x, torch.Tensor) and x.device != device:
        raise RuntimeError(f'Incompatible pytorch devices (current device {x.device} vs. requested device {device}).')
    return torch.as_tensor(x, dtype=dtype, device=device)

# for general data loading & saving

def read_image(filename: str, target_size=None, process=False, use_alpha=False):

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    img = imageio.imread(filename)
    if img.dtype == np.float32:
        pass
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype in (np.uint16, np.int32): # np.int32 is for compatibility with pillow
        img = img.astype(np.float32) / 65535.0
    else:
        raise RuntimeError('Unexpected image data type.')

    if target_size is not None and img.shape[:2] != target_size:
        img = resize(img, target_size)

    if process:
        img = torch.from_numpy(img)
        if len(img.shape) == 3:
            img = img.permute(2,0,1)
            if use_alpha and img.shape[0] == 3:
                img = torch.cat([
                    img,
                    torch.ones(1, img.shape[1], img.shape[2], device=torch.device('cpu'))
                    ], dim=0)
            elif not use_alpha and img.shape[0] == 4:
                img = img[:3,:,:]
        else:
            img = img.unsqueeze(0)

    return img

def write_image(filename: str, img, process=False):

    if process:
        img = img.permute(1, 2, 0) if img.shape[0] > 1 else img.squeeze(0)
        img = img.numpy()

    # See the following link for installing the OpenEXR plugin for imageio:
    # https://imageio.readthedocs.io/en/stable/format_exr-fi.html

    if np.any(np.isnan(img)):
        raise ValueError(f"Failed to write image to '{filename}': input image has nan pixels")
    if not np.all((img >= 0) & (img <= 1)):
        raise ValueError(f"Failed to write image to '{filename}': input image has out-of-bound pixel value")

    extension = os.path.splitext(filename)[1]
    if extension == '.exr':
        imageio.imwrite(filename, img)
    elif extension == '.png':
        imageio.imwrite(filename, (img * 255.0).round().astype(np.uint8))
    elif extension in ['.jpg', 'jpeg']:
        imageio.imwrite(filename, (img * 255.0).round().astype(np.uint8), quality=80)
    else:
        raise RuntimeError(f'Unexpected image filename extension {extension}.')

# random sampling function
def param_sampling_uniform(x_in, ptb_min_, ptb_max_, lb, ub):
    epsilon = 0.0
    return np.clip(x_in * (1.0 + random.uniform(ptb_min_, ptb_max_)) + epsilon, lb, ub)

# random sampling function
def param_sampling_normal(x_in, mu, sigma, lb, ub):
    epsilon = 0.0
    return np.clip(np.random.normal(mu, sigma) + x_in + epsilon, lb, ub)

# def gen_rand_params(params, params_spec, trainable_list, keys,
#     param_sampling_func, node_list=None):
#     '''
#     Given a dictionary of template parameters, this function generates a perturbed
#     dictionary of parameters by adding a random offset to each of variable

#     params: parameters saved as ordered dict
#     params_spec: specification for sampling, i.e. number of allowed variables, range of variations
#     trainable_list: list of list, the internal list contains name of trainable variables
#     keys: list of parameter keys
#     param_sampling_func: sampling function
#     node_list: pre-specified list of variable nodes, if provided won't do random generation of
#                variable nodes
#     '''
#     # random.seed(30)
#     if not node_list:
#         node_list = random.sample(range(params_spec['total_var']), params_spec['num_free_nodes'])
#     params_rand = copy.deepcopy(params)

#     if param_sampling_func is param_sampling_uniform:
#         param_a = params_spec['ptb_min']
#         param_b = params_spec['ptb_max']
#     else:
#         param_a = params_spec['mu']
#         param_b = params_spec['sigma']

#     for node_idx in node_list:
#         node_name = keys[node_idx]
#         trainable_var_name_list = trainable_list[node_idx]
#         node_prefix = node_name.split('_')[0]
#         # deal with special nodes
#         # gradient map & curve
#         if (node_prefix == "gradient" or node_prefix == "curve") and trainable_var_name_list[0] == "anchors":
#             # sample each parameter in the 2-d matrix
#             var = params_rand[keys[node_idx]]["anchors"]
#             if var is None:
#                 multiplier = 6 if node_prefix == "curve" else \
#                              4 + params_rand[keys[node_idx]]["use_alpha"] if params_rand[keys[node_idx]]["mode"] == "color" else 2
#                 var = [[0.0] * multiplier, [1.0] * multiplier]
#             if isinstance(var, list) or \
#                isinstance(var, np.ndarray) or \
#                isinstance(var, torch.Tensor):
#                 # replace the list
#                 for row_idx, row in enumerate(var):
#                     for col_idx, item in enumerate(row):
#                         var[row_idx][col_idx] = param_sampling_func(item, param_a, param_b, 0.0, 1.0)
#                 params_rand[keys[node_idx]]["anchors"] = var
#             else:
#                 raise RuntimeError("Unrecognized input type")
#         else:
#             for var_key in trainable_var_name_list:
#                 var = params_rand[keys[node_idx]][var_key]
#                 if isinstance(var, list) or \
#                    (isinstance(var, np.ndarray) and len(var.shape) == 1) or \
#                    (isinstance(var, torch.Tensor) and len(var.shape) == 1):
#                     for idx, item in enumerate(var):
#                         var[idx] = param_sampling_func(item, param_a, param_b, 0.0, 1.0)
#                 elif isinstance(var, numbers.Number) or \
#                      (isinstance(var, np.ndarray) and len(var.shape) == 0) or \
#                      (isinstance(var, torch.Tensor) and len(var.shape) == 0):
#                     var = param_sampling_func(var, param_a, param_b, 0.0, 1.0)
#                 params_rand[keys[node_idx]][var_key] = var

#     return params_rand

def gen_rand_light_params(render_params, params_spec, param_sampling_func):
    ptb_max = params_spec['ptb_max']
    ptb_min = params_spec['ptb_min']

    render_params_rand = copy.deepcopy(render_params)
    render_params_rand['size'] = param_sampling_func(render_params_rand['size'], ptb_min, ptb_max, 0.0, 1.0)
    render_params_rand['light_color'] = param_sampling_func(render_params_rand['light_color'], ptb_min, ptb_max, 0.0, 1.0)

    return render_params_rand

def nan_test(maps, use_rendering_only):
    if use_rendering_only:
        if torch.sum(maps != maps) > 0:
            raise RuntimeError("rendering has nan")
    else:
        # order: albedo, normal, metallic, roughness, render
        if torch.sum(maps[:,:3,:,:] != maps[:,:3,:,:]) > 0:
            raise RuntimeError("albedo has nan")
        if torch.sum(maps[:,3:6,:,:] != maps[:,3:6,:,:]) > 0:
            raise RuntimeError("normal has nan")
        if torch.sum(maps[:,6,:,:] != maps[:,6,:,:]) > 0:
            raise RuntimeError("metallic has nan")
        if torch.sum(maps[:,7,:,:] != maps[:,7,:,:]) > 0:
            raise RuntimeError("roughness has nan")
        if torch.sum(maps[:,8,:,:] != maps[:,8,:,:]) > 0:
            raise RuntimeError("rendered image has nan")

@input_check(tensor_args=['img'])
def bin_statistics(img, bin_shape='circular', num_bins=5, num_row_bins=2, num_col_bins=2):
    '''
    bin_shape: 'circular', 'grid'
    '''
    res_h = img.shape[2]
    res_w = img.shape[3]

    X,Y = torch.meshgrid(torch.linspace(-res_h//2,res_h//2-1, res_h), torch.linspace(-res_w//2,res_w//2-1, res_w), indexing='ij')
    dist = torch.sqrt(X*X + Y*Y).expand(img.shape[0],res_h,res_w)

    if bin_shape == 'circular':
        dist_step = (torch.max(dist)+1.0) / num_bins
        mean = torch.zeros(img.shape[1], num_bins)
        var = torch.zeros(img.shape[1], num_bins)
        for j in range(img.shape[1]):
            for i in range(num_bins):
                temp = img[:,j,:,:]
                mask = (dist >= dist_step*i) * (dist < dist_step*(i+1))
                mean[j,i] = torch.mean(temp[mask])
                var[j,i] = torch.var(temp[mask])
    elif bin_shape == 'grid':
        row_step = res_h // num_row_bins
        col_step = res_w // num_col_bins
        mean = torch.zeros(img.shape[1], num_row_bins, num_col_bins)
        var = torch.zeros(img.shape[1], num_row_bins, num_col_bins)
        for k in range(img.shape[1]):
            temp = img[:,k,:,:]
            for i in range(num_row_bins):
                for j in range(num_col_bins):
                    row_indices = torch.arange(row_step*i, row_step*(i+1))
                    col_indices = torch.arange(col_step*j, col_step*(j+1))
                    mean[k,i,j] = torch.mean(temp[:,row_indices,col_indices])
                    var[k,i,j] = torch.var(temp[:,row_indices,col_indices])

    return mean, var

@input_check(tensor_args=['fft_in'])
def fftshift2d(fft_in):
    input_shape = list(fft_in.size())

    fft_out = fft_in
    for axis in range(2, 4):
        split = (input_shape[axis] + 1) // 2
        mylist = torch.cat((torch.arange(split, input_shape[axis]), torch.arange(split)))
        fft_out = torch.index_select(fft_out, axis, mylist)
    return fft_out

@input_check(tensor_args=['fft_in'])
def ifftshift2d(fft_in):
    input_shape = list(fft_in.size())

    fft_out = fft_in
    for axis in range(2, 4):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = torch.cat((torch.arange(split, n), torch.arange(split)))
        fft_out = torch.index_select(fft_out, axis, mylist)
    return fft_out

@input_check(tensor_args=['img'])
def fourier_stats(img, scale='linear', output='intensity', normalize=True):
    '''
    scale: 'linear', 'log', 'log2'
    output: 'intensity', 'complex'
    '''
    if output == 'intensity':
        fourier_out = torch.zeros_like(img)
    elif output == 'complex':
        temp = torch.zeros_like(img)
        fourier_out = torch.stack([img, temp], dim=len(img.shape))
    else:
        raise RuntimeError("invalid output type")

    for i in range(img.shape[0]):
        slice_complex = torch.stack([img[i,:,:,:], torch.zeros_like(img[i,:,:,:])], dim=3)
        slice_fft = torch.fft(slice_complex, signal_ndim=2, normalized=normalize)

        if output == 'intensity':
            slice_intensity = torch.sqrt(slice_fft[:,:,:,0]*slice_fft[:,:,:,0] + \
                slice_fft[:,:,:,1]*slice_fft[:,:,:,1])
            fourier_out[i,:,:,:] = slice_intensity
        else:
            fourier_out[i,:,:,:,:] = slice_fft

    if scale == 'linear':
        pass
    elif scale == 'log10':
        fourier_out = torch.log10(fourier_out)
    elif scale == 'log2':
        fourier_out = torch.log2(fourier_out)
    elif scale == 'sqrt':
        fourier_out = torch.sqrt(fourier_out)
    else:
        raise RuntimeError('invalid scale type')

    return fourier_out

to_zero_one = lambda a : a / 2.0 + 0.5
from_zero_one = lambda a : (a-0.5) * 2.0

class MissingDependencyError(RuntimeError):
    pass

class InvalidNodeError(RuntimeError):
    pass

class UnsupportedNodeExpansionError(RuntimeError):
    pass

def randomize_tensor(val, rng, dist='uniform', offset_max=None, scale_offset_max=None, offset_std=None, std=None, val_min=0.0, val_max=1.0):
    if not isinstance(val, torch.Tensor):
        raise RuntimeError('Need to pass a tensor.')

    if dist == 'uniform':
        if offset_max is not None:
            range_min = torch.clamp(val-offset_max, min=val_min, max=val_max)
            range_max = torch.clamp(val+offset_max, min=val_min, max=val_max)
        elif scale_offset_max is not None:
            range_min = torch.clamp(val * (1.0 - scale_offset_max), min=val_min, max=val_max)
            range_max = torch.clamp(val * (1.0 + scale_offset_max), min=val_min, max=val_max)
        else:
            range_min = val_min
            range_max = val_max
        val.copy_(torch.rand(size=val.shape, generator=rng, device=val.device) * (range_max-range_min) + range_min)

    elif dist == 'normal':
        if offset_std is not None:
            mean = val
            std = offset_std
        elif std is not None:
            mean = (val_min+val_max)*0.5
        else:
            mean = (val_min+val_max)*0.5
            std = (val_max-val_min)*0.25
        val.copy_(torch.normal(size=val.shape, mean=mean, std=std, generator=rng, device=val.device).clamp_(min=val_min, max=val_max))

def generator_type_group(gen_type):
    tokens = gen_type.split('_')
    if len(tokens) > 1 and tokens[-1].isdigit():
        return '_'.join(tokens[:-1])
    else:
        return gen_type

def resolve_dependency_path(path, source_filename, resource_dirs):
    '''
    Resolve a dependency path into a filename.
    '''
    # convert a windows path to a unix path if necessary
    if platform.system() != 'Windows' and '\\' in path:
        # print('WARNING: dependency path appears to be in Windows format, but the current platform is not Windows.')
        path = path.replace('\\', '/')

    if path == '?himself':
        if source_filename is None:
            raise RuntimeError('Cannot resolve dependencies that are specified relative to the source filename, since the source filename was not provided.')
        # path points to current file
        return source_filename

    if '://' in path:
        tokens = path.split('://')
        if len(tokens) != 2:
            raise RuntimeError(f'Cannot parse dependency path format:\n{path}')
        prefix, path = tokens[0], tokens[1]
        if prefix not in resource_dirs:
            raise RuntimeError(f'Unknown resource location: {prefix}')
        # resource dir is relative to current working directory (not current sbs file) or absolute
        return os.path.join(os.path.normpath(os.path.abspath(resource_dirs[prefix])), path)

    # path is relative to current sbs file or absolute
    if not os.path.isabs(path):
        if source_filename is None:
            raise RuntimeError('Cannot resolve dependencies that are specified relative to the source filename, since the source filename was not provided.')
        path = os.path.join(os.path.dirname(source_filename), path)

    return os.path.normpath(path)

def missing_dependencies(deps, source_filename, resource_dirs):
    missing_deps = []
    for dep in deps:
        if not os.path.exists(resolve_dependency_path(path=dep.path, source_filename=source_filename, resource_dirs=resource_dirs)):
            missing_deps.append(dep)
    return missing_deps

def gen_unique_uids(count, existing_uids=None, exclude=None):
    if existing_uids is None:
        existing_uids = []
    if exclude is not None:
        existing_uids += exclude
    if len(existing_uids) == 0:
        existing_uids = [1000000000]
    return list(range(max(existing_uids)+1, max(existing_uids)+1+count))

def longest_nxgraph_distance(graph, from_nodes, to_nodes):
    max_dist = None
    graph_ud = graph.to_undirected()
    graph_components = [graph.subgraph(component) for component in networkx.algorithms.components.connected_components(graph_ud)]
    for comp in graph_components:
        comp_from_nodes = [node for node in from_nodes if node in comp.nodes]
        comp_to_nodes = [node for node in to_nodes if node in comp.nodes]
        if len(comp_from_nodes) == 0 or len(comp_to_nodes) == 0:
            continue

        for from_node in comp_from_nodes:
            for to_node in comp_to_nodes:
                try:
                    dist = networkx.shortest_path_length(comp, from_node, to_node)
                    if max_dist is None or dist > max_dist:
                        max_dist = dist
                except networkx.exception.NetworkXNoPath:
                    pass # ignore nodes that are not connected

    return max_dist

def json_to_networkx_slot_graph(json_nodes):
    nxgraph = networkx.DiGraph()

    # add nodes, input slots, and output slots as nodes
    # and connect input slot nodes and output slot nodes to their main node
    input_slots = {}
    for json_node in json_nodes:
        # add node
        nxgraph.add_node(json_node['name'])

        # add input slots as nodes and add intra-node edges from input slots to node
        for node_input in json_node['inputs']:
            input_slot_name = f'{json_node["name"]}_{node_input[0]}'
            nxgraph.add_node(input_slot_name)
            edge_name = f'{input_slot_name}_to_{json_node["name"]}'
            nxgraph.add_edge(input_slot_name, json_node["name"], key=edge_name)
            input_slots[input_slot_name] = node_input

        # add output slots as nodes and add intra-node edges from node to output slots
        for node_output in json_node['outputs']:
            output_slot_name = node_output[1]
            nxgraph.add_node(output_slot_name)
            edge_name = f'{json_node["name"]}_to_{output_slot_name}'
            nxgraph.add_edge(json_node["name"], output_slot_name, key=edge_name)

    # add inter-node edge from output slots to input slots
    for input_slot_name, input_slot in input_slots.items():
        output_slot_name = input_slot[1]
        if output_slot_name is not None:
            edge_name = f'{output_slot_name}_to_{input_slot_name}'
            nxgraph.add_edge(output_slot_name, input_slot_name, key=edge_name)

    return nxgraph

def json_to_networkx_graph(json_nodes, reverse_edges=True, directed=True):
    if directed:
        nxgraph = networkx.MultiDiGraph()
    else:
        nxgraph = networkx.MultiGraph()

    # add nodes
    output_parents = {}
    output_parent_slot_indices = {}
    for json_node in json_nodes:
        nxgraph.add_node(json_node['name'])
        for oi, output_info in enumerate(json_node['outputs']):
            output_name = output_info[1]
            if output_name in output_parents:
                raise RuntimeError(f'Duplicate node output name: {output_name}.')
            output_parents[output_name] = json_node['name']
            output_parent_slot_indices[output_name] = oi

    # add edges
    for json_node in json_nodes:
        for ii, node_input in enumerate(json_node['inputs']):
            if node_input[1] in output_parents:
                parent_name = output_parents[node_input[1]]
                parent_slot_index = output_parent_slot_indices[node_input[1]]
                edge_name = f'{parent_name}_{parent_slot_index}_to_{json_node["name"]}_{ii}'
                if reverse_edges:
                    nxgraph.add_edge(json_node['name'], parent_name, key=edge_name)
                else:
                    nxgraph.add_edge(parent_name, json_node['name'], key=edge_name)

    return nxgraph

# can the src dtype be cast to the tgt dtype without conversion? (i.e. is the src dtype a subtype of the tgt dtype?)
def dtype_castable(src_dtype, tgt_dtype):
    # dtypes are stored as bitwise flags, and more general types are bitwise ors of their more specialized childs
    # TODO: this is *not* true for INTEGER5, INTEGER6, FLOAT5 and FLOAT6
    # But since this function is only used for node input/output dtypes, this should be fine for now.
    # Need to change these 4 dtypes to have their own bit flags, but this probably requires re-generating the dataset and re-training the models.

    # castable if tgt_dtype is a generalization of src_dtype (src_dtype has no additional bitwise flags)
    return (~tgt_dtype & src_dtype) == 0

def center_normal_map(normals):

    # convert to normals
    normals = normals * 2.0 - 1.0

    # # normalize normals
    # normals = normals / torch.clamp((normals**2).sum(dim=-3, keepdim=True).sqrt(), min=1e-8)

    # center x and y components at 0
    normals[..., 0, :, :] -= normals[..., 0, :, :].mean() # center x component at 0
    normals[..., 1, :, :] -= normals[..., 1, :, :].mean() # center y component at 0

    # clamp xy vector magnitude <= 1
    normals[..., :2, :, :] /= normals[..., :2, :, :].norm(dim=-3, keepdim=True).clamp_min(1.0)

    # re-normalize normals by keeping x and y fixed and adjusting z
    normals[..., 2, :, :] = (1 - normals[..., :2, :, :].norm(dim=-3) ** 2).clamp_min(0.0).sqrt()

    # # re-normalize normals
    # normals = normals / torch.clamp((normals**2).sum(dim=-3, keepdim=True).sqrt(), min=1e-8)

    # convert back to normal map
    normals = ((normals + 1.0) * 0.5).clamp(min=0, max=1)

    return normals
