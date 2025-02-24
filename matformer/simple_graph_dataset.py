# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import os.path as pth
import glob
import json
from collections import OrderedDict, deque
import random
import numpy as np
import torch
# import struct
from PIL import Image
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
# from torchvision.transforms.functional import resize as F_resize
from abc import ABC, abstractmethod
import pickle
import cv2
from cv2 import resize as cv2_resize
from io import BytesIO

from .diffsbs.sbs_utils import read_image
from .diffsbs.io_json import apply_json_diff

from .simple_graph import SimpleGraph, SimpleOrderedNodes
from .sequencer import EdgeSequencer
from .sequencer.sequences import add_auxiliary_tokens, sort_output_nodes_to_fixed_order
from .utils import unique
from .sPickle import s_load as spkl_load


# a wrapper for image and text prompt inputs
class MultiModalInput:
    def __init__(self, data):
        self.data = data


class GraphValidationError(RuntimeError):
    pass


class Identity:
    def __call__(self, x):
        return x


class ConvertToRGB:
    def __call__(self, x):
        return x.convert('RGB')


class PILImageLoader:
    def __init__(self, img_res=None, to_tensor=True):
        # copied from clip.py;
        # we do not perform normalization here because we may need to extract other features form this image
        self.transform = Compose([Resize(img_res, interpolation=InterpolationMode.BICUBIC) if img_res is not None else Identity(),
                                  CenterCrop(img_res) if img_res is not None else Identity(),
                                  ConvertToRGB(),
                                  ToTensor() if to_tensor else Identity(),
                                  # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                  ])

    def __call__(self, image_file):
        # wrap encoded byte sequence in an IO buffer
        if isinstance(image_file, bytes):
            image_file = BytesIO(image_file)

        # read image
        if isinstance(image_file, (str, BytesIO)):
            pil_image = Image.open(image_file)
        elif isinstance(image_file, Image.Image):
            pil_image = image_file
        elif isinstance(image_file, np.ndarray):
            pil_image = Image.fromarray(image_file)
        else:
            raise RuntimeError(f'Unknown image file type: {type(image_file).__name__}')
        image = self.transform(pil_image)
        return image


class ImageLoader:
    def __init__(self, img_res=None, to_tensor=True):
        self.n_px = img_res
        self.to_tensor = to_tensor

    def __call__(self, image_file):
        # wrap encoded byte sequence in an IO buffer
        if isinstance(image_file, bytes):
            image_file = BytesIO(image_file)

        # read image
        if isinstance(image_file, (str, BytesIO)):
            image = read_image(image_file)
        elif isinstance(image_file, np.ndarray):
            image = image_file
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535
            elif image.dtype != np.float32:
                raise RuntimeError(f'Unexpected image data type: {type(image.dtype).__name__}')

        # resize image
        im_size = self.n_px
        if im_size is not None and image.shape[:2] != (im_size, im_size):
            image = cv2_resize(image, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
            image = np.clip(image, 0.0, 1.0)

        # convert to tensor
        if self.to_tensor:
            image = torch.from_numpy(image)
            if image.ndim == 3:
                image = image.permute(2, 0, 1)
                if image.shape[0] == 4:
                    image = image[:3, :, :]
            else:
                image = image.unsqueeze(0).expand(3, -1, -1)

        return image


def split_variation_name(variation_name):
    graph_name, variation_name = pth.split(variation_name)
    graph_name = graph_name.replace(pth.sep, '/')
    return graph_name, variation_name


# abstract data loader class
class AbstractDataLoader(ABC):
    def __init__(self, data, image_loader):
        self.data = data
        self.image_loader = image_loader

    def _make_data_dict(self, json_graph, image):
        return {
            'json_nodes': json_graph,
            'prerendered': image,
            'pil_prerendered': image if isinstance(self.image_loader, PILImageLoader) else None
        }

    # load prerendered image and json from file
    @abstractmethod
    def load_data(self, *args, **kwargs): ...

    # load a chunk of prerendered image and json files
    @abstractmethod
    def load_chunk(self, *args, **kwargs): ...

    # open and close methods maintain a file IO handle if necessary
    def open(self): ...
    def close(self): ...

    def __del__(self):
        self.close()


# data loader from dataset folder
class DataLoader(AbstractDataLoader):
    def __init__(self, data, image_loader, image_suffix='_rendered', image_ext='png'):
        super().__init__(data, image_loader)
        self.image_suffix = f'{image_suffix}.{image_ext}'

    def load_data(self, variation_name, json_suffix='_quantized'):
        # read image
        image_filename = pth.join(self.data, variation_name + self.image_suffix)
        image = self.image_loader(image_filename)

        # read json graph
        json_filename = pth.join(self.data, f'{variation_name}{json_suffix}.json')
        if not pth.exists(json_filename): # legacy
            json_filename = pth.join(self.data, variation_name, f'graph{json_suffix}.json')
        if not pth.exists(json_filename):
            json_filename = pth.join(self.data, variation_name, f'op_graph_nodes{json_suffix}.json') # legacy
        with open(json_filename) as f:
            json_graph = json.load(f)

        # apply json content to initial parameters
        graph_name = pth.dirname(variation_name)
        initial_json_filename = pth.join(self.data, graph_name, f'initial{json_suffix}.json')
        if pth.exists(initial_json_filename):
            with open(initial_json_filename) as f:
                initial_json_graph = json.load(f)
            json_graph = apply_json_diff(initial_json_graph, json_graph)

        return self._make_data_dict(json_graph, image)

    def load_chunk(self, *_, **__):
        raise NotImplementedError('Loading chunks is not supported by basic data loader')


# data loader from HDF5 dataset file
class H5DataLoader(AbstractDataLoader):
    def __init__(self, data, image_loader, chunksize=64):
        super().__init__(data, image_loader)
        self.chunksize = chunksize
        self.data_file = None

        # the dictionary for data index lookup (lazily initialized)
        self.data_index_dict = {}

    def _get_index_dict(self, graph_name, graph_data=None):
        index_dict = self.data_index_dict.get(graph_name)
        if index_dict is None:
            # get all variation names of a graph
            if graph_data is not None:
                names_data = graph_data['names'][:]
            else:
                with self as f:
                    names_data = f[graph_name]['names'][:]
            index_dict = {n.decode('utf-8'): i for i, n in enumerate(names_data)}
            self.data_index_dict[graph_name] = index_dict
        return index_dict

    def load_data(self, variation_name, json_suffix='_quantized'):
        graph_name, variation_name = split_variation_name(variation_name)

        with self as f:
            # lazily build the name-to-index mapping
            graph_data = f[graph_name]
            graph_data_index = self._get_index_dict(graph_name, graph_data)[variation_name]

            # read image data
            image_data = graph_data['images'][graph_data_index]
            image = self.image_loader(image_data)

            # read json data
            json_data = graph_data[f'jsons{json_suffix}/variations'][graph_data]
            json_graph = json.loads(json_data)

            # apply json content to initial parameters
            if f'jsons{json_suffix}/initial' in graph_data:
                initial_json_data = graph_data[f'jsons{json_suffix}/initial'][0]
                initial_json_graph = json.loads(initial_json_data)
                json_graph = apply_json_diff(initial_json_graph, json_graph)
            else:
                raise RuntimeError(f"Initial parameters not found for graph {graph_name}.")

        return self._make_data_dict(json_graph, image)

    def load_chunk(self, graph_name, chunk_id, entry_list=None, json_suffix='_quantized'):
        chunksize = self.chunksize

        with self as f:
            graph_data = f[graph_name]
            num_data = graph_data['images'].shape[0]
            chunk_start, chunk_end = chunk_id * chunksize, min((chunk_id + 1) * chunksize, num_data)

            # read image and JSON data
            image_data = graph_data['images'][chunk_start:chunk_end]
            json_data = graph_data[f'jsons{json_suffix}/variations'][chunk_start:chunk_end]

            # load initial parameters
            initial_json_graph = None
            if f'jsons{json_suffix}/initial' in graph_data:
                initial_json_data = graph_data[f'jsons{json_suffix}/initial'][0]
                initial_json_graph = json.loads(initial_json_data)

            # preprocess loaded data entries
            for entry_id in (range(len(chunksize)) if entry_list is None else entry_list):
                image = self.image_loader(image_data[entry_id])
                json_graph = json.loads(json_data[entry_id])
                if initial_json_graph is not None:
                    json_graph = apply_json_diff(initial_json_graph, json_graph)

                yield self._make_data_dict(json_graph, image)

    # open the HDF5 file
    def open(self):
        import h5py
        if self.data_file is None:
            self.data_file = h5py.File(self.data, rdcc_nbytes=64*1024**2)

    # close the opened HDF5 file
    def close(self):
        if self.data_file is not None:
            self.data_file.close()
        self.data_file = None

    # functions for context manager
    def __enter__(self):
        self.open()
        return self.data_file

    def __exit__(self, *_):
        pass


# data loader from streaming pickles
class SPickleDataLoader(AbstractDataLoader):
    def __init__(self, data, image_loader, chunksize=128):
        super().__init__(data, image_loader)
        self.chunksize = chunksize

        # read all variation names
        with open(pth.join(data, 'all_variations.json')) as f:
            all_variations = json.load(f)

        # initialize dictionary for data index lookup
        data_index_dict = {}
        for graph_name, variations in all_variations.items():
            data_index_dict[graph_name] = {n: i for i, n in enumerate(variations)}

        # load initial parameters (if any)
        initial_dict = {}
        for suffix in ('', '_quantized'):
            initial_file = pth.join(data, f'all_initial{suffix}.pkl')
            if pth.isfile(initial_file):
                with open(initial_file, 'rb') as f:
                    initial_dict[f'initial{suffix}'] = pickle.load(f)

        self.all_variations = all_variations
        self.data_index_dict = data_index_dict
        self.initial_dict = initial_dict

    def _get_index_dict(self, graph_name):
        return self.data_index_dict[graph_name]

    def _preprocess(self, spkl_data, json_suffix, initial_json_graph=None):
        # read image
        image = self.image_loader(spkl_data[0])

        # read JSON and apply to initial parameters if any
        if not json_suffix:
            json_graph = spkl_data[1]
        elif json_suffix == '_quantized':
            json_graph = spkl_data[2]
        else:
            raise ValueError(f'Unknown JSON file suffix: {json_suffix}')

        if initial_json_graph is not None:
            json_graph = apply_json_diff(initial_json_graph, json_graph)

        return self._make_data_dict(json_graph, image)

    def load_data(self, variation_name, json_suffix='_quantized'):
        graph_name, variation_name = split_variation_name(variation_name)

        # locate the data chunk
        variation_index = self.data_index_dict[graph_name][variation_name]
        chunk_id, entry_id = variation_index // self.chunksize, variation_index % self.chunksize

        # fetch data from the located chunk
        with open(pth.join(self.data, graph_name, f'part_{chunk_id:05d}.spkl'), 'rb') as f:
            spkl_gen = spkl_load(f)
            for _ in range(entry_id + 1):
                spkl_data = next(spkl_gen)

        # read initial parameters (if any)
        initial_json_graph = self.initial_dict.get(f'initial{json_suffix}', {}).get(graph_name)

        return self._preprocess(spkl_data, json_suffix, initial_json_graph=initial_json_graph)

    def load_chunk(self, graph_name, chunk_id, entry_list=None, json_suffix='_quantized'):
        # read initial parameters (if any)
        initial_json_graph = self.initial_dict.get(f'initial{json_suffix}', {}).get(graph_name)

        chunk_data_file = pth.join(self.data, graph_name, f'part_{chunk_id:05d}.spkl')

        if entry_list is None:
            with open(chunk_data_file, 'rb') as f:
                for i, spkl_data in enumerate(spkl_load(f)):
                    yield self._preprocess(spkl_data, json_suffix, initial_json_graph=initial_json_graph)

        else:
            entry_list = sorted(entry_list)
            with open(chunk_data_file, 'rb') as f:
                for i, spkl_data in enumerate(spkl_load(f)):
                    if i == entry_list[0]:
                        yield self._preprocess(spkl_data, json_suffix, initial_json_graph=initial_json_graph)

                        entry_list.pop(0)
                        if not entry_list:
                            break


# Create a data loader class based on image properties
def get_data_loader(data_dir, pil_image_loader=False, data_chunksize=None, image_res=None, image_suffix=None, image_ext=None, to_tensor=True):
    # create the image loader object
    image_loader = (PILImageLoader if pil_image_loader else ImageLoader)(img_res=image_res, to_tensor=to_tensor)

    if data_dir.endswith('.h5'):
        data_loader = H5DataLoader(data_dir, image_loader, chunksize=data_chunksize)
    elif pth.isdir(data_dir):
        if pth.exists(pth.join(data_dir, 'all_variations.json')):
            data_loader = SPickleDataLoader(data_dir, image_loader, chunksize=data_chunksize)
        else:
            data_loader = DataLoader(data_dir, image_loader, image_suffix=image_suffix, image_ext=image_ext)
    else:
        raise ValueError(f'Invalid data path: {data_dir}')
    return data_loader


def generator_type_group(gen_type):
    tokens = gen_type.split('_')
    if len(tokens) > 1 and tokens[-1].isdigit():
        return '_'.join(tokens[:-1])
    else:
        return gen_type


# back to front, breadth first, with branches sorted by the input slot indices
def get_nodes_reverse_breadth_first(graph, sorted_by_name, fixed_output_order):
    nodes = []
    node_depths = []

    output_root = [node for node in graph.nodes if node.type == 'output_root']
    if len(output_root) == 0:
        # do reverse BFS from sorted leaf nodes
        leaf_nodes = graph.get_leafs()
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
        node_queue = deque([(leaf_node, 0) for leaf_node in leaf_nodes])
        visited_nodes = set(leaf_nodes)
    elif len(output_root) == 1:
        # do reverse BFS from the single output root node
        output_root = output_root[0]
        node_queue = deque([(output_root, 0)])
        visited_nodes = {output_root}
    else:
        raise RuntimeError(f'Found multiple output_root nodes!')

    while len(node_queue) > 0:
        node, node_depth = node_queue.popleft()
        # if no empty input node, it's possible to have a None input
        if node is None:
            continue
        nodes.append(node)
        node_depths.append(node_depth)
        univisited_parent_nodes = [parent_node for parent_node in unique([n for n, _ in node.parents])
                                    if parent_node not in visited_nodes]
        visited_nodes.update(univisited_parent_nodes)
        node_queue.extend([(parent_node, node_depth+1) for parent_node in univisited_parent_nodes])

    return nodes, node_depths


# front to back, topologically sorted and otherwise random
def get_nodes_topological_random(graph, rng):
    nodes = []
    node_depths = []

    if len(graph.nodes) > 0:
        # generators = [node for node in graph.nodes if len(node.parents) == 0]
        generators = [node for node in graph.nodes if all(parent is None for parent, _ in node.parents)]
        if len(generators) == 0:
            raise RuntimeError('All nodes in the graph have parents, the graph must have a cycle.')

        candidates = generators
        candidate_depths = [0]*len(generators)
        visited_nodes = set(candidates)

        while len(candidates) > 0:
            # randomly choose a node from the list of candidates
            candidate_idx = rng.choice(len(candidates))
            node = candidates.pop(candidate_idx)
            node_depth = candidate_depths.pop(candidate_idx)

            # add node to the ordered list of nodes
            node.idx = len(nodes)
            nodes.append(node)
            node_depths.append(node_depth)

            # new candidates: all children of the node where all parents have already been added to the ordered node list
            # (this ensures topological sorting of the list)
            for child in node.children:
                if child not in visited_nodes and all(parent.idx is not None for parent, _ in child.parents if parent is not None):
                    candidates.append(child)
                    visited_nodes.add(child)
                    candidate_depths.append(max([node_depths[parent.idx] for parent, _ in child.parents if parent is not None])+1) # max distance to a generator node

    # reset all indices to None
    for node in graph.nodes:
        node.idx = None

    return nodes, node_depths


# front to back, breadth first, with branches sorted by the output slot indices, and random for the connections inside an output slot
def get_nodes_breadth_first_random(graph, rng):
    nodes = []
    node_depths = []

    if len(graph.nodes) > 0:

        generators = [node for node in graph.nodes if all(parent is None for parent, _ in node.parents)]
        if len(generators) == 0:
            raise RuntimeError('All nodes in the graph have parents, the graph must have a cycle.')

        # randomly permute the generators since they have no specific order
        generators = rng.permutation(generators).tolist()

        node_queue = deque(zip(generators, [0]*len(generators)))
        visited_nodes = set(generators)
        while len(node_queue) > 0:
            node, node_depth = node_queue.popleft()
            nodes.append(node)
            node_depths.append(node_depth)

            unvisited_child_node_and_slot_idx = tuple(zip(*[(child_node, child_slot_idx) for child_node, child_slot_idx in node.get_child_slots() if child_node not in visited_nodes]))
            if len(unvisited_child_node_and_slot_idx) > 0:
                # get unvisited child nodes
                unvisited_child_nodes, unvisited_child_slot_idx = unvisited_child_node_and_slot_idx

                # randomly permute child nodes
                perm_inds = rng.permutation(len(unvisited_child_nodes))
                unvisited_child_nodes = np.array(unvisited_child_nodes)[perm_inds].tolist()
                unvisited_child_slot_idx = np.array(unvisited_child_slot_idx)[perm_inds].tolist()

                # sort by output slot index (the sort should be stable, so child nodes in the same output slot should remain randomly permuted)
                unvisited_child_nodes, unvisited_child_slot_idx = zip(*sorted(zip(unvisited_child_nodes, unvisited_child_slot_idx), key=lambda x: x[1]))

                for child_node in unvisited_child_nodes:
                    if child_node not in visited_nodes:
                        visited_nodes.add(child_node)
                        node_queue.append((child_node, node_depth+1))

                # # add unvisited child nodes to the queue
                # visited_nodes.update(unvisited_child_nodes)
                # node_queue.extend([(child_node, node_depth+1) for child_node in unvisited_child_nodes])

    return nodes, node_depths


# same as get_nodes_reverse_breadth_first, but flipped
def get_nodes_reverse_breadth_first_flipped(graph, sorted_by_name, fixed_output_order):
    nodes, node_depths = get_nodes_reverse_breadth_first(graph=graph, sorted_by_name=sorted_by_name, fixed_output_order=fixed_output_order)

    return list(reversed(nodes)), list(reversed(node_depths))


# order nodes in a graph
def get_ordered_nodes(graph, node_order, rng, sorted_by_name, fixed_output_order):
    if node_order.startswith('reverse_breadth_first') and not node_order.startswith('reverse_breadth_first_flipped'):
        nodes, node_depths = get_nodes_reverse_breadth_first(graph=graph, sorted_by_name=sorted_by_name, fixed_output_order=fixed_output_order)
    elif node_order == 'topological_random':
        nodes, node_depths = get_nodes_topological_random(graph=graph, rng=rng)
    elif node_order == 'breadth_first_random':
        nodes, node_depths = get_nodes_breadth_first_random(graph=graph, rng=rng)
    elif node_order.startswith('reverse_breadth_first_flipped'):
        nodes, node_depths = get_nodes_reverse_breadth_first_flipped(graph=graph, sorted_by_name=sorted_by_name, fixed_output_order=fixed_output_order)
    else:
        raise RuntimeError(f'Unknown node order: {node_order}')
    return SimpleOrderedNodes(nodes, node_depths)


class SimpleGraphLoader:

    # list of features generated by the graph loader
    SUPPORTED_FEATURES = [
        'graph', 'name', 'category_idx', 'generator_count', 'active_generator_groups', 'rendered_output', 'rendered_output_v2', 'rendered_output_v2_filename',
        'rendered_output_v3', 'rendered_output_v3_filename', 'output_channels', 'node_outputs', 'nodes', 'node_depths', 'node_type_seq', 'node_depth_seq',
        'node_idx_seq', 'node_seq_mask', 'slot_node_type_seq', 'slot_node_depth_seq', 'slot_node_idx_seq', 'slot_idx_seq', 'slot_id_seq', 'slot_seq_mask',
        'edge_seq', 'edge_idx_seq', 'edge_elm_seq', 'edge_seq_mask', 'param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq',
        'param_array_elm_idx_seq', 'param_idx_seq', 'param_seq_mask', 'param_node_inds', 'edge_node_inds', 'prerendered', 'param_node_type_seq', 'param_node_depth_seq',
        'param_node_seq_mask', 'pil_prerendered'
    ]

    def __init__(self, data_dir, feature_list, use_alpha=True, seed=None, real_input=False, pil_image_loader=False,
                 img_res=224, image_ext='png', augment_images=None, avoid_black_variations=False, avoid_white_variations=False, validate=False,
                 generator_count_list=None, max_num_param_sets=None, legacy_node_names=False, categories=None,
                 sample_cat_list=None, node_order=None, param_node_order=None, legacy_json_loading=False,
                 node_sequencer=None, slot_sequencer=None, edge_sequencer=None, param_sequencer=None,
                 param_node_sequencer=None, param_per_node=True, revert_to_v0=False, sorted_by_name=True, fixed_output_order=False,
                 sample_text_prompts=False, text_prompts_sample_rate=0):

        self.data_dir = data_dir
        self.feature_list = feature_list
        self.use_alpha = use_alpha
        self.real_input = real_input
        self.img_res = img_res
        self.augment_images = augment_images
        self.avoid_black_variations = avoid_black_variations
        self.avoid_white_variations = avoid_white_variations
        self.validate = validate
        self.max_num_param_sets = max_num_param_sets
        self.legacy_node_names = legacy_node_names
        self.node_order = node_order if any(sq is not None for sq in (node_sequencer, slot_sequencer, edge_sequencer)) else param_node_order
        self.param_node_order = param_node_order
        self.legacy_json_loading = legacy_json_loading
        self.image_ext = image_ext
        self.revert_to_v0 = revert_to_v0
        self.sorted_by_name = sorted_by_name
        self.fixed_output_order = fixed_output_order
        self.sample_text_prompts = sample_text_prompts
        self.text_prompts_sample_rate = text_prompts_sample_rate

        self.node_sequencer = node_sequencer
        self.slot_sequencer = slot_sequencer
        self.edge_sequencer = edge_sequencer
        self.param_sequencer = param_sequencer
        self.param_node_sequencer = param_node_sequencer
        self.param_per_node = param_per_node

        sequencer_list = [self.node_sequencer, self.slot_sequencer, self.param_node_sequencer]
        if all(sq is None for sq in sequencer_list):
            raise RuntimeError('Must pass either a node sequencer or a slot sequencer (or both).')

        self.node_types = next(sq.node_types for sq in sequencer_list if sq is not None)
        self.node_type_names = [node_type['name'] for node_type in self.node_types]
        self.max_num_nodes = next(sq.max_num_nodes for sq in sequencer_list if sq is not None)
        self._legacy_flattened = getattr(self.node_types, '_legacy_flattened', True)

        self.rng = np.random.default_rng(seed=seed)

        # image loader
        self.image_loader = (PILImageLoader if pil_image_loader else ImageLoader)(img_res)
        self.pil_image_loader = PILImageLoader(img_res)

        # read sample categories if available
        self.categories = categories
        # if cat_list is not None:
        #     with open(cat_list, 'r') as f:
        #         self.categories = f.read().splitlines()
        #     self.categories = [cat for cat in self.categories if len(cat) > 0]

        self.sample_categories = None
        self.sample_category_inds = None
        if sample_cat_list is not None:
            if self.categories is None:
                raise RuntimeError('Need to pass a list of categories.')
            self.sample_categories = OrderedDict()
            self.sample_category_inds = OrderedDict()
            with open(sample_cat_list, 'r') as f:
                sample_cats = json.load(f)
            for sample_name in self.sample_names:
                cat = sample_cats[sample_name]
                if cat not in self.categories:
                    raise RuntimeError(f'Sample {sample_name} has category {cat}, which was not found in the list of categories.')
                self.sample_categories[sample_name] = cat
                self.sample_category_inds[sample_name] = self.categories.index(cat)

        # generator node types
        self.generator_node_types = [nt for nt in self.node_types if len(nt['input_names']) == 0 and nt['name'] not in ['parent_end', 'output_root', 'empty_input']]
        self.generator_node_type_groups = sorted(list(set(generator_type_group(gen_type=generator_type['name']) for generator_type in self.generator_node_types)))

        self.generator_node_type_freq = None
        self.generator_node_type_group_freq = None
        if generator_count_list is not None:
            with open(generator_count_list, 'r') as f:
                generator_count_json = json.load(f, object_pairs_hook=OrderedDict)
            if list(generator_count_json['sample_count_gen_type'].keys()) != [t['name'] for t in self.generator_node_types]:
                raise RuntimeError('Generator node types used in the generator count list do not match the generator node types of the dataset.')
            if list(generator_count_json['sample_count_gen_type_group'].keys()) != self.generator_node_type_groups:
                raise RuntimeError('Generator node type groups used in the generator count list do not match the generator node type groups of the dataset.')
            self.generator_node_type_freq = torch.tensor(list(generator_count_json['sample_count_gen_type'].values()), dtype=torch.float32) / generator_count_json['sample_count']
            self.generator_node_type_group_freq = torch.tensor(list(generator_count_json['sample_count_gen_type_group'].values()), dtype=torch.float32) / generator_count_json['sample_count']

        # validate the parameter statistics of all node types
        if self.param_sequencer is not None:
            for node_type in self.node_types:
                if len(node_type['parameters']) > self.param_sequencer.max_num_params+1:
                    raise GraphValidationError(f'Number of parameters exceeds maximum for node type {node_type["name"]}.')
                for param_type in node_type['parameters']:
                    param_name = param_type['name']
                    param_dtype = SimpleGraph.get_param_dtype(node_type=node_type['name'], param_type_info=param_type, use_alpha=use_alpha, legacy_flattened=self._legacy_flattened)
                    param_vec_dim = SimpleGraph.get_param_vector_dim(param_dtype=param_dtype)
                    if param_vec_dim > self.param_sequencer.max_vec_dim:
                        raise GraphValidationError(f'Node parameter vector dimension exceeds maximum for parameter {param_name} in node type {node_type["name"]}.')

    @staticmethod
    def sort_node_params(graph, node_types):
        node_type_names = [node_type['name'] for node_type in node_types]
        for node in graph.nodes:
            if len(node.param_names) > 0:
                param_types = node_types[node_type_names.index(node.type)]['parameters']
                param_type_names = [param_type['name'] for param_type in param_types]
                param_type_indices = [param_type_names.index(param_name) for param_name in node.param_names]
                # param_type_indices[node.param_names.index('param_end')] = 999999999 # param_end should always be last # not needed since parameter sequences have their own stop token

                _, node.param_names, node.param_vals = tuple(zip(*sorted(zip(param_type_indices, node.param_names, node.param_vals), key=lambda x: x[0])))
                node.param_names = list(node.param_names)
                node.param_vals = list(node.param_vals)

    @staticmethod
    def get_augmentation_params(img, augment_mode):

        if augment_mode == 'resize_flip_rot':
            return {
                'resized_crop': torchvision.transforms.RandomResizedCrop.get_params(img=img, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
                'hflip': torch.rand(1) < 0.5,
                'vflip': torch.rand(1) < 0.5,
                'rot_90_ccw': torch.rand(1) < 0.5,
                'rot_90_cw': torch.rand(1) < 0.5}
        elif augment_mode is None:
            return None
        else:
            raise RuntimeError(f'Unknown image augmentation type: {augment_mode}')

    @staticmethod
    def augmentation_transform_image(img, augment_mode, params):

        if augment_mode == 'resize_flip_rot':
            img = torchvision.transforms.functional.resized_crop(img, *params['resized_crop'], size=(img.shape[-2], img.shape[-1]))
            if params['hflip']:
                img = torchvision.transforms.functional.hflip(img)
            if params['vflip']:
                img = torchvision.transforms.functional.vflip(img)
            if params['rot_90_ccw']:
                img = torchvision.transforms.functional.rotate(img, angle=90.0)
            if params['rot_90_cw']:
                img = torchvision.transforms.functional.rotate(img, angle=-90.0)
        elif augment_mode is None:
            pass
        else:
            raise RuntimeError(f'Unknown image augmentation type: {augment_mode}')

        return img

    @staticmethod
    def flatten_param_node_inds(param_node_inds):
        node_subset_inds_flattened_x, node_subset_inds_flattened_y = [], []
        for b in range(len(param_node_inds)):
            node_subset_inds_flattened_x.extend([b]*len(param_node_inds[b]))
            node_subset_inds_flattened_y.extend(param_node_inds[b])
        return np.stack((node_subset_inds_flattened_x, node_subset_inds_flattened_y), axis=1)

    def __call__(self, graph_name, **prefetched_data):

        graph_category_idx = None
        if 'category_idx' in self.feature_list and self.sample_category_inds is not None:
            graph_category_idx = self.sample_category_inds[graph_name]

        # get graph
        graph = None
        if any(feat in self.feature_list for feat in [
            'nodes', 'node_depths', 'graph', 'generator_count', 'generator_group_count',
            'node_type_seq', 'node_depth_seq', 'node_idx_seq', 'node_seq_mask',
            'edge_seq', 'edge_idx_seq', 'edge_elm_seq', 'edge_seq_mask',
            'slot_node_type_seq', 'slot_node_idx_seq', 'slot_node_depth_seq', 'slot_id_seq', 'slot_idx_seq', 'slot_seq_mask',
            'edge_node_inds']):

            json_nodes = prefetched_data.get('json_nodes')
            if json_nodes is None:
                raise RuntimeError('JSON graphs must be prefetched using custom data loaders')

            graph = SimpleGraph.load_json(json_nodes=json_nodes, node_types=self.node_types, node_type_names=self.node_type_names, legacy_node_names=self.legacy_node_names, legacy_json_loading=self.legacy_json_loading)

            add_auxiliary_tokens(graph=graph, node_order=self.node_order, node_types=self.node_types, node_type_names=self.node_type_names,
                                 revert_to_v0=self.revert_to_v0, sorted_by_name=self.sorted_by_name, fixed_output_order=self.fixed_output_order)

            # sort the parameter list of each node by the order in which the parameters appear in the node parameter type definitions
            self.sort_node_params(graph, node_types=self.node_types)

            # check graph size
            if self.slot_sequencer is not None:
                for node in graph.nodes:
                    if len(node.parents) > self.slot_sequencer.max_num_parents:
                        raise GraphValidationError(f'Too many parents for node {node.name} in graph {graph_name} ({len(node.parents)} > {self.slot_sequencer.max_num_parents})')
            if self.max_num_nodes is not None and len(graph.nodes) > self.max_num_nodes:
                raise GraphValidationError(f'Too many nodes in graph {graph_name} ({len(graph.nodes)} > {self.max_num_nodes})')

            if self.validate:
                all_nodes = set()

                # validate nodes
                for node in graph.nodes:
                    if node.type not in self.node_type_names:
                        raise RuntimeError(f'Node type {node.type} is not in the type list (is the node a generator?) for node {node.name} in graph {graph_name}')
                        # I don't use generators for now, to keep the number of node types low
                    all_nodes.update([n for n, _ in node.parents if n is not None])
                    all_nodes.update(node.children)
                    for parent, _ in node.parents:
                        if self.node_order in ['reverse_breadth_first', 'reverse_breadth_first_flipped'] and parent is None:
                            raise RuntimeError(f'Empty parent has not been replaced by empty_input node in graph {graph_name}')
                        if parent is not None and node not in parent.children:
                            raise RuntimeError(f'Node parents and children are not consistent in graph {graph_name}')
                    for child in node.children:
                        if node not in [n for n, _ in child.parents]:
                            raise RuntimeError(f'Node parents and children are not consistent in graph {graph_name}')
                    if self.slot_sequencer is not None:
                        slot_inds = [slot_idx for n, slot_idx in node.parents if n is not None]
                        if len(slot_inds) > 0 and max(slot_inds) >= self.slot_sequencer.max_num_output_slots:
                            raise GraphValidationError(f'Number of output slots exceeds maximum for node {node.name} in graph {graph_name}')

                    # validate node parameters
                    if self.param_sequencer is not None:
                        node_type_info = self.node_types[self.node_type_names.index(node.type)]
                        if len(node.param_names) != len(node.param_vals):
                            raise RuntimeError(f'Number of parameter names does not match number of parameter values for node {node.name} in graph {graph_name}')
                        # if len(node.param_names) > self.param_sequencer.max_num_params+1: # +1 due to param_end tokens  # not needed since parameter sequences have their own stop token
                        if len(node.param_names) > self.param_sequencer.max_num_params:
                            raise GraphValidationError(f'Number of parameters exceeds maximum for node {node.name} in graph {graph_name}')
                        # param_seq_len = 1 # 1 for param_end # not needed since parameter sequences have their own stop token
                        param_seq_len = 0
                        param_type_names = [param_type['name'] for param_type in node_type_info['parameters']]
                        for param_name, param_val in zip(node.param_names, node.param_vals):
                            if param_name not in param_type_names:
                                raise RuntimeError(f'Parameter {param_name} of node {node.name} in graph {graph_name} not found in the parmeters for node type {node.type}.')
                            param_type_info = node_type_info['parameters'][param_type_names.index(param_name)]
                            param_dtype = SimpleGraph.get_param_dtype(node_type=node.type, param_type_info=param_type_info, use_alpha=self.use_alpha, legacy_flattened=self._legacy_flattened)
                            param_tensor_rank = SimpleGraph.get_param_tensor_rank(param_dtype)
                            param_vector_dim = SimpleGraph.get_param_vector_dim(param_dtype)
                            if param_tensor_rank == 'scalar':
                                if not isinstance(param_val, int):
                                    raise RuntimeError(f'Unexpected format for scalar parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                if not param_val < self.param_sequencer.quant_steps:
                                    raise RuntimeError(f'Parameter value out of bound for parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                param_seq_len += 1
                            elif param_tensor_rank == 'vector':
                                if not isinstance(param_val, list) or len(param_val) != param_vector_dim or not all(isinstance(v, int) for v in param_val):
                                    raise RuntimeError(f'Unexpected format for vector parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                if not all(v < self.param_sequencer.quant_steps for v in param_val):
                                    raise RuntimeError(f'Parameter value out of bound for parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                param_seq_len += param_vector_dim
                            elif param_tensor_rank == 'array':
                                if len(param_val) == 0:
                                    raise GraphValidationError(f'Unsupported empty array found in parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}.')
                                # if len(param_val) > 20:
                                #     print(f'**** ARRAY LEN : {len(param_val)}')
                                for param_val_entry in param_val:
                                    if not isinstance(param_val_entry, list) or len(param_val_entry) != param_vector_dim or not all(isinstance(v, int) for v in param_val_entry):
                                        raise RuntimeError(f'Unexpected format for array parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                    if not all(v < self.param_sequencer.quant_steps for v in param_val_entry):
                                        raise RuntimeError(f'Parameter value out of bound for parameter {param_name} of node {node.name} in graph {graph_name}: {param_val}')
                                    param_seq_len += param_vector_dim
                        if param_seq_len > self.param_sequencer.max_seq_len:
                            raise GraphValidationError(f'Parameter sequence length exceeds maximum for node {node.name} in graph {graph_name} ({param_seq_len} > {self.param_sequencer.max_seq_len})')

                # validate graph
                if len(all_nodes - set(graph.nodes)) > 0:
                    raise RuntimeError(f'Graph is inconsistent, some connected nodes are not in graph {graph_name}')
                if len(set(n.name for n in graph.nodes)) != len(graph.nodes):
                    raise RuntimeError(f'Graph is inconsistent, duplicate node names in graph {graph_name}')

        augmentation_params = None

        generator_count = None
        if 'generator_count' in self.feature_list:
            generator_count = torch.zeros(len(self.generator_node_types), dtype=torch.int64)
            generator_type_names = [nt['name'] for nt in self.generator_node_types]
            for node in graph.nodes:
                if node.type in generator_type_names:
                    generator_count[generator_type_names.index(node.type)] += 1

        active_generator_groups = None
        if 'active_generator_groups' in self.feature_list:
            active_generator_groups = torch.zeros(len(self.generator_node_type_groups), dtype=torch.bool)
            with open(os.path.join(self.data_dir, f'{graph_name}_activegens.json'), 'r') as f:
                gen_groups = json.load(f)
                for gen_group_name, gen_group_active in gen_groups.items():
                    if self.validate and gen_group_name not in self.generator_node_type_groups:
                        raise RuntimeError(f'Could not find generator group {gen_group_name} in graph {graph_name} in the list of all generator groups.')
                    if gen_group_active:
                        active_generator_groups[self.generator_node_type_groups.index(gen_group_name)] = True

        rendered_output = None
        if 'rendered_output' in self.feature_list:
            rendered_output = torch.from_numpy(read_image(os.path.join(self.data_dir, graph_name, f'rendered_output.{self.image_ext}'))).permute(2, 0, 1)
            if rendered_output.shape[1] != self.img_res or rendered_output.shape[2] != self.img_res:
                rendered_output = torchvision.transforms.functional.resize(rendered_output, size=(self.img_res, self.img_res))
            # apply augmentation
            if self.augment_images is not None:
                if augmentation_params is None:
                    augmentation_params = self.get_augmentation_params(img=rendered_output, augment_mode=self.augment_images)
                rendered_output = self.augmentation_transform_image(img=rendered_output, augment_mode=self.augment_images, params=augmentation_params)

        rendered_output_v2 = None
        rendered_output_v2_filename = None
        if 'rendered_output_v2' in self.feature_list:
            rendered_output_v2_filename = os.path.join(self.data_dir, graph_name[:-len('_graph')]+f'.{self.image_ext}')
            rendered_output_v2 = torch.from_numpy(read_image(rendered_output_v2_filename)).permute(2, 0, 1)
            # resize if necessary
            if rendered_output_v2.shape[1] != self.img_res or rendered_output_v2.shape[2] != self.img_res:
                rendered_output_v2 = torchvision.transforms.functional.resize(rendered_output_v2, size=(self.img_res, self.img_res))
            # make sure the rendered output is not close to all-zeros (these should have been filtered out)
            if self.validate and self.avoid_black_variations:
                fraction_black_pixels = (rendered_output_v2 < 0.07).all(dim=0).to(dtype=torch.float32).mean()
                if fraction_black_pixels > 0.9:
                    raise GraphValidationError(f'Rendered output, is nearly all-zero in graph {graph_name}')
            # if self.image_augmentation is not None:
            #     rendered_output_v2 = self.image_augmentation(rendered_output_v2)
            # apply augmentation
            if self.augment_images is not None:
                if augmentation_params is None:
                    augmentation_params = self.get_augmentation_params(img=rendered_output_v2, augment_mode=self.augment_images)
                rendered_output_v2 = self.augmentation_transform_image(img=rendered_output_v2, augment_mode=self.augment_images, params=augmentation_params)

        rendered_output_v3 = None
        rendered_output_v3_filename = None
        if 'rendered_output_v3' in self.feature_list:
            rendered_output_v3_filename = os.path.join(self.data_dir, f'{graph_name}.{self.image_ext}')
            rendered_output_v3 = torch.from_numpy(read_image(rendered_output_v3_filename)).permute(2, 0, 1)
            # resize if necessary
            if rendered_output_v3.shape[1] != self.img_res or rendered_output_v3.shape[2] != self.img_res:
                rendered_output_v3 = torchvision.transforms.functional.resize(rendered_output_v3, size=(self.img_res, self.img_res))
            # make sure the rendered output is not close to all-zeros (these should have been filtered out)
            if self.validate and self.avoid_black_variations:
                fraction_black_pixels = (rendered_output_v3 < 0.07).all(dim=0).to(dtype=torch.float32).mean()
                if fraction_black_pixels > 0.9:
                    raise GraphValidationError(f'Rendered output, is nearly all-zero in graph {graph_name}')
            # if self.image_augmentation is not None:
            #     rendered_output_v3 = self.image_augmentation(rendered_output_v3)
            # apply augmentation
            if self.augment_images is not None:
                if augmentation_params is None:
                    augmentation_params = self.get_augmentation_params(img=rendered_output_v3, augment_mode=self.augment_images)
                rendered_output_v3 = self.augmentation_transform_image(img=rendered_output_v3, augment_mode=self.augment_images, params=augmentation_params)

        output_channels = None
        if 'output_channels' in self.feature_list:
            output_channel_names = ['baseColor', 'normal', 'roughness', 'metallic']
            output_channels = []
            for output_channel_name in output_channel_names:
                output_channel_filename = os.path.join(self.data_dir, f'{graph_name}_{output_channel_name}.png')
                output_channel = torch.from_numpy(read_image(output_channel_filename))
                if output_channel.ndim == 2: # grayscale
                    output_channel = output_channel.unsqueeze(dim=0)
                else: # color
                    output_channel = output_channel.permute(2, 0, 1)
                # convert channels to their expect format (grayscale/color)
                if output_channel_name in ['baseColor', 'normal'] and output_channel.shape[0] == 1:
                    output_channel = output_channel.repeat(3, 1, 1) # convert to color
                elif output_channel_name in ['roughness', 'metallic'] and output_channel.shape[0] == 3:
                    output_channel = output_channel.mean(dim=0, keepdim=True) # convert to grayscale
                # resize if necessary
                if output_channel.shape[1] != self.img_res or output_channel.shape[2] != self.img_res:
                    output_channel = torchvision.transforms.functional.resize(output_channel, size=(self.img_res, self.img_res))
                output_channels.append(output_channel)
            output_channels = torch.cat(output_channels, dim=0)
            # pad output channels with zeros to make sure the number of channels is divisible by 3
            if output_channels.shape[0] % 3 != 0:
                output_channels = torch.cat([
                    output_channels,
                    torch.zeros(3-(output_channels.shape[0]%3), output_channels.shape[1], output_channels.shape[2], dtype=output_channels.dtype, device=output_channels.device)])
            # apply augmentation
            if self.augment_images is not None:
                if augmentation_params is None:
                    augmentation_params = self.get_augmentation_params(img=output_channels, augment_mode=self.augment_images)
                # would need to rotate/flip the normals at eachn pixel as well when rotating/flipping, but since I can't do that for the intermediate images either (I don't know which ones are normals),
                # its better to leave the normals as-is for consistency with the intermediate images
                output_channels = self.augmentation_transform_image(img=output_channels, augment_mode=self.augment_images, params=augmentation_params)

        node_outputs = None
        if 'node_outputs' in self.feature_list:
            node_outputs = {}
            for node in graph.nodes:
                if node.type in ['output_root', 'parent_end', 'empty_input'] or node.type.startswith('output_'):
                    continue # skip auxiliary nodes and output nodes (which originally do not have any outputs, but got connected to the output root node)
                node_type_info = self.node_types[self.node_type_names.index(node.type)]
                for _, output_slot_idx in node.get_child_slots():
                    output_slot_name = list(node_type_info['output_names'].keys())[output_slot_idx]
                    node_output_filename = os.path.join(self.data_dir, f'{graph_name}_node_outputs', f'{node.name}_{output_slot_name}.png')
                    node_output = torch.from_numpy(read_image(node_output_filename))
                    if node_output.ndim == 2:
                        # grayscale
                        node_output = node_output.unsqueeze(dim=0).repeat(3, 1, 1)
                    else:
                        # color
                        node_output = node_output.permute(2, 0, 1)
                        # node_output = node_output.mean(dim=0, keepdims=True) # convert to grayscale
                    # resize if necessary
                    if node_output.shape[1] != self.img_res or node_output.shape[2] != self.img_res:
                        node_output = torchvision.transforms.functional.resize(node_output, size=(self.img_res, self.img_res))
                    # apply augmentation
                    if self.augment_images is not None:
                        # print('***** WARNING! data augmentation may make node inputs/outputs inconsistent with the operation of the node')
                        if augmentation_params is None:
                            augmentation_params = self.get_augmentation_params(img=node_output, augment_mode=self.augment_images)
                        # the augmentation may make node inputs/outputs inconsistent with the operation of the node (for example, a normal node on a rotated height field would results in something else than the rotated output)
                        node_output = self.augmentation_transform_image(img=node_output, augment_mode=self.augment_images, params=augmentation_params)
                    node_outputs[f'{node.name}_{output_slot_idx}'] = node_output

        # get an ordered list of nodes
        nodes = None
        node_depths = None
        ordered_nodes = None
        if any(feat in self.feature_list for feat in [
            'nodes', 'node_depths',
            'node_type_seq', 'node_idx_seq', 'node_depth_seq', 'node_seq_mask',
            'edge_seq', 'edge_idx_seq', 'edge_elm_seq', 'edge_seq_mask',
            'slot_node_type_seq', 'slot_node_idx_seq', 'slot_node_depth_seq', 'slot_id_seq', 'slot_idx_seq', 'slot_seq_mask',
            'param_node_type_seq', 'param_node_depth_seq', 'param_node_seq_mask', 'edge_node_inds']):

            ordered_nodes = get_ordered_nodes(graph=graph, node_order=self.node_order,
                                              rng=self.rng, sorted_by_name=self.sorted_by_name,
                                              fixed_output_order=self.fixed_output_order)
            nodes, node_depths = ordered_nodes.nodes, ordered_nodes.depths

        node_type_seq = None
        node_idx_seq = None
        node_depth_seq = None
        node_seq_mask = None
        if any(feat in self.feature_list for feat in [
            'node_type_seq', 'node_idx_seq', 'node_depth_seq', 'node_seq_mask']):
            node_type_seq, node_idx_seq, node_depth_seq, node_seq_mask = self.node_sequencer.get_sequences(ordered_nodes)

        slot_node_type_seq = None
        slot_node_idx_seq = None
        slot_node_depth_seq = None
        slot_idx_seq = None
        slot_id_seq = None
        slot_seq_mask = None
        edge_seq = None
        edge_idx_seq = None
        edge_elm_seq = None # which element of the edge pair (edge start or end)
        edge_seq_mask = None
        if any(feat in self.feature_list for feat in [
            'edge_seq', 'edge_idx_seq', 'edge_elm_seq', 'edge_seq_mask',
            'slot_node_type_seq', 'slot_node_idx_seq', 'slot_node_depth_seq', 'slot_id_seq', 'slot_idx_seq', 'slot_seq_mask']):

            slot_node_type_seq, slot_node_idx_seq, slot_node_depth_seq, slot_id_seq, slot_idx_seq, slot_seq_mask, edge_seq, edge_idx_seq, edge_elm_seq, edge_seq_mask \
                = self.edge_sequencer.get_sequences(ordered_nodes, slot_sequencer=self.slot_sequencer)

        # update node order for the parameter generator
        if any(feat in self.feature_list for feat in [
            'param_node_type_seq', 'param_node_depth_seq', 'param_node_seq_mask', 'edge_node_inds',
            'param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq', 'param_array_elm_idx_seq', 'param_idx_seq', 'param_seq_mask', 'param_node_inds']):
            if self.param_node_order is not None and self.param_node_order != self.node_order:
                ordered_nodes = get_ordered_nodes(graph=graph, node_order=self.param_node_order,
                                                  rng=self.rng, sorted_by_name=self.sorted_by_name,
                                                  fixed_output_order=self.fixed_output_order)

        param_node_type_seq = None
        param_node_depth_seq = None
        param_node_seq_mask = None
        if any(feat in self.feature_list for feat in ['param_node_type_seq', 'param_node_depth_seq', 'param_node_seq_mask']):
            param_node_type_seq, _, param_node_depth_seq, param_node_seq_mask = self.param_node_sequencer.get_sequences(ordered_nodes)

        edge_node_inds = None
        if 'edge_node_inds' in self.feature_list:
            edge_node_inds = EdgeSequencer.get_edge_node_inds(ordered_nodes) # pass ordered nodes to get indices into the node sequence

        param_id_seq = None
        param_token_idx_seq = None
        param_val_seq = None
        param_vector_elm_idx_seq = None
        param_array_elm_idx_seq = None
        param_idx_seq = None
        param_seq_mask = None
        param_node_inds = None
        if any(feat in self.feature_list for feat in ['param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq', 'param_array_elm_idx_seq', 'param_idx_seq', 'param_seq_mask', 'param_node_inds']):

            if self.param_per_node:
                valid_node_inds = [node_idx for node_idx, node in enumerate(ordered_nodes) if node.type not in ['parent_end', 'empty_input', 'output_root']] # do not use auxiliary nodes that are known to have 0 parameters
                if self.max_num_param_sets is not None and len(valid_node_inds) > self.max_num_param_sets:
                    param_node_inds = np.random.choice(valid_node_inds, size=self.max_num_param_sets, replace=False).tolist()
                else:
                    param_node_inds = valid_node_inds

                selected_nodes = [ordered_nodes[idx] for idx in param_node_inds]

                param_id_seq = []
                param_token_idx_seq = []
                param_val_seq = []
                param_vector_elm_idx_seq = []
                param_array_elm_idx_seq = []
                param_idx_seq = []
                param_seq_mask = []
                for node in selected_nodes:
                    node_param_id_seq, node_param_token_idx_seq, node_param_val_seq, node_param_vector_elm_idx_seq, node_param_array_elm_idx_seq, node_param_idx_seq, node_param_seq_mask = self.param_sequencer.get_sequences_for_one_node(node=node)
                    param_id_seq.append(node_param_id_seq)
                    param_token_idx_seq.append(node_param_token_idx_seq)
                    param_val_seq.append(node_param_val_seq)
                    param_vector_elm_idx_seq.append(node_param_vector_elm_idx_seq)
                    param_array_elm_idx_seq.append(node_param_array_elm_idx_seq)
                    param_idx_seq.append(node_param_idx_seq)
                    param_seq_mask.append(node_param_seq_mask)
                # param_id_seq = torch.stack(param_id_seq, dim=0)
                # param_token_idx_seq = torch.stack(param_token_idx_seq, dim=0)
                # param_val_seq = torch.stack(param_val_seq, dim=0)
                # param_vector_elm_idx_seq = torch.stack(param_vector_elm_idx_seq, dim=0)
                # param_array_elm_idx_seq = torch.stack(param_array_elm_idx_seq, dim=0)
                # param_idx_seq = torch.stack(param_idx_seq, dim=0)
                # param_seq_mask = torch.stack(param_seq_mask, dim=0)
            else:
                param_id_seq, param_token_idx_seq, param_val_seq, param_vector_elm_idx_seq, param_array_elm_idx_seq, param_idx_seq, param_seq_mask, param_node_inds = self.param_sequencer.get_sequences(ordered_nodes)

        prerendered = None
        if 'prerendered' in self.feature_list:
            if not self.sample_text_prompts or random.random() >= self.text_prompts_sample_rate:
                prerendered = prefetched_data.get('prerendered')
                if prerendered is None:
                    if not self.real_input:
                        raise RuntimeError('Synthetic input images must be prefetched using custom data loader')
                    else: # real data
                        prerendered_filename = os.path.join(self.data_dir, graph_name)
                        prerendered = self.image_loader(prerendered_filename)

                # avoid black variations
                if self.validate and self.avoid_black_variations:
                    fraction_black_pixels = (prerendered < 0.05).all(dim=0).to(dtype=torch.float32).mean()
                    if fraction_black_pixels > 0.9:
                        raise GraphValidationError(f'Rendered output, is nearly all-zero in graph {graph_name}')
                # avoid white variations
                if self.validate and self.avoid_white_variations:
                    fraction_white_pixels = (prerendered > 0.95).all(dim=0).to(dtype=torch.float32).mean()
                    if fraction_white_pixels > 0.9:
                        raise GraphValidationError(f'Rendered output, is nearly all-one in graph {graph_name}')
            else:
                tokens = pth.basename(pth.dirname(graph_name.strip())).split('_')
                if tokens[-1].isnumeric():
                    tokens = tokens[:-1]
                text_prompt = ' '.join(tokens)
                prerendered = text_prompt

            if self.sample_text_prompts:
                prerendered = MultiModalInput(prerendered)

        pil_prerendered = None
        if 'pil_prerendered' in self.feature_list:
            pil_prerendered = prefetched_data.get('pil_prerendered')
            if pil_prerendered is None:
                if not self.real_input:
                    raise RuntimeError('Synthetic input images must be prefetched using custom data loader')
                else: # real data
                    prerendered_filename = os.path.join(self.data_dir, graph_name)
                    prerendered = self.pil_image_loader(prerendered_filename)

        # collect and return specified features
        # this list must match the supported features verbatim
        all_features = [
            graph, graph_name, graph_category_idx, generator_count, active_generator_groups, rendered_output, rendered_output_v2, rendered_output_v2_filename,
            rendered_output_v3, rendered_output_v3_filename, output_channels, node_outputs, nodes, node_depths, node_type_seq, node_depth_seq,
            node_idx_seq, node_seq_mask, slot_node_type_seq, slot_node_depth_seq, slot_node_idx_seq, slot_idx_seq, slot_id_seq, slot_seq_mask,
            edge_seq, edge_idx_seq, edge_elm_seq, edge_seq_mask, param_id_seq, param_token_idx_seq, param_val_seq, param_vector_elm_idx_seq,
            param_array_elm_idx_seq, param_idx_seq, param_seq_mask, param_node_inds, edge_node_inds, prerendered, param_node_type_seq, param_node_depth_seq,
            param_node_seq_mask, pil_prerendered
        ]
        if len(all_features) != len(self.SUPPORTED_FEATURES):
            raise RuntimeError('Supported and calculated features do not match.')

        return dict(zip(self.SUPPORTED_FEATURES, all_features))


class SimpleGraphDataset(torch.utils.data.Dataset):
    """Map-style simple graph dataset (most suitable for reading from individual files).
    """
    def __init__(self, data_dir, data_list, feature_list, shuffle=False, seed=None, image_input='render',
                 pil_image_loader=False, img_res=224, image_ext='png', graph_suffix=None, data_chunksize=64,
                 mp_rank=None, mp_world_size=None, **kwargs):
        # check input validity
        if not isinstance(data_dir, str):
            raise TypeError('The path to data directory is not a string.')
        if not isinstance(data_list, (str, list)):
            raise TypeError('The path to data list must be a string or a list of strings.')

        self.data_dir = data_dir
        self.feature_list = feature_list

        # read input prompts
        real_input = False
        text_input = False
        sample_names = None
        transformed_clip_embedding = None

        ## image prompts
        if data_dir:
            if data_list:
                if isinstance(data_list, str):
                    with open(data_list, 'r') as f:
                        sample_names = f.read().splitlines()
                else:
                    sample_names = data_list
            else:
                if not pth.isdir(data_dir):
                    raise ValueError('The data directory must be a folder.')
                sample_names = []
                image_exts = ('.jpg', '.jpeg', '.png')
                for ext in image_exts:
                    sample_names.extend(sorted(glob.glob(pth.join(data_dir, f'*{ext}'))))

                sample_names = [pth.basename(sample_name) for sample_name in sample_names]
                real_input = True

        ## text prompts
        elif data_list:
            if not isinstance(data_list, str):
                raise TypeError('The path to data list must be a string.')
            text_file_ext = pth.splitext(data_list)[1]
            if text_file_ext == '.txt':
                with open(data_list, 'r') as f:
                    sample_names = [x.strip() for x in f.read().splitlines()]
                text_input = True
            elif text_file_ext == '.json':
                with open(data_list, 'r') as f:
                    clip_file_dict = json.load(f)

                clip_data_dir = pth.dirname(data_list)
                sample_names = []
                transformed_clip_embedding = []
                for sample_name, clip_filename in clip_file_dict.items():
                    if isinstance(clip_filename, str):
                        sample_names.append(sample_name)
                        transformed_clip_embedding.append(pth.join(clip_data_dir, clip_filename))
                    else:
                        for clip_filename_ in clip_filename:
                            sample_names.append(sample_name)
                            transformed_clip_embedding.append(pth.join(clip_data_dir, clip_filename_))
                text_input = True
            else:
                raise RuntimeError('Unknown type of input text file.')

        if sample_names is None:
            raise RuntimeError('No data input.')

        self.sample_names = sample_names
        self.text_input = text_input
        self.transformed_clip_embedding = transformed_clip_embedding
        self.rank = max(mp_rank, 0) if mp_rank is not None else 0
        self.world_size = max(mp_world_size, 1) if mp_world_size is not None else 1

        # image input type
        suffix_dict = {
            'render': '_rendered',
            'basecolor': '_baseColor',
            'real': '',
        }
        if image_input not in suffix_dict:
            raise RuntimeError(f'Unknown type of input image: {image_input}')
        image_suffix = suffix_dict[image_input]

        # create the source data loader
        self.data_loader = get_data_loader(
            data_dir, pil_image_loader=pil_image_loader, data_chunksize=data_chunksize, image_res=img_res,
            image_suffix=image_suffix, image_ext=image_ext)
        self.graph_suffix = graph_suffix or '_quantized'

        # create the graph data loader
        self.graph_loader = SimpleGraphLoader(
            data_dir, feature_list, seed=seed, real_input=real_input, pil_image_loader=pil_image_loader,
            img_res=img_res, image_ext=image_ext, **kwargs)

        # shuffling indices
        shuffle_arr = None
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            shuffle_arr = rng.permutation(len(sample_names)).tolist()

        self.shuffle_arr = shuffle_arr

    # obsolete and for compatibility only, prefer using get_ordered_nodes directly
    @staticmethod
    def get_ordered_nodes(*args, **kwargs):
        return get_ordered_nodes(*args, **kwargs)

    def __len__(self):
        rank, world_size = self.rank, self.world_size
        return (len(self.sample_names) + world_size - rank - 1) // world_size

    def __getitem__(self, index):
        # index shuffling
        index = index * self.world_size + self.rank
        index = self.shuffle_arr[index] if self.shuffle_arr is not None else index

        # load graph data by the graph name
        graph_name = self.sample_names[index]
        prefetched_data = self.data_loader.load_data(graph_name, json_suffix=self.graph_suffix)
        feature_dict = self.graph_loader(graph_name, **prefetched_data)

        # additional features
        clip_embedding = None
        if 'clip_embedding' in self.feature_list:
            if self.text_input and self.transformed_clip_embedding is not None:
                with open(self.transformed_clip_embedding[index], "rb") as f:
                    data = f.read()
                # f = struct.unpack('f' * 768, data)
                # clip_embedding = torch.as_tensor(f, dtype=torch.float32)
                clip_embedding = torch.frombuffer(data, dtype=torch.float32)
                if len(clip_embedding) != 768:
                    raise RuntimeError('Failed to read the clip embedding into tensor.')
            else:
                raise RuntimeError('Clip features are only available for text prompts json dataset.')

            feature_dict['clip_embedding'] = clip_embedding

        if 'index' in self.feature_list:
            feature_dict['index'] = index

        # gather requested features
        for feature_name in self.feature_list:
            if feature_name not in feature_dict:
                raise KeyError('Unrecognized feature name:', feature_name)

        return tuple(feature_dict[feature_name] for feature_name in self.feature_list)


class SimpleGraphIterableDataset(torch.utils.data.IterableDataset):
    """Iterable-style simple graph dataset (most suitable for reading from HDF5).
    """
    def __init__(self, data_dir, data_list, feature_list, shuffle=False, pre_shuffle=False, seed=None, image_input='render',
                 pil_image_loader=False, img_res=224, image_ext='png', graph_suffix=None, data_chunksize=64,
                 target_shuffle_queue_size=512, mp_rank=None, mp_world_size=None, batch_alignment=None, **kwargs):
        # check input validity
        if not isinstance(data_dir, str):
            raise TypeError('The path to data directory is not a string.')
        if not isinstance(data_list, (str, list)):
            raise TypeError('The path to data list must be a string or a list of strings.')
        if not data_dir:
            raise NotImplementedError('Iterable dataset does not support text prompts.')
        if not data_list:
            raise ValueError('A list of graph names must be provided.')

        # unsupported features
        if image_input != 'render':
            raise NotImplementedError('Non-prerendered image input is not supported.')
        if 'clip_embedding' in feature_list:
            raise NotImplementedError('CLIP embedding feature is not available in an iterable dataset.')

        if isinstance(data_list, str):
            with open(data_list, 'r') as f:
                sample_names = f.read().splitlines()
        else:
            sample_names = data_list
        if not sample_names:
            raise RuntimeError('No data input.')

        self.data_dir = data_dir
        self.feature_list = feature_list
        self.sample_names = sample_names

        self.is_distributed = mp_world_size is not None and mp_world_size > 1
        self.rank = max(mp_rank, 0) if mp_rank is not None else 0
        self.world_size = max(mp_world_size, 1) if mp_world_size is not None else 1
        self.alignment = max(batch_alignment, 0) if batch_alignment is not None else 0

        # create the source data loader
        image_loader = (PILImageLoader if pil_image_loader else ImageLoader)(img_res)

        if data_dir.endswith('.h5'):
            data_loader = H5DataLoader(data_dir, image_loader, chunksize=data_chunksize)
        elif pth.isdir(data_dir):
            data_loader = SPickleDataLoader(data_dir, image_loader, chunksize=data_chunksize)
        else:
            raise ValueError(f'Invalid data path: {data_dir}')

        self.data_loader = data_loader
        self.graph_suffix = graph_suffix

        # generate a random seed and make sure it is the same across all processes
        if self.is_distributed and seed is None:
            seed_rank = np.random.default_rng().integers(2**32, dtype=np.int64) if self.rank == 0 else 0
            seed = torch.tensor(seed_rank, dtype=torch.int64, device='cuda')
            torch.distributed.broadcast(seed, src=0)
            seed = seed.item()

        self.rng = np.random.default_rng(seed=seed)
        self.shuffle = shuffle
        self.target_shuffle_queue_size = target_shuffle_queue_size

        # create the graph data loader
        self.graph_loader = SimpleGraphLoader(data_dir, feature_list, image_ext=image_ext, **kwargs)

        # build chunk index dictionary (read from cache if exists)
        data_list_name = pth.splitext(pth.basename(data_list))[0] if isinstance(data_list, str) else '<custom>'
        chunk_dict = {}
        print(f'Initializing iterable dataset ({data_list_name})...')

        for sample_id, sample_name in enumerate(sample_names):
            graph_name, variation_name = split_variation_name(sample_name)
            index_dict = data_loader._get_index_dict(graph_name)

            # get data chunk and entry index
            data_index = index_dict[variation_name]
            chunk_id, entry_id = data_index // data_chunksize, data_index % data_chunksize

            # insert data into the chunk dictionary
            chunk_entries = chunk_dict.setdefault(graph_name, {}).setdefault(chunk_id, [])
            chunk_entries.append((entry_id, sample_id))

        # compile chunk list
        graph_names = sorted(chunk_dict.keys())
        graph_index_dict = {n: i for i, n in enumerate(graph_names)}

        chunk_list = []
        for graph_name, graph_chunk_dict in chunk_dict.items():
            graph_id = graph_index_dict[graph_name]
            chunk_list.extend((graph_id, chunk_id, None) for chunk_id in sorted(graph_chunk_dict.keys()))

        # pre-shuffle chunk list
        if pre_shuffle:
            shuffle_arr = self.rng.permutation(len(chunk_list)).tolist()
            chunk_list = [chunk_list[i] for i in shuffle_arr]

        self.graph_names = graph_names
        self.chunk_dict = chunk_dict
        self.chunk_list = chunk_list

    def __len__(self):
        # roughly estimate the number of samples in each split
        # the actual number of samples may be different due to chunk shuffling
        return (len(self.sample_names) + self.world_size - self.rank - 1) // self.world_size

    # extend a chunk list with masked entries until reaching dataset_split_size
    def _extend_chunk_list(self, chunk_list, target_size, rng=None):
        # calculate the number of samples
        graph_names, chunk_dict = self.graph_names, self.chunk_dict
        split_size = sum(len(chunk_dict[graph_names[graph_id]][chunk_id])
                         for graph_id, chunk_id, _ in chunk_list)

        # continue extending the chunk list until reaching the target size
        N = len(chunk_list)

        while split_size < target_size:
            # shuffle chunks
            shuffle_arr = rng.permutation(N).tolist() if rng is not None else None

            for i in range(N):
                # check chunk size
                graph_id, chunk_id, _ = chunk_list[shuffle_arr[i] if shuffle_arr is not None else i]
                chunk_size = len(chunk_dict[graph_names[graph_id]][chunk_id])

                # append the entire chunk to the list
                if split_size + chunk_size <= target_size:
                    chunk_list.append((graph_id, chunk_id, None))
                    split_size += chunk_size

                # append the chunk to the list and mask some entries
                else:
                    chunk_mask = np.zeros(chunk_size, dtype=bool)
                    chunk_mask[:target_size-split_size] = True
                    if rng is not None:
                        rng.shuffle(chunk_mask)

                    chunk_list.append((graph_id, chunk_id, chunk_mask))
                    split_size += chunk_mask.sum()

                if split_size >= target_size:
                    break

        # final sanity check
        if split_size != target_size:
            raise RuntimeError('Failed to extend the chunk list')

        return chunk_list

    # generate streamed and preprocessed data
    def _data_generator(self, chunk_list):
        # loop over all chunks, optionally in a shuffled order
        suffix = self.graph_suffix or '_quantized'

        for graph_id, chunk_id, chunk_mask in chunk_list:
            # fetch the chunk data
            graph_name = self.graph_names[graph_id]
            entry_list, sample_ids = tuple(zip(*self.chunk_dict[graph_name][chunk_id]))
            chunk_data_iter = self.data_loader.load_chunk(
                graph_name, chunk_id, entry_list=entry_list, json_suffix=suffix)

            # sequentially generate data entries
            if chunk_mask is None:
                for sample_id, sample_data in zip(sample_ids, chunk_data_iter):
                    sample_name = self.sample_names[sample_id]
                    yield sample_id, sample_name, sample_data

            # only generate masked data entries
            else:
                for sample_id, sample_data, sample_flag in zip(sample_ids, chunk_data_iter, chunk_mask):
                    if sample_flag:
                        sample_name = self.sample_names[sample_id]
                        yield sample_id, sample_name, sample_data

    def __iter__(self):
        # obtain worker info
        worker_info = torch.utils.data.get_worker_info()
        is_worker = worker_info is not None
        worker_id_rank = worker_info.id if is_worker else 0
        num_workers_rank = worker_info.num_workers if is_worker else 1

        rank, world_size = self.rank, self.world_size
        worker_id = worker_id_rank + rank * num_workers_rank
        num_workers = num_workers_rank * world_size

        # pre-shuffle the chunk list
        num_chunks = len(self.chunk_list)

        if self.shuffle:
            shuffle_arr = self.rng.permutation(num_chunks).tolist()
            chunk_list = [self.chunk_list[i] for i in shuffle_arr]
        else:
            chunk_list = self.chunk_list.copy()

        # extend the chunk list to match the number of workers
        if num_chunks < num_workers:
            chunk_list = chunk_list * ((num_workers + num_chunks - 1) // num_chunks)
            chunk_list = chunk_list[:num_workers]

        # assign the worker chunk list
        worker_chunk_list = chunk_list[worker_id::num_workers] if num_workers > 1 else chunk_list
        if not worker_chunk_list:
            raise RuntimeError('No data chunk assigned to worker', worker_id)

        # count the number of samples for each worker
        chunk_dict, graph_names = self.chunk_dict, self.graph_names

        if num_workers > 1:
            num_worker_samples = sum(len(chunk_dict[graph_names[graph_id]][chunk_id]) for graph_id, chunk_id, _ in worker_chunk_list)
        else:
            num_worker_samples = len(self.sample_names)

        # align the number of samples across workers for multi-GPU training
        if self.is_distributed:
            all_num_worker_samples = [0] * num_workers
            for i, (graph_id, chunk_id, _) in enumerate(chunk_list):
                all_num_worker_samples[i % num_workers] += len(chunk_dict[graph_names[graph_id]][chunk_id])
            target_num_worker_samples = max(all_num_worker_samples)
        else:
            target_num_worker_samples = num_worker_samples

        # align the number of samples with the user-specified stride
        alignment = self.alignment
        if alignment > 0:
            target_num_worker_samples = (target_num_worker_samples + alignment - 1) // alignment * alignment

        # set up the worker-specific random number generator
        if not self.shuffle:
            rng = None
        elif num_workers == 1:
            rng = self.rng
        else:
            rng = np.random.default_rng(self.rng.integers(2**32) + worker_id)

        # extend the chunk list if necessary
        if target_num_worker_samples > num_worker_samples:
            worker_chunk_list = self._extend_chunk_list(worker_chunk_list, target_num_worker_samples, rng=rng)

        # print the number of samples for each worker
        if self.is_distributed:
            print(f'[Rank {rank} - Worker {worker_id_rank}] {len(worker_chunk_list)} chunks, {target_num_worker_samples} samples')

        # instantiate the data generator
        data_gen = self._data_generator(worker_chunk_list)
        next_data = next(data_gen, None)

        # main loop
        shuffle_queue = []
        target_size = self.target_shuffle_queue_size

        while shuffle_queue or next_data:
            # fill the shuffling queue
            if len(shuffle_queue) < target_size and next_data:
                shuffle_queue.append(next_data)
                next_data = next(data_gen, None)

            # randomly eject a data entry from the shuffle queue
            else:
                queue_id = rng.integers(len(shuffle_queue), size=None) if rng is not None else 0
                sample_id, sample_name, prefetched_data = shuffle_queue.pop(queue_id)
                feature_dict = self.graph_loader(sample_name, **prefetched_data)

                # assign sample index
                if 'index' in self.feature_list:
                    feature_dict['index'] = sample_id

                # gather requested features
                for feature_name in self.feature_list:
                    if feature_name not in feature_dict:
                        raise KeyError('Unrecognized feature name:', feature_name)

                yield tuple(feature_dict[fn] for fn in self.feature_list)


class RealImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_list, nn_file, data_dir, feature_list, seed=None,
                 image_ext='png', img_res=224, pil_image_loader=False, augment_image=None, augment_graph=None,
                 shuffle=False, shuffle_mode=None, graph_suffix=None, data_chunksize=64, mp_rank=None, mp_world_size=None,
                 **json_dataset_kwargs):
        # check input validity
        if not isinstance(image_dir, str):
            raise TypeError('The path to image directory is not a string.')
        if not isinstance(image_list, (str, list)):
            raise TypeError('The path to image list must be a string or a list of strings.')

        # read image list
        if isinstance(image_list, str):
            with open(image_list, 'r') as f:
                image_list = f.read().splitlines()

        # read nearest neighbor dictionary
        try:
            with open(nn_file, 'r') as f:
                nn_dict = json.load(f)
        except FileNotFoundError:
            nn_dict = {}

        self.image_dir = image_dir
        self.image_list = image_list
        self.nn_dict = nn_dict
        self.feature_list = feature_list
        self.image_ext = image_ext
        self.augment_image = augment_image
        self.augment_graph = augment_graph
        self.rank = max(mp_rank, 0) if mp_rank is not None else 0
        self.world_size = max(mp_world_size, 1) if mp_world_size is not None else 1

        # create image loaders
        self.image_loader = (PILImageLoader if pil_image_loader else ImageLoader)(img_res)
        self.pil_image_loader = PILImageLoader(img_res)

        # create the JSON data loader
        self.json_loader = get_data_loader(data_dir, data_chunksize=data_chunksize)
        self.graph_suffix = graph_suffix or '_quantized'

        # create the graph data loader
        self.graph_loader = SimpleGraphLoader(data_dir, feature_list, **json_dataset_kwargs)

        # size multipliers for augmentation
        self.multiplier_image = {'rot': 2, 'rot4': 4, 'all': 8}[augment_image] if augment_image else 1
        self.multiplier_nn = max(min(len(next(iter(nn_dict.values()))), augment_graph), 1) if augment_graph else 1
        self.multiplier = self.multiplier_image * self.multiplier_nn

        # extended data length for mixing
        self.extended_length = None

        # shuffling indices
        shuffle_arr = None
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            num_images = len(self.image_list)

            # shuffle only at the image level
            if shuffle_mode == 'image':
                shuffle_arr = rng.permutation(num_images)
                shuffle_arr = (shuffle_arr[:, None] * self.multiplier + np.arange(self.multiplier)).flatten().tolist()

            # shuffle only at the augmented image level
            elif shuffle_mode == 'aug_image':
                shuffle_arr = rng.permutation(num_images * self.multiplier_image)
                shuffle_arr = (shuffle_arr[:, None] * self.multiplier_nn + np.arange(self.multiplier_nn)).flatten().tolist()

            # shuffle all
            elif shuffle_mode is None or shuffle_mode == 'all':
                shuffle_arr = rng.permutation(len(self)).tolist()
            else:
                raise ValueError('Unknown shuffling mode:', shuffle_mode)

        self.shuffle_arr = shuffle_arr

    def __len__(self):
        rank, world_size = self.rank, self.world_size
        size = self.extended_length or len(self.image_list) * self.multiplier
        return (size + world_size - rank - 1) // world_size

    def extend(self, length):
        self.extended_length = None
        if length < len(self):
            raise ValueError('The extended length must be greater than the default dataset size.')
        self.extended_length = length

    def _augment_image(self, img, augment_mode):
        # augment mode must be within [0, 7]
        if augment_mode < 0 or augment_mode > 7:
            raise RuntimeError('Invalid augmentation mode:', augment_mode)

        # flip and rotate the image
        img = img.flip(2) if augment_mode >= 4 else img
        img = img.rot90(augment_mode % 4, dims=(1, 2)) if augment_mode % 4 else img
        return img

    def __getitem__(self, idx):
        prefetched_data = {}

        # index shuffling
        idx = (idx * self.world_size + self.rank) % (len(self.image_list) * self.multiplier)
        idx = self.shuffle_arr[idx] if self.shuffle_arr is not None else idx

        # get the image name
        aug_img_idx, nn_idx = divmod(idx, self.multiplier_nn)
        img_idx, aug_mode = divmod(aug_img_idx, self.multiplier_image)
        image_name = self.image_list[img_idx]

        # load the image
        image_filename = os.path.join(self.image_dir, f'{image_name}.{self.image_ext}')

        if 'prerendered' in self.feature_list:
            image = self.image_loader(image_filename)
            prefetched_data['prerendered'] = self._augment_image(image, aug_mode)

        if 'pil_prerendered' in self.feature_list:
            pil_image = self.pil_image_loader(image_filename)
            prefetched_data['pil_prerendered'] = self._augment_image(pil_image, aug_mode)

        # get the nearest neighbor graph name
        aug_image_name = image_name + f'.aug{aug_mode}' if aug_mode else image_name
        graph_name = self.nn_dict[aug_image_name][nn_idx] if self.nn_dict else aug_image_name

        # load the JSON data from the nearest neighbors
        if any(feat not in ('prerendered', 'pil_prerendered', 'name', 'index') for feat in self.feature_list):
            json_nodes = self.json_loader.load_data(graph_name, json_suffix=self.graph_suffix)['json_nodes']
            prefetched_data['json_nodes'] = json_nodes

        # obtain other features
        feature_dict = self.graph_loader(graph_name, **prefetched_data)

        # replace the graph name with image name
        if 'name' in self.feature_list:
            item_name = aug_image_name.replace('.', '_') + (f'_nn{nn_idx}' if nn_idx else '')
            feature_dict['name'] = item_name

        # assign image index
        if 'index' in self.feature_list:
            feature_dict['index'] = aug_img_idx

        # gather requested features
        for feature_name in self.feature_list:
            if feature_name not in feature_dict:
                raise KeyError('Unrecognized feature name:', feature_name)

        return tuple(feature_dict[fn] for fn in self.feature_list)
