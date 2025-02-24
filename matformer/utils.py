# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import json
import math
from collections import OrderedDict
import time
import torch
import numpy as np
import torchvision
import graphviz # pip install graphviz
import re
import sys
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import functional as TF

from .diffsbs.sbs_utils import write_image


def logits_regularize(logits, temperature=1, top_k=0, filter_value=-1e9, return_mask=False):
    if top_k > 0:
        logits_rank = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        logits_mask = logits_rank > top_k
        logits[logits_mask] = filter_value

    if temperature != 1:
        logits /= temperature

    return (logits, logits_mask) if return_mask else logits


def nucleus_filtering(logits, top_p=0.9, filter_value=-1e9, return_mask=False):
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative probability above the threshold
    prob_mask = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    prob_mask[..., 1:] = prob_mask[..., :-1].clone()
    prob_mask[..., 0] = False

    prob_mask = prob_mask.gather(-1, sorted_indices.argsort(dim=-1))
    logits[prob_mask] = filter_value

    return (logits, prob_mask) if return_mask else logits


def unique(x):
    seen = set()
    seen_add = seen.add
    return [elm for elm in x if not (elm in seen or seen_add(elm))]


def load_node_types(filename, verbose=True):
    with open(filename, 'r') as f:
        node_types = json.load(f, object_pairs_hook=OrderedDict)

    # insert compatibility flags
    node_types._legacy_flattened = any(':' in k for k in node_types.keys())

    return node_types


def load_model_state(model, state_dict, device, exclude_lm_head=False):
    # load model state between data parallel / non data parallel models
    if isinstance(state_dict, str):
        weight_dict = torch.load(state_dict, map_location=device)
    elif isinstance(state_dict, dict):
        weight_dict = state_dict
    else:
        raise TypeError(f'Invalid state dict type: {type(state_dict).__name__}')

    is_data_parallel = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    fixed_weight_dict = {}
    for k, v in weight_dict.items():
        if not is_data_parallel and k.startswith('module.'):
            fixed_weight_dict[k[len('module.'):]] = v
        elif is_data_parallel and not k.startswith('module.'):
            fixed_weight_dict[f'module.{k}'] = v
        else:
            fixed_weight_dict[k] = v

    # remove all fields pertaining to the LM head for value networks
    # the removed fields should contain (decoder.[*_generator.]lm_head)
    if exclude_lm_head:
        pattern_lm_head = re.compile(r'(^|\.)decoder(\.[^\.]+_(generator|decoder))?\.lm_head')
        fixed_weight_dict = {k: v for k, v in fixed_weight_dict.items() if pattern_lm_head.search(k) is None}

    # read state dict and resolve unmatched portions
    wrong_keys = model.load_state_dict(fixed_weight_dict, strict=False)

    if wrong_keys.unexpected_keys:
        raise RuntimeError(f'Unexpected keys: {wrong_keys.unexpected_keys}')

    # check for missing keys, filter out any missing key that belongs to CLIP or VGG19
    pattern_frozen = re.compile(r'(^|\.)(clip_encoder|vgg19)\.')
    missing_keys = [k for k in wrong_keys.missing_keys if pattern_frozen.search(k) is None]
    if exclude_lm_head:
        missing_keys = [k for k in missing_keys if pattern_lm_head.search(k) is None]
    if missing_keys:
        raise RuntimeError(f'Missing keys: {missing_keys}')


def save_model_state(model, state_dict_filename=None):
    # save model state to local file, removing freezed sections
    pattern = re.compile(r'(^|\.)(clip_encoder|vgg19)\.')
    state_dict = {k: v for k, v in model.state_dict().items() if pattern.search(k) is None}
    if state_dict_filename is not None:
        torch.save(state_dict, state_dict_filename)
    return state_dict


def load_optim_state(optimizer, lr_scheduler, state_dict_filename):
    # load optimizer state dict
    state_dict = torch.load(state_dict_filename)
    optimizer.load_state_dict(state_dict['optim'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dict['sched'])

    return state_dict['epoch'], state_dict['step']


def save_optim_state(epoch, step, optimizer, lr_scheduler, state_dict_filename):
    # save optimizer state dict
    state_dict = {
        'epoch': epoch,
        'step': step,
        'optim': optimizer.state_dict(),
        'sched': lr_scheduler.state_dict() if lr_scheduler is not None else None
    }
    torch.save(state_dict, state_dict_filename)


class CosineWarmup:
    """Cosine warmup learning rate function.
    """
    def __init__(self, warmup_steps=0, annealing_steps=1000, min_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        self.min_ratio = min_ratio

    def __call__(self, step):
        if step < self.warmup_steps:
            ratio = step / self.warmup_steps
        else:
            step -= self.warmup_steps
            if step < self.annealing_steps:
                alpha = (1 + math.cos(math.pi * step / self.annealing_steps)) * 0.5
                ratio = self.min_ratio + (1 - self.min_ratio) * alpha
            else:
                ratio = self.min_ratio

        return ratio


# data features for a single sample should be given as a tuple of elements.
# Each element can either be a tensor, a list/tuple of tensors (where each element is collated recursively), or a string.
# Lists of tensors are always stacked (so they need to have the same length), lists of floats or ints are converted to tensors, and no changes are made otherwise.
def collate_data_features(batch):
    if not isinstance(batch, (list, tuple)):
        raise RuntimeError('Data features must be given as a list or tuple.')

    batched_data_feats = list(zip(*batch))

    for feat_idx in range(len(batched_data_feats)):
        batched_data_feats[feat_idx] = stack_tensor_lists(batched_data_feats[feat_idx])

    return batched_data_feats


def stack_tensor_lists(batch, pad_value=0):

    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        # determine output tensor size
        max_dims = [max(x.shape[i] for x in batch) for i in range(elem.ndim)]
        numel = len(batch) * math.prod(max_dims)

        # copied from pytorch 'default_collate'
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            storage = elem._typed_storage() if hasattr(elem, '_typed_storage') else elem.storage()
            out = elem.new(storage._new_shared(numel)).resize_(len(batch), *max_dims)
        else:
            out = torch.empty(numel, dtype=elem.dtype, device=elem.device).resize_(len(batch), *max_dims)

        # try stacking tensors
        try:
            torch.stack(batch, dim=0, out=out)
        # if stacking fails, fill with pad value and copy tensors one by one
        except RuntimeError:
            if pad_value is not None:
                out.fill_(pad_value)
            for i, x in enumerate(batch):
                out[(i, *(slice(0, s) for s in x.shape))] = x

        return out

    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)

    return batch


def prepare_batch(args, data, dataset_features, device, non_blocking=False):
    prepared_data = {}
    param_full = getattr(args, 'full', True)

    for key, value in zip(dataset_features, data):
        # compose data
        if not param_full and key in (
            'param_id_seq', 'param_token_idx_seq', 'param_val_seq', 'param_vector_elm_idx_seq',
            'param_array_elm_idx_seq', 'param_idx_seq', 'param_seq_mask'
        ):
            if isinstance(value, torch.Tensor):
                value = value.flatten(0, 1)
            elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                value = torch.cat(value, dim=0)
            else:
                raise ValueError(f'Cannot compose data type: {type(value).__name__}[{type(value[0]).__name__}]')

        prepared_data[key] = value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value

    return prepared_data


def split_at_indices(tensor, before_split_inds, dim=0):
    if len(before_split_inds) == 0:
        subseq_lengths = [tensor.shape[dim]]
    else:
        subseq_lengths = [0]*(len(before_split_inds)+1)
        subseq_lengths[0] = before_split_inds[0]+1
        subseq_lengths[1:-1] = before_split_inds[1:]-before_split_inds[:-1]
        subseq_lengths[-1] = tensor.shape[dim]-(before_split_inds[-1]+1)
    return torch.split(tensor, split_size_or_sections=subseq_lengths, dim=dim)


def vis_simple_graph(graph, filename, include_parent_end=False, colorized_nodes=None):
    if len(set([n.name for n in graph.nodes])) != len(graph.nodes):
        raise RuntimeError('Node names are not unique, need unique node names.')

    dotgraph = graphviz.Digraph()

    # add nodes
    for node in graph.nodes:
        if not include_parent_end and node.type == 'parent_end':
            continue
        if colorized_nodes is None or node not in colorized_nodes:
            dotgraph.node(name=node.name.replace(':','__'), label=node.type, style='rounded', shape='box')
        else:
            dotgraph.node(name=node.name.replace(':', '__'), label=node.type, style='filled', shape='box', fillcolor='red')
    for node in graph.nodes:
        for pi in range(len(node.parents)):
            if node.parents[pi][0] is None:
                # empty input slot
                continue
            if not include_parent_end and (node.type == 'parent_end' or node.parents[pi][0].type == 'parent_end'):
                continue
            dotgraph.edge(tail_name=node.parents[pi][0].name.replace(':','__'), head_name=node.name.replace(':','__'))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dotgraph.render(os.path.splitext(filename)[0])


def vis_comp_batch_images(gt_imgs, pred_imgs, filename):
    # resize ground truth images to match prediction images
    if gt_imgs.shape[-2:] != pred_imgs.shape[-2:]:
        gt_imgs = TF.resize(gt_imgs, size=pred_imgs.shape[-2:], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        gt_imgs.clamp_(0, 1)

    # interleave ground truth and prediction images
    all_imgs = torch.cat((gt_imgs[:, None], pred_imgs[:, None]), dim=1).flatten(0, 1)

    # save image
    n_rows = min(len(gt_imgs) * 2, 8)
    write_image(filename, torchvision.utils.make_grid(all_imgs, nrow=n_rows).cpu(), process=True)


class Timer:
    def __init__(self):
        self.start_time = []

    def begin(self, output=''):
        if output != '':
            print(output)
        self.start_time.append(time.time())

    def end(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time.pop()
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def lap(self, output=''):
        if len(self.start_time) == 0:
            raise RuntimeError("Timer stack is empty!")
        t = self.start_time[-1]
        elapsed_time = time.time() - t
        print(output, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


class IterProfiler:
    def __init__(self, sequence, title=''):
        self.seq = sequence
        self.title = title if title else 'Data loader'
        self.iter = None

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.iter = iter(self.seq)
        return self

    def __next__(self):
        t_start = time.time()
        val = next(self.iter)
        t_end = time.time()
        print(f'{self.title}: {t_end - t_start:.3f} s')
        return val


# helper function for printing tqdm progress bar to file
def tqdm_file(iterable, write_to_file=False, update_interval=1000, total=None, **kwargs):
    # behaves the same as tqdm by default
    if not write_to_file:
        yield from tqdm(iterable, **kwargs)
        return

    # read total size
    if total is None:
        if not hasattr(iterable, '__len__'):
            raise ValueError("The input iterable must support '__len__'.")
        total = len(iterable)

    # create the pbar and start iteration
    with open(os.devnull, 'w') as f:
        pbar = tqdm(total=total, file=f, **kwargs)

        for element in iterable:
            yield element

            # update and print progress bar
            pbar.update()
            if not pbar.n % update_interval:
                print(str(pbar))
                sys.stdout.flush()

        # print final result
        if pbar.n % update_interval:
            print(str(pbar))
            sys.stdout.flush()


# unwrap a model from a DDP wrapper
def unwrap_ddp(model):
    if isinstance(model, DDP):
        return model.module
    return model


# render a text label on an image
def render_text_label(image, text, font_size=32, font_color=(255, 255, 255), bg_color=(0, 0, 0), bg_alpha=0.5, padding=10):
    # convert image to PIL image
    original_type = 'pil'

    if isinstance(image, torch.Tensor):
        original_type = 'tensor'
        image = TF.to_pil_image(image).convert('RGB')

    elif isinstance(image, np.ndarray):
        original_type = 'numpy'
        if image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise TypeError(f'Invalid image data type: {image.dtype}')
        image = Image.fromarray(image).convert('RGB')

    elif not isinstance(image, Image.Image):
        raise TypeError(f'Invalid image type: {type(image).__name__}')

    # convert colors to RGB
    if any(isinstance(c, float) for c in font_color):
        font_color = tuple(int(c * 255 + 0.5) for c in font_color)
    if any(isinstance(c, float) for c in bg_color):
        bg_color = tuple(int(c * 255 + 0.5) for c in bg_color)
    if isinstance(bg_alpha, float):
        bg_alpha = int(bg_alpha * 255 + 0.5)

    # create font
    FONT_PATH = '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf'
    font = ImageFont.truetype(FONT_PATH, font_size)

    # draw text label with rectangle background
    draw = ImageDraw.Draw(image, 'RGBA')
    _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)

    x, y = image.width - text_width - padding * 2, image.height - text_height - padding * 2
    draw.rectangle((x, y, image.width, image.height), fill=(*bg_color, bg_alpha))
    draw.text((x + padding, y + padding), text, font=font, fill=font_color)

    # convert image back to tensor
    if original_type == 'tensor':
        image = TF.to_tensor(image)
    elif original_type == 'numpy':
        image = np.array(image, dtype=np.float32) / 255

    return image


# generate a comparison grid image
def gen_comp_grid(image_input, *out_images, n_cols=10, placeholder=None, val_labels=None, **label_kwargs):
    # apply value labels to output images if given
    if val_labels is not None:
        if len(val_labels) != len(out_images):
            raise ValueError('Number of value labels must match number of output images.')

        # draw value labels on output images
        out_images = list(out_images)
        for i, (img, label) in enumerate(zip(out_images, val_labels)):
            out_images[i] = render_text_label(img, label, **label_kwargs)

    # arrange images in a line
    if len(out_images) <= n_cols:
        grid = torch.cat((image_input, *out_images), dim=-1)
    else:
        placeholder = torch.ones_like(image_input) if placeholder is None else placeholder
        grid_imgs = []
        for i in range((len(out_images) + n_cols - 1) // n_cols):
            row_imgs = [placeholder if i else image_input]
            row_imgs.extend(out_images[i*n_cols:(i+1)*n_cols])
            row_imgs.extend([placeholder] * (n_cols + 1 - len(row_imgs)))
            grid_imgs.append(torch.cat(row_imgs, dim=-1))
        grid = torch.cat(grid_imgs, dim=-2)

    return grid


# generate a comparison grid image with vertical layout
def gen_vertical_comp_grid(target, *img_stacks, val_labels=None, method_labels=None, row_padding=4,
                           label_padding=10, font_size=42, font_file=None, **label_kwargs):
    # apply value labels to output images if given
    if val_labels is not None:
        if len(val_labels) != len(img_stacks):
            raise ValueError('Number of value labels must match number of output images.')

        # draw value labels on output images
        img_stacks = list(img_stacks)
        for i, (img_row, val_row) in enumerate(zip(img_stacks, val_labels)):
            for i, (img, val) in enumerate(zip(img_row, val_row)):
                img_row[i] = render_text_label(img, val, **label_kwargs)

    # convert the image stack into a grid
    img_rows = [torch.cat(row, dim=-1) for row in img_stacks]
    grid = torch.cat([F.pad(row, [0, 0, row_padding if i else 0, 0], value=1.0) for i, row in enumerate(img_rows)], dim=-2)

    # apply labels to the grid
    font = ImageFont.truetype(font_file or '/home/beichen/Downloads/linux_libertine/LinLibertine_R.ttf', font_size)

    # draw method labels
    if method_labels is not None:
        # Expand the image to reserve space for method labels
        grid = F.pad(grid, (font_size + label_padding, 0), value=1.0)

        # Draw the method labels vertically on the left side of the image
        grid = TF.to_pil_image(grid).convert('RGB')
        draw = ImageDraw.Draw(grid)

        img_res = target.shape[-2]
        for i, label in enumerate(method_labels):

            # draw a standalone image for the label
            label_img = Image.new('RGBA', (img_res, font_size), color=(255, 255, 255))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((img_res * 0.5, 0), label, font=font, fill='black', anchor='mt', align='center')

            # rotate the label image and paste it on the grid
            label_img = label_img.rotate(90, expand=True)
            grid.paste(label_img, (0, i * (img_res + row_padding)), label_img)

        grid = TF.to_tensor(grid)

    # pad the target image to match grid height
    target = F.pad(target, [label_padding * 2, label_padding * 2, 0, grid.shape[-2] - target.shape[-2]], value=1.0)
    grid = torch.cat((target, grid), dim=-1)

    return grid
