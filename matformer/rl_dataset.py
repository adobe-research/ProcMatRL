# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from abc import ABC, abstractmethod
from argparse import Namespace
from itertools import starmap
import copy
import math
import os
import os.path as pth

from multiprocessing import Pool
from kornia.color import rgb_to_lab
from lpips import LPIPS
import torch
import torch.nn.functional as F
import numpy as np

from .diffsbs.sbs_graph import SBSGraph

from .simple_graph.convert_simple_graph_parameters import unconvert_simple_graph_parameters
from .simple_graph.convert_and_filter_simple_graph_parameters import unconvert_clamped_simple_graph_parameters
from .model_def import CondMatFormer
from .render_graph import render_graph_output
from .swd import swd
from .utils import prepare_batch, stack_tensor_lists, vis_comp_batch_images
from .vgg import VGGTextureDescriptor


def render_worker(json_nodes, graph_filename, load_json_kwargs, render_worker_kwargs):
    output_image = None

    try:
        # debug for file handlers
        if render_worker_kwargs.get('use_networkx', True):
            raise RuntimeError('Using networkx for rendering could cause resource leak.')

        # read SBS graph
        graph_sbs = SBSGraph.load_json(graph_name=pth.basename(graph_filename), json_data=json_nodes, **load_json_kwargs)
        graph_sbs.force_directx_normals()
        graph_sbs.update_node_dtypes(harmonize_signatures=True)

        # invoke SAT to render the graph (with time limit)
        channels = graph_sbs.run_sat(graph_filename=f'{graph_filename}.sbs', output_name=graph_filename,
                                     timeout=60, **render_worker_kwargs)
        output_image = render_graph_output(output_channels=channels, normal_format='dx')

    except RuntimeError as e:
        print(f'[WARNING] Rendering {graph_filename} failed (error message: {e.args[0]}).')

    return output_image


class RewardFunction:
    def __init__(self, vgg_coeff=10.0, vgg_td_level=2, ds_l1_coeff=0.5, ds_lab_l1_coeff=0.0, lpips_coeff=0.0, swd_coeff=0.0, kl_coeff=0.001,
                 lab_weights=[0.2, 1.0, 1.0], tmp_dir=None, vis_dir=None, num_render_procs=0, conversion_kwargs={}, rendering_kwargs={}, device='cpu'):
        # coefficients for different reward (loss) terms
        self.vgg_coeff = vgg_coeff
        self.ds_l1_coeff = ds_l1_coeff
        self.ds_lab_l1_coeff = ds_lab_l1_coeff
        self.lab_weights = lab_weights
        self.lpips_coeff = lpips_coeff
        self.swd_coeff = swd_coeff
        self.kl_coeff = kl_coeff
        self.img_size = 224

        # create feature extractors
        self.vgg_metric = VGGTextureDescriptor(device=device, td_level=vgg_td_level) if vgg_coeff > 0 else None
        self.lpips_metric = LPIPS(net='vgg', version='0.1').to(device) if lpips_coeff > 0 else None

        # graph conversion options
        self.conversion_kwargs = {
            'node_types_conv': None,
            'node_types_unconv': None,
            'sat_dir': None,
            'sbsrc_dir': None,
            'clamped': True,
            'param_quant_steps': 4096,
            'use_alpha': True,
            'output_res': 9,
            'output_usages': None,
            **conversion_kwargs
        }

        # graph rendering options
        self.rendering_kwargs = {
            'image_format': 'png',
            'output_usages': None,
            'engine': 'sse2',
            **rendering_kwargs
        }

        # get the current process rank
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # create a temporary directory for rendering
        if tmp_dir:
            self.tmp_dir = pth.join(tmp_dir, f'proc_{rank}')
            os.makedirs(self.tmp_dir, exist_ok=True)
        else:
            self.tmp_dir = '.'

        # create a directory for visualizing target and rendered images
        if vis_dir:
            self.vis_dir = pth.join(vis_dir, f'proc_{rank}')
            os.makedirs(self.vis_dir, exist_ok=True)
        else:
            self.vis_dir = None

        # create a pool of processes for rendering
        self.pool = Pool(num_render_procs) if num_render_procs > 1 else None

    def _render_graphs(self, ordered_nodes, device):
        ck, rk = self.conversion_kwargs, self.rendering_kwargs

        # convert ordered nodes to material graphs
        simple_graphs = [on.to_graph() for on in ordered_nodes]

        # options for unconverting parameters
        if ck['clamped']:
            unconvert_func = unconvert_clamped_simple_graph_parameters
        else:
            unconvert_func = unconvert_simple_graph_parameters

        # options for loading JSON files
        resource_dirs = {'sbs': pth.join(ck['sat_dir'], 'resources', 'packages'), 'sbsrc': ck['sbsrc_dir']}
        load_json_kwargs = {
            'resource_dirs': resource_dirs,
            'use_alpha': ck['use_alpha'],
            'res': [ck['output_res']] * 2,
            'output_usages': ck['output_usages'],
            'prune_inactive_nodes': False,
            'expand_unsupported_nodes': False,
            'expand_unsupported_fnodes': False,
            'allow_unsupported_nodes': True,
            'condition_active_node_params': False
        }

        # options for rendering
        render_worker_kwargs = {
            'sat_dir': ck['sat_dir'],
            'resource_dirs': resource_dirs,
            'output_usages': rk['output_usages'],
            'randomize_generators': True,
            'generators_only': False,
            'image_format': rk['image_format'],
            'center_normals': True,
            'engine': rk['engine'],
            'use_networkx': False,
            'write_output_channels': False
        }

        # unconvert the node parameters
        worker_args = []

        for i, graph in enumerate(simple_graphs):
            graph_filename = pth.join(self.tmp_dir, f'graph_{i:05d}')

            # dequantize the graph in JSON format
            json_nodes_quantized = graph.save_json(node_types=ck['node_types_conv'], use_alpha=ck['use_alpha'])
            json_nodes = unconvert_func(
                json_nodes_quantized, node_types=ck['node_types_unconv'], step_count=ck['param_quant_steps'],
                use_alpha=ck['use_alpha'])

            # pre-save worker arguments
            worker_args.append((json_nodes, graph_filename, load_json_kwargs, render_worker_kwargs))

        # render the SBS graphs
        # note that passing SBSGraph directly to the worker will cause resource leak
        if self.pool is not None:
            out_images = self.pool.starmap(render_worker, worker_args)
        else:
            worker_map = starmap(render_worker, worker_args)
            out_images = [img for img in worker_map]

        # fill in missing images
        if any(img is None for img in out_images):
            default_image = torch.zeros(1, 3, 2 ** ck['output_res'], 2 ** ck['output_res'])
            for i, img in enumerate(out_images):
                out_images[i] = default_image if img is None else img

        # delete temporary files
        for i in range(len(ordered_nodes)):
            for fn in os.listdir(self.tmp_dir):
                if fn.startswith(f'graph_{i:05d}'):
                    os.remove(pth.join(self.tmp_dir, fn))

        return torch.cat([img.to(device, non_blocking=True) for img in out_images], dim=0)

    def _get_mipmap(self, images, mipmap_level=4):
        # iteratively downsample the images
        if mipmap_level > 0:
            images = F.avg_pool2d(images, kernel_size=2 ** mipmap_level, stride=2 ** mipmap_level)

        return images

    def get_image_reward(self, rendered_images, images, reduce=True):
        # resize images to the same internal size
        interp_kwargs = {'size': self.img_size, 'mode': 'bicubic', 'antialias': True, 'align_corners': False}
        if images.shape[-2:] != (self.img_size, self.img_size):
            images = F.interpolate(images, **interp_kwargs).clamp_(0.0, 1.0)
        if rendered_images.shape[-2:] != (self.img_size, self.img_size):
            rendered_images = F.interpolate(rendered_images, **interp_kwargs).clamp_(0.0, 1.0)

        # constant max reward
        image_reward, stats = 1.0, {}

        # calculate VGG image loss
        if self.vgg_coeff > 0:
            image_ft = self.vgg_metric(images)
            rendered_image_ft = self.vgg_metric(rendered_images)
            vgg_loss = F.l1_loss(image_ft, rendered_image_ft, reduction='none').mean(dim=1)

            image_reward -= self.vgg_coeff * vgg_loss
            stats['vgg'] = vgg_loss.mean().item() if reduce else vgg_loss.tolist()

        # calculate sliced wasserstein loss
        if self.swd_coeff > 0:
            swd_loss = swd(images, rendered_images)
            image_reward -= self.swd_coeff * swd_loss
            stats['swd'] = swd_loss.mean().item() if reduce else swd_loss.tolist()

        # calculate downsampled loss
        if max(self.ds_l1_coeff, self.ds_lab_l1_coeff) > 0:
            image_ft = self._get_mipmap(images, mipmap_level=4)
            rendered_image_ft = self._get_mipmap(rendered_images, mipmap_level=4)

            if self.ds_l1_coeff > 0:
                ds_l1_loss = F.l1_loss(image_ft, rendered_image_ft, reduction='none').mean(dim=(1, 2, 3))
                image_reward -= self.ds_l1_coeff * ds_l1_loss
                stats['ds_l1'] = ds_l1_loss.mean().item() if reduce else ds_l1_loss.tolist()

            if self.ds_lab_l1_coeff > 0:
                lab_weights = torch.tensor(self.lab_weights, device=images.device).view(3, 1, 1)
                ds_lab_l1_loss = F.l1_loss(rgb_to_lab(image_ft), rgb_to_lab(rendered_image_ft), reduction='none')
                ds_lab_l1_loss = (ds_lab_l1_loss * lab_weights).mean(dim=(1, 2, 3))
                image_reward -= self.ds_lab_l1_coeff * ds_lab_l1_loss
                stats['ds_lab_l1'] = ds_lab_l1_loss.mean().item() if reduce else ds_lab_l1_loss.tolist()

        # calculate LPIPS image loss
        if self.lpips_coeff > 0:
            lpips_loss = self.lpips_metric(rendered_images * 2 - 1, images * 2 - 1)
            image_reward -= self.lpips_coeff * lpips_loss
            stats['lpips'] = lpips_loss.mean().item() if reduce else lpips_loss.tolist()

        # record image reward
        stats['image'] = image_reward.mean().item() if reduce else image_reward.tolist()

        return image_reward, stats

    def get_graph_reward(self, ordered_nodes, images, vis_dir=None, batch_label=None):
        # render material graphs
        device = images.device
        rendered_images = self._render_graphs(ordered_nodes, device)

        # save ground-truth and rendered images
        vis_dir = vis_dir or self.vis_dir
        if vis_dir and batch_label:
            vis_comp_file = pth.join(vis_dir, f'{batch_label}.jpg')
            vis_comp_batch_images(images, rendered_images, vis_comp_file)

        # calculate image reward
        return self.get_image_reward(rendered_images, images)

    def __call__(self, ordered_nodes, images, data, kl_div, batch_label=None):
        # render material graphs
        device = images.device
        image_reward, stats = self.get_graph_reward(ordered_nodes, images, batch_label=batch_label)

        # combine rewards and KL divergence
        reward = {k: -self.kl_coeff * v for k, v in kl_div.items()}
        last_key = list(reward.keys())[-1]

        inds_i = torch.arange(len(ordered_nodes), device=device)
        inds_j = data[f'{last_key}_seq_mask'].sum(dim=1) - 2
        reward[last_key][inds_i, inds_j] += image_reward

        # return stats
        masks = {k: data[f'{k}_seq_mask'][:, 1:] for k in kl_div}
        avg_kl = sum((v * masks[k]).sum().item() for k, v in kl_div.items())
        avg_kl /= sum(masks[k].count_nonzero().item() for k in kl_div)
        stats['kl_div'] = avg_kl

        return reward, stats

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def unbind_arr(arr):
    if isinstance(arr, torch.Tensor):
        arr_list = arr.unbind(0)
    elif isinstance(arr, np.ndarray):
        arr_list = [a.squeeze(0) for a in np.split(arr, arr.shape[0])]
    elif isinstance(arr, (list, tuple)):
        arr_list = arr
    elif isinstance(arr, dict):
        arr_list = {k: unbind_arr(v) for k, v in arr.items()}
    else:
        raise RuntimeError(f'Unsupported data type during unbinding: {type(arr).__name__}')

    return arr_list


def stack_arr(arr_list):
    if not isinstance(arr_list, (list, tuple)):
        raise TypeError(f'Expect input type to be list or tuple. Got {type(arr_list).__name__}.')
    if not arr_list:
        raise RuntimeError('Cannot stack empty list.')

    if isinstance(arr_list[0], torch.Tensor):
        arr = stack_tensor_lists(arr_list)
    elif isinstance(arr_list[0], dict):
        arr = {k: stack_arr([item[k] for item in arr_list]) for k in arr_list[0]}
    else:
        arr = list(arr_list)

    return arr


def extend_arr(arr, arr_ext):
    if isinstance(arr, list) and isinstance(arr_ext, (list, tuple)):
        arr.extend(arr_ext)
    elif isinstance(arr, dict) and isinstance(arr_ext, dict):
        for k, v in arr_ext.items():
            extend_arr(arr.setdefault(k, [] if isinstance(v, (list, tuple)) else {}), v)
    else:
        raise RuntimeError(f'Unsupported data types during extension: '
                           f'{type(arr).__name__} and {type(arr_ext).__name__}')


def segment_indices(traj_inds, state_inds, seq_lens):
    group_inds = {}
    keys = CondMatFormer.GENERATOR_TYPES

    # group indices by segments
    for i, j in zip(traj_inds, state_inds):
        for k in keys:
            if k not in seq_lens:
                continue
            lens = seq_lens[k]
            if j < lens[i]:
                group = group_inds.setdefault(k, [[], []])
                group[0].append(i)
                group[1].append(j)
            else:
                j -= lens[i]

    return group_inds


def index_batch(data, group_inds):
    data_arr = []

    # index the data array using grouped indices
    for k, (traj_inds, state_inds) in group_inds.items():
        if k not in data:
            continue
        arr = data[k]
        if isinstance(arr, (torch.Tensor, np.ndarray)):
            data_arr.append(arr[traj_inds, state_inds])
        else:
            data_arr.append(np.array([arr[i][j] for i, j in zip(traj_inds, state_inds)], dtype=np.float32))

    # concatenate the data
    if not data_arr:
        raise RuntimeError('No data in the current batch.')
    if isinstance(data_arr[0], torch.Tensor):
        data_arr = torch.cat(data_arr, dim=0)
    else:
        data_arr = np.concatenate(data_arr, axis=0)

    return data_arr


class BaseBuffer(ABC):
    """A rollout buffer stores trajectories for PPO training.
    """
    def __init__(self, buffer_size=8192, gamma=0.99, gae_lambda=0.95, seed=None, use_advantage=True, device='cpu'):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_advantage = use_advantage

        self.rng = np.random.default_rng(seed=seed)
        self.device = device

        # list of trajectories
        self.trajectories = {}
        self.num_samples = 0

    def reset(self):
        self.trajectories.clear()
        self.num_samples = 0

    def is_full(self):
        return self.num_samples >= self.buffer_size

    def _compute_returns(self, values, rewards, seq_lens):
        gamma, gae_lambda = self.gamma, self.gae_lambda

        # compute returns using discounted rewards
        if not self.use_advantage:
            # initialize discounted rewards
            returns = {k: np.zeros_like(v) for k, v in rewards.items()}
            batch_size = len(next(iter(returns.values())))
            next_rewards = np.zeros(batch_size)

            for seq_name in reversed(CondMatFormer.GENERATOR_TYPES):
                if seq_name not in values:
                    continue

                # compute returns using discounted rewards
                ret_seq, rew_seq = returns[seq_name], rewards[seq_name]
                seq_len = seq_lens[seq_name]

                for i in reversed(range(rew_seq.shape[1])):
                    mask = i < seq_len
                    ret_seq[:, i] = np.where(mask, rew_seq[:, i] + gamma * next_rewards, 0.0)
                    next_rewards[:] = np.where(mask, ret_seq[:, i], next_rewards)

            # use returns as advantages
            advantages = returns

        else:
            # initialize advantages
            advantages = {k: np.zeros_like(v) for k, v in rewards.items()}
            batch_size = len(next(iter(advantages.values())))
            next_values, next_advantages = np.zeros(batch_size), np.zeros(batch_size)

            for seq_name in reversed(CondMatFormer.GENERATOR_TYPES):
                if seq_name not in values:
                    continue

                # compute advantages using GAE
                adv_seq, val_seq, rew_seq = advantages[seq_name], values[seq_name], rewards[seq_name]
                seq_len = seq_lens[seq_name]

                for i in reversed(range(adv_seq.shape[1])):
                    delta = rew_seq[:, i] + gamma * next_values - val_seq[:, i]
                    mask = i < seq_len

                    # Only update the advantages for the valid samples
                    adv_seq[:, i] = np.where(mask, delta + gamma * gae_lambda * next_advantages, 0.0)
                    next_values[:] = np.where(mask, val_seq[:, i], next_values)
                    next_advantages[:] = np.where(mask, adv_seq[:, i], next_advantages)

            # compute returns and convert to tensors
            returns = {k: v + advantages[k] for k, v in values.items()}

        return advantages, returns

    @abstractmethod
    def add(self, data, log_probs, values, rewards, max_samples=None): ...

    @abstractmethod
    def sample(self, batch_size): ...


class RolloutBuffer(BaseBuffer):
    """Rollout buffer for storing trajectories for PPO training. The buffer generates mini-batches
    for training using transitions from the stored trajectories.
    """
    def add(self, data, log_probs, values, rewards, max_samples=None):
        # calculate trajectory lengths
        seq_lens = {key: data[f'{key}_seq_mask'].sum(dim=1).cpu().numpy() - 1 for key in log_probs}
        traj_lens = sum(seq_lens.values()).tolist()

        # calculate how many samples each trajectory will contribute
        total_samples = sum(traj_lens)
        target_samples = min(self.num_samples + (max_samples or self.buffer_size), self.buffer_size)

        if self.num_samples + total_samples <= target_samples:
            traj_samples, added_samples = traj_lens, total_samples
            self.num_samples += total_samples
        else:
            traj_samples = [0] * len(traj_lens)

            for i in range(max(traj_lens)):
                for j, traj_len in enumerate(traj_lens):
                    if traj_len > i and self.num_samples < target_samples:
                        traj_samples[j] += 1
                        self.num_samples += 1
                if self.num_samples >= target_samples:
                    break

            added_samples = sum(traj_samples)

        # convert tensors to numpy arrays
        log_probs = {k: v.cpu().numpy() for k, v in log_probs.items()}
        values = {k: v.cpu().numpy() for k, v in values.items()}
        rewards = {k: v.cpu().numpy() for k, v in rewards.items()}

        # compute advantages and returns
        advantages, returns = self._compute_returns(values, rewards, seq_lens)

        # move the data to the device
        data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        # split the data into trajectories
        traj_data = [unbind_arr(item) for item in (data, log_probs, values, advantages, returns, seq_lens, traj_samples)]
        traj_keys = ('data', 'old_log_probs', 'old_values', 'advantages', 'returns', 'seq_lens', 'samples')
        extend_arr(self.trajectories, dict(zip(traj_keys, traj_data)))

        print(f'Length of {len(traj_lens)} trajectories: avg = {np.array(traj_lens).mean():.1f}, max = {max(traj_lens)}.')

        return added_samples

    def sample(self, batch_size):
        trajs = self.trajectories

        # create trajectory indices for samples in the buffer
        N, pos = self.num_samples, 0
        traj_inds, state_inds = np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)

        for i, M in enumerate(trajs['samples']):
            traj_inds[pos:pos+M] = i
            state_inds[pos:pos+M] = np.arange(M)
            pos += M

        # shuffle the trajectory indices
        shuffle_inds = self.rng.permutation(N)
        traj_inds, state_inds = traj_inds[shuffle_inds], state_inds[shuffle_inds]

        # sample minibatches from trajectories
        for start_ind in range(0, N, batch_size):
            # group indices by segments
            batch_traj_inds = traj_inds[start_ind:start_ind+batch_size]
            batch_state_inds = state_inds[start_ind:start_ind+batch_size]
            batch_inds = segment_indices(batch_traj_inds, batch_state_inds, trajs['seq_lens'])

            # get unique trajectories and build grouped indices
            unique_traj_inds = sorted(set(batch_traj_inds.tolist()))
            unique_traj_mapping = {i: j for j, i in enumerate(unique_traj_inds)}

            batch_data_inds = copy.deepcopy(batch_inds)
            for g in batch_data_inds.values():
                g[0] = [unique_traj_mapping[i] for i in g[0]]

            # fetch data unique trajectories and compose the batch
            batch_data = {k: stack_arr([v[i] for i in unique_traj_inds]) for k, v in trajs['data'].items()}

            # construct the batch helper function
            def _batch_helper(log_probs, values, entropy):
                return {
                    'log_probs': index_batch(log_probs, batch_data_inds),
                    'values': index_batch(values, batch_data_inds),
                    'entropy': index_batch(entropy, batch_data_inds),
                    **{k: torch.from_numpy(index_batch(trajs[k], batch_inds)).to(self.device, non_blocking=True)
                       for k in ('old_log_probs', 'old_values', 'advantages', 'returns')}
                }

            # yield the batch
            yield batch_data, _batch_helper


class TrajectoryBuffer(BaseBuffer):
    """Trajectory buffer for storing trajectories for PPO training. The buffer generates mini-batches of
    stored trajectories for training.
    """
    def add(self, data, log_probs, values, rewards, max_samples=None):
        # calculate trajectory lengths
        seq_lens = {key: data[f'{key}_seq_mask'].sum(dim=1).cpu().numpy() - 1 for key in log_probs}

        # calculate how many samples each trajectory will contribute
        total_samples = len(next(iter(log_probs.values())))
        target_samples = min(self.num_samples + (max_samples or self.buffer_size), self.buffer_size)
        added_samples = min(total_samples, target_samples - self.num_samples)

        if not added_samples:
            return 0

        self.num_samples += added_samples

        # truncate the trajectories
        if added_samples < total_samples:
            data, log_probs, values, rewards, seq_lens = tuple(
                {k: v[:added_samples] for k, v in d.items()}
                for d in (data, log_probs, values, rewards, seq_lens))

        # convert tensors to numpy arrays
        values_np = {k: v.cpu().numpy() for k, v in values.items()}
        rewards_np = {k: v.cpu().numpy() for k, v in rewards.items()}

        # compute advantages and returns
        advantages_np, returns_np = self._compute_returns(values_np, rewards_np, seq_lens)

        # move the data to the device
        data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        log_probs = {k: v.to(self.device, non_blocking=True) for k, v in log_probs.items()}
        advantages = {k: torch.from_numpy(v).to(self.device, non_blocking=True) for k, v in advantages_np.items()}
        returns = {k: torch.from_numpy(v).to(self.device, non_blocking=True) for k, v in returns_np.items()}

        # split the data into trajectories
        traj_data = [unbind_arr(item) for item in (data, log_probs, values, advantages, returns)]
        traj_keys = ('data', 'old_log_probs', 'old_values', 'advantages', 'returns')
        extend_arr(self.trajectories, dict(zip(traj_keys, traj_data)))

        # print average trajectory length
        traj_lens = sum(seq_lens.values())
        print(f'Length of {added_samples} trajectories: avg = {traj_lens.mean():.1f}, max = {traj_lens.max()}.')

        return added_samples

    def sample(self, batch_size):
        trajs, N = self.trajectories, self.num_samples

        # shuffle the trajectory indices
        traj_inds = self.rng.permutation(N)

        # sample minibatches from trajectories
        for start_ind in range(0, N, batch_size):

            # compose the batch using the selected trajectory indices
            batch_traj_inds = traj_inds[start_ind:start_ind+batch_size]
            batch_data, old_log_probs, old_values, advantages, returns = tuple(
                {k: stack_arr([v[i] for i in batch_traj_inds]) for k, v in trajs[f].items()}
                for f in ('data', 'old_log_probs', 'old_values', 'advantages', 'returns'))

            # get the batch sequence masks
            batch_masks = {k: batch_data[f'{k}_seq_mask'][:, 1:].bool() for k in old_log_probs}
            mask_batch = lambda d: torch.cat([d[k][m] for k, m in batch_masks.items() if k in d], dim=0)

            # construct the batch helper function
            def _batch_helper(log_probs, values, entropy):
                # check the changes in log probabilities
                with torch.no_grad():
                    invalid_inds = {k: torch.where((log_probs[k] - old_log_probs[k]) * batch_masks[k] > 10.0)
                                    for k in batch_masks}

                    # print warning if significant changes are detected
                    if any(len(v[0]) > 0 for v in invalid_inds.values()):
                        print('Warning: detected significant changes in log probabilities. See details below:')

                        for k, v in invalid_inds.items():
                            if len(v[0]) > 0:
                                print(f"  - Sequence '{k}': ", end='')
                                details = []
                                for i, j in zip(*(l.tolist() for l in v)):
                                    details.append(f'[{i}, {j}] ({old_log_probs[k][i, j].item():.4g} -> {log_probs[k][i, j].item():.4g})')
                                print(', '.join(details))

                return {
                    'log_probs': mask_batch(log_probs),
                    'values': mask_batch(values),
                    'entropy': mask_batch(entropy),
                    'old_log_probs': mask_batch(old_log_probs),
                    'old_values': mask_batch(old_values),
                    'advantages': mask_batch(advantages),
                    'returns': mask_batch(returns)
                }

            # yield the batch
            yield batch_data, _batch_helper


class RLDataset:
    """Iterable dataset for PPO. Samples trajectories from input images.
    """
    def __init__(self, source_dataloader, source_data_features, policy_model, value_model, ref_policy_model, reward_function,
                 trainable=None, fix_nontrainable=False, buffer_type='rollout', buffer_size=8192, batch_size=64,
                 gen_batch_size=32, gamma=0.99, gae_lambda=0.95, use_advantage=True, seed=None, device='cpu'):
        super().__init__()

        self.source_dataloader = source_dataloader              # source dataloader for input images
        self.source_data_features = source_data_features        # features to extract from the source dataloader
        self.policy_model = policy_model                        # policy network
        self.value_model = value_model                          # value network
        self.reward_fn = reward_function                        # reward function

        self.trainable = trainable                              # trainable generators
        self.fix_nontrainable = fix_nontrainable                # non-trainable generators behave deterministically
        self.batch_size = batch_size                            # mini-batch size for training
        self.gen_batch_size = gen_batch_size                    # batch size for generating trajectories
        self.device = device

        # reference policy network
        self.ref_policy_model = ref_policy_model if ref_policy_model is not None else copy.deepcopy(policy_model)

        # iterator for the dataloader
        self.source_iter = iter(self.source_dataloader)

        # buffer for storing trajectories
        buffer_kwargs = dict(buffer_size=buffer_size, gamma=gamma, gae_lambda=gae_lambda, seed=seed, device=device,
                             use_advantage=use_advantage)

        if buffer_type == 'rollout':
            self.buffer = RolloutBuffer(**buffer_kwargs)
        elif buffer_type == 'trajectory':
            self.buffer = TrajectoryBuffer(**buffer_kwargs)
        else:
            raise ValueError(f'Unsupported buffer type: {buffer_type}')

    def _extend_data(self, data):
        # already reaches the batch size for trajectory generation
        target_size = self.gen_batch_size
        if len(data[0]) >= target_size:
            return data

        # data shuffling is handled by the dataloader, so we can just repeat the data
        num_repeats = math.ceil(target_size / len(data[0]))

        for i, value in enumerate(data):
            if isinstance(value, torch.Tensor):
                value = value.repeat(num_repeats, *([1] * (value.ndim - 1)))
            elif isinstance(value, (list, tuple)):
                ext_value = list(value)
                # must use deepcopy to make sure each data is unique
                while len(ext_value) < target_size:
                    ext_value.extend(copy.deepcopy(value))
                value = ext_value
            else:
                raise RuntimeError(f'Unsupported data type during extension: {type(value).__name__}')

            data[i] = value[:target_size] if len(value) > target_size else value

        return data

    def _sample_trajectories(self, data, max_samples=None, batch_label=None):
        # determine which components generate sequences deterministically
        if self.trainable is not None and self.fix_nontrainable:
            non_trainable = [k for k in self.policy_model.GENERATOR_TYPES
                             if self.policy_model.has_generator(k) and k not in self.trainable]
            deterministic = non_trainable or False
        else:
            deterministic = False

        # generate trajectories using the policy network
        ordered_nodes, outputs = self.policy_model.generate(data, return_sequences=True, deterministic=deterministic)

        # update the data with the CLIP embedding and generated token sequences
        images = data['prerendered']
        data = {'prerendered': outputs.pop('clip_embedding')}
        for (seqs,) in outputs.values():
            data.update(seqs)

        # recompute logits using the policy network and the reference policy network
        eval_kwargs = {'gen_types': self.trainable, 'bypass_clip': True}

        with torch.no_grad():
            raw_log_probs, log_probs = self.policy_model.get_log_probs(data, **eval_kwargs)[:2]
            ref_log_probs = self.ref_policy_model.get_log_probs(data, **eval_kwargs)[0]
            values = self.value_model.get_values(data, **eval_kwargs)

        # compute KL divergence
        kl_div = {k: raw_log_probs[k] - ref_log_probs[k] for k in log_probs}

        # compute final rewards by rendering the generated material graphs
        rewards, reward_stats = self.reward_fn(ordered_nodes, images, data, kl_div, batch_label=batch_label)

        # add trajectories to the buffer
        num_samples = self.buffer.add(data, log_probs, values, rewards, max_samples=max_samples)

        return num_samples, reward_stats

    def reset_buffer(self):
        self.buffer.reset()

    def refill_buffer(self, max_samples=None, buffer_label=None):
        num_batches, num_samples = 0, 0
        reward_stats = {}
        args = Namespace()  # dummy args

        # reset the buffer
        self.reset_buffer()

        # fill the buffer with trajectories
        while not self.buffer.is_full():

            # get the next batch of source data
            try:
                data = next(self.source_iter)
            except StopIteration:
                self.source_iter = iter(self.source_dataloader)
                data = next(self.source_iter)

            # repeat the data to match the batch size for trajectory generation
            data = self._extend_data(data)

            # sample trajectories using the current policy
            data = prepare_batch(args, data, self.source_data_features, device=self.device)
            batch_label = f'{buffer_label}_batch_{num_batches:03d}' if buffer_label else None

            num_new_samples, batch_reward_stats = self._sample_trajectories(
                data, max_samples=max_samples - num_samples if max_samples else None,
                batch_label=batch_label)

            # report progress
            num_batches += 1
            num_samples += num_new_samples

            other_losses = ''
            if any(k not in ('kl_div', 'image') for k in batch_reward_stats):
                other_losses = ', '.join(f'{k} = {v:.4f}' for k, v in batch_reward_stats.items()
                                         if k not in ('kl_div', 'image'))
                other_losses = f' ({other_losses})'

            percentage = 100 * num_samples / self.buffer.buffer_size
            print(f'Generated {num_new_samples} samples. '
                  f'KL div = {batch_reward_stats["kl_div"]:.4f}, image reward = {batch_reward_stats["image"]:.4f}{other_losses}. '
                  f'[Total: {num_samples} samples ({percentage:.1f} %)]')

            # update reward stats
            for k, v in batch_reward_stats.items():
                reward_stats.setdefault(k, 0)
                reward_stats[k] += v

            # clear device memory cache
            # if self.device.startswith('cuda'):
            #     torch.cuda.empty_cache()

            # exit if the required number of samples is reached
            if max_samples and num_samples >= max_samples:
                break

        # normalize reward stats
        for k in reward_stats:
            reward_stats[k] /= num_batches

        return num_samples, reward_stats

    def __len__(self):
        return math.ceil(self.buffer.num_samples / self.batch_size)

    def __iter__(self):
        # Check if the buffer is empty
        if self.buffer.num_samples == 0:
            raise RuntimeError('The rollout buffer is empty.')

        # generate batches from the rollout buffer
        yield from self.buffer.sample(self.batch_size)
