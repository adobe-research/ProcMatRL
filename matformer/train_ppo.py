# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from argparse import ArgumentParser
import json
import os
import os.path as pth
import math
import multiprocessing as mp
import sys
import warnings

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import spawn
from torch.distributed import (
    init_process_group, destroy_process_group, get_rank, get_world_size,
    reduce, all_reduce, ReduceOp
)
import torch

from .eval_image_cond_generators import ModelBuilder, get_model_args, load_dataset
from .rl_dataset import RLDataset, RewardFunction
from .sequencer.sequences import convert_node_types
from .utils import load_node_types, load_optim_state, save_model_state, save_optim_state, prepare_batch


# limit each SAT instance to a single thread
os.environ['OMP_NUM_THREADS'] = '1'

# disable warnings
warnings.filterwarnings('ignore', category=UserWarning)


def loss_ppo(args, log_probs, values, entropy, old_log_probs, old_values, returns, advantages):
    # normalize advantage
    if args.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # policy loss
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
    policy_loss = torch.min(surr1, surr2).mean()

    # value loss
    if args.clip_range_vf is not None:
        values = old_values + torch.clamp(values - old_values, -args.clip_range_vf, args.clip_range_vf)
    value_loss = torch.nn.functional.mse_loss(values, returns)

    # entropy loss
    entropy_loss = entropy.mean()

    # total loss and loss dictionary
    total_loss = value_loss * args.value_coeff - policy_loss - entropy_loss * args.entropy_coeff
    loss_dict = {
        'total': total_loss.detach().item(),
        'policy': policy_loss.detach().item(),
        'value': value_loss.detach().item(),
        'entropy': entropy_loss.detach().item()
    }

    # approximate KL divergence
    with torch.no_grad():
        approx_kl_div = (ratio - 1 - log_ratio).mean().item()

    return total_loss, loss_dict, approx_kl_div


def eval_policy(args, dataloader, dataset_features, policy_model, reward_fn, valid_dir, valid_label=None, max_samples=None):
    # create validation directory
    os.makedirs(valid_dir, exist_ok=True)

    # estimate the number of batches
    max_samples = len(dataloader.dataset) if max_samples is None else min(max_samples, len(dataloader.dataset))
    num_batches = math.ceil(max_samples / dataloader.batch_size)

    # start validation loop
    tot_reward, num_samples = 0.0, 0
    device = args.devices[0]
    pbar = tqdm(dataloader, desc='Validating policy', total=num_batches, disable=args.disable_tqdm)

    for i, data in enumerate(dataloader):
        # truncate batch if necessary
        if max_samples is not None and max_samples - num_samples < len(data[0]):
            for k, v in enumerate(data):
                data[k] = v[:max_samples - num_samples]

        # evaluate policy
        data = prepare_batch(args, data, dataset_features, device)
        ordered_nodes, _ = policy_model.generate(data, deterministic=True)

        reward, _ = reward_fn.get_graph_reward(
            ordered_nodes, data['prerendered'], vis_dir=valid_dir,
            batch_label=f'{valid_label}_batch_{i:03d}' if valid_label is not None else None)

        # count reward and samples
        tot_reward += sum(reward.tolist())
        num_samples += len(next(iter(data.values())))

        # update progress bar
        pbar.update(1)

        # break if the maximum number of samples is reached
        if max_samples is not None and num_samples >= max_samples:
            break

    pbar.close()

    # print validation statistics
    avg_reward = tot_reward / num_samples
    print(f'Validation statistics: image reward = {avg_reward:.4g}')

    return avg_reward


# reduce statistics to the main process
def reduce_stats(stats, device, is_main_rank=False):
    new_stats = {}
    for k, v in stats.items():
        v_t = torch.tensor(v, device=device)
        reduce(v_t, dst=0)
        new_stats[k] = v_t.item() / get_world_size() if is_main_rank else v

    return new_stats


def train_ppo(args):
    # enable TF32 precision optimization
    if args.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

    # set training device
    rank = get_rank() if args.distributed else 0
    device, is_main_rank = args.devices[rank], not rank

    # build policy and value models
    model_args = get_model_args(args)
    build_model = ModelBuilder(model_args, device=device, train_from_scratch=args.train_from_scratch)

    policy_model = build_model(distributed=args.distributed)
    ref_policy_model = build_model(bypass_clip=True, requires_grad=False)
    value_model = build_model(is_value_network=True, bypass_clip=True, distributed=args.distributed)

    # synchronize policy network parameters
    if args.train_from_scratch:
        policy_state = save_model_state(policy_model)
        ref_policy_model.load_from_checkpoint(policy_state, device)
        value_model.load_from_checkpoint(policy_state, device, exclude_lm_head=True)

    # compile dataset features
    dataset_features = ['name', 'prerendered']

    # parameter only mode requires ground-truth uninitialized nodes
    if 'param' in args.eval_modes and 'gen' not in args.eval_modes:
        dataset_features += ['nodes', 'node_depths']

    print('Dataset features:', dataset_features)

    # load image dataset
    args.valid_dataset = args.valid_dataset or args.train_dataset
    train_dataloader = load_dataset(
        model_args, args.custom_data_dir, args.train_dataset, dataset_features, args.gen_batch_size, *build_model.sequencers,
        real_data_list=args.real_train_dataset, real_data_ratio=args.real_data_ratio, is_map_dataset=True)
    valid_dataloader = load_dataset(
        model_args, args.custom_data_dir, args.valid_dataset, dataset_features, args.gen_batch_size, *build_model.sequencers,
        real_data_list=args.real_valid_dataset, real_data_ratio=args.real_data_ratio, is_map_dataset=True,
        shuffle=False, pre_shuffle=True, distributed=False)

    # create result folder
    result_dir = pth.join(args.result_dir, args.exp_name)
    if args.eval_suffix is not None:
        result_dir += args.eval_suffix
    os.makedirs(result_dir, exist_ok=True)

    # build reward function using CLIP encoder
    tmp_dir = pth.join(result_dir, 'tmp')
    vis_dir = pth.join(result_dir, 'vis_comp') if is_main_rank else None
    valid_dir = pth.join(result_dir, 'valid')

    node_types = load_node_types(args.custom_node_type_list)
    node_types_conv = convert_node_types(node_types)
    conversion_kwargs = {
        'node_types_conv': node_types_conv,
        'node_types_unconv': node_types,
        'sat_dir': args.sat_dir,
        'sbsrc_dir': args.sbsrc_dir,
        'clamped': model_args['param'].node_param_quant_steps != 4096,
        'param_quant_steps': model_args['param'].node_param_quant_steps,
    }
    reward_fn = RewardFunction(
        vgg_coeff=args.vgg_coeff, vgg_td_level=args.vgg_td_level, ds_l1_coeff=args.ds_l1_coeff, ds_lab_l1_coeff=args.ds_lab_l1_coeff,
        lpips_coeff=args.lpips_coeff, swd_coeff=args.swd_coeff, kl_coeff=args.kl_coeff, lab_weights=args.lab_weights,
        tmp_dir=tmp_dir, vis_dir=vis_dir, num_render_procs=args.num_render_procs, conversion_kwargs=conversion_kwargs, device=device)

    # build the RL dataset
    buffer_kwargs_fields = ['trainable', 'fix_nontrainable', 'buffer_type', 'buffer_size', 'batch_size', 'gen_batch_size',
                            'gamma', 'gae_lambda', 'use_advantage', 'seed']
    rl_dataset = RLDataset(
        train_dataloader, dataset_features, policy_model, value_model, ref_policy_model, reward_fn, device=device,
        **{k: getattr(args, k) for k in buffer_kwargs_fields})

    # build the optimizer
    all_parameters = [*policy_model.parameters(), *value_model.parameters()]
    optimizer = torch.optim.AdamW(all_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # create tensorboard writer
    log_dir = pth.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir) if is_main_rank else None

    # save training configuration
    with open(pth.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # load from checkpoint
    load_from_dir = args.load_from_dir or result_dir
    start_epoch, buffers_per_epoch = 0, math.ceil(args.epoch_data_size / args.buffer_size)

    if args.load_from_epoch is not None:
        policy_model.load_from_checkpoint(pth.join(load_from_dir, f'{args.load_from_epoch}_policy.pth'), device)
        value_model.load_from_checkpoint(pth.join(load_from_dir, f'{args.load_from_epoch}_value.pth'), device)

        # fine-tune mode, also load the reference policy model
        if args.reset_after_load:
            ref_policy_model.load_from_checkpoint(pth.join(load_from_dir, f'{args.load_from_epoch}_policy.pth'), device)
        else:
            load_epoch, _ = load_optim_state(optimizer, None, pth.join(load_from_dir, f'{args.load_from_epoch}_optim.pth'))
            start_epoch = load_epoch + 1

    # initial validation
    eval_policy_with_label = lambda label: eval_policy(
        args, valid_dataloader, dataset_features, policy_model, reward_fn, valid_dir,
        valid_label=label, max_samples=args.num_valid_samples)

    if is_main_rank and args.initial_valid:
        print('Performing initial validation...')
        valid_reward = eval_policy_with_label('initial' if start_epoch == 0 else f'epoch_{start_epoch-1:03d}')
        best_reward = valid_reward
        writer.add_scalar('reward/image_valid', valid_reward, global_step=start_epoch * buffers_per_epoch)
    else:
        best_reward = -math.inf

    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'************* [{args.exp_name}] epoch {epoch} *************')

        # reset buffer and sample counters
        num_buffers, num_samples = 0, 0

        # sample data until reaching the maximum value
        while num_samples < args.epoch_data_size:
            if is_main_rank and not num_buffers:
                buffer_label = f'epoch_{epoch:03d}_buffer_{num_buffers:03d}'
            else:
                buffer_label = None

            # fill the rollout buffer
            num_new_samples, reward_stats = rl_dataset.refill_buffer(args.epoch_data_size - num_samples, buffer_label=buffer_label)
            num_samples += num_new_samples
            print(f'Buffer refilled. Epoch progress: {num_samples}/{args.epoch_data_size}')

            # try to free up memory
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            # reduce the reward statistics to the main process
            if args.distributed:
                reward_stats = reduce_stats(reward_stats, device, is_main_rank)

            # log training reward
            if is_main_rank:
                for k in ['image', 'kl_div']:
                    writer.add_scalar(f'reward/{k}_train', reward_stats[k],
                                      global_step=epoch * buffers_per_epoch + num_buffers + 1)

            # reset early stopping switch
            kl_stopping_switch = False

            # set the models to training mode
            if args.set_training_mode:
                policy_model.train()
                value_model.train()

            # train using the rollout buffer
            for buffer_epoch in range(args.buffer_epochs):
                # initialize buffer training statistics
                buffer_train_stats = dict(zip(['total', 'policy', 'value', 'entropy', 'approx_kl_div'],
                                              [[] for _ in range(5)]))

                # iterate over mini-batches from the rollout buffer
                for batch, batch_helper in rl_dataset:

                    # compute log probabilities and values
                    eval_kwargs = {'gen_types': args.trainable, 'bypass_clip': True}
                    log_probs, entropy = policy_model.get_log_probs(batch, **eval_kwargs)[1:]
                    values = value_model.get_values(batch, **eval_kwargs)

                    # construct the actual batch data and compute loss
                    loss_kwargs = batch_helper(log_probs, values, entropy)
                    loss, loss_dict, approx_kl_div = loss_ppo(args, **loss_kwargs)

                    # reduce the approximate KL divergence across processes by maximum
                    if args.distributed:
                        approx_kl_div_t = torch.tensor(approx_kl_div, device=device)
                        all_reduce(approx_kl_div_t, op=ReduceOp.MAX)
                        approx_kl_div = approx_kl_div_t.item()

                    # early stopping if the KL divergence is too large
                    if approx_kl_div > args.target_kl * 1.5:
                        print(f'Early stopping at buffer epoch {buffer_epoch} due to reaching max KL divergence (approx. KL = {approx_kl_div:.4f}).')
                        kl_stopping_switch = True
                        break

                    # update buffer training statistics
                    for k, v in loss_dict.items():
                        buffer_train_stats[k].append(v)
                    buffer_train_stats['approx_kl_div'].append(approx_kl_div)

                    # compute gradients
                    optimizer.zero_grad()
                    loss.backward()
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(all_parameters, args.grad_clip_norm)
                    optimizer.step()

                # average buffer training statistics
                # reduce the buffer training statistics to the main process in distributed training
                buffer_train_stats = {k: sum(v) / max(len(v), 1) for k, v in buffer_train_stats.items()}
                if args.distributed:
                    buffer_train_stats = reduce_stats(buffer_train_stats, device, is_main_rank)

                # log buffer training statistics
                stat_str = ', '.join([f'{k} = {v:.4g}' for k, v in buffer_train_stats.items()])
                print(f'Buffer epoch {buffer_epoch + 1}/{args.buffer_epochs}: {stat_str}')

                # force write-out
                sys.stdout.flush()

                # early stopping if the KL divergence is too large
                if kl_stopping_switch:
                    break

            num_buffers += 1

            # set the models to evaluation mode
            if args.set_training_mode:
                policy_model.eval()
                value_model.eval()

            # try to free up memory
            rl_dataset.reset_buffer()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            if is_main_rank:
                # log buffer training statistics
                if buffer_train_stats['total']:
                    for k in ['policy', 'value', 'entropy']:
                        writer.add_scalar(f'loss/{k}_train', buffer_train_stats[k],
                                          global_step=epoch * buffers_per_epoch + num_buffers)

                # validate the model
                if num_buffers % args.valid_interval == 0 or num_buffers == buffers_per_epoch:
                    print('Performing validation...')
                    valid_reward = eval_policy_with_label(f'epoch_{epoch:03d}_buffer_{num_buffers:03d}')
                    writer.add_scalar('reward/image_valid', valid_reward, global_step=epoch * buffers_per_epoch + num_buffers)

                # save the model if it is the best so far
                if valid_reward > best_reward:
                    best_reward = valid_reward
                    save_model_state(policy_model, pth.join(result_dir, 'best_policy.pth'))
                    save_model_state(value_model, pth.join(result_dir, 'best_value.pth'))
                    save_optim_state(epoch, num_buffers, optimizer, None, pth.join(result_dir, 'best_optim.pth'))

            # force write-out
            sys.stdout.flush()

        # save the model
        if is_main_rank and (epoch + 1) % args.save_interval == 0:
            save_model_state(policy_model, pth.join(result_dir, f'{epoch}_policy.pth'))
            save_model_state(value_model, pth.join(result_dir, f'{epoch}_value.pth'))
            save_optim_state(epoch, num_buffers, optimizer, None, pth.join(result_dir, f'{epoch}_optim.pth'))


def launch_distributed(rank, args):
    # initialize distributed training
    port_name = ''.join(chr(c) for c in [77, 65, 83, 84, 69, 82])
    os.environ[f'{port_name}_ADDR'] = 'localhost'
    os.environ[f'{port_name}_PORT'] = str(args.ddp_port)

    init_process_group(backend='nccl', rank=rank, world_size=len(args.devices))

    # disable printing from non-main ranks
    if rank:
        text_io = open(os.devnull, 'w')
        sys.stdout, sys.stderr = text_io, text_io

    # run training
    train_ppo(args)

    destroy_process_group()


def create_arg_parser():
    # set up argument parser
    p = ArgumentParser(description='Fine-tune conditional MatFormer using PPO')

    # I/O related
    p.add_argument('--config', default=None, type=str, help='Path to a config file.')
    p.add_argument('--custom_data_dir', default=None, type=str, help='Path to data directory.')
    p.add_argument('--custom_node_type_list', default=None, type=str, help='Path to node type list.')
    p.add_argument('--train_dataset', default=None, type=str, help='Path to training dataset.')
    p.add_argument('--valid_dataset', default=None, type=str, help='Path to validation dataset.')
    p.add_argument('--model_dir', default=None, type=str, help='Path to models directory for all experiments.')
    p.add_argument('--log_dir', default=None, type=str, help='Path to log directory.')
    p.add_argument('--result_dir', default=None, type=str, help='Path to results directory.')
    p.add_argument('--sat_dir', default=None, type=str, help='Location of the Substance Automation Toolkit.')
    p.add_argument('--sbsrc_dir', default=None, type=str, help='Location of the directory containing substance source.')

    # real image dataset
    p.add_argument('--real_image_dir', default=None, type=str, help='Path to real image directory.')
    p.add_argument('--real_train_dataset', default=None, type=str, help='Path to real image training dataset.')
    p.add_argument('--real_valid_dataset', default=None, type=str, help='Path to real image validation dataset.')
    p.add_argument('--real_data_ratio', default=1.0, type=float, help='Ratio of real images in the training dataset.')
    p.add_argument('--real_nn_file', default=None, type=str, help='Path to real nearest neighbor file.')
    p.add_argument('--real_augment_image', default=None, type=str, help='Whether to augment real images.')
    p.add_argument('--real_augment_graph', default=0, type=int, help='Number of nearest neighbor graph structures considered for real images.')
    p.add_argument('--image_ext', default=None, type=str, help='Image file extension.')

    # experiment names and checkpoints
    p.add_argument('--exp_name', default=None, type=str, help='Name of the RL fine-tuning experiment.')
    p.add_argument('--eval_suffix', type=str, default=None, help='Suffix to add to the experiment name that describes the evaluation details.')
    p.add_argument('--param_exp_name', default=None, type=str, help='Name of the parameter generator experiment.')
    p.add_argument('--param_model_step', default=None, help='Training step of the parameter generator model to load.')
    p.add_argument('--train_from_scratch', default=False, action='store_true', help='Whether to train from scratch.')

    # model configurations
    p.add_argument('--eval_modes', default=['param'], nargs='+', choices=['param'], help='Modes to evaluate.')
    p.add_argument('--trainable', default=['param'], nargs='+', choices=['param'], help='Trainable components.')
    p.add_argument('--fix_nontrainable', default=False, action='store_true', help='Whether to let non-trainable components predict deterministically.')
    p.add_argument('--semantic_validate', default=False, action='store_true', help='Whether to perform semantic validation.')
    p.add_argument('--devices', default=['cpu'], nargs='+', help='Devices to use for evaluation.')
    p.add_argument('--distributed', default=False, action='store_true', help='Whether to use distributed training.')
    p.add_argument('--ddp_port', default=12355, type=int, help='Port for distributed training.')
    p.add_argument('--allow_tf32', default=False, action='store_true', help='Whether to allow TF32.')
    p.add_argument('--use_fast_attn', default=False, action='store_true', help='Whether to use fast attention.')
    p.add_argument('--max_gen_nodes', default=None, type=int, help='Maximum number of nodes allowed to generate in one graph.')

    # dataset
    p.add_argument('--use_alpha', default=False, action='store_true',help='If using alpha channel.')
    p.add_argument('--data_chunksize', default=128, type=int, help='Chunksize for coalesced access to HDF5 dataset.')

    # replay buffer
    p.add_argument('--buffer_type', default='rollout', choices=['rollout', 'trajectory'], help='Type of the replay buffer.')
    p.add_argument('--buffer_size', default=8192, type=int, help='Size of the rollout buffer.')
    p.add_argument('--gen_batch_size', default=16, type=int, help='Batch size for trajectory generation.')
    p.add_argument('--gamma', default=0.99, type=float, help='Discount factor.')
    p.add_argument('--gae_lambda', default=0.95, type=float, help='Lambda for GAE.')
    p.add_argument('--use_advantage', default=True, action='store_true', help='Whether to use advantage instead of return.')
    p.add_argument('--temperature', default=1.0, type=float, help='Temperature of softmax')
    p.add_argument('--prob_k', default=0, type=int, help='Sample from top-k probability')
    p.add_argument('--nucleus_top_p', default=None, type=float, help='Nucleus sampling top-p probability.')

    # reward function
    p.add_argument('--vgg_coeff', default=1.0, type=float, help='Coefficient for the VGG loss term in the reward function.')
    p.add_argument('--vgg_td_level', default=2, type=int, help='Mipmap pyramid layers to use for the VGG texture descriptor.')
    p.add_argument('--ds_l1_coeff', default=0.0, type=float, help='Coefficient for the downsampled L1 loss term in the reward function.')
    p.add_argument('--ds_lab_l1_coeff', default=0.0, type=float, help='Coefficient for the downsampled LAB L1 loss term in the reward function.')
    p.add_argument('--lpips_coeff', default=0.0, type=float, help='Coefficient for the LPIPS loss term in the reward function.')
    p.add_argument('--swd_coeff', default=0.0, type=float, help='Coefficient for the SWD loss term in the reward function.')
    p.add_argument('--kl_coeff', default=0.001, type=float, help='Coefficient for the KL divergence term in the loss function.')
    p.add_argument('--lab_weights', default=[0.2, 1.0, 1.0], nargs=3, type=float, help='Weights for the LAB loss term in the reward function.')
    p.add_argument('--clip_range', default=0.2, type=float, help='Clip range for PPO.')
    p.add_argument('--clip_range_vf', default=None, type=float, help='Clip range for value function.')
    p.add_argument('--value_coeff', default=0.5, type=float, help='Coefficient for the value function term in the loss function.')
    p.add_argument('--entropy_coeff', default=0.01, type=float, help='Coefficient for the entropy term in the loss function.')
    p.add_argument('--num_render_procs', default=1, type=int, help='Number of processes for rendering.')

    # training
    p.add_argument('--epochs', default=100, type=int, help='Number of epochs to train for.')
    p.add_argument('--save_interval', default=5, type=int, help='Number of epochs between saving the model.')
    p.add_argument('--initial_valid', default=False, action='store_true', help='Whether to validate the model before training.')
    p.add_argument('--num_valid_samples', default=64, type=int, help='Number of samples to use for validation.')
    p.add_argument('--valid_interval', default=4, type=int, help='Number of epochs between validation.')
    p.add_argument('--buffer_epochs', default=10, type=int, help='Number of epochs to train using the rollout buffer.')
    p.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
    p.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay.')
    p.add_argument('--target_kl', default=0.2, type=float, help='Target KL divergence.')
    p.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    p.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading.')
    p.add_argument('--target_shuffle_queue_size', default=512, type=int, help='Size of the target shuffle queue.')
    p.add_argument('--epoch_data_size', default=131072, type=int, help='Number of samples to use per epoch.')
    p.add_argument('--normalize_advantage', default=False, action='store_true', help='Whether to normalize the advantage scores in each minibatch.')
    p.add_argument('--grad_clip_norm', default=None, type=float, help='Gradient clipping norm.')
    p.add_argument('--seed', default=414646787, type=int, help='Random seed.')
    p.add_argument('--load_from_epoch', default=None, type=str, help='Epoch to load from.')
    p.add_argument('--load_from_dir', default=None, type=str, help='Directory to load from.')
    p.add_argument('--reset_after_load', default=False, action='store_true', help='Whether to reset the epoch counter.')
    p.add_argument('--disable_tqdm', default=False, action='store_true', help='Whether to disable tqdm progress bar.')
    p.add_argument('--set_training_mode', default=False, action='store_true', help='Whether to set the model to training mode during policy update.')

    return p


def main():
    # setup multiprocessing
    mp.set_start_method('spawn')

    # parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # reset defaults from config file
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)

        # check unrecognized arguments
        if any(json_name not in vars(args).keys() for json_name in config.keys()):
            unrecognized_args = [json_name for json_name in config.keys() if json_name not in vars(args).keys()]
            raise RuntimeError(f'Unrecognized argument(s) in config file: {unrecognized_args}')

        parser.set_defaults(**config)
        args = parser.parse_args()

        config_name = pth.splitext(pth.basename(args.config))[0]
        if not config_name.startswith('train_ppo_') or config_name[len('train_ppo_'):] != args.exp_name:
            print('************************************************************')
            print('* WARNING: experiment name does not match config filename. *')
            print('************************************************************')

    # print arguments
    print('Arguments:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')
    print('-' * 50)

    # launch on multiple GPUs
    if args.distributed:
        spawn(launch_distributed, args=(args,), nprocs=len(args.devices), join=True)

    # launch on a single GPU
    else:
        train_ppo(args)


if __name__ == '__main__':
    main()
