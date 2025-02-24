# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import os.path as pth
import sys
import argparse
import json
import random
import copy
import re
import warnings

os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.multiprocessing as mp
import pandas as pd

from .eval_image_cond_generators import (
    get_model_args, ModelBuilder, eval_token_loss, generate_graphs, render_graphs, collect_metrics,
    run_match_optimization, compile_results
)


def eval_graph_generator(args, rank=None, world_size=None, queue=None):
    # set device
    device = args.devices[rank if rank is not None else 0]
    use_cuda = torch.cuda.is_available() and device.startswith('cuda')

    # distributed evaluation
    is_main_rank = rank is None or rank == 0
    dist_kwargs = dict(rank=rank, world_size=world_size, queue=queue)

    # disable screen output for non-main ranks
    if not is_main_rank:
        warnings.filterwarnings('ignore')
        text_io = open(os.devnull, 'w')
        sys.stdout, sys.stderr = text_io, text_io

    # enable TF32 precision optimization
    if use_cuda and args.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

    # set random seed
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    # prepare model arguments
    model_args = get_model_args(args)

    # model-related evaluations
    if any(mode in ('loss', 'param_loss', 'gen', 'param') for mode in args.eval_modes):

        # build model
        build_model = ModelBuilder(model_args, device=device)
        model = build_model()

        # disable gradient computation
        for p in model.parameters():
            p.requires_grad = False

        # load model checkpoint
        if getattr(args, 'checkpoint_exp_name', None) is not None:
            ckpt_model_step = args.checkpoint_model_step

            # load the best checkpoint or the specified checkpoint
            if ckpt_model_step == 'best' or ckpt_model_step.isdigit() or isinstance(ckpt_model_step, int):
                ckpt_file_name = f'{ckpt_model_step}_policy.pth'

            # load the last checkpoint available
            elif ckpt_model_step == 'last':
                ckpt_pattern = re.compile(r'(\d+)_policy.pth')
                all_ckpt_files = [f for f in os.listdir(pth.join(args.checkpoint_dir, args.checkpoint_exp_name))
                                  if ckpt_pattern.fullmatch(f)]
                if not all_ckpt_files:
                    raise RuntimeError('No checkpoint file found.')
                ckpt_file_name = sorted(all_ckpt_files, key=lambda f: int(ckpt_pattern.match(f).group(1)))[-1]

            model.load_from_checkpoint(pth.join(args.checkpoint_dir, args.checkpoint_exp_name, ckpt_file_name), device)
            print(f"Loaded checkpoint '{ckpt_file_name}'")

        # evaluate token losses
        if any(mode in ('loss', 'param_loss') for mode in args.eval_modes):
            eval_token_loss(model_args, model, build_model.sequencers, **dist_kwargs)

        # generate nodes, edges, and parameters
        if any(mode in ('gen', 'param') for mode in args.eval_modes):
            generate_graphs(model_args, model, build_model.sequencers, **dist_kwargs)

        # remove model and release memory
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    # parallel section finished
    if not is_main_rank:
        return

    # render generated graphs
    if 'render' in args.eval_modes:
        render_graphs(model_args)

    # collect rendered image metrics
    if 'metrics' in args.eval_modes:
        collect_metrics(model_args)
        if use_cuda:
            torch.cuda.empty_cache()

    # run post-optimization using MATch v2
    if 'match' in args.eval_modes and args.match_profiles:
        run_match_optimization(model_args)

    # compile results
    results = None
    if 'results' in args.eval_modes:
        results = compile_results(model_args)

    return results


def create_arg_parser():
    p = argparse.ArgumentParser(description='Evaluate a parameter generator.')

    # I/O related
    p.add_argument('--config', default=None, type=str, help='Path to a config file.')
    p.add_argument('--batch_config', default=None, type=str, help='Path to a batch config file.')
    p.add_argument('--exp_name', default=None, type=str, help='Name of the graph generator experiment.')
    p.add_argument('--eval_suffix', type=str, default=None, help='Suffix to add to the experiment name that describes the evaluation details.')
    p.add_argument('--param_exp_name', default=None, type=str, help='Name of the parameter generator experiment.')
    p.add_argument('--param_model_step', default=None, help='Training step of the parameter generator model to load.')
    p.add_argument('--checkpoint_exp_name', default=None, type=str, help='Name of the checkpoint experiment.')
    p.add_argument('--checkpoint_model_step', default=None, help='Training step of the checkpoint to load.')
    p.add_argument('--model_dir', default=None, type=str, help='Path to models directory for all experiments.')
    p.add_argument('--checkpoint_dir', default=None, type=str, help='Path to checkpoint directory.')
    p.add_argument('--result_dir', default=None, type=str, help='Path to results directory.')
    p.add_argument('--sat_dir', default=None, type=str, help='Location of the Substance Automation Toolkit.')
    p.add_argument('--sbsrc_dir', default=None, type=str, help='Location of the directory containing substance source.')
    p.add_argument('--label', default='gen', type=str, help='Label of output file folder')

    # dataset configuration
    p.add_argument('--custom_data_dir', default=None, type=str, help='Overwrite the data directory.')
    p.add_argument('--custom_node_type_list', default=None, type=str, help='Overwrite the node types path.')
    p.add_argument('--test_dataset', default=None, type=str, help='Path to test set.')
    p.add_argument('--use_alpha', default=False, action='store_true',help='If using alpha channel.')
    p.add_argument('--data_chunksize', default=64, type=int, help='Chunksize for coalesced access to HDF5 dataset.')
    p.add_argument('--target_shuffle_queue_size', default=512, type=int, help='Size of the target shuffle queue.')
    p.add_argument('--pre_shuffle', default=False, action='store_true', help='Pre-shuffle the dataset instead of shuffling it in runtime.')
    p.add_argument('--image_ext', default='png', type=str, help='Extension of input image.')
    p.add_argument('--real_image_dir', default=None, type=str, help='Path to real image directory.')
    p.add_argument('--real_test_dataset', default=None, type=str, help='Path to real test set.')
    p.add_argument('--real_nn_file', default=None, type=str, help='Path to nearest neighbor file.')
    p.add_argument('--real_augment_graph', default=0, type=int, help='Augment real graph with nearest neighbors.')
    p.add_argument('--real_pre_shuffle_mode', default=None, type=str, help='Pre-shuffle mode for real image dataset.')

    # evaluation configuration
    ## overall control
    p.add_argument('--eval_modes', type=str, default=None, nargs='+',
                   choices=('param', 'param_loss', 'render', 'metrics', 'match', 'results'),
                   help='Specify one or multiple evaluation steps.')

    ## token loss
    p.add_argument('--vis_per_token_loss', metavar=['STAGE', 'SEQ'], default=None, type=str, nargs=2, help='Visualize per-token loss for a given token type.')
    p.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    p.add_argument('--num_workers', default=4, type=int, help='Number of threads (workers) for data loading.')

    ## graph generation
    p.add_argument('--num_gen_samples', default=100, type=int, help='Number of generated samples.')
    p.add_argument('--deterministic', default=False, action='store_true', help='Generate a deterministic sequence (using the maximum activation)')
    p.add_argument('--k_subsamples', default=10, type=int, help='Number of subsamples generated by predicted probability.')
    p.add_argument('--semantic_validate', default=False, action='store_true', help='Extra semantic validation')
    p.add_argument('--temperature', default=1.0, type=float, help='Temperature of softmax')
    p.add_argument('--prob_k', default=5, type=int, help='Sample from top-k probability')
    p.add_argument('--nucleus_top_p', default=1.0, type=float, help='Nucleus sampling top-p')

    ## rendering
    p.add_argument('--num_processes', default=16, type=int, help='Number of processes to render generated graphs')
    p.add_argument('--render_channels',  default=False, action='store_true', help='Render material channels.')

    ## metric
    p.add_argument('--metric_batch_size', default=64, type=int, help='Batch size for metric computation.')
    p.add_argument('--metric_num_workers', default=4, type=int, help='Number of threads (workers) for data loading in metric computation.')

    ## match optimization
    p.add_argument('--match_profiles', default=None, type=str, nargs='+', help='Configurations for MATch v2 optimization')
    p.add_argument('--match_profile_mode', default='sample', type=str, choices=('sample', 'graph'), help='Mode for MATch v2 optimization profiles')
    p.add_argument('--match_timeout', default=None, type=int, help='Timeout for MATch v2 optimization in seconds')

    ## result
    p.add_argument('--output_folder_name', default='output', type=str, help='Name of the output folder.')
    p.add_argument('--output_num_cand_graphs', default=None, type=int, help='Number of candidate graphs for output')
    p.add_argument('--output_num_cand_samples', default=None, type=int, help='Number of candidate samples per graph for output')
    p.add_argument('--output_deterministic', default=False, action='store_true', help='Output deterministic predictions only')
    p.add_argument('--output_sample_reduction', default='max', type=str, choices=('max', 'none'), help='Reduction method for samples of each graph')
    p.add_argument('--rank_output', default=False, action='store_true', help='Rank output images')
    p.add_argument('--rank_vgg_td_level', default=0, type=int, help='VGG level for ranking output images')
    p.add_argument('--rank_vgg_coeff', default=10.0, type=float, help='VGG coefficient for ranking output images')
    p.add_argument('--rank_l1_coeff', default=0.5, type=float, help='L1 coefficient for ranking output images')
    p.add_argument('--rank_lab_l1_coeff', default=0.0, type=float, help='L1 coefficient for ranking output images in LAB space')
    p.add_argument('--rank_lpips_coeff', default=0.0, type=float, help='LPIPS coefficient for ranking output images')
    p.add_argument('--rank_swd_coeff', default=0.0, type=float, help='SWD coefficient for ranking output images')
    p.add_argument('--rank_top_k', default=0, type=int, help='Top k results to output after ranking')
    p.add_argument('--rank_lab_weights', default=[0.2, 1.0, 1.0], nargs=3, type=float, help='LAB weights for ranking output images')
    p.add_argument('--rank_match_profile', default=None, type=str, help='Configurations for MATch v2 ranking')
    p.add_argument('--high_res_image_dir', default=None, type=str, help='Path to high resolution data directory.')
    p.add_argument('--output_num_imgs', default=64, type=int, help='Maximum number of images in the output image grid')
    p.add_argument('--max_img_cols', default=16, type=int, help='Maximum number of columns in the output image grid')
    p.add_argument('--output_histogram', default=False, action='store_true', help='Output histogram of image scores')
    p.add_argument('--output_csv', default=None, type=str, help='Output csv file for results')

    # system configuration
    p.add_argument('--devices', default=None, type=str, nargs='+', help='Devices to run on.')
    p.add_argument('--seed', default=414646787, type=int, help='Seed for repeatability.')
    p.add_argument('--distributed', default=False, action='store_true', help='Use distributed training.')
    p.add_argument('--allow_tf32', default=False, action='store_true', help='Allow TF32 precision for model evaluation')
    p.add_argument('--use_fast_attn', default=False, action='store_true', help='Use fast attention for model evaluation')

    return p


def run_batched_experiments(args):
    # load batched config
    with open(args.batch_config, 'r') as f:
        batch_config = json.load(f)

        # update config with batched config
        for k, v in batch_config.items():
            if not isinstance(v, list):
                setattr(args, k, v)

    # check batched config format
    for k, v in batch_config.items():
        if not hasattr(a, k):
            raise RuntimeError(f"Batched config key '{k}' not found in the original config.")

    num_exps = max(len(v) if isinstance(v, list) else 1 for v in batch_config.values())
    results, failed_exps = [], []

    for i in range(num_exps):
        print(f'------ Running batched experiment {i+1}/{num_exps} ... ------')

        # update config
        a_i = copy.deepcopy(args)
        for k, v in batch_config.items():
            if isinstance(v, list) and i < len(v):
                setattr(a_i, k, v[i])

        try:
            ret = eval_graph_generator(args=a_i)
            if ret is not None:
                match_label = '_match' if a_i.rank_match_profile is not None else ''
                results.append((a_i.exp_name + match_label, ret))
        except Exception as e:
            print(f'Experiment {i+1} ERROR: {e}')
            print('Skipping to the next experiment ...')
            failed_exps.append(f'Exp. {i+1}/{num_exps} ERROR: {e}')
            raise e from e

    # print failed experiments
    if failed_exps:
        print('************************************************************')
        print('* WARNING: some experiments failed. See below for details. *')
        print('************************************************************')
        for e in failed_exps:
            print(e)

    # save results
    if results:

        # print message header
        msg = [
            'ranked' if args.rank_output else 'unranked',
            f'{args.output_num_cand_graphs}-NN' if args.output_num_cand_graphs is not None else 'all',
            f'top-{args.rank_top_k}' if args.rank_top_k > 0 else 'all',
            'deterministic' if args.output_deterministic else 'random'
        ]
        print('-' * 50)
        print(f"Results ({', '.join(msg)}):")

        # collect categories and metric names
        result_dict = results[0][1]
        categories = list(result_dict.keys())
        metrics = list(result_dict[categories[0]].keys())

        # print results
        for k in categories:
            print(f'  {k}:')
            for exp_name, ret in results:
                print(f"    {exp_name}: {', '.join(f'{t} = {v:.4f}' for t, v in ret[k].items())}")

        def translate(name):
            out = []
            for n in name.split('_'):
                # dataset
                if n == 'init':
                    out.append('Supervised')
                elif n == 'real':
                    out.append('Real data')
                elif n == 'syn':
                    out.append('Synthetic data')
                elif n.startswith('mix'):
                    out.append(f'Mixed data ({int(n[3:])}% real)')
                elif n == 'only':
                    continue

                # hyperparameters
                elif n[0] == 'r' and n[1:].isdigit():
                    out.append(f'VGG = {int(n[1:])/100:.2g}')
                elif n[0] == 'l' and n[1:].isdigit():
                    out.append(f'DS L1 = {int(n[1:])/100:.2g}')
                elif n.startswith('lab') and n[3:].isdigit():
                    out.append(f'DS LAB L1 = {int(n[3:])/100:.2g}')

                # other comments
                elif n == 'noaug':
                    out.append('No augmentation')
                elif n == 'entropy':
                    out.append('Favor entropy')
                elif n == 'longterm':
                    out.append('Favor long seqs')

                # system
                elif n == 'sensei':
                    out.append('Sensei')

                # unknown
                else:
                    out.append(n)

            return ', '.join(out)

        column_dict = {'score': 'Reward', 'lpips': 'LPIPS', 'clip': 'CLIP', 'swd': 'SWD'}

        # save results to csv
        if args.output_csv is not None:
            output_dir = pth.join(args.result_dir, args.output_folder_name)
            os.makedirs(output_dir, exist_ok=True)

            columns = [f'{column_dict[m]} ({c})' for m in metrics for c in categories]
            data = [[ret[c][m] for m in metrics for c in categories] for _, ret in results]
            index = [translate(exp_name) for exp_name, _ in results]
            pd.DataFrame(data=data, index=index, columns=columns).to_csv(pth.join(output_dir, args.output_csv))


if __name__ == '__main__':
    use_networkx = False
    mp.set_start_method('spawn')

    # if platform.system() == 'Windows':
    if os.name == 'nt':
        import win32file
        win32file._setmaxstdio(4096)

    parser = create_arg_parser()

    a = parser.parse_args()
    assert a.config is not None

    if a.config is not None:
        config_parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        config_path = a.config
        with open(config_path, 'r') as f:
            json_config = json.load(f)

        if any(json_name not in vars(a).keys() for json_name in json_config.keys()):
            unrecognized_args = [json_name for json_name in json_config.keys() if json_name not in vars(a).keys()]
            raise RuntimeError(f'Unrecognized argument(s) in config file: {unrecognized_args}')

        config_parser.set_defaults(**json_config)
        a = config_parser.parse_args()

        config_name = os.path.splitext(os.path.basename(config_path))[0]
        if not config_name.startswith('eval_') or config_name[len('eval_'):] != a.exp_name:
            print('************************************************************')
            print('* WARNING: experiment name does not match config filename. *')
            print('************************************************************')
            # r = input('Proceed anyway? (Y/N) ')
            # if r == 'N':
            #     sys.exit()
        a.config = config_path

    for k, v in a.__dict__.items():
        print(f'{k}: {v}')
    print('-'*50)

    # run batched experiments
    if a.batch_config is not None:
        if a.distributed:
            raise RuntimeError('Batched experiments are not supported in distributed mode.')

        run_batched_experiments(args=a)

    # run single experiment in distributed mode
    elif a.distributed:
        world_size = len(a.devices)
        queue = mp.Queue()

        # spawn processes
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=eval_graph_generator, args=(a, rank, world_size, queue))
            p.start()
            processes.append(p)

        # join processes
        for p in processes:
            p.join()

    # run single experiment
    else:
        eval_graph_generator(args=a)
