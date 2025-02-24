# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import argparse
import logging
import os
import os.path as pth
import sys
import time
import traceback


def differentiable_optimize_v2(sbs_path, img_path, sat_dir, reward_config={}, device='cuda', line_search=False, img_format='png', timeout=None):
    '''Optimize materials using DiffMat v2.
    '''
    from diffmat import MaterialGraphTranslator as MGT
    from diffmat.core.io import read_image
    from diffmat.optim import Optimizer, HybridOptimizer
    import torch.nn.functional as F

    # Specify output directories
    suffix = '_opt_v2' + ('_ls' if line_search else '')
    output_dir = pth.join(pth.dirname(sbs_path), pth.basename(sbs_path)[:-4] + suffix)
    ext_input_dir = pth.join(output_dir, 'external_input')

    # Load and translate SBS graph
    translator = MGT(sbs_path, res=9, external_noise=False, toolkit_path=sat_dir)
    graph = translator.translate(external_input_folder=ext_input_dir, img_format=img_format, keep_input_sbs=False, device=device)
    graph.compile()

    # Read target image and convert to PyTorch tensor
    target_img = read_image(img_path, device=device)[None, :3]
    if target_img.shape[-2:] != (512, 512):
        target_img = F.interpolate(target_img, (512, 512), mode='bicubic', align_corners=False)
        target_img.clamp_(0, 1)

    # Set up optimizer
    config = reward_config or {
        'vgg': {'weight': 2.0, 'td_kwargs': {'td_level': 0, 'resize_image': True}},
        'ds_lab': {'weight': 0.05}
    }
    optimizer = Optimizer(
        graph, lr=1e-3, algorithm='line' if line_search else 'adam', metric='combine',
        metric_kwargs={'config': config})
    optim_kwargs = {'save_option': 'render', 'img_format': img_format}

    # Line search
    if line_search:
        optimizer.optimize(target_img, num_iters=100, result_dir=output_dir, c1=0.01, **optim_kwargs)
        opt_img_path = pth.join(output_dir, 'render', 'optimized.jpg')

    # MATch v2
    else:
        opt_img_path = None

        # Start timer
        start_time = time.time()

        # Run staged optimization with timeout
        for stage_id in range(3):
            stage_kwargs = {
                'result_dir': pth.join(output_dir, f'stage_{stage_id}'),
                'timeout': timeout - (time.time() - start_time) if timeout is not None else None,
                **optim_kwargs
            }

            # Run differentiable optimization
            if stage_id == 0:
                optimizer.optimize(target_img, num_iters=1000, **stage_kwargs)

            # Run integer optimization
            elif stage_id == 1:
                int_optimizer = HybridOptimizer(graph, filter_integer=1, metric='combine', metric_kwargs={'config': config})
                if graph.num_integer_parameters() > 0:
                    int_optimizer.optimize(target_img, num_iters=5, **stage_kwargs)

            # Run fine-tuning differentiable optimization
            else:
                graph.train()
                ft_optimizer = Optimizer(graph, lr=1e-4, metric='combine', metric_kwargs={'config': config})
                ft_optimizer.optimize(target_img, num_iters=500, **stage_kwargs)

            opt_img_path = pth.join(output_dir, f'stage_{stage_id}', 'render', f'optimized.{img_format}')

    return opt_img_path


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Optimize materials using DiffMat v2.')
    parser.add_argument('sbs_path', metavar='PATH', type=str, help='Path to the Substance Designer graph file.')
    parser.add_argument('img_path', metavar='PATH', type=str, help='Path to the target image.')
    parser.add_argument('--sat_dir', metavar='DIR', type=str, required=True, help='Path to the Substance Automation Toolkit directory.')
    parser.add_argument('--vgg_coeff', metavar='FLOAT', type=float, default=2.0, help='VGG loss coefficient.')
    parser.add_argument('--vgg_td_level', metavar='INT', type=int, default=0, help='VGG texture pyramid level.')
    parser.add_argument('--ds_l1_coeff', metavar='FLOAT', type=float, default=0.0, help='Downsampled L1 loss coefficient.')
    parser.add_argument('--ds_lab_coeff', metavar='FLOAT', type=float, default=0.05, help='Downsampled LAB loss coefficient.')
    parser.add_argument('--lpips_coeff', metavar='FLOAT', type=float, default=0.0, help='LPIPS loss coefficient.')
    parser.add_argument('--device_id', metavar='INT', type=int, default=None, help='GPU device ID to run optimization.')
    parser.add_argument('--timeout', metavar='INT', type=int, default=None, help='Timeout in seconds.')

    args = parser.parse_args()

    # Set CUDA device
    if args.device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
        device = 'cuda'
    else:
        device = 'cpu'

    # Set up reward configuration
    reward_config = {
        'vgg': {'weight': args.vgg_coeff, 'td_kwargs': {'td_level': args.vgg_td_level, 'resize_image': True}},
        'ds': {'weight': args.ds_l1_coeff},
        'ds_lab': {'weight': args.ds_lab_coeff},
        'lpips': {'weight': args.lpips_coeff}
    }
    reward_config = {k: v for k, v in reward_config.items() if v['weight'] > 0}

    # Run optimization
    differentiable_optimize_v2(
        args.sbs_path, args.img_path, args.sat_dir, reward_config=reward_config, device=device, img_format='jpg',
        timeout=args.timeout)


if __name__ == '__main__':
    try:
        # Set logging level
        logging.basicConfig(level=logging.WARNING)

        main()

    # Capture error and print related information to stderr
    except Exception as e:
        print('Error occurred during optimization:', file=sys.stderr)
        print(str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print('End of error message.', file=sys.stderr)

        sys.exit(1)
