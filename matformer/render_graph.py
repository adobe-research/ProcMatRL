# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch

from .render_material import render_material

def default_render_params(device):
    return {
        'size': torch.tensor([0.10], dtype=torch.float32, device=device) * torch.tensor(300.0, dtype=torch.float32, device=device),
        'camera': torch.tensor([0.0, 0.0, 25.0], dtype=torch.float32, device=device).view(3, 1, 1),
        'light_color' : torch.tensor([0.33], dtype=torch.float32, device=device) * torch.tensor([10000.0, 10000.0, 10000.0], dtype=torch.float32, device=device).view(3, 1, 1),
        'f0' : torch.tensor(0.04, dtype=torch.float32, device=device)}

# render graph output channels into a single image with lighting
def render_graph_output(output_channels, render_params=None, output_usages=None, normal_format='gl', force_ogl_normal=False):

    if len(output_channels) == 0:
        raise RuntimeError('Output channels are empty.')

    device = list(output_channels.values())[0].device

    if render_params is None:
        render_params = default_render_params(device=device)

    # render output and save channels
    output_image = None
    mat_channels = []
    if output_usages is None:

        output_usages = ['baseColor', 'normal', 'roughness', 'metallic']
        if any(channel_name not in output_channels for channel_name in output_usages):
            # this should not happen as the forward pass generates defaults for all necessary material channels
            raise RuntimeError('A material channel was not generated in the forward pass.')
        output_image = render_material(
            basecolor=output_channels['baseColor'],
            normal=output_channels['normal'],
            roughness=output_channels['roughness'],
            metallic=output_channels['metallic'],
            normal_format=normal_format,
            force_ogl_normal=force_ogl_normal,
            **render_params)
        for output_usage in ['baseColor', 'normal', 'roughness', 'metallic']:
            mat_channels.append(output_channels[output_usage])

    elif len(output_usages) == 1:

        if any(channel_name not in output_channels for channel_name in output_usages):
            raise RuntimeError('A material channel was not generated in the forward pass.')
        output_image = output_channels[output_usages[0]]
        mat_channels.append(output_channels[output_usages[0]])

        # make sure the output image has exactly 3 channels
        if output_image.shape[1] == 4:
            output_image = output_image[:, :3] # remove alpha
        elif output_image.shape[1] == 1:
            output_image = output_image.expand(-1, 3, -1 ,-1) # grayscale to color

    else:
        raise RuntimeError('Don''t know how to generate output variation for the given output usages.')

    return output_image
