# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import math

import numpy as np
import torch
import scipy.ndimage as spi

from .sbs_utils import input_check, roll_row, roll_col, normalize, color_input_check, grayscale_input_check, to_zero_one, get_shape_from_any, as_comp_tensor

'''
Implmented atomic functions:
    - Blend
    - Blur
    - Channels Shuffle (no trainable variables)
    - Curve
    - Directional Blur
    - Directional Warp
    - Distance
    - Emboss
    - Gradient Map
    - Grayscale Conversion (c2g)
    - Levels
    - Normal
    - Sharpen
    - Transformation 2D
    - Uniform Color
    - Warp

Not implemented atomic functions:
    - Gradient Dynamic (Gradient Map is enough)
    - HSL (non-differentiable)
    - FX Map, Pixel/Value Processor (Define custom function directly when needed)
    - SVG/TEXT (not useful in current context)

Others:
    - linear_to_srgb
    - srgb_to_linear
    - curvature

TODO:
    - implement world-space normal
'''

@input_check(tensor_args=['img_fg', 'img_bg', 'blend_mask'])
def blend(img_fg=None, img_bg=None, blend_mask=None, blending_mode='copy', cropping=[0.0,1.0,0.0,1.0], opacity=1.0, alphablending='use_source', outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: blend (https://substance3d.adobe.com/documentation/sddoc/blending-modes-description-132120605.html)
        img_fg: foreground image (G or RGB(A))
        img_bg: background image (G or RGB(A))
        blend_mask: blending mask (G only)
        blending mode: copy, add, subtract, multiply, add_sub, max, min, divide, switch, overlay, screen, soft_light
        cropping: [left, right, top, bottom]
        opacity: optional alpha mask
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        only support use_source_alpha, if not graph parser will raise error
    '''
    # defaults
    if img_fg is None:
        if img_bg is not None:
            img_fg = torch.zeros(size=[img_bg.shape[0], img_bg.shape[1], img_bg.shape[2], img_bg.shape[3]], dtype=torch.float32, device=img_bg.device)
        elif blend_mask is not None:
            img_fg = torch.zeros(size=[blend_mask.shape[0], 1, blend_mask.shape[2], blend_mask.shape[3]], dtype=torch.float32, device=blend_mask.device)
        else:
            img_fg = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if img_bg is None:
        img_bg = torch.zeros(size=[img_fg.shape[0], img_fg.shape[1], img_fg.shape[2], img_fg.shape[3]], dtype=torch.float32, device=img_fg.device)
    # default for blend_mask is handled below

    if alphablending != 'use_source':
        print('WARNING: only \'use_source\' alpha blending is supported, output may be incorrect.')

    use_alpha = False
    if img_fg.shape[1] == 4:
        img_fg_alpha = img_fg[:,[3],:,:]
        img_fg = img_fg[:,:3,:,:]
        use_alpha = True
    else:
        img_fg_alpha = 0.0

    if img_bg.shape[1] == 4:
        img_bg_alpha = img_bg[:,[3],:,:]
        img_bg = img_bg[:,:3,:,:]
        use_alpha = True
    else:
        img_bg_alpha = 0.0

    # if len(img_fg.shape) and len(img_bg.shape):
    if img_fg.shape[1] == 1 and img_bg.shape[1] == 3:
        img_fg = img_fg.expand(-1, 3, -1, -1)
    elif img_fg.shape[1] == 3 and img_bg.shape[1] == 1:
        img_bg = img_bg.expand(-1, 3, -1, -1)
    assert img_fg.shape[1] == img_bg.shape[1], 'foreground and background image type does not match'

    if blend_mask is not None:
        blend_mask = as_comp_tensor(blend_mask, dtype=torch.float32, device=img_fg.device)
        blend_mask = grayscale_input_check(blend_mask, 'blend mask')
        weight = blend_mask * opacity
    else:
        weight = opacity

    input_shape = get_shape_from_any([img_fg, img_bg, blend_mask], default_shape=[1, 1, 1 << outputsize[0], 1 << outputsize[1]])

    # compute output alpha channel
    # if len(img_fg.shape):
    #     if img_fg.shape[1] == 4:
    #         img_fg_alpha = img_fg[:,[3],:,:]
    #         img_fg = img_fg[:,:3,:,:]
    # if len(img_bg.shape):
    #     if img_bg.shape[1] == 4:
    #         img_bg_alpha = img_bg[:,[3],:,:]
    #         img_bg = img_bg[:,:3,:,:]
    if use_alpha:
        if blending_mode == 'switch':
            img_out_alpha = img_fg_alpha * weight + img_bg_alpha * (1.0 - weight)
        else:
            weight = weight * img_fg_alpha
            img_out_alpha = weight + img_bg_alpha * (1.0 - weight)

    clamp_max = 1.0
    clamp_min = 0.0
    if blending_mode == 'copy':
        img_out = torch.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'add':
        img_out = torch.clamp(img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'subtract':
        img_out = torch.clamp(-img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'multiply':
        img_fg = img_fg * weight + (1.0 - weight)
        img_out = torch.clamp(img_fg * img_bg, clamp_min, clamp_max)
    elif blending_mode == 'add_sub':
        img_fg = (img_fg - 0.5) * 2.0
        img_out = torch.clamp(img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'max':
        img_fg = torch.clamp(torch.max(img_fg, img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'min':
        img_fg = torch.clamp(torch.min(img_fg, img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'divide':
        img_fg = img_bg / (img_fg + 1e-15)
        img_out = torch.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'switch':
        if blend_mask is None and weight == 1.0:
            img_out = img_fg
        elif blend_mask is None and weight == 0.0:
            img_out = img_bg
        else:
            img_out = torch.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'overlay':
        img_out = torch.zeros(input_shape, dtype=torch.float32, device=img_fg.device)
        mask = img_bg < 0.5
        img_out[mask] = torch.clamp(2.0 * img_fg * img_bg, clamp_min, clamp_max)[mask]
        img_out[~mask] = torch.clamp(1.0 - 2.0 * (1.0 - img_fg) * (1.0 - img_bg), clamp_min, clamp_max)[~mask]
        img_out = img_out * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'screen':
        img_fg = torch.clamp(1.0 - (1.0 - img_fg) * (1.0 - img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'soft_light':
        img_fg = torch.clamp(img_bg + (img_fg * 2.0 - 1.0) * img_bg * (1.0 - img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    else:
        raise 'unknown blending_mode'

    # apply cropping
    if cropping[0] == 0.0 and cropping[1] == 1.0 and cropping[2] == 0.0 and cropping[3] == 1.0:
        img_out_crop = img_out
    else:
        start_row = math.floor(cropping[2] * img_out.shape[2])
        end_row = math.floor(cropping[3] * img_out.shape[2])
        start_col = math.floor(cropping[0] * img_out.shape[3])
        end_col = math.floor(cropping[1] * img_out.shape[3])
        img_out_crop = img_bg.clone()
        img_out_crop[:,:,start_row:end_row, start_col:end_col] = img_out[:,:,start_row:end_row, start_col:end_col]

    if use_alpha == True:
        img_out_crop = torch.cat([img_out_crop, img_out_alpha], dim=1)

    return img_out_crop

@input_check(tensor_args=['img_in'])
def blur(img_in, intensity=0.5, max_intensity=20.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: blur (simple box blur: https://substance3d.adobe.com/documentation/sddoc/blur-172825121.html)
        img_in (G or RGB(A))
        intensity: normalized box filter side length [0,1], need to multiply max_intensity
        max_intensity: maximum blur intensity
        return image (G or RGB(A))
    Implementation:
        Use separable convolution for better performance
    Behavior:
        identical to sbs
        alpha changes with blur
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    num_group = img_in.shape[1]
    img_size = img_in.shape[2]
    intensity = as_comp_tensor(intensity * img_size / 256.0, dtype=img_in.dtype, device=img_in.device) * max_intensity
    kernel_len = (torch.ceil(intensity + 0.5) * 2.0 - 1.0).type(torch.int)
    if kernel_len <= 1:
        return img_in.clone()
    # create 2d kernel
    blur_idx = -torch.abs(torch.linspace(-kernel_len//2, kernel_len//2, kernel_len, device=img_in.device))
    blur_1d  = torch.clamp(blur_idx + intensity + 0.5, 0.0, 1.0)
    blur_row = blur_1d.view(1,1,1,kernel_len).expand(num_group,1,1,kernel_len)
    blur_col = blur_1d.view(1,1,kernel_len,1).expand(num_group,1,kernel_len,1)
    # manually pad input image
    p2d = [kernel_len//2, kernel_len//2, kernel_len//2, kernel_len//2]
    img_in  = torch.nn.functional.pad(img_in, p2d, mode='circular')
    # perform depth-wise convolution without implicit padding
    img_out = torch.nn.functional.conv2d(img_in, blur_row, groups=num_group, padding = 0)
    img_out = torch.nn.functional.conv2d(img_out, blur_col, groups=num_group, padding = 0)
    img_out = torch.clamp(img_out / (intensity ** 2 * 4.0), torch.tensor(0.0, device=img_in.device), torch.tensor(1.0, device=img_in.device))
    return img_out


@input_check(tensor_args=['img_in', 'img_in_aux'])
def channel_shuffle(img_in, img_in_aux=None, use_alpha=False, shuffle_idx=[0,1,2,3], outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: channel shuffle (https://substance3d.adobe.com/documentation/sddoc/channel-shuffle-172825126.html)
        img_in (G or RGB(A))
        img_in_aux(G or RGB(A)): auxilary input image for swapping channels between images
        shuffle_idx: vector of length 3 (for RGB channel)
        return imge (RGB(A) only)
    '''
    # defaults
    if img_in is None:
        if img_in_aux is not None:
            img_in = torch.zeros(size=[img_in_aux.shape[0], 1, img_in_aux.shape[2], img_in_aux.shape[3]], dtype=torch.float32, device=img_in_aux.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    # default for img_in_aux is handled below

    # tensor conversion
    # shuffle_idx = as_comp_tensor(shuffle_idx, dtype=img_in.dtype, device=img_in.device)

    # img_in_aux type and shape check
    if img_in_aux is not None:
        img_in_aux = as_comp_tensor(img_in_aux, dtype=img_in.dtype, device=img_in.device)
        assert len(img_in_aux.shape) == 4, 'img_in_aux is not a 4D Pytorch tensor'
        assert img_in.shape[0] == img_in_aux.shape[0] and \
                img_in.shape[2] == img_in_aux.shape[2] and \
                img_in.shape[3] == img_in_aux.shape[3], 'the shape of img_in and img_in_aux does not match'

    if img_in.shape[1] == 1:
        img_in = img_in.expand(img_in.shape[0], 4 if use_alpha else 3, img_in.shape[2], img_in.shape[3])

    if img_in_aux is not None and img_in_aux.shape[1] == 1:
        img_in_aux = img_in_aux.expand(img_in_aux.shape[0], 4 if use_alpha else 3, img_in_aux.shape[2], img_in_aux.shape[3])

    # output has the same shape as the input
    img_out = img_in.clone()
    for i in range(4 if use_alpha else 3):
        if i != shuffle_idx[i]:
            if shuffle_idx[i] <= 3 and shuffle_idx[i] >= 0:
                # channels should come from the first input image
                channels_in = img_in
                channel_idx = shuffle_idx[i]
            elif shuffle_idx[i] >= 4 and shuffle_idx[i] <= 7:
                # channels should come from the second input image
                if img_in_aux is None:
                    print('WARNING: channel shuffle index refers to second input image, but no second input image was given. Using the first input image instead.')
                    channels_in = img_in
                else:
                    channels_in = img_in_aux
                channel_idx = shuffle_idx[i]-3
            else:
                raise RuntimeError(f'shuffle idx {shuffle_idx[i]} is invalid')

            channel_out = None
            if channels_in.shape[1] == 3:
                channel_out = 1.0 if channel_idx == 3 else channels_in[:,channel_idx,:,:]
            elif channels_in.shape[1] == 4:
                channel_out = channels_in[:,channel_idx,:,:]
            else:
                raise RuntimeError('Unexpected number of channels.')

            img_out[:,i,:,:] = channel_out

    return img_out


@input_check(tensor_args=['img_in'])
def curve(img_in, anchors=None, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: curve (https://substance3d.adobe.com/documentation/sddoc/curve-172825175.html)
        img_in (G or RGB(A))
        anchors: matrix with shape [num_anchors, 6]
            - [:, :2] indicates the position of the anchor itself
            - [:, 2:4] indicates the position of the left control point of the anchor
            - [:, 4:] indicates the position of the right control point of the anchor
        return imge (G or RGB, same as img_in)
    Implementation:
        Piecewise Bezier curve interpolation (using input control points)
    Behavior:
        identical to sbs
    TODO:
        - support per channel adjustment
        - support alpha adjustment
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    # Process input anchor table
    if anchors is None:
        anchors = torch.stack([torch.zeros(6, device=img_in.device), torch.ones(6, device=img_in.device)])
    else:
        anchors = as_comp_tensor(anchors, device=img_in.device)
        # assert anchors.shape == (num_anchors, 6), 'shape of anchors is not [num_anchors, 6]'
        assert anchors.shape[0] > 0 and anchors.shape[1] == 6, 'shape of anchors is not [num_anchors, 6]'
        # Sort input anchors based on [:,0] in ascendng order
        if anchors.shape[0] == 1:
            print('WARNING: only a single anchor available for curve node, but need at least two anchors. Using all-zeros as additional anchor.')
            anchors = torch.cat([torch.zeros_like(anchors), anchors], dim=0)
        anchors = anchors[torch.argsort(anchors[:, 0]), :]
    num_anchors = anchors.shape[0]

    # Determine the size of the sample grid
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_size_t = max(res_h, res_w) * 2
    sample_size_x = sample_size_t

    # First sampling pass (parameter space)
    p1 = anchors[:-1, :2].t()
    p2 = anchors[:-1, 4:].t()
    p3 = anchors[1:, 2:4].t()
    p4 = anchors[1:, :2].t()
    A = p4 - p1 + (p2 - p3) * 3.0
    B = (p1 + p3 - p2 * 2.0) * 3.0
    C = (p2 - p1) * 3.0
    D = p1

    t = torch.linspace(0.0, 1.0, sample_size_t, device=img_in.device)
    inds = torch.sum((t.unsqueeze(0) >= anchors[:, [0]]), 0)
    inds = torch.clamp(inds - 1, 0, num_anchors - 2)
    t_ = (t - p1[0, inds]) / (p4[0, inds] - p1[0, inds] + 1e-8)
    bz_t = ((A[:, inds] * t_ + B[:, inds]) * t_ + C[:, inds]) * t_ + D[:, inds]
    bz_t = torch.where((t <= p1[0, 0]).unsqueeze(0), torch.stack([t, p1[1, 0].expand_as(t)]), bz_t)
    bz_t = torch.where((t >= p4[0, -1]).unsqueeze(0), torch.stack([t, p4[1, -1].expand_as(t)]), bz_t)

    # Second sampling pass (x space)
    x = torch.linspace(0.0, 1.0, sample_size_x, device=img_in.device)
    inds = torch.sum((x.unsqueeze(0) >= bz_t[0].view(sample_size_t, 1)), 0)
    inds = torch.clamp(inds - 1, 0, sample_size_t - 2)
    x_ = (x - bz_t[0, inds]) / (bz_t[0, inds + 1] - bz_t[0, inds] + 1e-8)
    bz_x = bz_t[1, inds] * (1 - x_) + bz_t[1, inds + 1] * x_

    # Third sampling pass (color space)
    bz_x = bz_x.expand(img_in.shape[0] * img_in.shape[1], 1, 1, sample_size_x)
    col_grid = img_in.view(img_in.shape[0] * img_in.shape[1], res_h, res_w, 1) * 2.0 - 1.0
    sample_grid = torch.cat([col_grid, torch.zeros_like(col_grid)], 3)
    img_out = torch.nn.functional.grid_sample(bz_x, sample_grid, align_corners=True)
    img_out = img_out.view_as(img_in)

    # Append the original alpha channel
    if use_alpha:
        img_out = torch.cat([img_out, img_in_alpha], dim=1)

    return img_out

# update
@input_check(tensor_args=['img_in'])
def d_blur(img_in, intensity=0.5, max_intensity=20.0, angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: directional blur (1d line blur filter: https://substance3d.adobe.com/documentation/sddoc/directional-blur-172825181.html)
        img_in (G or RGB(A))
        intensity: normalized filter length
        max_intensity: max blur intensity
        angle: filter angle (from [0.0, 1.0])
        return image (G or RGB(A))
    Implementation:
        1) pad input image to prevent rotation out of bound
        1) rotate image by the angle
        2) apply horizontal 1d filter
        3) unrotate the image by angle
    Behavior:
        identical to sbs
        alpha changes with d_blur
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    num_group = img_in.shape[1]
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    res_angle = torch.remainder(as_comp_tensor(angle, dtype=img_in.dtype, device=img_in.device), 0.5)
    angle = as_comp_tensor(angle, dtype=img_in.dtype, device=img_in.device) * np.pi * 2.0
    intensity = as_comp_tensor(intensity * num_row / 512, dtype=img_in.dtype, device=img_in.device) * max_intensity
    if intensity <= 0.25:
        return img_in.clone()

    # compute convolution kernel
    kernel_len = (torch.ceil(2*intensity+0.5)*2-1).type(torch.int)
    blur_idx = -torch.abs(torch.linspace(-kernel_len//2, kernel_len//2, kernel_len, device=img_in.device))
    blur_1d  = torch.nn.functional.hardtanh(blur_idx + intensity*2.0 + 0.5, 0.0, 1.0)
    kernel_1d  = blur_1d / torch.sum(blur_1d)
    kernel_1d  = kernel_1d.view(1,1,1,kernel_len).expand(num_group, 1, 1, kernel_len)

    # Special case for small intensity
    ab_cos = torch.abs(torch.cos(angle))
    ab_sin = torch.abs(torch.sin(angle))
    sc_max, sc_min = torch.max(ab_cos, ab_sin), torch.min(ab_cos, ab_sin)
    dist_1 = (intensity * 2.0 - 0.5) * sc_max
    dist_2 = (intensity * 2.0 - 0.5) * sc_min
    if dist_1 <= 1.0:
        kernel_len = 3

    # circularly pad the image & update num_row, num_col
    # padding is specified as left, right, top, bottom
    conv_p2d = [
        min(img_in.shape[3], kernel_len//2), min(img_in.shape[3], kernel_len//2),
        min(img_in.shape[2], kernel_len//2), min(img_in.shape[2], kernel_len//2)]
    img_in  = torch.nn.functional.pad(img_in, conv_p2d, mode='circular')

    # Compute directional motion blur in different algorithms
    # Special condition (3x3 kernel) when intensity is small
    if dist_1 <= 1.0:
        k_00 = torch.where(res_angle < 0.25, dist_2, torch.tensor(0.0, device=img_in.device))
        k_01 = torch.where(res_angle > 0.125 and res_angle < 0.375, dist_1 - dist_2, torch.tensor(0.0, device=img_in.device))
        k_02 = torch.where(res_angle > 0.25, dist_2, torch.tensor(0.0, device=img_in.device))
        k_10 = torch.where(res_angle < 0.125 or res_angle > 0.375, dist_1 - dist_2, torch.tensor(0.0, device=img_in.device))
        k_11 = torch.tensor(1.0, device=img_in.device)
        kernel_2d = torch.stack([torch.stack([k_00, k_01, k_02]),
                              torch.stack([k_10, k_11, k_10]),
                              torch.stack([k_02, k_01, k_00])])
        kernel_2d = (kernel_2d / torch.sum(kernel_2d)).expand(num_group, 1, 3, 3)
        img_out = torch.nn.functional.conv2d(img_in, kernel_2d, groups=num_group, padding=0)
        img_out = torch.clamp(img_out, torch.tensor(0.0, device=img_in.device), torch.tensor(1.0, device=img_in.device))
    # Compute kernel from rotated small kernels
    elif intensity <= 1.1:
        assert kernel_len == 5
        kernel_2d = torch.zeros(kernel_len, kernel_len, device=img_in.device)
        kernel_2d[[kernel_len//2],:] = blur_1d
        kernel_2d = kernel_2d.expand(num_group, 1, kernel_len, kernel_len)
        sin_res_angle = torch.sin(res_angle * np.pi * 2.0)
        cos_res_angle = torch.cos(res_angle * np.pi * 2.0)
        kernel_2d = transform_2d(kernel_2d, tile_mode=0, mipmap_mode='manual', x1=to_zero_one(cos_res_angle), x2=to_zero_one(-sin_res_angle),
                                 y1=to_zero_one(sin_res_angle), y2=to_zero_one(cos_res_angle))
        kernel_2d = kernel_2d / torch.sum(kernel_2d[0,0])
        img_out = torch.nn.functional.conv2d(img_in, kernel_2d, groups=num_group, padding=0)
        img_out = torch.clamp(img_out, torch.tensor(0.0, device=img_in.device), torch.tensor(1.0, device=img_in.device))
    # Rotation -> convolution -> reversed rotation
    else:
        # compute rotation padding
        num_row = img_in.shape[2]
        num_col = img_in.shape[3]
        num_row_new = torch.abs(torch.cos(-angle))*num_row + torch.abs(torch.sin(-angle))*num_col
        num_col_new = torch.abs(torch.cos(-angle))*num_col + torch.abs(torch.sin(-angle))*num_row
        row_pad = ((num_row_new - num_row) / 2.0).ceil().type(torch.int)
        col_pad = ((num_col_new - num_col) / 2.0).ceil().type(torch.int)
        rot_p2d  = [row_pad, row_pad, col_pad, col_pad]
        img_in  = torch.nn.functional.pad(img_in, rot_p2d, mode='constant')
        num_row = img_in.shape[2]
        num_col = img_in.shape[3]
        # rotate the image
        row_grid, col_grid = torch.meshgrid(torch.linspace(0, num_row-1, num_row, dtype=img_in.dtype, device=img_in.device), torch.linspace(0, num_col-1, num_col, dtype=img_in.dtype, device=img_in.device))
        row_grid = (row_grid + 0.5) / num_row * 2.0 - 1.0
        col_grid = (col_grid + 0.5) / num_col * 2.0 - 1.0
        col_grid_rot = torch.cos(-angle) * col_grid + torch.sin(-angle) * row_grid
        row_grid_rot = -torch.sin(-angle) * col_grid + torch.cos(-angle) * row_grid
        # sample grid
        sample_grid = torch.stack([col_grid_rot, row_grid_rot], 2).expand(img_in.shape[0], num_row, num_col, 2)
        img_rot = torch.nn.functional.grid_sample(img_in, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
        # perform depth-wise convolution without implicit padding
        img_blur = torch.nn.functional.conv2d(img_rot, kernel_1d, groups=num_group, padding = 0)
        # pad back the columns comsumed by conv2d
        img_blur = torch.nn.functional.pad(img_blur, [conv_p2d[0], conv_p2d[1], 0, 0], mode='constant')
        # unrotate the image
        col_grid_unrot = torch.cos(angle) * col_grid + torch.sin(angle) * row_grid
        row_grid_unrot = -torch.sin(angle) * col_grid + torch.cos(angle) * row_grid
        sample_grid = torch.stack([col_grid_unrot, row_grid_unrot], 2).expand(img_in.shape[0], num_row, num_col, 2)
        img_out = torch.nn.functional.grid_sample(img_blur, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
        # remove padding
        full_pad = as_comp_tensor(conv_p2d, device=img_in.device) + as_comp_tensor(rot_p2d, device=img_in.device)
        img_out = img_out[:,:,full_pad[0]:img_out.shape[2]-full_pad[1], full_pad[2]:img_out.shape[3]-full_pad[3]]

    return img_out


@input_check(tensor_args=['img_in', 'intensity_mask'])
def d_warp(img_in, intensity_mask, intensity=0.5, max_intensity=20.0, angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: directional warp (https://substance3d.adobe.com/documentation/sddoc/directional-warp-172825190.html)
        img_in (G or RGB(A))
        intensity_mask: intensity mask for computing displacement (G only)
        intensity: normalized intensity_mask multiplier
        max_intensity: maximum intensity_mask multiplier
        angle: direction to shift, 0 degree points to left, range from [0,1]
        return imge (G or RGB(A))
    Implementation:
        1) compute scaling factor which sbs applies to d_warp
        2) multiply the mask with intensity, scale and sin/cos of angle
        3) shift the pixel according to the mask
    Behavior:
        identical to sbs
        alpha changes with d_warp
    '''
    # defaults
    if img_in is None:
        if intensity_mask is not None:
            img_in = torch.zeros(size=[intensity_mask.shape[0], 1, intensity_mask.shape[2], intensity_mask.shape[3]], dtype=torch.float32, device=intensity_mask.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if intensity_mask is None:
        intensity_mask = torch.zeros(size=[img_in.shape[0], 1, img_in.shape[2], img_in.shape[3]], dtype=torch.float32, device=img_in.device)

    intensity_mask = grayscale_input_check(intensity_mask, 'input mask')

    # tensor conversion
    angle = as_comp_tensor(angle, dtype=img_in.dtype, device=img_in.device) * np.pi * 2.0
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    row_scale = num_row / 256.0 # magic number
    col_scale = num_col / 256.0 # magic number
    intensity_mask = intensity_mask * intensity
    row_shift = intensity_mask * torch.sin(angle) * row_scale
    col_shift = intensity_mask * torch.cos(angle) * col_scale
    row_grid, col_grid = torch.meshgrid(torch.linspace(0, num_row-1, num_row, device=img_in.device), torch.linspace(0, num_col-1, num_col, device=img_in.device))
    # mod the index to behavior as tiling
    row_grid = torch.remainder((row_grid + row_shift + 0.5) / num_row * 2.0, 2.0) - 1.0
    col_grid = torch.remainder((col_grid + col_shift + 0.5) / num_col * 2.0, 2.0) - 1.0
    row_grid = row_grid * num_row / (num_row + 2)
    col_grid = col_grid * num_col / (num_col + 2)
    # sample grid
    sample_grid = torch.cat([col_grid, row_grid], 1).permute(0,2,3,1).expand(intensity_mask.shape[0], num_row, num_col, 2)
    in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], mode='circular')
    img_out = torch.nn.functional.grid_sample(in_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
    return img_out

@input_check(tensor_args=['img_mask', 'img_source'])
def distance(img_mask, img_source=None, mode='gray', combine=True, use_alpha=False, dist=10.0 / 256.0, max_dist=256.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: distance (https://substance3d.adobe.com/documentation/sddoc/distance-172825194.html)
        img_mask: will be binarized using threshold 0.5 (G only)
        img_source: colors will be fetch using img_mask (G or RGB(A))
        mode: controls whether the output is in color or grayscale if img_source is not provided
        combine: controls whether the image mask is combined with the source image (if present)
        dist: normalized propagation distance (euclidean distance, will be multiplied by 3.0 to match substance designer's behavior)
        max_dist: maximum propagation distance
        return imge (G or RGB, same as img_source or G if img_source is None)
    Implementation:
        1) use scipy's distance trasform function to find the closest non-zero point index for each pixel
        2) compute the pixel value based on the euclidean distance
    Behavior:
        almost identical to sbs (some indices might be slightly different)
        alpha changes with distance
    '''
    # defaults
    if img_mask is None:
        if img_source is not None:
            img_mask = torch.zeros(size=[img_source.shape[0], 1, img_source.shape[2], img_source.shape[3]], dtype=torch.float32, device=img_source.device)
        else:
            img_mask = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    # default for img_source is handled below

    assert mode in ('color', 'gray')
    img_mask = grayscale_input_check(img_mask, 'image mask')
    num_rows = img_mask.shape[2]
    num_cols = img_mask.shape[3]

    if img_source is not None:
        # img_source type and shape check
        # img_source = as_comp_tensor(img_source, dtype=img_source.dtype, device=img_source.device)
        assert len(img_source.shape) == 4, 'img_source is must be a 4D Pytorch tensor'
        assert img_mask.shape[0] == img_source.shape[0] and \
               num_rows == img_source.shape[2] and \
               num_cols == img_source.shape[3], 'the shape of img_mask and img_source does not match'

        # Manually add an alpha channel if necessary
        if combine and img_source.shape[1] > 1:
            # assert use_alpha, 'Alpha channel must be enabled for this case.'
            use_alpha = True
            if img_source.shape[1] == 3:
                img_source = torch.cat([img_source, torch.ones_like(img_mask)], dim=1)

    # Rescale distance
    dist = as_comp_tensor(dist, dtype=img_mask.dtype, device=img_mask.device) * max_dist * num_rows / 256

    # Special cases for small distances
    if dist <= 1.0:
        img_mask = torch.zeros_like(img_mask) if dist == 0.0 else (img_mask > 0.5).float()
        if img_source is None:
            return img_mask.expand(img_mask.shape[0], 1 if mode == 'gray' else 3 if not use_alpha else 4, num_rows, num_cols)
        elif not combine:
            return img_source
        elif img_source.shape[1] == 1:
            return img_source * img_mask
        else:
            img_out = img_source.clone()
            img_out[:,[3],:,:] = img_out[:,[3],:,:] * img_mask
            return img_out

    # Calculate padding
    pad_dist = int(np.ceil(dist.item() if isinstance(dist, torch.Tensor) else dist)) + 1
    pad_cols = min(num_cols // 2, pad_dist)
    pad_rows = min(num_rows // 2, pad_dist)
    p2d = [pad_cols, pad_cols, pad_rows, pad_rows]

    if img_source is not None:
        img_out = torch.zeros_like(img_source)
    else:
        img_out = torch.zeros(img_mask.shape[0], 1 if mode == 'gray' else 3 if not use_alpha else 4, num_rows, num_cols, device=img_mask.device)

    # loop through batch
    for i in range(img_mask.shape[0]):
        # compute mask
        binary_mask = (img_mask[i,0,:,:] <= 0.5).unsqueeze(0).unsqueeze(0)
        binary_mask = torch.nn.functional.pad(binary_mask, p2d, mode='circular')
        binary_mask_np = binary_mask[0, 0].detach().cpu().numpy()
        # compute manhattan distance, closest point indices, non-zero mask
        # !! speed bottleneck !!
        dist_mtx, indices = spi.morphology.distance_transform_edt(binary_mask_np, return_distances=True, return_indices=True)
        dist_mtx = as_comp_tensor(dist_mtx.astype(np.float32), device=img_mask.device)
        dist_mtx = dist_mtx[p2d[2]:p2d[2]+num_rows, p2d[0]:p2d[0]+num_cols].unsqueeze(0).unsqueeze(0)
        dist_weights = torch.clamp(1.0 - dist_mtx / dist, 0.0, 1.0)
        indices = as_comp_tensor(indices[::-1, p2d[2]:p2d[2]+num_rows, p2d[0]:p2d[0]+num_cols].astype(np.float32), device=img_mask.device)

        if img_source is None:
            img_out[i,:,:,:] = dist_weights
        else:
            # normalize to screen coordinate
            indices[0,:,:] = (torch.remainder(indices[0,:,:] - p2d[0], num_cols) + 0.5) / num_cols * 2.0 - 1.0
            indices[1,:,:] = (torch.remainder(indices[1,:,:] - p2d[2], num_rows) + 0.5) / num_rows * 2.0 - 1.0
            # reshape to (1, num_rows, num_cols, 2) and convert to torch tensor
            sample_grid = indices.permute(1,2,0).unsqueeze(0)
            # sample grid and apply distance operator
            cur_img = torch.nn.functional.grid_sample(img_source[[i],:,:,:], sample_grid, mode='nearest', padding_mode='zeros', align_corners=False)
            if not combine:
                dist_mask = (dist_mtx >= dist).expand_as(cur_img)
                cur_img[dist_mask] = img_source[dist_mask]
            elif img_source.shape[1] == 1:
                cur_img = cur_img * dist_weights
            else:
                cur_img[:,[3],:,:] = cur_img[:,[3],:,:] * dist_weights

            img_out[i,:,:,:] = cur_img

    return img_out


@input_check(tensor_args=['img_in', 'height_map'])
def emboss(img_in, height_map, intensity=0.5, max_intensity=10.0, light_angle=0.0,
    highlight_color=[1.0, 1.0, 1.0, 1.0], shadow_color=[0.0, 0.0, 0.0, 1.0], outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: emboss (https://substance3d.adobe.com/documentation/sddoc/emboss-172825208.html)
        img_in (G or RGB(A))
        height_map (G only)
        intensity: normalized height_map multiplier
        light angle: angle of illumination (0 degree points to left)
        highlight color: as the name indicates
        shadow color: as the name indicates
        max_intensity: maximum height_map multiplier
        return imge (G or RGB(A))
    Behavior:
        very close to sbs
        **don't support alpha channel for highlight and shadow
    '''
    # defaults
    if img_in is None:
        if height_map is not None:
            img_in = torch.zeros(size=[height_map.shape[0], 1, height_map.shape[2], height_map.shape[3]], dtype=torch.float32, device=height_map.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if height_map is None:
        height_map = torch.zeros(size=[img_in.shape[0], 1, img_in.shape[2], img_in.shape[3]], dtype=torch.float32, device=img_in.device)

    height_map = grayscale_input_check(height_map, 'height map')
    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    light_angle = as_comp_tensor(light_angle, dtype=img_in.dtype, device=img_in.device) * np.pi * 2.0
    num_channels = img_in.shape[1]
    highlight_color = as_comp_tensor(highlight_color[:num_channels], dtype=img_in.dtype, device=img_in.device).view(1,num_channels,1,1).expand(*img_in.shape)
    shadow_color = as_comp_tensor(shadow_color[:num_channels], dtype=img_in.dtype, device=img_in.device).view(1,num_channels,1,1).expand(*img_in.shape)

    num_rows = img_in.shape[2]
    intensity = intensity * num_rows / 512
    N = normal(height_map, mode='object_space', intensity=intensity, max_intensity=max_intensity)
    weight = (N[:,0,:,:] * torch.cos(np.pi-light_angle) - N[:,1,:,:] * torch.sin(np.pi-light_angle)).unsqueeze(0).expand(*img_in.shape)

    img_out = torch.zeros_like(img_in)
    highlight_color = 2.0 * highlight_color - 1.0
    shadow_color = 2.0 * shadow_color - 1.0
    img_out[weight >= 0.0] = img_in[weight >= 0.0] + highlight_color[weight >= 0.0] * weight[weight >= 0.0]
    img_out[weight < 0.0]  = img_in[weight < 0.0]  + shadow_color[weight < 0.0] * (-weight[weight < 0.0])
    img_out = torch.clamp(img_out, 0.0, 1.0)

    if use_alpha:
        img_out = torch.cat([img_out, img_in_alpha], dim=1)

    return img_out


@input_check(tensor_args=['img_in'])
def gradient_map(img_in, interpolate=True, mode='color', use_alpha=False, anchors=None, interpmode=0, addressingrepeat=False, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: gradient map (https://substance3d.adobe.com/documentation/sddoc/gradient-map-172825246.html)
        img_in (G only)
        mode: "color" or "gray"
        anchors: matrix with shape [num_anchors,2];
                [:,0]  indicates the color before
                [:,>0] indicates the color after
        return imge (G or RGB)
    Implementation:
        linear interpolation between anchors
    Behavior:
        identical to sbs
        interpolate alpha if use_alpha
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, "input image")

    num_col = 2 if mode == 'gray' else 4 + use_alpha
    if anchors is None:
        anchors = torch.linspace(0.0, 1.0, 2, device=img_in.device).view(2, 1).repeat(1, num_col)
    else:
        anchors = as_comp_tensor(anchors, dtype=torch.float32, device=img_in.device)
        if anchors.shape[1] != num_col:
            if num_col == 2 and anchors.shape[1] >= 4:
                # average colors to get grayscale value
                anchors = torch.cat([anchors[:, [0]], anchors[:, 1:4].mean(dim=1, keepdims=True)], dim=1)
            elif num_col >= 4 and anchors.shape[1] == 2:
                # repeat grayscale value to get colors
                anchors = anchors[:, [0, 1, 1, 1]]

            if num_col == 5 and anchors.shape[1] == 4:
                # add alpha channel
                anchors = anchors[:, [0, 1, 2, 3, 3]]
                anchors[:, 4] = 1.0

            if anchors.shape[1] != num_col:
                raise RuntimeError('Could not convert anchors to the requested shape.')
                # this should only happen if the anchor shape or num_col is not in the expected set [2, 4, 5]

            # # print('WARNING: color mode does not match anchor shape.')
            # num_col = anchors.shape[1]

        # assert anchors.shape[1] == num_col, "shape of anchors doesn't match color mode"
        shuffle_idx = torch.argsort(anchors[:,0])
        anchors = anchors[shuffle_idx, :]

    num_anchors = anchors.shape[0]

    if num_anchors == 0:
        return img_in

    # compute mapping
    img_out = torch.zeros(img_in.shape[0], img_in.shape[2], img_in.shape[3], num_col-1, device=img_in.device)
    img_in = img_in.squeeze(1)

    img_out[img_in < anchors[0,0], :] = anchors[0,1:]
    img_out[img_in >= anchors[num_anchors-1,0], :] = anchors[num_anchors-1,1:]
    for j in range(num_anchors-1):
        a = (img_in.unsqueeze(3) - anchors[j,0]) / (anchors[j+1,0] - anchors[j,0] + 1e-8)
        if not interpolate:  # this should be a Bezier curve
            a = a ** 2 * 3 - a ** 3 * 2
        img_map = (1 - a) * anchors[j,1:] + a * anchors[j+1,1:]
        cond = (img_in >= anchors[j,0]) & (img_in < anchors[j+1,0])
        img_out[cond,:] = img_map[cond,:]

    # revert the order of dimensions
    img_out = img_out.permute(0,3,1,2)
    return img_out

@input_check(tensor_args=['img_in', 'img_gradient'])
def gradient_map_dyn(img_in, img_gradient, orientation='horizontal', use_alpha = False, position=0.0, addressingrepeat=False, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: gradient map (https://substance3d.adobe.com/documentation/sddoc/gradient-map-172825246.html)
        img_in (G only)
        img_gradient (G or RGB(A))
        ...
        return imge (G or RGB)
    Implementation:
        linear interpolation between anchors
    Behavior:
        identical to sbs
        add opaque alpha if use_alpha
    '''
    # defaults
    if img_in is None:
        if img_gradient is not None:
            img_in = torch.zeros(size=[img_gradient.shape[0], 1, img_gradient.shape[2], img_gradient.shape[3]], dtype=torch.float32, device=img_gradient.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if img_gradient is None:
        img_gradient = torch.zeros(size=[img_in.shape[0], 3, img_in.shape[2], img_in.shape[3]], dtype=torch.float32, device=img_in.device)

    img_in = grayscale_input_check(img_in, "input image")
    if img_gradient.shape[1] == 3 or img_gradient.shape[1] == 4:
        img_gradient = img_gradient[:,:3,:,:]
        mode = 'color'
    else:
        mode = 'gray'

    assert img_gradient.shape[0] == 1, "please input a single gradient image"

    h_res, w_res = img_in.shape[2], img_in.shape[3]
    grad_h_res, grad_w_res = img_gradient.shape[2], img_gradient.shape[3]
    gs_interp_mode = 'bilinear'
    gs_padding_mode ='zeros'

    img_in_perm = img_in.permute(0, 2, 3, 1)
    if orientation == 'vertical':
        row_grid = img_in_perm * 2.0 - 1.0
        col_grid = torch.tensor((position * 2.0 - 1.0) * (grad_w_res - 1) / grad_w_res, dtype=torch.float32, device=img_in.device).expand_as(img_in_perm)
    else:
        row_grid = torch.tensor((position * 2.0 - 1.0) * (grad_h_res - 1) / grad_h_res, dtype=torch.float32, device=img_in.device).expand_as(img_in_perm)
        col_grid = img_in_perm * 2.0 - 1.0

    sample_grid = torch.cat([col_grid, row_grid], dim=3)
    img_out = torch.nn.functional.grid_sample(img_gradient, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)

    return img_out

@input_check(tensor_args=['img_in'])
def c2g(img_in, flatten_alpha=False, rgba_weights=[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0], bg=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: grayscale conversion (https://substance3d.adobe.com/documentation/sddoc/grayscale-conversion-172825250.html)
        img_in (RGB(A) only)
        rgba_weights: a weight vector of length 4 for RGBA channels
        bg: bg ground color for alpha blending
        return image (G only)
    Behavior:
        identical to sbs
        use alpha blending with background when flatten_alpha=True
    TODO
        - check why output a uniform image (when setting alpha to 1.0) will cause an invalid .exr file
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image')

    rgba_weights = as_comp_tensor(rgba_weights, dtype=img_in.dtype, device=img_in.device)
    img_out = (img_in * rgba_weights[:img_in.shape[1]].view(1,img_in.shape[1],1,1)).sum(dim=1, keepdim=True)
    if flatten_alpha and img_in.shape[1] == 4:
        img_out = img_out * img_in[:,3,:,:] + bg * (1.0 - img_in[:,3,:,:])

    return img_out


@input_check(tensor_args=['img_in'])
def hsl(img_in, hue=0.5, saturation=0.5, lightness=0.5, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: hsl, the actual formula is hsv (https://substance3d.adobe.com/documentation/sddoc/hsl-172825254.html)
        img_in (RGB(A) only)
        hue: hue adjustment
        saturation: saturation adjustment
        lightness: lightness adjustment
        return image (RGB(A) only)
    Behavior:
        identical to sbs
        alpha won't be modified
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image')

    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    r = img_in[:,0,:,:]
    g = img_in[:,1,:,:]
    b = img_in[:,2,:,:]

    # compute s,v
    max_vals, _ = torch.max(img_in, 1, False)
    min_vals, _ = torch.min(img_in, 1, False)
    delta = max_vals - min_vals
    delta_mask = delta > 0.0
    l = (max_vals + min_vals) / 2.0
    s = torch.zeros_like(delta)
    s_mask = (l > 0.0) * (l < 1.0)
    # still need a small constant...
    s[s_mask] = delta[s_mask] / (1.0 - torch.abs(2*l[s_mask] - 1.0) + 1e-8)
    h = torch.zeros_like(s)

    # compute h
    red_mask = (img_in[:,0,:,:] == max_vals) * delta_mask
    green_mask = (img_in[:,1,:,:] == max_vals) * delta_mask
    blue_mask = (img_in[:,2,:,:] == max_vals) * delta_mask
    h[red_mask] = torch.remainder((g[red_mask]-b[red_mask])/delta[red_mask], 6.0) / 6.0
    h[green_mask] = ((b[green_mask]-r[green_mask])/delta[green_mask] + 2.0) / 6.0
    h[blue_mask] = ((r[blue_mask]-g[blue_mask])/delta[blue_mask] + 4.0) / 6.0

    # modify hsv
    h = torch.remainder(h + (hue-0.5) * 2.0 + 2.0, 1.0)
    l = torch.clamp(l + 2.0*lightness - 1.0, 0.0, 1.0)
    s = torch.clamp(s + 2.0*saturation - 1.0, 0.0, 1.0)

    # convert back to rgb
    c = (1.0 - torch.abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - torch.abs( torch.remainder(h/(1.0/6.0), 2.0) - 1.0))
    m = l - c/2.0

    r_out = torch.zeros_like(r)
    g_out = torch.zeros_like(g)
    b_out = torch.zeros_like(b)
    h_1_mask = (h >= 0.0) * (h < 1.0/6.0)
    h_2_mask = (h >= 1.0/6.0) * (h < 2.0/6.0)
    h_3_mask = (h >= 2.0/6.0) * (h < 3.0/6.0)
    h_4_mask = (h >= 3.0/6.0) * (h < 4.0/6.0)
    h_5_mask = (h >= 4.0/6.0) * (h < 5.0/6.0)
    h_6_mask = (h >= 5.0/6.0) * (h <= 6.0/6.0)
    r_out[h_1_mask + h_6_mask] = c[h_1_mask + h_6_mask]
    r_out[h_2_mask + h_5_mask] = x[h_2_mask + h_5_mask]
    g_out[h_1_mask + h_4_mask] = x[h_1_mask + h_4_mask]
    g_out[h_2_mask + h_3_mask] = c[h_2_mask + h_3_mask]
    b_out[h_3_mask + h_6_mask] = x[h_3_mask + h_6_mask]
    b_out[h_4_mask + h_5_mask] = c[h_4_mask + h_5_mask]

    rgb_out = torch.stack([r_out, g_out, b_out], dim=1) + m

    if use_alpha:
        rgb_out = torch.cat([rgb_out, img_in_alpha], dim=1)

    return rgb_out


@input_check(tensor_args=['img_in'])
def levels(img_in, in_low=0.0, in_mid=0.5, in_high=1.0, out_low=0.0, out_high=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: levels (https://substance3d.adobe.com/documentation/sddoc/levels-172825279.html)
        img_in (G or RGB)
        in_low  : lowest bound(threshold) for input
        in_mid  : middle point for calculating the gamma correction
        in_high : highest bound(threshold) for input
        out_low : lowest bound(threshold) for output
        out_high: highest bound(threshold) for output
        return image (G or RGB)
    Implementation:
        - threshold image based on in_low, in_high
        - use in_mid to compute the gamma correction term and apply gamma correction
        - remap based on out_low, out_high
    Behavior:
        identical to sbs
        alpha channel is processed according to the last list element
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    def param_process(param_in, default_val):
        if not isinstance(param_in, (list, torch.Tensor)):
            param_in = [param_in]
        param_in = as_comp_tensor(param_in, dtype=img_in.dtype, device=img_in.device)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1: # grayscale to color
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and img_in.shape[1] == 4: # add alpha
            param_in = torch.cat([param_in, torch.tensor([default_val], device=param_in.device)])
        return param_in

    num_channels = img_in.shape[1]
    limit = torch.tensor(9.0, device=img_in.device)
    in_low = param_process(in_low, 0.0)
    in_mid = param_process(in_mid, 0.5)
    in_high = param_process(in_high, 1.0)
    out_low = param_process(out_low, 0.0)
    out_high = param_process(out_high, 1.0)

    img_out = torch.zeros_like(img_in)
    for i in range(num_channels):
        if in_low[i] > in_high[i]:
            img_in_slice = 1.0 - img_in[:,i,:,:].clone()
            left, right = in_high[i], in_low[i]
        else:
            img_in_slice = img_in[:,i,:,:].clone()
            left, right = in_low[i], in_high[i]
        if left == right:
            right = right + 0.0001

        gamma_corr = 1.0 + (8.0 * torch.abs(2.0 * in_mid[i] - 1.0))
        gamma_corr = torch.min(gamma_corr, limit)
        if in_mid[i] < 0.5:
            gamma_corr = 1.0 / gamma_corr

        img_in_slice = torch.min(torch.max(img_in_slice, left), right)
        # magic number 1e-15
        img_slice = torch.pow((img_in_slice - left + 1e-15) / (right - left + 1e-15), gamma_corr)

        if out_low[i] > out_high[i]:
            img_slice = 1.0 - img_slice
            left, right = out_high[i], out_low[i]
        else:
            left, right = out_low[i], out_high[i]
        img_out_slice = img_slice * (right - left) + left
        img_out_slice = torch.min(torch.max(img_out_slice, left), right)
        img_out[:,i,:,:] = img_out_slice

    return img_out

    # in_low_high = torch.zeros(2, num_channels)
    # img_in_ = img_in.clone()
    # for i in range(num_channels):
    #     if in_low[i] > in_high[i]:
    #         in_low_high[0,i] = in_high[i]
    #         in_low_high[1,i] = in_low[i]
    #         img_in_[:,i,:,:] = 1.0 - img_in_[:,i,:,:]
    #     elif in_low[i] == in_high[i]:
    #         in_low_high[0,i] = in_low[i]
    #         in_low_high[1,i] = in_high[i] + 1e-4
    #     else:
    #         in_low_high[0,i] = in_low[i]
    #         in_low_high[1,i] = in_high[i]
    # in_mid = in_mid[:num_channels].view(1,num_channels,1,1)
    # in_low = in_low_high[0,:].view(1,num_channels,1,1)
    # in_high = in_low_high[1,:].view(1,num_channels,1,1)

    # gamma_corr = 1.0 + (8.0 * torch.abs(2.0 * in_mid - 1.0))
    # gamma_corr = torch.min(gamma_corr, limit)
    # gamma_mask = in_mid < 0.5
    # gamma_corr[gamma_mask] = 1.0 / gamma_corr[gamma_mask]

    # img_in_ = torch.min(torch.max(img_in_, in_low), in_high)
    # # magic number 1e-15
    # img_in_ = torch.pow((img_in_ - in_low + 1e-15) / (in_high - in_low + 1e-15), gamma_corr)

    # out_low_high = torch.zeros(2, num_channels)
    # for i in range(num_channels):
    #     if out_low[i] > out_high[i]:
    #         out_low_high[0,i] = out_high[i]
    #         out_low_high[1,i] = out_low[i]
    #         img_in_[:,i,:,:] = 1.0 - img_in_[:,i,:,:]
    #     else:
    #         out_low_high[0,i] = out_low[i]
    #         out_low_high[1,i] = out_high[i]
    # out_low = out_low_high[0,:].view(1,num_channels,1,1)
    # out_high = out_low_high[1,:].view(1,num_channels,1,1)

    # img_out = img_in_ * (out_high - out_low) + out_low
    # img_out = torch.min(torch.max(img_out, out_low), out_high)

    # return img_out

@input_check(tensor_args=['img_in'])
def normal(img_in, mode='tangent_space', normal_format='dx', use_input_alpha=False, use_alpha=False, intensity=1.0/3.0, max_intensity=3.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: normal (https://substance3d.adobe.com/documentation/sddoc/normal-172825289.html)
        img_in (G only)
        intensity: normalized height map multiplier on dx, dy
        mode: 'tangent_space' or 'object_space'
        max_intensity: maximum height map multiplier
        normal_format: flips Y coordinates of normals
        return imge (RGB(A) only)
    Behavior:
        identical to sbs
        if use_alpha, add an opaque alpha mask
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, "input height field")

    img_size = img_in.shape[2]
    intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
    dx = roll_col(img_in, -1) - img_in
    dy = roll_row(img_in, -1) - img_in
    if normal_format == 'gl':
        img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), 1)
    elif normal_format == 'dx':
        img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
    else:
        img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
    img_out = normalize(img_out)
    if mode == 'tangent_space':
        img_out = img_out / 2.0 + 0.5

    if use_alpha == True:
        if use_input_alpha:
            img_out = torch.cat([img_out, img_in], dim=1)
        else:
            img_out = torch.cat([img_out, torch.ones(img_out.shape[0], 1, img_out.shape[2], img_out.shape[3], device=img_in.device)], dim=1)

    return img_out


@input_check(tensor_args=['img_in'])
def sharpen(img_in, intensity=1.0 / 3.0, max_intensity=3.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: sharpen (https://substance3d.adobe.com/documentation/sddoc/sharpen-172825322.html)
        img_in (G or RGB)
        intensity: normalized unsharp mask multiplier
        max_intensity: maximum unsharp mask multiplier
        return image (G or RGB)
    Implementation
        img_out = img_in + intensity*(img_in - img_in_blured)
        blur kernel uses a gaussian kernel with sigma=0.5, subject to change
    Behavior:
        identical to sbs
        alpha mask is treated the same as the color channels, so no special treatment is required
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    num_group = img_in.shape[1]
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    kernel = as_comp_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=img_in.dtype, device=img_in.device)
    kernel = kernel.view(1,1,3,3).expand(num_group, 1, 3, 3)
    # manually pad input image
    p2d = [1, 1, 1, 1]
    in_pad  = torch.nn.functional.pad(img_in, p2d, mode='circular')
    # perform depth-wise convolution without implicit padding
    unsharp_mask = torch.nn.functional.conv2d(in_pad, kernel, groups=num_group, padding = 0)
    img_out = torch.clamp(img_in + unsharp_mask * intensity, torch.tensor(0.0, device=img_in.device), torch.tensor(1.0, device=img_in.device))
    return img_out

@input_check(tensor_args=['img_in'])
def transform_2d(img_in, tile_mode=3, sample_mode='bilinear', mipmap_mode='auto', mipmap_level=0, x1=1.0, x1_max=1.0, x2=0.5, x2_max=1.0,
                 x_offset=0.5, x_offset_max=1.0, y1=0.5, y1_max=1.0, y2=1.0, y2_max=1.0, y_offset=0.5, y_offset_max=1.0, mattecolor=[0.0, 0.0, 0.0, 0.0], outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: transform 2d (https://substance3d.adobe.com/documentation/sddoc/transformation-2d-172825332.html)
        img_in (G or RGB(A))
        tile_mode: 0=no tile, 1=horizontal tile, 2=vertical tile, 3=horizontal and vertical tile
        x1, x2, x_offset, y1, y2, y_offset: element in the affine transformation matrix
        return image (G or RGB(A), same as img_in)
    Implementation:
        compute affine transformation sampling grid, then use torch's grid_sample
    Behavior:
        identical to sbs
        alpha mask is treated the same as the color channels, so no special treatment is required
    Additional notes from Beichen:
        The overall process is: creating mipmap levels -> resizing output -> transformation.
        The 'automatic' mipmap mode samples the input progressively, while the 'manual' mode samples only once.
        Under 'automatic' mode, scale factors greater than 1 also count into mipmap levels.
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    assert sample_mode in ('bilinear', 'nearest')
    assert mipmap_mode in ('auto', 'manual')

    gs_padding_mode = 'zeros'
    gs_interp_mode = sample_mode

    x1 = as_comp_tensor((x1 * 2.0 - 1.0) * x1_max, dtype=img_in.dtype, device=img_in.device).squeeze()
    x2 = as_comp_tensor((x2 * 2.0 - 1.0) * x2_max, dtype=img_in.dtype, device=img_in.device).squeeze()
    x_offset = as_comp_tensor((x_offset * 2.0 - 1.0) * x_offset_max, dtype=img_in.dtype, device=img_in.device).squeeze()
    y1 = as_comp_tensor((y1 * 2.0 - 1.0) * y1_max, dtype=img_in.dtype, device=img_in.device).squeeze()
    y2 = as_comp_tensor((y2 * 2.0 - 1.0) * y2_max, dtype=img_in.dtype, device=img_in.device).squeeze()
    y_offset = as_comp_tensor((y_offset * 2.0 - 1.0) * y_offset_max, dtype=img_in.dtype, device=img_in.device).squeeze()

    # compute mipmap level
    mm_level = mipmap_level
    det = torch.abs(x1 * y2 - x2 * y1)
    if det < 1e-6:
        print('Warning: singular transformation matrix may lead to unexpected results.')
        mm_level = 0
    elif mipmap_mode == 'auto':
        inv_h1 = torch.sqrt(x2 * x2 + y2 * y2)
        inv_h2 = torch.sqrt(x1 * x1 + y1 * y1)
        max_compress_ratio = torch.max(inv_h1, inv_h2)
        # !! this is a hack !!
        upper_limit = 2895.329
        thresholds = torch.tensor([upper_limit / (1 << i) for i in reversed(range(12))], device=img_in.device)
        mm_level = torch.sum(max_compress_ratio > thresholds).item()
        # Special cases
        is_pow2 = lambda x: torch.remainder(torch.log2(x), 1.0) == 0
        if torch.abs(x1) == torch.abs(y2) and x2 == 0 and y1 == 0 and is_pow2(torch.abs(x1)) or \
           torch.abs(x2) == torch.abs(y1) and x1 == 0 and y2 == 0 and is_pow2(torch.abs(x2)):
            scale = torch.max(torch.abs(x1), torch.abs(x2))
            if torch.remainder(x_offset * scale, 1.0) == 0 and torch.remainder(y_offset * scale, 1.0) == 0:
                mm_level = max(0, mm_level - 1)

    # mipmapping (optional)
    if mm_level > 0:
        mm_level = min(mm_level, min(int(np.floor(np.log2(img_in.shape[2]))), int(np.floor(np.log2(img_in.shape[3])))))
        img_mm = automatic_resize(img_in, -mm_level)
        img_mm = manual_resize(img_mm, mm_level)
        assert img_mm.shape == img_in.shape
    else:
        img_mm = img_in

    # compute sampling tensor
    res_x, res_y = img_in.shape[3], img_in.shape[2]
    theta_first_row = torch.stack([x1, y1, x_offset * 2.0])
    theta_second_row = torch.stack([x2, y2, y_offset * 2.0])
    theta = torch.stack([theta_first_row, theta_second_row]).unsqueeze(0).expand(img_in.shape[0],2,3)
    sample_grid = torch.nn.functional.affine_grid(theta, img_in.shape, align_corners=False).to(dtype=img_mm.dtype)

    if tile_mode in (1, 3):
        sample_grid[:,:,:,0] = (torch.remainder(sample_grid[:,:,:,0] + 1.0, 2.0) - 1.0) * res_x / (res_x + 2)
    if tile_mode in (2, 3):
        sample_grid[:,:,:,1] = (torch.remainder(sample_grid[:,:,:,1] + 1.0, 2.0) - 1.0) * res_y / (res_y + 2)

    # Pad input image
    if tile_mode == 0:
        img_pad = img_mm
    else:
        pad_arr = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
        img_pad = torch.nn.functional.pad(img_mm, pad_arr[tile_mode], mode='circular')

    # compute output
    img_out = torch.nn.functional.grid_sample(img_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)

    return img_out

@input_check(tensor_args=['img_in'])
def special_transform(img_in, tile_mode=3, sample_mode='bilinear', scale=0.25, scale_max=4.0, x_offset=0.5, x_offset_max=1.0,
                      y_offset=0.5, y_offset_max=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    special transform for only changing the scale and offset of the image
        img_in (G or RGB)
        tile_mode: 0=no tile, 1=horizontal tile, 2=vertical tile, 3=horizontal and vertical tile
        scale, x_offset, y_offset: for changing scale and offset
        return imge (G or RGB, same as img_in)
    Implementation:
        call transform_2d internally
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    assert sample_mode in ('bilinear', 'nearest')
    scale = scale / 2.0 + 0.5
    img_out = transform_2d(img_in, tile_mode, sample_mode, 'manual', 0, scale, scale_max, 0.5, 1.0, x_offset, x_offset_max, 0.5, 1.0, scale, scale_max, y_offset, y_offset_max)

    return img_out

@input_check(tensor_args=[])
def uniform_color(mode='color', num_imgs=1, outputsize=[9, 9], use_alpha=False, rgba=[0.0, 0.0, 0.0, 1.0], device=torch.device('cpu')):
    '''
    Atomic function: uniform color (https://substance3d.adobe.com/documentation/sddoc/uniform-color-172825339.html)
        mode: 'gray' (use red channel) or 'color'
        rgb: vector of rgb or grayscale value
        return image (G or RGB)
    Behavior:
        identical to sbs
    '''
    def param_process(param_in):
        param_in = as_comp_tensor(param_in, dtype=torch.float32, device=device)
        if not isinstance(param_in, torch.Tensor):
            param_in = param_in.to(device)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1:
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and use_alpha:
            param_in = torch.cat([param_in, torch.tensor([1.0], device=param_in.device)])
        return param_in

    res_h = 1 << outputsize[0]
    res_w = 1 << outputsize[1]

    rgba = param_process(rgba)
    if mode == 'gray':
        img_out = torch.ones(num_imgs,1,res_h,res_w, device=rgba.device) * rgba[0]
    else:
        if use_alpha:
            img_out = torch.ones(num_imgs,4,res_h,res_w, device=rgba.device) * rgba.view(1,4,1,1)
        else:
            img_out = torch.ones(num_imgs,3,res_h,res_w, device=rgba.device) * rgba[:3].view(1,3,1,1)
    return img_out


@input_check(tensor_args=['img_in', 'intensity_mask'])
def warp(img_in, intensity_mask, intensity=0.5, max_intensity=2.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Atomic function: warp (https://substance3d.adobe.com/documentation/sddoc/warp-172825344.html)
        img_in (G or RGB(A))
        intensity_mask: intensity mask for computing displacement (default: all ones, this gives constant intensity defined by 'intensity')
        intensity: normalized intensity_mask multiplier
        max_intensity: maximum intensity mask multiplier
        return image (G or RGB(A))
    Implementation:
        1) compute gradient of the intensity mask
        2) multiply the mask with intensity and image resolution
        3) shift the pixel according to the mask
    Behavior:
        identical to sbs
        alpha changes with warp
    '''
    # defaults
    if img_in is None:
        if intensity_mask is not None:
            img_in = torch.zeros(size=[intensity_mask.shape[0], 1, intensity_mask.shape[2], intensity_mask.shape[3]], dtype=torch.float32, device=intensity_mask.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if intensity_mask is None:
        intensity_mask = torch.zeros(size=[img_in.shape[0], 1, *img_in.shape[2:]], dtype=img_in.dtype, device=img_in.device)

    intensity_mask = grayscale_input_check(intensity_mask, "intensity mask")

    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    row_scale = num_row / 256.0 # magic number, similar to d_warp
    col_scale = num_col / 256.0 # magic number
    shift = torch.cat([intensity_mask - roll_row(intensity_mask, -1), intensity_mask - roll_col(intensity_mask, -1)], 1)
    row_grid, col_grid = torch.meshgrid(torch.linspace(0, num_row-1, num_row, device=img_in.device), torch.linspace(0, num_col-1, num_col, device=img_in.device))
    row_shift = shift[:,[0],:,:] * intensity * num_row * row_scale
    col_shift = shift[:,[1],:,:] * intensity * num_col * col_scale
    # mod the index to behavior as tiling
    row_grid = torch.remainder((row_grid + row_shift + 0.5) / num_row * 2.0, 2.0) - 1.0
    col_grid = torch.remainder((col_grid + col_shift + 0.5) / num_col * 2.0, 2.0) - 1.0
    row_grid = row_grid * num_row / (num_row + 2)
    col_grid = col_grid * num_col / (num_col + 2)
    sample_grid = torch.cat([col_grid, row_grid], 1).permute(0,2,3,1).expand(intensity_mask.shape[0], num_row, num_col, 2)
    in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], mode='circular')
    img_out = torch.nn.functional.grid_sample(in_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
    return img_out

@input_check(tensor_args=['img_in'])
def passthrough(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Helper function: dot node (https://substance3d.adobe.com/documentation/sddoc/graph-items-186056822.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    return img_in


# ---------------------------------------------
# Non-atomic functions

# add sbsnode support
@input_check(tensor_args=['img_in'])
def linear_to_srgb(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: Convert to sRGB (https://substance3d.adobe.com/documentation/sddoc/convert-to-srgb-159449240.html)
        input (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha doesn't change
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if img_in.shape[1] == 3 or img_in.shape[1]==4:
        img_out = levels(img_in, in_mid=[0.425,0.425,0.425,0.5])
    else:
        img_out = levels(img_in, in_mid=[0.425])

    return img_out

# add sbsnode support
@input_check(tensor_args=['img_in'])
def srgb_to_linear(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: Convert to linear (https://substance3d.adobe.com/documentation/sddoc/convert-to-linear-159449235.html)
        input (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha doesn't change
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if img_in.shape[1] == 3 or img_in.shape[1]==4:
        img_out = levels(img_in, in_mid=[0.575,0.575,0.575,0.5])
    else:
        img_out = levels(img_in, in_mid=[0.575])
    return img_out

@input_check(tensor_args=['normal'])
def curvature(normal, normal_format='dx', emboss_intensity=0.1, emboss_max_intensity=10.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: curvature (https://substance3d.adobe.com/documentation/sddoc/curvature-filter-node-159450514.html)
        normal (RGB(A) only)
        emboss_intensity: normalized intensity for the internal emboss node
        emboss_max_intensity: maximum emboss intensity
        normal_format: 'dx' or 'gl'
        return image (G only)
    Behavior:
        identical to sbs
        no alpha in output
    '''
    # defaults
    if normal is None:
        normal = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    normal = color_input_check(normal, 'input normal')

    normal_col_shifted = roll_col(normal, 1)[:,[0],:,:]
    normal_row_shifted = roll_row(normal, 1)[:,[1],:,:]
    gray_outputsize = [int(math.log2(normal_col_shifted.shape[2])), int(math.log2(normal_col_shifted.shape[3]))]
    gray = uniform_color(mode='gray', rgba=0.5, outputsize=gray_outputsize, device=normal.device)
    pixel_size = 2048 / normal_col_shifted.shape[2] * 0.1
    embossed_col = emboss(gray, normal_col_shifted, emboss_intensity * pixel_size, emboss_max_intensity)
    embossed_row = emboss(gray, normal_row_shifted, emboss_intensity * pixel_size, emboss_max_intensity,
                          light_angle=0.25 if normal_format == 'dx' else 0.75)
    img_out = blend(embossed_col, embossed_row, opacity=0.5, blending_mode='add_sub')

    return img_out


@input_check(tensor_args=['img_in'])
def invert(img_in, invert_switch=True, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: invert (https://substance3d.adobe.com/documentation/sddoc/invert-159449221.html)
        img_in (G or RGB(A))
        ruturn image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha doesn't change
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if invert_switch:
        img_out = img_in.clone()
        if img_out.shape[1] == 1:
            img_out = torch.clamp(1.0 - img_out, 0.0, 1.0)
        else:
            img_out[:,:3,:,:] = torch.clamp(1.0 - img_in[:,:3,:,:].clone(), 0.0, 1.0)
        return img_out
    else:
        return img_in

@input_check(tensor_args=['img_in'])
def histogram_scan(img_in, invert_position=False, position=0.0, contrast=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: histogram scan (https://substance3d.adobe.com/documentation/sddoc/histogram-scan-159449213.html)
        img_in (G only)
        position: used to shift the middle point
        contrast: used to adjust the contrast of input
        invert_position: boolean indicates whether to invert the final output
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')

    position = as_comp_tensor(position, dtype=img_in.dtype, device=img_in.device)
    contrast = as_comp_tensor(contrast, dtype=img_in.dtype, device=img_in.device)
    position_ = position if invert_position else 1.0 - position
    start_low = (torch.max(as_comp_tensor(0.5, dtype=img_in.dtype, device=img_in.device), position_) - 0.5) * 2.0
    end_low = torch.min(position_ * 2.0, as_comp_tensor(1.0, dtype=img_in.dtype, device=img_in.device))
    weight_low = torch.clamp(contrast * 0.5, 0.0, 1.0)
    in_low = torch.clamp(lerp(start_low, end_low, weight_low), 0.0, 1.0)
    in_high = torch.clamp(lerp(end_low, start_low, weight_low), 0.0, 1.0)
    img_out = levels(img_in, in_low, 0.5, in_high, 0.0, 1.0)
    return img_out

@input_check(tensor_args=['img_in'])
def histogram_range(img_in, ranges=0.5, position=0.5, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: histogram range (https://substance3d.adobe.com/documentation/sddoc/histogram-range-159449207.html)
        img_in (G only)
        ranges: How much to reduce the range down from. This is similar to moving both Levels min and Max sliders inwards.
        position: Offset for the range reduction, setting a different midpoint for the range reduction.
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')

    ranges = as_comp_tensor(ranges, dtype=img_in.dtype, device=img_in.device)
    position = as_comp_tensor(position, dtype=img_in.dtype, device=img_in.device)
    out_low  = torch.clamp(1.0 - torch.min(ranges * 0.5 + (1.0 - position), (1.0 - position) * 2.0), 0.0, 1.0)
    out_high = torch.clamp(torch.min(ranges * 0.5 + position, position * 2.0), 0.0, 1.0)
    img_out = levels(img_in, 0.0, 0.5, 1.0, out_low, out_high)
    return img_out

@input_check(tensor_args=['img_in'])
def histogram_select(img_in, position=0.5, ranges=0.25, contrast=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: histogram select (https://substance3d.adobe.com/documentation/sddoc/histogram-select-166363409.html)
        img_in (G only)
        position: Sets the middle position where the range selection happens.
        ranges: Sets width of the selection range.
        Contrast: Adjusts the contrast/falloff of the result.
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')
    position = as_comp_tensor(position, dtype=img_in.dtype, device=img_in.device)
    ranges = as_comp_tensor(ranges, dtype=img_in.dtype, device=img_in.device)
    contrast = as_comp_tensor(contrast, dtype=img_in.dtype, device=img_in.device)

    if ranges == 0.0:
        return torch.ones_like(img_in)
    img = torch.clamp(1.0 - torch.abs(img_in - position) / ranges, 0.0, 1.0)
    img_out = levels(img, contrast * 0.5, 0.5, 1.0 - contrast * 0.5, 0.0, 1.0)
    return img_out

@input_check(tensor_args=['img_in'])
def edge_detect(img_in, invert_flag=False, edge_width=2.0/16.0, max_edge_width=16.0, edge_roundness=4.0/16.0, max_edge_roundness=16.0,
                tolerance=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: edge detect (https://substance3d.adobe.com/documentation/sddoc/edge-detect-159450524.html)
        img_in (G only)
        edge_width: Width of the detected areas around the edges.
        edge_roundness: Rounds, blurs and smooths together the generated mask.
        tolerance: Tolerance treshold factor for where edges should appear.
        Invert: Inverts the result.
        return image (G only)
    Behavior:
        almost identical to sbs (difference caused by 'distance')
    TODO:
        - optimize performance
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')

    # Process input image
    edge_width = as_comp_tensor(edge_width, device=img_in.device) * max_edge_width
    edge_roundness = as_comp_tensor(edge_roundness, device=img_in.device) * max_edge_roundness
    tolerance = as_comp_tensor(tolerance, device=img_in.device)

    # Edge detect
    img_scale = 256.0 / min(img_in.shape[2], img_in.shape[3])
    in_blur = blur(img_in, img_scale, 1.0)
    blend_sub_1 = blend(in_blur, img_in, blending_mode='subtract')
    blend_sub_2 = blend(img_in, in_blur, blending_mode='subtract')
    img_out = blend(blend_sub_1, blend_sub_2, blending_mode='add')
    img_out = levels(img_out, 0.0, 0.5, 0.05, 0.0, 1.0)
    levels_out_high = lerp(as_comp_tensor(0.2, device=img_in.device), as_comp_tensor(1.2, device=img_in.device), tolerance) / 100.0
    img_out = levels(img_out, 0.002, 0.5, levels_out_high, 0.0, 1.0)

    # Add edge width
    max_dist = torch.max(edge_width - 1.0, as_comp_tensor(0.0, device=img_in.device))
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)

    # Edge roundness
    max_dist = torch.max(torch.ceil(edge_roundness), as_comp_tensor(0.0, device=img_in.device))
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)
    img_out = 1.0 - img_out
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)
    img_out = 1.0 - img_out if invert_flag else img_out
    return img_out

@input_check(tensor_args=['img_in'])
def safe_transform(img_in, tile=1, tile_safe_rot=True, symmetry='none', tile_mode=3, mipmap_mode='auto', mipmap_level=0,
                   offset_mode='manual', offset_x=0.0, offset_y=0.0, angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: safe transform (https://substance3d.adobe.com/documentation/sddoc/safe-transform-159450643.html)
        img_in (G or RGB(A))
        tile: Scales the input down by tiling it.
        tile_safe_rot: Determines the behaviors of the rotation, whether it should snap to safe values that don't blur any pixels.
        symmetry: Performs symmetric transformation on the input.
        offset(_x, _y): Moves or translates the result. Makes sure pixels are snapped and not interpolated.
        angle: Rotates input along angle.
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with transformation
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    # initial transform
    if symmetry == 'X':
        img_out = torch.flip(img_in, dims=[2])
    elif symmetry == 'Y':
        img_out = torch.flip(img_in, dims=[3])
    elif symmetry == 'X+Y':
        img_out = torch.flip(torch.flip(img_in, dims=[3]), dims=[2])
    elif symmetry=='none':
        img_out = img_in
    else:
        raise RuntimeError('unknown symmetry mode')
    # main transform
    angle = as_comp_tensor(angle, device=img_in.device)
    tile = as_comp_tensor(tile, device=img_in.device)
    offset_tile = torch.remainder(tile + 1.0, 2.0) * as_comp_tensor(0.5, dtype=img_in.dtype, device=img_in.device)
    if tile_safe_rot:
        angle = torch.floor(angle * 8.0) / 8.0
        angle_res = torch.remainder(torch.abs(angle), 0.25) * (np.pi * 2.0)
        tile = tile * (torch.cos(angle_res) + torch.sin(angle_res))
    offset_x = torch.floor(as_comp_tensor(offset_x, dtype=img_in.dtype, device=img_in.device) * num_col) / num_col + offset_tile
    offset_y = torch.floor(as_comp_tensor(offset_y, dtype=img_in.dtype, device=img_in.device) * num_row) / num_row + offset_tile
    # compute affine transformation matrix
    angle = angle * np.pi * 2.0
    scale_matrix = as_comp_tensor([[torch.cos(angle), -torch.sin(angle)],[torch.sin(angle), torch.cos(angle)]], dtype=img_in.dtype, device=img_in.device)
    rotation_matrix = as_comp_tensor([[tile, 0.0],[0.0, tile]], dtype=img_in.dtype, device=img_in.device)
    scale_rotation_matrix = torch.mm(rotation_matrix, scale_matrix)
    img_out = transform_2d(img_out, tile_mode=tile_mode, mipmap_mode=mipmap_mode, mipmap_level=mipmap_level,
                           x1=to_zero_one(scale_rotation_matrix[0,0]), x2=to_zero_one(scale_rotation_matrix[0,1]), x_offset=to_zero_one(offset_x),
                           y1=to_zero_one(scale_rotation_matrix[1,0]), y2=to_zero_one(scale_rotation_matrix[1,1]), y_offset=to_zero_one(offset_y))
    return img_out

@input_check(tensor_args=['img_in'])
def blur_hq(img_in, high_quality=False, intensity=10.0 / 16.0, max_intensity=16.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: blur HQ (https://substance3d.adobe.com/documentation/sddoc/blur-hq-159450455.html)
        img_in (G or RGB(A))
        intensity: Strength (Radius) of the blur. The higher this value, the further the blur will reach.
        max_intensity: Maximum value of 'intensity'.
        quality: Increases internal sampling amount for even higher quality, at reduced computation speed.
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with blur
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    # blur path 1s
    blur_intensity = intensity * 0.66
    blur_1 = d_blur(img_in, blur_intensity, 1.0, 0.0)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.125)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.25)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.875)
    if high_quality:
        # blur path 2
        blur_2 = d_blur(img_in, blur_intensity, 1.0, 0.0625)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.4375)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.1875)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.3125)
        # blending
        img_out = blend(blur_1, blur_2, opacity=0.5)
    else:
        img_out = blur_1
    return img_out

@input_check(tensor_args=['img_in', 'img_mask'])
def non_uniform_blur(img_in, img_mask, samples=4, blades=5, intensity=0.2, max_intensity=50.0, anisotropy=0.0, asymmetry=0.0, angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: non-uniform blur (https://substance3d.adobe.com/documentation/sddoc/non-uniform-blur-159450461.html)
        img_in (G or RGB(A))
        img_mask: blur map (G only).
        samples: Amount of samples, determines quality. Multiplied by amount of Blades.
        blades: Amount of sampling sectors, determines quality. Multiplied by amount of Samples.
        intensity: Maximum strength to apply the blur with.
        max_intensity: Maximum value of 'intensity'.
        anisotropy: Optionally adds directionality to the blur effect. Driven by the Angle parameter.
        asymmetry: Optionally adds a bias to the sampling. Driven by the Angle parameter.
        angle: Angle to set directionality and sampling bias.
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with blur
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    intensity = as_comp_tensor(intensity, device=img_in.device) * max_intensity
    anisotropy = as_comp_tensor(anisotropy, device=img_in.device)
    asymmetry = as_comp_tensor(asymmetry, device=img_in.device)
    angle = as_comp_tensor(angle, device=img_in.device)
    assert isinstance(samples, int) and samples >= 1 and samples <= 16
    assert isinstance(blades, int) and blades >= 1 and blades <= 9

    # compute progressive warping results based on 'samples'
    def non_uniform_blur_sample(img_in, img_mask, intensity=10.0, inner_rotation=0.0):
        img_out = img_in
        for i in range(1, blades + 1):
            e_vec = ellipse(blades, i, intensity, anisotropy, angle, inner_rotation, asymmetry)
            warp_intensity = e_vec.norm()
            warp_angle = torch.atan2(e_vec[1] + 1e-15, e_vec[0] + 1e-15) / (np.pi * 2.0)
            img_warp = d_warp(img_in, img_mask, warp_intensity, 1.0, warp_angle)
            img_out = blend(img_warp, img_out, None, 'switch', opacity=1.0 / (i + 1))
        return img_out

    # compute progressive blurring based on 'samples' and 'intensity'
    samples_level = torch.min(as_comp_tensor(samples, device=img_in.device), torch.ceil(intensity * np.pi).long())
    img_out = non_uniform_blur_sample(img_in, img_mask, intensity, 1 / samples)
    for i in range(1, samples_level):
        blur_intensity = intensity * torch.exp(-i * np.sqrt(np.log(1e3) / np.e) / samples_level.float()) ** 2
        img_out = non_uniform_blur_sample(img_out, img_mask, blur_intensity, 1 / (samples * (i + 1)))
    return img_out

@input_check(tensor_args=['img_in', 'custom_curve'])
def bevel(img_in, custom_curve, non_uniform_blur_flag=True, use_alpha=False, dist=0.75, max_dist=1.0, smoothing=0.0, max_smoothing=5.0, \
        normal_intensity=0.2, max_normal_intensity=50.0, corner_type='Round', use_custom_curve=False, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: bevel (https://substance3d.adobe.com/documentation/sddoc/bevel-filter-node-159450511.html)
        img_in (G only)
        custom_curve (G only)
        dist: How far the bevel effect should reach.
        smoothing: How much additional smoothing (blurring) to perform after the bevel.
        max_smoothing: Maximum value of 'smoothing' parameter.
        non_uniform_blur_flag: Whether smoothing should be done non-uniformly.
        return image (G only)
    Notes:
        - Current implementation assumes that corner type is 'round', and neither custom curve input nor normal output is included.
    Behavior:
        almost identical to sbs (difference caused by 'distance')
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if use_custom_curve:
        raise NotImplementedError('Support for custom curves is not implemented in the bevel function.')
    if corner_type != 'Round':
        raise NotImplementedError('Support for the given corner type is not implemented in the bevel function.')

    img_in = grayscale_input_check(img_in, 'input image')

    dist = as_comp_tensor(dist * 2.0 - 1.0, dtype=img_in.dtype, device=img_in.device) * max_dist
    smoothing = as_comp_tensor(smoothing, dtype=img_in.dtype, device=img_in.device) * max_smoothing

    # height
    height = img_in
    if dist > 0:
        height = distance(height, None, combine=True, dist=dist * 128, max_dist=1.0)
    elif dist < 0:
        height = invert(height)
        height = distance(height, None, combine=True, dist=-dist * 128, max_dist=1.0)
        height = invert(height)
    if smoothing > 0:
        if non_uniform_blur_flag:
            img_blur = blur(height, 0.5, 1.0)
            img_blur = levels(img_blur, 0.0, 0.5, 0.0214, 0.0, 1.0)
            height = non_uniform_blur(height, img_blur, 6, 5, smoothing, 1.0, 0.0, 0.0, 0.0)
        else:
            height = blur_hq(height, False, smoothing, 1.0)

    # normal
    normal_intensity = as_comp_tensor(normal_intensity, dtype=img_in.dtype, device=img_in.device) * max_normal_intensity
    normal_one = transform_2d(height, mipmap_mode='manual', x1=to_zero_one(-1.0), y2=to_zero_one(-1.0))
    normal_one = normal(normal_one, use_alpha=use_alpha, intensity=1.0, max_intensity=normal_intensity)
    normal_one = transform_2d(normal_one, mipmap_mode='manual', x1=to_zero_one(-1.0), y2=to_zero_one(-1.0))
    normal_one = levels(normal_one, [1.0,1.0,0.0,0.0], [0.5]*4, [0.0,0.0,1.0,1.0], [0.0,0.0,0.0,1.0], [1.0]*4)

    normal_two = normal(height, use_alpha=use_alpha, intensity=1.0, max_intensity=normal_intensity)
    normal_two = levels(normal_two, [0.0]*4, [0.5]*4, [1.0]*4, [0.0,0.0,0.0,1.0], [1.0]*4)

    normal_out = blend(normal_one, normal_two, None, 'copy', opacity=0.5)

    return height, normal_out

@input_check(tensor_args=['img_in', 'img_mask'])
def slope_blur(img_in, img_mask, samples=1, mode='blur', intensity=10.0 / 16.0, max_intensity=16.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: slope blur (https://substance3d.adobe.com/documentation/sddoc/slope-blur-159450467.html)
        img_in (G or RGB(A))
        img_mask (G only) the angle of the blur anisotropy
        samples: Amount of samples, affects the quality at the expense of speed.
        intensity: Blur amount or strength.
        max_intensity: Maximum value of 'intensity'.
        mode: Blending mode for consequent blur passes. "Blur" behaves more like a standard Anisotropic Blur, while
              Min will "eat away" existing areas and Max will "smear out" white areas.
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with blur

    '''
    # defaults
    if img_in is None:
        if img_mask is not None:
            img_in = torch.zeros(size=[img_mask.shape[0], 3, img_mask.shape[2], img_mask.shape[3]], dtype=torch.float32, device=img_mask.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if img_mask is None:
        img_mask = torch.zeros(size=[img_in.shape[0], 1, *img_in.shape[2:]], dtype=img_in.dtype, device=img_in.device)

    img_mask = grayscale_input_check(img_mask, 'slope map')

    assert isinstance(samples, int) and samples >= 1 and samples <= 32
    assert mode in ['blur', 'min', 'max']
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    if intensity == 0 or torch.min(img_in) == torch.max(img_in):
        return img_in
    # progressive warping and blending
    warp_intensity = intensity / samples
    img_warp = warp(img_in, img_mask, warp_intensity, 1.0)
    img_out = img_warp
    blending_mode = 'copy' if mode == 'blur' else mode
    for i in range(2, samples + 1):
        img_warp_next = warp(img_warp, img_mask, warp_intensity, 1.0)
        img_out = blend(img_warp_next, img_out, None, blending_mode, opacity=1 / i)
        img_warp = img_warp_next
    return img_out

@input_check(tensor_args=['img_in', 'img_mask'])
def mosaic(img_in, img_mask, samples=1, intensity=0.5, max_intensity=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: mosaic (https://substance3d.adobe.com/documentation/sddoc/mosaic-159450535.html)
        img_in (G or RGB(A))
        img_mask (G only)
        samples: Determines multi-sample quality.
        intensity: Strength of the effect. [0.0-1.0]
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with mosaic
    '''
    # defaults:
    if img_in is None:
        if img_mask is not None:
            img_in = torch.zeros_like(img_mask)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if img_mask is None:
        img_mask = torch.zeros(img_in.shape[0], 1, img_in.shape[2], img_in.shape[3], dtype=img_in.dtype, device=img_in.device)

    img_mask = grayscale_input_check(img_mask, 'warp map')

    assert isinstance(samples, int) and samples >= 1 and samples <= 16
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    if intensity == 0 or torch.min(img_in) == torch.max(img_in):
        return img_in
    # progressive warping
    warp_intensity = intensity / samples
    img_out = img_in
    for i in range(samples):
        img_out = warp(img_out, img_mask, warp_intensity, 1.0)
    return img_out

@input_check(tensor_args=['img_in'])
def auto_levels(img_in, quality=0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: auto levels (https://substance3d.adobe.com/documentation/sddoc/auto-levels-159449154.html)
        img_in (G only)
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')

    max_val, min_val = torch.max(img_in), torch.min(img_in)
    # when input is a uniform image, and pixel value smaller (greater) than 0.5
    # output a white (black) image
    if max_val == min_val and max_val <= 0.5:
        img_out = (img_in - min_val + 1e-15) / (max_val - min_val + 1e-15)
    else:
        img_out = (img_in - min_val) / (max_val - min_val + 1e-15)
    return img_out

@input_check(tensor_args=['img_in'])
def ambient_occlusion(img_in, spreading=0.15, max_spreading=1.0, equalizer=[0.0, 0.0, 0.0], levels_param=[0.0, 0.5, 1.0], outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: ambient occlusion (deprecated)
        img_in (G only)
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')

    # Process parameters
    spreading = as_comp_tensor(spreading, device=img_in.device) * max_spreading
    equalizer = as_comp_tensor(equalizer, device=img_in.device)
    levels_param = as_comp_tensor(levels_param, device=img_in.device)

    # Initial processing
    img_blur = blur_hq(1.0 - img_in, intensity=spreading, max_intensity=128.0)
    img_ao = blend(img_blur, img_in, blending_mode='add')
    img_ao = levels(img_ao, in_low=0.5)
    img_gs = c2g(normal(img_in, intensity=1.0, max_intensity=16.0), rgba_weights=[0.0, 0.0, 1.0])
    img_ao_2 = blend(img_ao, 1.0 - img_gs, blending_mode='add')
    img_ao = blend(img_ao, img_ao_2, blending_mode='multiply')

    # Further processing
    img_ao_blur = blur_hq(manual_resize(img_ao, -1), intensity=1.0, max_intensity=2.2)
    img_ao_blur_2 = blur_hq(manual_resize(img_ao_blur, -1), intensity=1.0, max_intensity=3.3)
    img_blend = blend(manual_resize(1.0 - img_ao_blur, 1), img_ao, blending_mode='add_sub', opacity=0.5)
    img_blend_1 = blend(manual_resize(1.0 - img_ao_blur_2, 1), img_ao_blur, blending_mode='add_sub', opacity=0.5)

    img_ao_blur_2 = levels(img_ao_blur_2, in_mid=(equalizer[0] + 1) * 0.5)
    img_blend_1 = blend(img_blend_1, manual_resize(img_ao_blur_2, 1), blending_mode='add_sub',
                        opacity=torch.clamp(equalizer[1] + 0.5, 0.0, 1.0))
    img_blend = blend(img_blend, manual_resize(img_blend_1, 1), blending_mode='add_sub',
                      opacity=torch.clamp(equalizer[2] + 0.5, 0.0, 1.0))
    img_ao = levels(img_blend, in_low=levels_param[0], in_mid=levels_param[1], in_high=levels_param[2])

    return img_ao

@input_check(tensor_args=['img_in'])
def hbao(img_in, quality=4, depth=0.1, radius=1.0, height_scale=1.0, surface_size=1.0, gpu_optim=False, use_world_units=False, non_square=True, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: ambient occlusion (HBAO) (https://substance3d.adobe.com/documentation/sddoc/ambient-occlusion-hbao-filter-node-159450550.html)
        img_in (G only)
        return image (G only)
    Notes:
        - sampling in all cones can be parallelized at the cost of high GPU memory usage (>8G is recommended)
    Behavior:
        identical to sbs
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = grayscale_input_check(img_in, 'input image')
    assert quality in [4, 8, 16], 'quality must be 4, 8, or 16'
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    pixel_size = 1.0 / max(num_row, num_col)
    min_size_log2 = int(np.log2(min(num_row, num_col)))

    # Performance triggers
    full_upsampling = True      # Enable full-sized mipmap sampling (identical results to sbs)
    batch_processing = False    # Enable batched HBAO cone sampling (20% faster; higher GPU memory cost)

    # Process input parameters
    depth = as_comp_tensor(depth, dtype=img_in.dtype, device=img_in.device) * min(num_row, num_col)
    radius = as_comp_tensor(radius, dtype=img_in.dtype, device=img_in.device)

    # Create mipmap stack
    in_low = levels(img_in, 0.0, 0.5, 1.0, 0.0, 0.5)
    in_high = levels(img_in, 0.0, 0.5, 1.0, 0.5, 1.0)
    mipmaps_level = 11
    mipmaps = create_mipmaps(in_high, mipmaps_level, keep_size=full_upsampling)

    # Precompute weights
    weights = [hbao_radius(min_size_log2, i + 1, radius) for i in range(mipmaps_level)]

    # HBAO cone sampling
    img_out = torch.zeros_like(img_in)
    row_grid_init, col_grid_init = torch.meshgrid(torch.linspace(0, num_row - 1, num_row, device=img_in.device), torch.linspace(0, num_col - 1, num_col, device=img_in.device))
    row_grid_init = (row_grid_init + 0.5) / num_row
    col_grid_init = (col_grid_init + 0.5) / num_col

    # Sampling all cones together
    if batch_processing:
        sample_grid_init = torch.stack([col_grid_init, row_grid_init], 2)
        angle_vec = lambda i: torch.tensor([np.cos(i * np.pi * 2.0 / quality), np.sin(i * np.pi * 2.0 / quality)], device=img_in.device)
        img_sample = torch.zeros_like(img_in)

        # perform sampling on each mipmap level
        for mm_idx, img_mm in enumerate(mipmaps):
            mm_scale = 2.0 ** (mm_idx + 1)
            mm_row, mm_col = img_mm.shape[2], img_mm.shape[3]
            sample_grid = torch.stack([torch.remainder(sample_grid_init + mm_scale * pixel_size * angle_vec(i), 1.0) * 2.0 - 1.0 \
                                    for i in range(quality)])
            sample_grid = sample_grid * torch.tensor([mm_col / (mm_col + 2), mm_row / (mm_row + 2)], device=img_in.device)
            img_mm = img_mm.view(1, img_in.shape[0], mm_row, mm_col).expand(quality, img_in.shape[0], mm_row, mm_col)
            img_mm_pad = torch.nn.functional.pad(img_mm, [1, 1, 1, 1], mode='circular')
            img_mm_gs = torch.nn.functional.grid_sample(img_mm_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)

            img_diff = (img_mm_gs - in_low - 0.5) / mm_scale
            img_max = torch.max(img_max, img_diff) if mm_idx else img_diff
            img_sample = lerp(img_sample, img_max, weights[mm_idx])

        # integrate into sampled image
        img_sample = img_sample * depth * 2.0
        img_sample = img_sample / torch.sqrt(img_sample * img_sample + 1.0)
        img_out = torch.sum(img_sample, 0, keepdim=True).view_as(img_in)

    # Sampling each cone individually
    else:
        for i in range(quality):
            cone_angle = i * np.pi * 2.0 / quality
            sin_angle = np.sin(cone_angle)
            cos_angle = np.cos(cone_angle)
            img_sample = torch.zeros_like(img_in)

            # perform sampling on each mipmap level
            for mm_idx, img_mm in enumerate(mipmaps):
                mm_scale = 2.0 ** (mm_idx + 1)
                mm_row, mm_col = img_mm.shape[2], img_mm.shape[3]
                row_grid = torch.remainder(row_grid_init + mm_scale * pixel_size * sin_angle, 1.0) * 2.0 - 1.0
                col_grid = torch.remainder(col_grid_init + mm_scale * pixel_size * cos_angle, 1.0) * 2.0 - 1.0
                row_grid = row_grid * mm_row / (mm_row + 2)
                col_grid = col_grid * mm_col / (mm_col + 2)
                sample_grid = torch.stack([col_grid, row_grid], 2).expand(img_in.shape[0], num_row, num_col, 2)
                img_mm_pad = torch.nn.functional.pad(img_mm, [1, 1, 1, 1], mode='circular')
                img_mm_gs = torch.nn.functional.grid_sample(img_mm_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)

                img_diff = (img_mm_gs - in_low - 0.5) / mm_scale
                img_max = img_diff if mm_idx == 0 else torch.max(img_max, img_diff)
                img_sample = lerp(img_sample, img_max, weights[mm_idx])

            # integrate into sampled image
            img_sample = img_sample * depth * 2.0
            img_sample = img_sample / torch.sqrt(img_sample * img_sample + 1.0)
            img_out = img_out + img_sample

    # final output
    img_out = torch.clamp(1.0 - img_out / quality, 0.0, 1.0)

    return img_out

@input_check(tensor_args=['img_in'])
def highpass(img_in, radius=6.0/64.0, max_radius=64.0, outputsize=[9, 9], device=torch.device('cpu')):
    """
    Non-atomic function: highpass (https://substance3d.adobe.com/documentation/sddoc/highpass-159449203.html)
        img_in (G or RGB(A))
        radius: A small radius removes small differences, a bigger radius removes large areas.
        max_radius: Maximum value of 'radius'.
        return image (G or RGB(A))
    Behavior:
        identical to sbs
        alpha changes with highpass

    """
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    radius = as_comp_tensor(radius, dtype=img_in.dtype, device=img_in.device) * max_radius
    img_out = blur(img_in, radius, 1.0)
    img_out = invert(img_out)
    img_out = blend(img_out, img_in, None, 'add_sub', opacity=0.5)
    return img_out

@input_check(tensor_args=['normal'])
def normal_normalize(normal, outputsize=[9, 9], device=torch.device('cpu')):
    """
    Non-atomic function: normal_normalize (https://substance3d.adobe.com/documentation/sddoc/normal-normalize-159450586.html)
        img_in (RGB(A) only)
        return image (RGB(A) only)
    """
    # defaults:
    if normal is None:
        normal = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    normal = color_input_check(normal, 'input image')

    # reimplementation of the internal pixel processor (latest implementation)
    normal_rgb = normal[:,:3,:,:] if normal.shape[1] == 4 else normal
    normal_rgb = normal_rgb * 2.0 - 1.0
    normal_length = torch.norm(normal_rgb, p=2, dim=1)
    normal_rgb = normal_rgb / normal_length
    normal_rgb = normal_rgb / 2.0 + 0.5
    normal = torch.cat([normal_rgb, normal[:,3,:,:].unsqueeze(1)], dim=1) if normal.shape[1] == 4 else normal_rgb
    return normal

@input_check(tensor_args=['normal_one', 'normal_two'])
def normal_combine(normal_one, normal_two, mode='whiteout', outputsize=[9, 9], device=torch.device('cpu')):
    """
    Non-atomic function: normal_combine (https://substance3d.adobe.com/documentation/sddoc/normal-combine-159450580.html)
        normal_one (RGB(A))
        normal_two (RGB(A))
        mode: "whiteout" / "channel_mixer" / "detail_oriented"
        return normal(RGB(A))
    Behavior:
        identical to sbs
        output opaque alpha
    """
    # defaults:
    if normal_one is None:
        if normal_two is not None:
            normal_one = torch.zeros(size=[1, 3, normal_two.shape[2], normal_two.shape[3]], dtype=torch.float32, device=normal_two.device)
        else:
            normal_one = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if normal_two is None:
        normal_two = torch.zeros(size=[1, 3, normal_one.shape[2], normal_one.shape[3]], dtype=torch.float32, device=normal_one.device)

    normal_one = color_input_check(normal_one, 'normal one')
    normal_two = color_input_check(normal_two, 'normal two')
    assert normal_one.shape == normal_two.shape, "two input normals don't have same shape"

    # top branch
    normal_one_r = normal_one[:,0,:,:].unsqueeze(1)
    normal_one_g = normal_one[:,1,:,:].unsqueeze(1)
    normal_one_b = normal_one[:,2,:,:].unsqueeze(1)

    normal_two_r = normal_two[:,0,:,:].unsqueeze(1)
    normal_two_g = normal_two[:,1,:,:].unsqueeze(1)
    normal_two_b = normal_two[:,2,:,:].unsqueeze(1)


    if mode == 'whiteout':
        r_blend = blend(normal_one_r, normal_two_r, None, 'add_sub', opacity=0.5)
        g_blend = blend(normal_one_g, normal_two_g, None, 'add_sub', opacity=0.5)
        b_blend = blend(normal_one_b, normal_two_b, None, 'multiply', opacity=1.0)

        rgb_blend = torch.cat([r_blend, g_blend, b_blend], dim=1)
        normal_out = normal_normalize(rgb_blend)

        if normal_one.shape[1] == 4:
            normal_out = torch.cat([
                normal_out,
                torch.ones(normal_out.shape[0], 1, normal_out.shape[2], normal_out.shape[3], device=normal_one.device)], dim=1)

    elif mode == 'channel_mixer':
        # middle left branch
        normal_two_levels_one = levels(normal_two, [0.5,0.5,0.0,0.0], [0.5]*4, [1.0] * 4, [0.5,0.5,0.0,1.0], [1.0,1.0,0.0,1.0])
        normal_two_levels_one[:,:2,:,:] = normal_two_levels_one[:,:2,:,:] - 0.5

        normal_two_levels_two = levels(normal_two, [0.0]*4, [0.5]*4, [0.5,0.5,1.0,1.0], [0.0,0.0,0.0,1.0], [0.5,0.5,1.0,1.0])
        normal_two_levels_two[:,:2,:,:] = -normal_two_levels_two[:,:2,:,:] + 0.5
        normal_two_levels_two[:,2,:,:] = -normal_two_levels_two[:,2,:,:] + 1.0

        # bottom left branch
        grayscale_blend = blend(normal_two_b, normal_one_b, None, 'min', opacity=1.0)

        # bottom middle branch
        cm_normal_one_blend = blend(normal_two_levels_two, normal_one, None, 'subtract', opacity=1.0)
        normal_out = blend(normal_two_levels_one, cm_normal_one_blend, None, 'add', opacity=1.0)
        normal_out[:,2,:,:] = grayscale_blend

    elif mode == 'detail_oriented':
        # implement pixel processorggb_rgb_temp
        r_one_temp = normal_one_r * 2.0 - 1.0
        g_one_temp = normal_one_g * 2.0 - 1.0
        b_one_temp = normal_one_b * 2.0 - 1.0
        b_invert_one_temp = 1.0 / (normal_one_b + 1.0)
        rg_one_temp = -r_one_temp * g_one_temp
        rgb_one_temp = b_invert_one_temp * rg_one_temp
        rrb_one_temp = 1.0 - r_one_temp * r_one_temp * b_invert_one_temp
        ggb_one_temp = 1.0 - g_one_temp * g_one_temp * b_invert_one_temp

        rrb_rgb_temp = torch.cat([rrb_one_temp, rgb_one_temp, -r_one_temp], dim=1)
        rrb_rgb_if   = torch.zeros_like(rrb_rgb_temp)
        rrb_rgb_if[:,1,:,:] = -1.0
        rrb_rgb_temp[normal_one_b.expand(-1,3,-1,-1) < -0.9999] = rrb_rgb_if[normal_one_b.expand(-1,3,-1,-1) < -0.9999]
        ggb_rgb_temp = torch.cat([rgb_one_temp, ggb_one_temp, -g_one_temp], dim=1)
        ggb_rgb_if   = torch.zeros_like(ggb_rgb_temp)
        ggb_rgb_if[:,0,:,:] = -1.0
        ggb_rgb_temp[normal_one_b.expand(-1,3,-1,-1) < -0.9999] = ggb_rgb_if[normal_one_b.expand(-1,3,-1,-1) < -0.9999]

        rrb_rgb_temp = rrb_rgb_temp * (normal_two_r * 2.0 - 1.0)
        ggb_rgb_temp = ggb_rgb_temp * (normal_two_g * 2.0 - 1.0)
        b_rgb_temp = (normal_one[:,:3,:,:] * 2.0 - 1.0) * (normal_two_b * 2.0 - 1.0)
        normal_out = (rrb_rgb_temp + ggb_rgb_temp + b_rgb_temp) * 0.5 + 0.5

        if normal_one.shape[1] == 4:
            normal_out = torch.cat([
                normal_out,
                torch.ones(normal_out.shape[0], 1, normal_out.shape[2], normal_out.shape[3], device=normal_one.device)], dim=1)

    else:
        raise RuntimeError("Can't recognized the mode")

    return normal_out

@input_check(tensor_args=['img_in'])
def channel_mixer(img_in, monochrome=False, red=[0.75,0.5,0.5,0.5], green=[0.5,0.75,0.5,0.5], blue=[0.5,0.5,0.75,0.5], outputsize=[9, 9], device=torch.device('cpu')):
    """
    Non-atomic function: normal_combine (https://substance3d.adobe.com/documentation/sddoc/channel-mixer-159449157.html)
        img_in (RGB(A))
        red: mix weight for red channel        valid range [-2, 2]
        green: mix weight for green channel    valid range [-2, 2]
        blue: mix weight for blue channel      valid range [-2, 2]
        return image (RGB(A))
    Behavior:
        identical to sbs
    """
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image')

    img_out = torch.zeros_like(img_in)

    if img_in.shape[1] == 4:
        # copy alpha channel from img_in
        img_out[:, 3] = img_in[:, 3]

    # scale to range [-2,2]
    red = (as_comp_tensor(red, device=img_in.device) - 0.5) * 4
    green = (as_comp_tensor(green, device=img_in.device) - 0.5) *4
    blue = (as_comp_tensor(blue, device=img_in.device) - 0.5)* 4

    weight = [red, green, blue]
    active_channels = 1 if monochrome else 3
    for i in range(active_channels):
        for j in range(3):
            img_out[:,i,:,:] = img_out[:,i,:,:] + img_in[:,j,:,:] * weight[i][j]
        img_out[:,i,:,:] = img_out[:,i,:,:] + weight[i][3]
    img_out = torch.clamp(img_out, 0.0, 1.0)

    if monochrome:
        # make monochrome: use red channel for r,g,b
        img_out[:, :3, :, :] = img_out[:, [0], : , :]

    return img_out

@input_check(tensor_args=['img_in'])
def height_to_normal_world_units(img_in, normal_format='gl', sampling_mode='standard', use_alpha=False, surface_size=0.3, max_surface_size=1000.0,
                                height_depth=0.16, max_height_depth=100.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: height to normal world units (https://substance3d.adobe.com/documentation/sddoc/height-to-normal-world-units-159450573.html)
        img_in (G only)
        normal_format: Inverts the green channel. ('dx' or 'gl')
        sampling_mode: Switches between two sampling modes determining accuracy. ('standard' or 'sobel')
        surface_size: Dimensions of the input Heightmap (cm).
        max_surface_size: The range of 'surface_size'.
        height_depth: Maximum depth of Heightmap details (cm).
        max_height_depth: The range of 'height_depth'.
        return image (RGB(A))
    Behavior:
        identical to sbs
        output opaque alpha
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    # Check input validity
    img_in = grayscale_input_check(img_in, 'input image')
    assert normal_format in ('dx', 'gl'), "normal format must be 'dx' or 'gl'"
    assert sampling_mode in ('standard', 'sobel'), "sampling mode must be 'standard' or 'sobel'"

    surface_size = as_comp_tensor(surface_size, dtype=img_in.dtype, device=img_in.device) * max_surface_size
    height_depth = as_comp_tensor(height_depth, dtype=img_in.dtype, device=img_in.device) * max_height_depth
    res_x, inv_res_x = img_in.shape[2], 1.0 / img_in.shape[2]
    res_y, inv_res_y = img_in.shape[3], 1.0 / img_in.shape[3]

    # Standard normal conversion
    if sampling_mode == 'standard':
        img_out = normal(img_in, normal_format=normal_format, use_alpha=use_alpha, intensity=height_depth / surface_size, max_intensity=256.0)
    # Sobel sampling
    else:
        # Convolution
        db_x = d_blur(img_in, inv_res_x, 256.0)
        db_y = d_blur(img_in, inv_res_y, 256.0, angle=0.25)
        db_x = torch.nn.functional.pad(db_x, (0, 0, 1, 1), mode='circular')
        db_y = torch.nn.functional.pad(db_y, (1, 1, 0, 0), mode='circular')
        sample_x = torch.nn.functional.conv2d(db_y, torch.linspace(1.0, -1.0, 3, device=img_in.device).view((1, 1, 1, 3)))
        sample_y = torch.nn.functional.conv2d(db_x, torch.linspace(-1.0, 1.0, 3, device=img_in.device).view((1, 1, 3, 1)))

        # Multiplier
        mult_x = res_x * height_depth * 0.5 / surface_size
        mult_y = (-1.0 if normal_format == 'dx' else 1.0) * res_y * height_depth * 0.5 / surface_size
        sample_x = sample_x * mult_x * (1.0 if res_x < res_y else res_y / res_x)
        sample_y = sample_y * mult_y * (res_x / res_y if res_x < res_y else 1.0)

        # Output
        scale = 0.5 / torch.sqrt(sample_x ** 2 + sample_y ** 2 + 1)
        img_out = torch.cat([sample_x, sample_y, torch.ones_like(img_in)], dim=1) * scale + 0.5

        # Add opaque alpha channel
        if use_alpha:
            img_out = torch.cat([img_out, torch.ones_like(img_in)], dim=1)

    return img_out

@input_check(tensor_args=['img_in'])
def normal_to_height(img_in, normal_format='dx', relief_balance=[0.5, 0.5, 0.5], opacity=0.36, max_opacity=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Non-atomic function: normal to height (https://substance3d.adobe.com/documentation/sddoc/normal-to-height-159450591.html)
        img_in (RGB(A))
        normal_format: Inverts the green channel.                                       ('dx' or 'gl')
        low_freq: The influence of low frequencies in the input image on the result.    (range: [0, 1])
        mid_freq: The influence of mid frequencies in the input image on the result.    (range: [0, 1])
        high_freq: The influence of high frequencies in the input image on the result.  (range: [0, 1])
        opacity: Adjusts the global opacity of the effect.                              (range: [0, 1])
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image')
    assert img_in.shape[2] == img_in.shape[3], 'input image must be in square shape'
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    assert in_size_log2 >= 7, 'input size must be at least 128'

    # Construct variables
    low_freq = as_comp_tensor(relief_balance[0], device=img_in.device)
    mid_freq = as_comp_tensor(relief_balance[1], device=img_in.device)
    high_freq = as_comp_tensor(relief_balance[2], device=img_in.device)
    opacity = as_comp_tensor(opacity, device=img_in.device) * max_opacity

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2,:,:], normal_format)
    img_blend = [None, None]

    # Low frequencies (for 16x16 images only)
    for i in range(4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i]
            blend_opacity = torch.clamp(0.0625 * 2 * (8 >> i) * low_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = img_i_c if img_blend[c] is None else blend(img_i_c, img_blend[c], blending_mode='add_sub', opacity=blend_opacity)

    # Mid frequencies
    for i in range(min(2, len(img_freqs[0]) - 4)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 4]
            blend_opacity = torch.clamp(0.0156 * 2 * (2 >> i) * mid_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # High frequencies
    for i in range(min(6, len(img_freqs[0]) - 6)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 6]
            blend_opacity = torch.clamp(0.0078 * 0.0625 * (32 >> i) * high_freq * 100 * opacity, 0.0, 1.0) if i < 5 else \
                            torch.clamp(0.0078 * 0.0612 * high_freq * 100 * opacity)
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # Combine both channels
    img_out = blend(img_blend[0], img_blend[1], blending_mode='add_sub', opacity=0.5)
    return img_out

@input_check(tensor_args=['img_in'])
def curvature_smooth(img_in, normal_format='dx', outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Curvature Smooth (https://substance3d.adobe.com/documentation/sddoc/curvature-smooth-159450517.html)
        img_in (RGB(A) only): the input normal map
        normal_format: the input normal format ('dx' or 'gl')
        return image (G only)
    Behavior:
        identical to sbs
    '''
    # defaults:
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    # Check input validity
    img_in = color_input_check(img_in, 'input image')
    assert img_in.shape[2] == img_in.shape[3], 'input image must be in square shape'
    assert img_in.shape[2] >= 16, 'input size must be at least 16'
    assert normal_format in ('dx', 'gl')

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2,:,:], normal_format)
    img_blend = [img_freqs[0][0], img_freqs[1][0]]

    # Low frequencies (for 16x16 images only)
    for i in range(1, 4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i]
            img_blend[c] = blend(img_i_c, img_blend[c], blending_mode='add_sub', opacity=0.25)

    # Other frequencies
    for i in range(len(img_freqs[0]) - 4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 4]
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=1.0 / (i + 5))

    # Combine both channels
    img_out = blend(img_blend[0], img_blend[1], blending_mode='add_sub', opacity=0.5)
    return img_out

@input_check(tensor_args=[f'input_{i+1}' for i in range(20)])
def multi_switch(
    input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
    input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20,
    input_number, input_selection, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Multi Switch(https://substance3d.adobe.com/documentation/sddoc/multi-switch-159450377.html)
        img_list: a list of input images (G or RGB(A))
        input_number: number of total input
        input_selection: index of selected input
    Behavior:
        identical to sbs
    '''
    # defaults further below

    img_list = [
        input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
        input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20]
    assert input_number <= len(img_list), f'the max. number of inputs is {len(img_list)}'
    # assert all(img_list[i] is None for i in range(input_number, len(img_list))), 'only inputs slots <= input_number can have a value' # this does not seem to be needed, and is in fact not true for some graphs (connections are not removed if the input number is reduced)
    assert input_selection <= input_number, 'input selection should be less equal then the input number'
    # if img_list[0].shape[1] == 1:
    #     for i, img in enumerate(img_list):
    #         img = grayscale_input_check(img, 'input image %d' % i)
    # else:
    #     for i, img in enumerate(img_list):
    #         color_input_check(img, 'input image %d' % i)

    # defaults
    if img_list[input_selection-1] is None:
        img_list[input_selection-1] = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    if img_list[input_selection-1].shape[1] == 1:
        img_list[input_selection-1] = grayscale_input_check(img_list[input_selection-1], f'input image {input_selection-1}')
    else:
        img_list[input_selection-1] = color_input_check(img_list[input_selection-1], f'input image {input_selection-1}')
    return img_list[input_selection-1]

@input_check(tensor_args=['rgba'])
def rgba_split(rgba, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    RGBA Split (https://substance3d.adobe.com/documentation/sddoc/rgba-split-159450492.html)
        rgba (RGB(A) only)
        return images (all G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if rgba is None:
        rgba = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    rgba = color_input_check(rgba, 'input image')
    if rgba.shape[1] == 3:
        # use all-ones alpha if the alpha channel is not given
        rgba = torch.cat([rgba, torch.zeros(rgba.shape[0], 1, rgba.shape[2], rgba.shape[3], dtype=rgba.dtype, device=rgba.device)], dim=1)
    # assert rgba.shape[1] == 4, 'input image must contain alpha channel'
    r = rgba[:,[0],:,:].clone()
    g = rgba[:,[1],:,:].clone()
    b = rgba[:,[2],:,:].clone()
    a = rgba[:,[3],:,:].clone()
    return r, g, b, a

@input_check(tensor_args=['r', 'g', 'b', 'a'])
def rgba_merge(r, g, b, a=None, use_alpha=False, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    merge per channel input to into an RGBA image (https://substance3d.adobe.com/documentation/sddoc/rgba-merge-159450486.html)
        r: red channel (G only)
        g: green channel (G only)
        b: blue channel (G only)
        a: alpha channel (G only)
        return image (RGB(A) only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if r is None:
        if g is not None:
            r = torch.zeros_like(g)
        elif b is not None:
            r = torch.zeros_like(b)
        else:
            r = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if g is None:
        g = torch.zeros_like(r)
    if b is None:
        b = torch.zeros_like(r)

    channels = [r,g,b,a]
    active_channels = 4 if use_alpha else 3
    for i in range(3):
        channels[i] = grayscale_input_check(channels[i], 'channel image')
    assert channels[0].shape == channels[1].shape == channels[2].shape, "rgb channels doesn't have same shape"

    if use_alpha and channels[3] is None:
        channels[3] = torch.ones_like(channels[0])
    img_out = torch.cat(channels[:active_channels], dim=1)
    return img_out

    # num_batch = 0
    # res_h = 0
    # res_w = 0
    # for channel in channels:
    #     if channel is not None:
    #         channel = grayscale_input_check(channel, 'channel image')
    #         num_batch = max(num_batch, channel.shape[0])
    #         res_h = max(res_h, channel.shape[2])
    #         res_w = max(res_w, channel.shape[3])

    # for i, channel in enumerate(channels):
    #     if channel is None:
    #         channels[i] = torch.zeros(num_batch, 1, res_h, res_w)
    #     else:
    #         assert res_h//channel.shape[2] == res_w//channel.shape[3], "rectangular input is not supported"
    #         if res_h//channel.shape[2] != 1:
    #             channels[i] = manual_resize(channel, res_h//channel.shape[2])[0]

    # img_out = torch.cat(channels, dim=1)
    # return img_out

@input_check(tensor_args=['base_color', 'roughness', 'metallic'])
def pbr_converter(base_color, roughness, metallic, use_alpha=False, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    pbr_converter (https://substance3d.adobe.com/documentation/sddoc/basecolor-metallic-roughness-converter-159451054.html)
        basecolor (G or RGB(A))
        roughness (G only)
        metallic (G only)
        return images:
        diffuse (G or RGB(A))
        specular (G or RGB(A))
        glossiness (G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if base_color is None:
        if roughness is not None:
            base_color = torch.zeros(size=[roughness.shape[0], 4 if use_alpha else 3, roughness.shape[2], roughness.shape[3]], dtype=torch.float32, device=roughness.device)
        elif metallic is not None:
            base_color = torch.zeros(size=[metallic.shape[0], 4 if use_alpha else 3, metallic.shape[2], metallic.shape[3]], dtype=torch.float32, device=metallic.device)
        else:
            base_color = torch.zeros(size=[1, 4 if use_alpha else 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if roughness is None:
        roughness = torch.zeros(size=[base_color.shape[0], 1, base_color.shape[2], base_color.shape[3]], dtype=torch.float32, device=base_color.device)
    if metallic is None:
        metallic = torch.zeros(size=[base_color.shape[0], 1, base_color.shape[2], base_color.shape[3]], dtype=torch.float32, device=base_color.device)

    roughness = grayscale_input_check(roughness, 'roughness')
    metallic = grayscale_input_check(metallic, 'metallic')

    # compute diffuse
    invert_metallic = levels(metallic, out_low=1.0, out_high=0.0)
    invert_metallic_sRGB = linear_to_srgb(invert_metallic)
    invert_metallic_sRGB = levels(invert_metallic_sRGB, out_low=1.0, out_high=0.0)
    black = torch.zeros_like(base_color)
    if use_alpha and base_color.shape[1] == 4:
        black[:,3,:,:] = 1.0
    diffuse = blend(black, base_color, invert_metallic_sRGB)

    # compute specular
    base_color_linear = srgb_to_linear(base_color)
    specular_blend = blend(black, base_color_linear, invert_metallic)
    specular_levels = torch.clamp(levels(invert_metallic, out_high = 0.04), 0.0, 1.0)
    if use_alpha:
        specular_levels = torch.cat([specular_levels,specular_levels,specular_levels,torch.ones_like(specular_levels)], dim=1)
    else:
        specular_levels = torch.cat([specular_levels,specular_levels,specular_levels], dim=1)

    specular_blend_2 = blend(specular_levels, specular_blend)
    specular = linear_to_srgb(specular_blend_2)

    # compute glossiness
    glossiness = levels(roughness, out_low=1.0, out_high=0.0)
    return diffuse, specular, glossiness

@input_check(tensor_args=['rgba'])
def alpha_split(rgba, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    alpha split (https://substance3d.adobe.com/documentation/sddoc/alpha-split-159450495.html)
        rgba (RGBA only)
        return images:
        rgb (RGB only)
        a (G only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if rgba is None:
        rgba = torch.zeros(size=[1, 4, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    rgba = color_input_check(rgba, 'rgba image', with_alpha=True)
    assert rgba.shape[1] == 4, 'image input must contain alpha channel'
    rgb = torch.cat([rgba[:,:3,:,:], torch.ones(rgba.shape[0], 1, rgba.shape[2], rgba.shape[3], device=rgba.device)], 1)
    a = rgba[:,[3],:,:].clone()
    return rgb, a

@input_check(tensor_args=['rgb', 'a'])
def alpha_merge(rgb, a, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    alpha merge (https://substance3d.adobe.com/documentation/sddoc/alpha-merge-159450489.html)
        rgb (RGB only)
        a (G only)
        return images:
        rgba (RGBA only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if rgb is None:
        if a is not None:
            rgb = torch.zeros(size=[a.shape[0], 3, a.shape[2], a.shape[3]], dtype=torch.float32, device=a.device)
        else:
            rgb = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if a is None:
        a = torch.zeros(size=[rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]], dtype=torch.float32, device=rgb.device)

    rgb = color_input_check(rgb, 'rgb image', with_alpha=False)
    a = grayscale_input_check(a, 'alpha image')

    channels = [rgb, a]
    res_h = max(rgb.shape[2], a.shape[2])
    res_w = max(rgb.shape[3], a.shape[3])

    for i, channel in enumerate(channels):
        assert res_h//channel.shape[2] == res_w//channel.shape[3], "rectangular input is not supported"
        if res_h//channel.shape[2] != 1:
            channels[i] = manual_resize(channel, res_h//channel.shape[2])

    img_out = torch.cat(channels, dim=1)
    return img_out

@input_check(tensor_args=['img_1', 'img_2'])
def switch(img_1, img_2, flag=True):
    '''
    switch (https://substance3d.adobe.com/documentation/sddoc/switch-159450385.html)
        img_1 (G or RGB(A))
        img_2 (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    if flag:
        return img_1
    else:
        return img_2

@input_check(tensor_args=['normal_fg', 'normal_bg', 'mask'])
def normal_blend(normal_fg, normal_bg, mask=None, use_mask=True, opacity=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    normal_blend (https://substance3d.adobe.com/documentation/sddoc/normal-blend-159450576.html)
        normal_fg (RGB(A) only)
        normal_bg (RGB(A) only)
        mask (G only)
        return image (RGB(A) only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if normal_bg is None and normal_fg is not None:
        normal_bg = torch.zeros_like(normal_fg)
    elif normal_fg is None and normal_bg is not None:
        normal_fg = torch.zeros_like(normal_bg)
    elif normal_fg is None and normal_bg is None:
        normal_fg = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
        normal_bg = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    normal_fg = color_input_check(normal_fg, 'normal foreground')
    normal_bg = color_input_check(normal_bg, 'normal background')
    assert normal_fg.shape == normal_bg.shape, 'the shape of normal fg and bg does not match'
    if mask is not None:
        mask = grayscale_input_check(mask, 'mask')
        assert normal_fg.shape[2] == mask.shape[2] and normal_fg.shape[3] == mask.shape[3], 'the shape of normal fg and bg does not match'

    opacity = as_comp_tensor(opacity, dtype=normal_fg.dtype, device=normal_fg.device)
    if use_mask and mask is not None:
        mask_blend = blend(mask, torch.zeros_like(mask), opacity=opacity)
    else:
        dummy_mask = torch.ones(normal_fg.shape[0], 1, normal_fg.shape[2], normal_fg.shape[3], device=normal_fg.device)
        dummy_mask_2 = torch.zeros(normal_fg.shape[0], 1, normal_fg.shape[2], normal_fg.shape[3], device=normal_fg.device)
        mask_blend = blend(dummy_mask, dummy_mask_2, opacity=opacity)

    out_normal = blend(normal_fg[:,:3,:,:], normal_bg[:,:3,:,:], mask_blend)

    # only when both normal inputs have alpha, process and append alpha
    if normal_fg.shape[1] == 4 and normal_bg.shape[1] == 4:
        out_normal_alpha = blend(normal_fg[:,3:4,:,:], normal_bg[:,3:4,:,:], mask_blend)
        out_normal = torch.cat([out_normal, out_normal_alpha], dim=1)

    out_normal = normal_normalize(out_normal)
    return out_normal

@input_check(tensor_args=['img_in'])
def mirror(img_in, mirror_axis=0, invert_axis=False, corner_type=0, offset=0.5, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    !!NOT SURE IF DIFFERENTIABLE!!
    mirror (https://substance3d.adobe.com/documentation/sddoc/mirror-filter-node-159450617.html)
        img_in (G or RGB(A))
        mirror_axis: 'x', 'y', 'corner'
        invert_axis: whether flip direction
        corner_type: 'tl', 'tr', 'bl', 'br'
        offset: where the axis locates
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    mirror_axis_list = ['x', 'y', 'corner']
    corner_type_list = ['tl', 'tr', 'bl', 'br']
    mirror_axis = mirror_axis_list[mirror_axis]
    corner_type = corner_type_list[corner_type]

    offset = as_comp_tensor(offset, device=img_in.device)

    if mirror_axis == 'x':
        axis_w = res_w * offset
        if (axis_w==0 and invert_axis==True) or (axis_w==res_w and invert_axis==False):
            return img_in

        if invert_axis:
            # invert image first
            img_in = torch.flip(img_in, dims=[3])
            axis_w = res_w - axis_w

        # compute img_out_two
        double_offset = int(np.ceil(axis_w.item()*2))
        axis_w_floor = int(np.floor(axis_w.item()))
        axis_w_ceil = int(np.ceil(axis_w.item()))
        if double_offset % 2 == 1:
            img_out = torch.zeros_like(img_in)
            img_out[:,:,:,:axis_w_floor] = img_in[:,:,:,:axis_w_floor]
            if double_offset < res_w:
                img_out[:,:,:,axis_w_floor:double_offset] = torch.flip(img_in[:,:,:,:axis_w_floor+1], dims=[3])

                img_out[:,:,:,axis_w_floor:double_offset] = \
                    img_out[:,:,:,axis_w_floor:double_offset].clone() * (1 - (double_offset - axis_w*2)) + \
                    img_out[:,:,:,axis_w_floor+1:double_offset+1].clone() * (double_offset - axis_w*2)

            else:
                img_out[:,:,:,axis_w_floor:res_w] = \
                    torch.flip(img_in[:,:,:,axis_w_floor-(res_w-axis_w_floor)+1:axis_w_floor+1], dims=[3]) * (1 - (double_offset - axis_w*2)) + \
                    torch.flip(img_in[:,:,:,axis_w_floor-(res_w-axis_w_floor):axis_w_floor], dims=[3]) * (double_offset - axis_w*2)
        else:
            img_out = torch.zeros_like(img_in)
            img_out[:,:,:,:axis_w_ceil] = img_in[:,:,:,:axis_w_ceil]
            if double_offset < res_w:
                # img_out[:,:,:,axis_w_ceil:double_offset] = torch.flip(img_in[:,:,:,:axis_w_ceil].clone().contiguous(), dims=[3])
                img_out[:,:,:,axis_w_ceil:double_offset] = torch.flip(img_in[:,:,:,:axis_w_ceil], dims=[3])
                img_out[:,:,:,axis_w_ceil:double_offset] = \
                    img_out[:,:,:,axis_w_ceil:double_offset].clone() * (1 - (double_offset - axis_w*2)) + \
                    img_out[:,:,:,axis_w_ceil+1:double_offset+1].clone() * (double_offset - axis_w*2)
            else:
                img_out[:,:,:,axis_w_ceil:res_w] = \
                    torch.flip(img_in[:,:,:,axis_w_ceil-(res_w-axis_w_ceil)+1:axis_w_ceil+1], dims=[3]) * (1 - (double_offset - axis_w*2)) + \
                    torch.flip(img_in[:,:,:,axis_w_ceil-(res_w-axis_w_ceil):axis_w_ceil], dims=[3]) * (double_offset - axis_w*2)

        if invert_axis:
            img_out = torch.flip(img_out, dims=[3])

    elif mirror_axis == 'y':
        axis_h = res_h * (1 - offset)
        if (axis_h==0 and invert_axis==True) or (axis_h==res_h and invert_axis==False):
            return img_in

        if invert_axis:
            # invert image first
            img_in = torch.flip(img_in, dims=[2])
            axis_h = res_h - axis_h

        # compute img_out_two
        double_offset = int(np.ceil(axis_h.item()*2))
        axis_h_floor = int(np.floor(axis_h.item()))
        axis_h_ceil = int(np.ceil(axis_h.item()))
        if double_offset % 2 == 1:
            img_out = torch.zeros_like(img_in)
            img_out[:,:,:axis_h_floor,:] = img_in[:,:,:axis_h_floor,:]
            if double_offset < res_h:
                img_out[:,:,axis_h_floor:double_offset,:] = torch.flip(img_in[:,:,:axis_h_floor+1,:], dims=[2])
                img_out[:,:,axis_h_floor:double_offset,:] = \
                    img_out[:,:,axis_h_floor:double_offset,:].clone() * (1 - (double_offset - axis_h*2)) + \
                    img_out[:,:,axis_h_floor+1:double_offset+1,:].clone() * (double_offset - axis_h*2)
            else:
                img_out[:,:,axis_h_floor:res_h,:] = \
                    torch.flip(img_in[:,:,axis_h_floor-(res_h-axis_h_floor)+1:axis_h_floor+1,:], dims=[2]) * (1 - (double_offset - axis_h*2)) + \
                    torch.flip(img_in[:,:,axis_h_floor-(res_h-axis_h_floor):axis_h_floor,:], dims=[2]) * (double_offset - axis_h*2)
        else:
            img_out = torch.zeros_like(img_in)
            img_out[:,:,:axis_h_ceil,:] = img_in[:,:,:axis_h_ceil,:]
            if double_offset < res_h:
                img_out[:,:,axis_h_ceil:double_offset,:] = torch.flip(img_in[:,:,:axis_h_ceil,:], dims=[2])
                img_out[:,:,axis_h_ceil:double_offset,:] = \
                    img_out[:,:,axis_h_ceil:double_offset,:].clone() * (1 - (double_offset - axis_h*2)) + \
                    img_out[:,:,axis_h_ceil+1:double_offset+1,:].clone() * (double_offset - axis_h*2)
            else:
                img_out[:,:,axis_h_ceil:res_h,:] = \
                    torch.flip(img_in[:,:,axis_h_ceil-(res_h-axis_h_ceil)+1:axis_h_ceil+1,:], dims=[2]) * (1 - (double_offset - axis_h*2)) + \
                    torch.flip(img_in[:,:,axis_h_ceil-(res_h-axis_h_ceil):axis_h_ceil,:], dims=[2]) * (double_offset - axis_h*2)

        if invert_axis:
            img_out = torch.flip(img_out, dims=[2])

    elif mirror_axis == 'corner':
        img_out = img_in
        if corner_type == 'tl':
            # top right
            img_out[:,:, :res_h//2, res_w//2:] = torch.flip(img_out[:,:, :res_h//2, :res_w//2], dims=[3])
            # bottom
            img_out[:,:, res_h//2:, :] = torch.flip(img_out[:,:, :res_h//2, :], dims=[2])
        elif corner_type == 'tr':
            # top left
            img_out[:,:, :res_h//2, :res_w//2] = torch.flip(img_out[:,:, :res_h//2, res_w//2:], dims=[3])
            # bottom
            img_out[:,:, res_h//2:, :] = torch.flip(img_out[:,:, :res_h//2, :], dims=[2])
        elif corner_type == 'bl':
            # bottom right
            img_out[:,:, res_h//2:, res_w//2:] = torch.flip(img_out[:,:, res_h//2:, :res_w//2], dims=[3])
            # top
            img_out[:,:, :res_h//2, :] = torch.flip(img_out[:,:, res_h//2:, :], dims=[2])
        elif corner_type == 'br':
            # bottom left
            img_out[:,:, res_h//2:, :res_w//2] = torch.flip(img_out[:,:, res_h//2:, res_w//2:], dims=[3])
            # top
            img_out[:,:, :res_h//2, :] = torch.flip(img_out[:,:, res_h//2:, :], dims=[2])
    else:
        raise RuntimeError("unknown mirror options")

    return img_out

@input_check(tensor_args=['img_in'])
def make_it_tile_patch(img_in, octave=3, seed=0, use_alpha=False, mask_size=1.0, max_mask_size=1.0, mask_precision=0.5, max_mask_precision=1.0,
                       mask_warping=0.5, max_mask_warping=100.0, pattern_width=0.2, max_pattern_width=1000.0, pattern_height=0.2, max_pattern_height=1000.0,
                       disorder=0.0, max_disorder=1.0, size_variation=0.0, max_size_variation=100.0, rotation=0.5, rotation_variation=0.0,
                       background_color=[0.0, 0.0, 0.0, 1.0], color_variation=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    make it tile patch (https://substance3d.adobe.com/documentation/sddoc/make-it-tile-patch-159450499.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs (appears different due to randomness)
        However, the result is not tiling correctly if pattern is too large.
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    def param_process(param_in):
        param_in = as_comp_tensor(param_in, dtype=img_in.dtype, device=img_in.device)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1:
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and use_alpha:
            param_in = torch.cat([param_in, torch.tensor([1.0], dtype=param_in.dtype, device=param_in.device)])
        return param_in

    # Process input parameters
    tensor = torch.as_tensor
    mask_size = tensor(mask_size, device=img_in.device) * max_mask_size
    mask_precision = tensor(mask_precision, device=img_in.device) * max_mask_precision
    mask_warping = tensor(mask_warping * 2.0 - 1.0, device=img_in.device) * max_mask_warping
    pattern_width = tensor(pattern_width, device=img_in.device) * max_pattern_width
    pattern_height = tensor(pattern_height, device=img_in.device) * max_pattern_height
    disorder = tensor(disorder, device=img_in.device) * max_disorder
    size_variation = tensor(size_variation, device=img_in.device) * max_size_variation
    rotation = tensor(rotation * 2.0 - 1.0, device=img_in.device)
    rotation_variation = tensor(rotation_variation, device=img_in.device)
    background_color = param_process(background_color)
    color_variation = tensor(color_variation, device=img_in.device)
    grid_size = 1 << octave

    # Mode switch
    mode_color = img_in.shape[1] > 1

    # Set random seed
    torch.manual_seed(seed)

    # Gaussian pattern (sigma is approximated)
    x = torch.linspace(-31 / 32, 31 / 32, 32, device=img_in.device).expand(32, 32)
    x = x ** 2 + x.transpose(1, 0) ** 2
    img_gs = torch.exp(-0.5 * x / 0.089).expand(1, 1, 32, 32)
    img_gs = automatic_resize(img_gs, int(np.log2(img_in.shape[2] >> 5)))
    img_gs = levels(img_gs, [1.0 - mask_size], [0.5], [1 - mask_precision * mask_size])

    # Add alpha channel
    if mask_warping > 0.0:
        img_in_gc = c2g(img_in) if mode_color else img_in
        img_a = d_blur(img_in_gc, 1.6, 1.0)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.125)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.25)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.875)
        img_a = warp(img_gs, img_a, mask_warping * 0.05, 1.0)
    else:
        img_a = img_gs

    img_patch = img_in[:, :3, :, :] if mode_color else img_in.expand(img_in.shape[0], 3, img_in.shape[2], img_in.shape[3])
    img_patch = torch.cat([img_patch, img_a], dim=1)

    # 'blend' operation with alpha processing
    def alpha_blend(img_fg, img_bg):
        fg_alpha = img_fg[:, [3], :, :]
        return torch.cat([img_fg[:, :3, :, :] * fg_alpha, fg_alpha], dim=1) + img_bg * (1.0 - fg_alpha)

    # 'transform_2d' operation using only scaling and without tiling
    def scale_without_tiling(img_patch, height_scale, width_scale):
        if height_scale > 1.0 or width_scale > 1.0:
            print('Warning: the result might not be tiling correctly.')
        if width_scale != 1.0 or height_scale != 1.0:
            img_patch_rgb = transform_2d(img_patch[:, :3, :, :], mipmap_mode='manual', x1=to_zero_one(1.0 / width_scale), y2=to_zero_one(1.0 / height_scale))
            img_patch_a = transform_2d(img_patch[:, [3], :, :], tile_mode=0, mipmap_mode='manual', x1=to_zero_one(1.0 / width_scale), y2=to_zero_one(1.0 / height_scale))
        return torch.cat([img_patch_rgb, img_patch_a], dim=1)

    # Pre-computation for transformed pattern (for non-random cases)
    if pattern_height == grid_size * 100 and pattern_width == grid_size * 100:
        img_patch_sc = img_patch
    else:
        img_patch_sc = scale_without_tiling(img_patch, pattern_height / (100 * grid_size), pattern_width / (100 * grid_size))
    if rotation == 0.0:
        img_patch_rot = img_patch_sc
    else:
        angle = rotation * np.pi * 2.0
        sin_angle, cos_angle = torch.sin(angle), torch.cos(angle)
        img_patch_rot = transform_2d(img_patch_sc, mipmap_mode='manual', x1=to_zero_one(cos_angle), x2=to_zero_one(-sin_angle),
                                     y1=to_zero_one(sin_angle), y2=to_zero_one(cos_angle))
    img_patch_double = alpha_blend(img_patch_rot, img_patch_rot)

    # Randomly transform the input patch (scaling, rotation, translation, color adjustment)
    def random_transform(img_patch, pos):
        size_delta = torch.rand(1, device=img_in.device) * size_variation - torch.rand(1, device=img_in.device) * size_variation
        h_size = torch.clamp((pattern_height + size_delta) * 0.01, 0.0, 10.0)
        w_size = torch.clamp((pattern_width + size_delta) * 0.01, 0.0, 10.0)
        rot_angle = (rotation + torch.rand(1, device=img_in.device) * rotation_variation) * np.pi * 2.0
        off_angle = torch.rand(1, device=img_in.device) * np.pi * 2.0
        pos_x = pos[0] + disorder * torch.cos(off_angle)
        pos_y = pos[1] + disorder * torch.sin(off_angle)
        col_scale = torch.cat([1.0 - torch.rand(3, device=img_in.device) * color_variation, torch.ones(1, device=img_in.device)])

        # Scaling
        if size_variation == 0.0:
            img_patch = img_patch_sc
        else:
            img_patch = scale_without_tiling(img_patch, h_size / grid_size, w_size / grid_size)

        # Rotation and translation
        sin_angle, cos_angle = torch.sin(rot_angle), torch.cos(rot_angle)
        img_patch = transform_2d(img_patch, mipmap_mode='manual', x1=to_zero_one(cos_angle), x2=to_zero_one(-sin_angle), y1=to_zero_one(sin_angle), y2=to_zero_one(cos_angle))
        img_patch = transform_2d(img_patch, mipmap_mode='manual', x_offset=to_zero_one(pos_x), y_offset=to_zero_one(pos_y))
        return img_patch * col_scale.view(1, 4, 1, 1)

    # Create two layers of randomly transformed patterns (element for FX-Map)
    def gen_double_pattern(img_patch, pos):
        if size_variation == 0.0 and rotation_variation == 0.0 and disorder == 0.0 and \
           color_variation == 0.0:
            return transform_2d(img_patch_double, mipmap_mode='manual', x_offset=to_zero_one(pos[0]), y_offset=to_zero_one(pos[1]))
        else:
            return alpha_blend(random_transform(img_patch, pos), random_transform(img_patch, pos))

    # Calculate FX-Map
    fx_map = uniform_color(use_alpha=True,
                           rgba=background_color if mode_color else [color_variation] * 3 + [1.0],
                           outputsize=outputsize, device=img_in.device)
    for i in range(grid_size):
        for j in range(grid_size):
            pos = [(i + 0.5) / grid_size - 0.5, (j + 0.5) / grid_size - 0.5]
            fx_map = alpha_blend(gen_double_pattern(img_patch, pos), fx_map)

    # Output channel conversion (if needed)
    img_out = fx_map if mode_color else c2g(fx_map)
    img_out = img_out[:, :3, :, :] if mode_color and not use_alpha else img_out

    return img_out

@input_check(tensor_args=['img_in'])
def make_it_tile_photo(img_in, mask_warping_x=0.5, max_mask_warping_x=100.0, mask_warping_y=0.5, max_mask_warping_y=100.0,
                       mask_size_x=0.1, max_mask_size_x=1.0, mask_size_y=0.1, max_mask_size_y=1.0,
                       mask_precision_x=0.5, max_mask_precision_x=1.0, mask_precision_y=0.5, max_mask_precision_y=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    make it tile photo (https://substance3d.adobe.com/documentation/sddoc/make-it-tile-photo-159450503.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    # Process input parameters
    tensor = torch.as_tensor
    mask_warping_x = tensor(mask_warping_x * 2.0 - 1.0, device=img_in.device)
    mask_warping_y = tensor(mask_warping_y * 2.0 - 1.0, device=img_in.device)
    mask_size_x = tensor(mask_size_x, device=img_in.device) * max_mask_size_x
    mask_size_y = tensor(mask_size_y, device=img_in.device) * max_mask_size_y
    mask_precision_x = tensor(mask_precision_x, device=img_in.device) * max_mask_precision_x
    mask_precision_y = tensor(mask_precision_y, device=img_in.device) * max_mask_precision_y

    # Check input shape
    assert img_in.shape[2] == img_in.shape[3], "Input image is required to be in square shape"
    res = img_in.shape[2]

    # Split channels
    if img_in.shape[1] >= 3:
        img_rgb = img_in[:, :3, :, :]
        img_gs = c2g(img_rgb)
        if img_in.shape[1] == 4:
            img_a = img_in[:, [3], :, :]
    else:
        img_gs = img_in

    # Create pyramid pattern
    vec_grad = torch.linspace((-res + 1) / res, (res - 1) / res, res, device=img_in.device)
    img_grad_x = torch.abs(vec_grad.unsqueeze(0).expand(res, res))
    img_grad_y = torch.abs(vec_grad.view(res, 1).expand(res, res))
    img_pyramid = torch.clamp((1.0 - torch.max(img_grad_x, img_grad_y)) * 7.39, 0.0, 1.0).view(1, 1, res, res)

    # Create cross mask
    img_grad = 1.0 - vec_grad ** 2
    img_grad_x = img_grad.unsqueeze(0).expand(1, 1, res, res)
    img_grad_y = img_grad.view(res, 1).expand(1, 1, res, res)
    img_grad_x = levels(img_grad_x, [1.0 - mask_size_x], [0.5], [1.0 - mask_size_x * mask_precision_x])
    img_grad_y = levels(img_grad_y, [1.0 - mask_size_y], [0.5], [1.0 - mask_size_y * mask_precision_y])

    img_gs = blur_hq(img_gs.view(1, 1, res, res), intensity=1.0, max_intensity=2.75)
    if mask_warping_x != 0:
        img_grad_x = d_warp(img_grad_x, img_gs, mask_warping_x, max_mask_warping_x)
        img_grad_x = d_warp(img_grad_x, img_gs, mask_warping_x, max_mask_warping_x, 0.5)
    if mask_warping_y != 0:
        img_grad_y = d_warp(img_grad_y, img_gs, mask_warping_y, max_mask_warping_y, 0.25)
        img_grad_y = d_warp(img_grad_y, img_gs, mask_warping_y, max_mask_warping_y, 0.75)
    img_cross = blend(img_grad_x, img_grad_y, None, 'max', opacity=1.0)
    img_cross = blend(img_pyramid, img_cross, blending_mode='multiply')

    # Create sphere mask
    img_grad = vec_grad ** 2 * 16
    img_grad_x = img_grad.unsqueeze(0).expand(res, res)
    img_grad_y = img_grad.view(res, 1).expand(res, res)
    img_grad = torch.clamp(1.0 - (img_grad_x + img_grad_y), 0.0, 1.0).view(1, 1, res, res)
    img_sphere = blend(torch.cat((img_grad[:, :, res >> 1:, :], img_grad[:, :, :res >> 1, :]), dim=2),
                       torch.cat((img_grad[:, :, :, res >> 1:], img_grad[:, :, :, :res >> 1]), dim=3),
                       blending_mode='add')
    img_sphere = warp(img_sphere, img_gs, 0.24, 1.0)

    # Fix tiling for an image
    def fix_tiling(img_in):
        img = blend(img_in, transform_2d(img_in, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.5), y_offset=to_zero_one(0.5)), img_cross)
        img = blend(transform_2d(img, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.25), y_offset=to_zero_one(0.25)),
                    transform_2d(img, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.5), y_offset=to_zero_one(0.5)), img_sphere)
        return img

    # Process separate channels
    if img_in.shape[1] == 3:
        return fix_tiling(img_rgb)
    elif img_in.shape[1] == 4:
        return torch.cat((fix_tiling(img_rgb), fix_tiling(img_a)), dim=1)
    else:
        return fix_tiling(img_in)

@input_check(tensor_args=['img_in'])
def replace_color(img_in, source_color, target_color, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    replace color (https://substance3d.adobe.com/documentation/sddoc/replace-color-159449260.html)
        img_in (RGB(A) only)
        return image (RGB(A) only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 3, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image')
    target_color = as_comp_tensor(target_color, dtype=torch.float32, device=img_in.device)
    source_color = as_comp_tensor(source_color, dtype=torch.float32, device=img_in.device)
    target_hsl = rgb2hsl(target_color)
    source_hsl = rgb2hsl(source_color)
    diff_hsl = (target_hsl - source_hsl) * 0.5 + 0.5
    img_out = hsl(img_in, diff_hsl[0], diff_hsl[1], diff_hsl[2])
    return img_out

@input_check(tensor_args=[])
def normal_color(normal_format='dx', num_imgs=1, outputsize=[9, 9], use_alpha=False, direction=0.0, slope_angle=0.0, device=torch.device('cpu')):
    '''
    normal color (a non-atomic function of uniform_color)
        return image (RGB(A) only)
    Behavior:
        identical to sbs
    '''
    assert normal_format in ('dx', 'gl')
    # res_h = 1 << outputsize[0]
    # res_w = 1 << outputsize[1]
    direction = as_comp_tensor(direction, dtype=torch.float32, device=device) * (np.pi * 2)
    slope_angle = as_comp_tensor(slope_angle, dtype=torch.float32, device=direction.device) * (np.pi * 2)
    vec = torch.stack([-torch.cos(direction), torch.sin(direction) * (1.0 if normal_format == 'gl' else -1.0)])
    vec = vec * torch.sin(slope_angle) * 0.5 + 0.5
    rgba = torch.cat([vec, torch.ones(2, device=direction.device)])
    img_out = uniform_color(mode='color', num_imgs=num_imgs, use_alpha=use_alpha, rgba=rgba, outputsize=outputsize, device=direction.device)
    return img_out

@input_check(tensor_args=['img_in', 'rotation_map'])
def normal_vector_rotation(img_in, rotation_map=None, normal_format='dx', rotation_angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Normal Vector Rotation (https://substance3d.adobe.com/documentation/sddoc/normal-vector-rotation-172819817.html)
        img_in (RGB(A) only)
        rotation_map (G only)
        return image (RGB(A) only)
    Behavior:
        identical to sbs (? - still untested)
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 4, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    if rotation_map is None:
        rotation_map = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    # rotation angles are given in [0,1], convert to [0,2*pi]
    rotation_angle = as_comp_tensor(rotation_angle, dtype=torch.float32, device=device) * (np.pi * 2)
    rotation_map = rotation_map * (np.pi * 2)

    # split normal map into xy and z (z optionally with alpha)
    normal_za = img_in[:, 2:, :, :] # includes alpha if present
    normal_xy = img_in[:, :2, :, :]

    # the rotation angle is a global offset for the per-pixel rotations given in the rotation map
    rotation_map = rotation_map + rotation_angle

    # counterclockwise rotation if OpenGL, clockwise if DirectX
    if normal_format == 'dx':
        rotation_map = -rotation_map
    normal_xy = normal_xy * 2 - 1 # from [0,1] to [-1,1]
    normal_xy = torch.cat([
        torch.cos(rotation_map) * normal_xy[:, [0], :, :] + -torch.sin(rotation_map) * normal_xy[:, [1], :, :],
        torch.sin(rotation_map) * normal_xy[:, [0], :, :] + torch.cos(rotation_map) * normal_xy[:, [1], :, :]
        ], dim=1)
    normal_xy = (normal_xy + 1) / 2 # from [-1,1] to [0,1]

    return torch.cat([normal_xy, normal_za], dim=1)

@input_check(tensor_args=['img_in', 'vector_field'])
def vector_morph(img_in, vector_field=None, amount=1.0, max_amount=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    vector morph (https://substance3d.adobe.com/documentation/sddoc/vector-morph-166363405.html)
        img_in (G or RGB(A))
        vector_field (RGB(A) only)
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        if vector_field is not None:
            img_in = torch.zeros(size=[1, 1, vector_field.shape[2], vector_field.shape[3]], dtype=torch.float32, device=vector_field.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    # default for vector_field is handled below

    # Check input
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    if vector_field is None:
        vector_field = img_in.expand(img_in.shape[0], 2, res_h, res_w) if img_in.shape[1] == 1 else img_in[:, :2, :, :]
    else:
        vector_field = color_input_check(vector_field, 'vector field')
        vector_field = vector_field[:, :2, :, :]

    # Process parameter
    amount = as_comp_tensor(amount, dtype=img_in.dtype, device=img_in.device)
    if amount == 0.0:
        return img_in

    # Progressive vector field sampling
    row_grid, col_grid = torch.meshgrid(torch.linspace(1, res_h * 2 - 1, res_h, device=img_in.device) / (res_h * 2), \
                                        torch.linspace(1, res_w * 2 - 1, res_w, device=img_in.device) / (res_w * 2))
    sample_grid = torch.stack((col_grid, row_grid), dim=2).expand(img_in.shape[0], res_h, res_w, 2)

    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    vector_field_pad = torch.nn.functional.pad(vector_field, [1, 1, 1, 1], 'circular')
    for i in range(16):
        if i == 0:
            vec = vector_field
        else:
            sample_grid_sp = (sample_grid * 2.0 - 1.0) * torch.tensor([res_w / (res_w + 2), res_h / (res_h + 2)], device=img_in.device)
            vec = torch.nn.functional.grid_sample(vector_field_pad, sample_grid_sp, gs_interp_mode, gs_padding_mode, align_corners=False)
        sample_grid = torch.remainder(sample_grid + (vec.permute(0, 2, 3, 1) - 0.5) * amount * 0.0625, 1.0)

    # Final image sampling
    sample_grid = (sample_grid * 2.0 - 1.0) * torch.tensor([res_w / (res_w + 2), res_h / (res_h + 2)], device=img_in.device)
    img_in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = torch.nn.functional.grid_sample(img_in_pad, sample_grid, gs_interp_mode, gs_padding_mode, align_corners=False)

    return img_out

@input_check(tensor_args=['img_in', 'vector_map'])
def vector_warp(img_in, vector_map=None, vector_format='dx', intensity=1.0, max_intensity=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    vector warp (https://substance3d.adobe.com/documentation/sddoc/vector-warp-159450546.html)
        img_in (G or RGB(A))
        vector_map (RGB(A) only)
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        if vector_map is not None:
            img_in = torch.zeros(size=[1, 1, vector_map.shape[2], vector_map.shape[3]], dtype=torch.float32, device=vector_map.device)
        else:
            img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)
    # default for vector_map is handled below

    # Check input
    assert vector_format in ('dx', 'gl')
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    if vector_map is None:
        vector_map = torch.zeros(img_in.shape[0], 2, res_h, res_w, device=img_in.device)
    else:
        vector_map = color_input_check(vector_map, 'vector map')
        vector_map = vector_map[:, :2, :, :]

    # Process input parameters
    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device)
    if intensity == 0.0:
        return img_in

    # Calculate displacement field
    vector_map = vector_map * 2.0 - 1.0
    if vector_format == 'gl':
        vector_map[:, [1], :, :] = -vector_map[:, [1], :, :]
    vector_map = vector_map * torch.sqrt(torch.sum(vector_map ** 2, 1, keepdim=True)) * intensity

    # Sample input image
    row_grid, col_grid = torch.meshgrid(torch.linspace(1, res_h * 2 - 1, res_h, device=img_in.device) / (res_h * 2), \
                                        torch.linspace(1, res_w * 2 - 1, res_w, device=img_in.device) / (res_w * 2))
    sample_grid = torch.stack((col_grid, row_grid), dim=2).expand(img_in.shape[0], res_h, res_w, 2)
    sample_grid = torch.remainder(sample_grid + vector_map.permute(0, 2, 3, 1), 1.0)
    sample_grid = (sample_grid * 2.0 - 1.0) * torch.tensor([res_w / (res_w + 2), res_h / (res_h + 2)], device=img_in.device)

    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    img_in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = torch.nn.functional.grid_sample(img_in_pad, sample_grid, gs_interp_mode, gs_padding_mode, align_corners=False)

    return img_out

@input_check(tensor_args=['img_in'])
def contrast_luminosity(img_in, contrast=0.5, luminosity=0.5, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    contrast/luminosity (https://substance3d.adobe.com/documentation/sddoc/contrast-luminosity-159449189.html)
        img_in (G or RGB)
        return image (G or RGB)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    # Process input parameters
    contrast = as_comp_tensor(contrast, dtype=img_in.dtype, device=img_in.device) * 2.0 - 1.0
    luminosity = as_comp_tensor(luminosity, dtype=img_in.dtype, device=img_in.device) * 2.0 - 1.0

    in_low = torch.clamp(contrast * 0.5, 0.0, 0.5)
    in_high = torch.clamp(1.0 - contrast * 0.5, 0.5, 1.0)
    temp = torch.abs(torch.min(contrast, torch.tensor(0.0, device=img_in.device))) * 0.5
    out_low = torch.clamp(temp + luminosity, 0.0, 1.0)
    out_high = torch.clamp(luminosity + 1.0 - temp, 0.0, 1.0)
    img_out = levels(img_in, in_low, 0.5, in_high, out_low, out_high)

    return img_out

@input_check(tensor_args=['img_in'])
def p2s(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Pre-Multiplied to Straight (https://substance3d.adobe.com/documentation/sddoc/pre-multiplied-to-straight-159450478.html)
        img_in (RGBA only)
        return image (RGBA only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 4, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image', with_alpha=True)
    assert img_in.shape[1] == 4, 'input image must contain alpha channel'

    rgb = img_in[:,:3,:,:]
    a = img_in[:,[3],:,:]
    img_out = torch.cat([(rgb / (a + 1e-15)).clamp(0.0, 1.0), a], 1)
    return img_out

@input_check(tensor_args=['img_in'])
def s2p(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Straight to Pre-Multiplied (https://substance3d.adobe.com/documentation/sddoc/straight-to-pre-multiplied-159450483.html)
        img_in (RGBA only)
        return image (RGBA only)
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 4, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    img_in = color_input_check(img_in, 'input image', with_alpha=True)
    assert img_in.shape[1] == 4, 'input image must contain alpha channel'

    rgb = img_in[:,:3,:,:]
    a = img_in[:,[3],:,:]
    img_out = torch.cat([rgb * a, a], 1)
    return img_out

@input_check(tensor_args=['img_in'])
def clamp(img_in, clamp_alpha=True, low=0.0, high=1.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Clamp (https://substance3d.adobe.com/documentation/sddoc/clamp-159449164.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    low = as_comp_tensor(low, dtype=img_in.dtype, device=img_in.device)
    high = as_comp_tensor(high, dtype=img_in.dtype, device=img_in.device)
    if img_in.shape[1] == 4 and not clamp_alpha:
        img_out = torch.cat([img_in[:,:3,:,:].clamp(low, high), img_in[:,[3],:,:]], 1)
    else:
        img_out = img_in.clamp(low, high)
    return img_out

@input_check(tensor_args=['img_in'])
def pow(img_in, exponent=0.4, max_exponent=10.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Pow (https://substance3d.adobe.com/documentation/sddoc/pow-159449251.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    exponent = as_comp_tensor(exponent, dtype=img_in.dtype, device=img_in.device) * max_exponent
    use_alpha = False
    if img_in.shape[1] == 4:
        use_alpha = True
        img_in_alpha = img_in[:,[3],:,:]
        img_in = img_in[:,:3,:,:]

    # Levels
    in_mid = (exponent - 1.0) / 16.0 + 0.5 if exponent >= 1.0 else \
             0.5625 if exponent == 0 else (1.0 / exponent - 9.0) / -16.0
    img_out = levels(img_in, in_mid=in_mid)

    if use_alpha:
        img_out = torch.cat([img_out, img_in_alpha], 1)
    return img_out

@input_check(tensor_args=['img_in'])
def quantize(img_in, quantize_number=3, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Quantize (https://substance3d.adobe.com/documentation/sddoc/quantize-159449255.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        almost identical to sbs (minor quantization error)
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    qn = (as_comp_tensor(quantize_number, dtype=img_in.dtype, device=img_in.device) - 1) / 255.0
    qt_shift = 1.0 - 286.0 / 512.0
    img_in = levels(img_in, out_high=qn)
    img_qt = torch.floor(img_in * 255.0 + qt_shift) / 255.0
    img_out = levels(img_qt, in_high=qn)
    return img_out

@input_check(tensor_args=['img_in'])
def anisotropic_blur(img_in, high_quality=False, intensity=10.0/16.0, max_intensity=16.0, anisotropy=0.5, angle=0.0, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Anisotropic Blur (https://substance3d.adobe.com/documentation/sddoc/anisotropic-blur-159450450.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    intensity = as_comp_tensor(intensity, dtype=img_in.dtype, device=img_in.device) * max_intensity
    anisotropy = as_comp_tensor(anisotropy, dtype=img_in.dtype, device=img_in.device)
    angle = as_comp_tensor(angle, dtype=img_in.dtype, device=img_in.device)
    quality_factor = 0.6 if high_quality else 1.0

    # Two-pass directional blur
    img_out = d_blur(img_in, intensity * quality_factor, 1.0, angle)
    img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, 1.0, angle + 0.25)
    if high_quality:
        img_out = d_blur(img_out, intensity * quality_factor, 1.0, angle)
        img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, 1.0, angle + 0.25)

    return img_out

@input_check(tensor_args=['img_in'])
def glow(img_in, glow_amount=0.5, clear_amount=0.5, size=0.5, max_size=20.0, color=[1.0, 1.0, 1.0, 1.0], outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Glow (https://substance3d.adobe.com/documentation/sddoc/glow-159450531.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    glow_amount = as_comp_tensor(glow_amount, dtype=img_in.dtype, device=img_in.device)
    clear_amount = as_comp_tensor(clear_amount, dtype=img_in.dtype, device=img_in.device)
    size = as_comp_tensor(size, dtype=img_in.dtype, device=img_in.device) * max_size
    color = as_comp_tensor(color, dtype=img_in.dtype, device=img_in.device)

    # Calculate glow mask
    num_channels = img_in.shape[1]
    img_mask = img_in[:,:3,:,:].sum(dim=1, keepdim=True) / 3 if num_channels > 1 else img_in
    img_mask = levels(img_mask, in_low=clear_amount - 0.01, in_high=clear_amount + 0.01)
    img_mask = blur_hq(img_mask, intensity=size, max_intensity=1.0)

    # Blending in glow effect
    if num_channels > 1:
        img_out = blend(color[:num_channels].view(1,num_channels,1,1).expand_as(img_in), img_in, img_mask * glow_amount, 'add')
    else:
        img_out = blend(img_mask, img_in, None, 'add', opacity=glow_amount)
    return img_out

@input_check(tensor_args=['img_in'])
def car2pol(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Cartesian to Polar (https://substance3d.adobe.com/documentation/sddoc/cartesian-to-polar-159450598.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    row_grid, col_grid = torch.meshgrid(torch.linspace(0.5, res_h - 0.5, res_h, device=img_in.device) / res_h - 0.5,
                                     torch.linspace(0.5, res_w - 0.5, res_w, device=img_in.device) / res_w - 0.5)
    rad_grid = torch.remainder(torch.sqrt(row_grid ** 2 + col_grid ** 2) * 2.0, 1.0) * 2.0 - 1.0
    ang_grid = torch.remainder(-torch.atan2(row_grid, col_grid) / (np.pi * 2), 1.0) * 2.0 - 1.0
    rad_grid = rad_grid * res_h / (res_h + 2)
    ang_grid = ang_grid * res_w / (res_w + 2)
    sample_grid = torch.stack([ang_grid, rad_grid], 2).expand(img_in.shape[0], res_h, res_w, 2)
    in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = torch.nn.functional.grid_sample(in_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)
    return img_out

@input_check(tensor_args=['img_in'])
def pol2car(img_in, outputsize=[9, 9], device=torch.device('cpu')):
    '''
    Polar to Cartesian (https://substance3d.adobe.com/documentation/sddoc/polar-to-cartesian-159450602.html)
        img_in (G or RGB(A))
        return image (G or RGB(A))
    Behavior:
        identical to sbs
    '''
    # defaults
    if img_in is None:
        img_in = torch.zeros(size=[1, 1, 1 << outputsize[0], 1 << outputsize[1]], dtype=torch.float32, device=device)

    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    row_grid, col_grid = torch.meshgrid(torch.linspace(0.5, res_h - 0.5, res_h, device=img_in.device) / res_h,
                                     torch.linspace(0.5, res_w - 0.5, res_w, device=img_in.device) / res_w)
    ang_grid = -col_grid * (np.pi * 2.0)
    rad_grid = row_grid * 0.5
    x_grid = torch.remainder(rad_grid * torch.cos(ang_grid) + 0.5, 1.0) * 2.0 - 1.0
    y_grid = torch.remainder(rad_grid * torch.sin(ang_grid) + 0.5, 1.0) * 2.0 - 1.0
    x_grid = x_grid * res_w / (res_w + 2)
    y_grid = y_grid * res_h / (res_h + 2)
    sample_grid = torch.stack([x_grid, y_grid], 2).expand(img_in.shape[0], res_h, res_w, 2)
    in_pad = torch.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = torch.nn.functional.grid_sample(in_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)
    return img_out

'''
Mathematical functions used in the implementation of nodes.
'''
def lerp(start, end, weight):
    assert weight >= 0.0 and weight <= 1.0, 'weight should be in [0,1]'
    return start + (end-start) * weight

def rgb2hsl(rgb):
    # rgb = as_comp_tensor(rgb, dtype=torch.float32)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    # compute s,v
    max_vals = torch.max(rgb)
    min_vals = torch.min(rgb)
    delta = max_vals - min_vals
    l = (max_vals + min_vals) / 2.0
    if delta == 0:
        s = torch.tensor(0.0, device=rgb.device)
    else:
        s = delta / (1.0 - torch.abs(2*l - 1.0))
    h = torch.zeros_like(s)

    # compute h
    if delta == 0:
        h = torch.tensor(0.0, device=rgb.device)
    elif rgb[0] == max_vals:
        h = torch.remainder((g-b)/delta, 6.0) / 6.0
    elif rgb[1] == max_vals:
        h = ((b-r)/delta + 2.0) / 6.0
    elif rgb[2] == max_vals:
        h = ((r-g)/delta + 4.0) / 6.0

    return torch.stack([h,s,l])

def ellipse(samples, sample_number, radius, ellipse_factor, rotation, inner_rotation, center_x):
    '''
    Ellipse sampling function. (No detailed explanation in SBS)
    '''
    radius = as_comp_tensor(radius)
    ###
    angle_1 = (as_comp_tensor(sample_number, device=radius.device).float() / as_comp_tensor(samples, device=radius.device).float() + as_comp_tensor(inner_rotation, device=radius.device)) * np.pi * 2.0
    angle_2 = -as_comp_tensor(rotation, device=radius.device) * np.pi * 2.0
    sin_1, cos_1 = torch.sin(angle_1), torch.cos(angle_1)
    sin_2, cos_2 = torch.sin(angle_2), torch.cos(angle_2)
    factor_1 = (1.0 - as_comp_tensor(ellipse_factor, device=radius.device)) * sin_1
    # factor_2 = torch.lerp(cos_1, torch.abs(cos_1), torch.max(as_comp_tensor(center_x) * 0.5, as_comp_tensor(0.0)))
    factor_2 = lerp(cos_1, torch.abs(cos_1), torch.max(as_comp_tensor(center_x, device=radius.device) * 0.5, as_comp_tensor(0.0, device=radius.device)))
    # assemble results
    res_x = radius * (factor_1 * sin_2 + factor_2 * cos_2)
    res_y = radius * (factor_1 * cos_2 - factor_2 * sin_2)
    return torch.cat((res_x.unsqueeze(0), res_y.unsqueeze(0)))

def hbao_radius(min_size_log2, mip_level, radius):
    '''
    'hbao_radius_function' in sbs, used in HBAO.
    '''
    radius = as_comp_tensor(radius)
    min_size_log2 = as_comp_tensor(min_size_log2, device=radius.device).float()
    mip_level = as_comp_tensor(mip_level, device=radius.device).float()
    radius = radius * 2.0 ** (min_size_log2 - mip_level) - 1
    return torch.clamp(radius, 0.0, 1.0)

def create_mipmaps(img_in, mipmaps_level, keep_size=False):
    '''
    Create mipmap levels for an input image using box filtering.
    '''
    mipmaps = []
    img_mm = img_in
    # last_shape = [img_in.shape[2], img_in.shape[3]]
    for i in range(mipmaps_level):
        reached_minsize = min(img_mm.shape[2], img_mm.shape[3]) <= 1
        img_mm = img_mm if reached_minsize else manual_resize(img_mm, -1)
        if not keep_size:
            mipmaps.append(img_mm)
        elif reached_minsize:
            mipmaps.append(mipmaps[-1]) if len(mipmaps) > 0 else img_mm
        elif max(img_mm.shape[2], img_mm.shape[3]) == 1:
            mipmaps.append(img_mm.expand_as(img_in))
        else:
            automatic_resize(img_mm, i + 1)
        # mipmaps.append(img_mm if not keep_size else \
        #                mipmaps[-1] if reached_minsize else \
        #                img_mm.expand_as(img_in) if max(img_mm.shape[2], img_mm.shape[3]) == 1 else \
        #                automatic_resize(img_mm, i + 1))
        # last_shape = [img_mm.shape[2], img_mm.shape[3]]
    return mipmaps

def frequency_transform(img_in, normal_format='dx'):
    '''
    Calculate convolution at multiple frequency levels.
    '''
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))

    # Create mipmap levels for R and G channels
    img_in = img_in[:, :2, :, :]
    mm_list = [img_in]
    if in_size_log2 > 4:
        mm_list.extend(create_mipmaps(img_in, in_size_log2 - 4))

    # Define convolution operators
    def conv_x(img):
        img_bw = torch.clamp(img - roll_col(img, -1), -0.5, 0.5)
        img_fw = torch.clamp(roll_col(img, 1) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5

    def conv_y(img):
        dr = -1 if normal_format == 'dx' else 1
        img_bw = torch.clamp(img - roll_row(img, dr), -0.5, 0.5)
        img_fw = torch.clamp(roll_row(img, -dr) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5

    conv_ops = [conv_x, conv_y]

    # Init blended images
    img_freqs = [[], []]

    # Low frequencies (for 16x16 images only)
    img_4 = mm_list[-1]
    img_4_scale = [None, None, None, img_4]
    for i in range(3):
        img_4_scale[i] = transform_2d(img_4, x1=to_zero_one(2.0 ** (3 - i)), y2=to_zero_one(2.0 ** (3 - i)))
    for i, scale in enumerate([8.0, 4.0, 2.0, 1.0]):
        for c in (0, 1):
            img_4_c = conv_ops[c](img_4_scale[i][:, [c], :, :])
            if scale > 1.0:
                img_4_c = transform_2d(img_4_c, mipmap_mode='manual', x1=to_zero_one(1.0 / scale), y2=to_zero_one(1.0 / scale))
            img_freqs[c].append(img_4_c)

    # Other frequencies
    for i in range(len(mm_list) - 1):
        for c in (0, 1):
            img_i_c = conv_ops[c](mm_list[-2 - i][:, [c], :, :])
            img_freqs[c].append(img_i_c)

    return img_freqs

def automatic_resize(img_in, scale_log2, filtering='bilinear'):
    '''
    Progressively resize an input image.
        img_in (G only or RGB(A))
        scale_log2: size change relative to input (after log2).
        filtering: bilinear or nearest sampling.
        return image (G only or RGB(A))
    '''
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size_log2 = [int(np.log2(img_in.shape[2])), int(np.log2(img_in.shape[3]))] # h, w
    scale_log2 = max(scale_log2, -min(in_size_log2)) # can down-sample only until one dimension is 1 pixel wide (size_log2=0)
    # out_size_log2 = np.array([max(in_size_log2[0] + scale_log2, 0), max(in_size_log2[1] + scale_log2, 0)])

    # Down-sampling (regardless of filtering)
    if scale_log2 <= 0:
        img_out = img_in
        for _ in range(-scale_log2): # stop down-sampling as soon as one dimension hits 1 pixel wide
            img_out = manual_resize(img_out, -1)
    # Up-sampling (progressive bilinear filtering)
    elif filtering == 'bilinear':
        img_out = img_in
        for _ in range(scale_log2):
            img_out = manual_resize(img_out, 1)
    # Up-sampling (nearest sampling)
    else:
        img_out = manual_resize(img_in, scale_log2, filtering)

    return img_out

def manual_resize(img_in, scale_log2, filtering='bilinear'):
    '''
    Manually resize an input image (all-in-one sampling).
        img_in (G only or RGB(A))
        scale_log2: size change relative to input (after log2).
        filtering: bilinear or nearest sampling.
        return image (G only or RGB(A))
    '''
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size = [img_in.shape[2], img_in.shape[3]]
    in_size_log2 = [int(np.log2(in_size[0])), int(np.log2(in_size[1]))] # h, w
    scale_log2 = max(scale_log2, -min(in_size_log2)) # can down-sample only until one dimension is 1 pixel wide (size_log2=0)
    out_size_log2 = [in_size_log2[0] + scale_log2, in_size_log2[1] + scale_log2] # h, w
    out_size = [1 << out_size_log2[0], 1 << out_size_log2[1]] # h, w

    # Equal size
    if scale_log2 == 0:
        img_out = img_in
    else:
        row_grid, col_grid = torch.meshgrid(
            torch.linspace(1, out_size[0] * 2 - 1, out_size[0], device=img_in.device),
            torch.linspace(1, out_size[1] * 2 - 1, out_size[1], device=img_in.device))
        sample_grid = torch.stack([col_grid, row_grid], dim=2).expand(img_in.shape[0], out_size[0], out_size[1], 2)
        sample_grid[:, :, :, 0] = (sample_grid[:, :, :, 0] / (out_size[1] * 2)) * 2.0 - 1.0 # sample x-coordinates in [-1, 1]
        sample_grid[:, :, :, 1] = (sample_grid[:, :, :, 1] / (out_size[0] * 2)) * 2.0 - 1.0 # sample y-coordinates in [-1, 1]
        # Down-sampling
        if scale_log2 < 0:
            img_out = torch.nn.functional.grid_sample(img_in, sample_grid, filtering, 'zeros', align_corners=False)
        # Up-sampling
        else:
            sample_grid[:, :, :, 0] = sample_grid[:, :, :, 0] * in_size[1] / (in_size[1] + 2)
            sample_grid[:, :, :, 1] = sample_grid[:, :, :, 1] * in_size[0] / (in_size[0] + 2)
            # sample_grid = sample_grid * in_size / (in_size + 2)
            img_in_pad = torch.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
            img_out = torch.nn.functional.grid_sample(img_in_pad, sample_grid, filtering, 'zeros', align_corners=False)

    return img_out

# def affine_grid_2d(theta, img_shape, align_corners=False):
#     '''
#     Generate a differentiable affine grid; same behavior as torch.nn.functional.affine_grid for 2d matrix
#     '''
#     assert theta.shape[1] == 2 and theta.shape[2] == 3, 'please input an affine matrix with shape [n,2,3]'
#     assert theta.shape[0] == img_shape[0], 'batch number of theta does not match batch number of image shape'
#     if align_corners:
#         # X changes a colunm, Y changes a Row
#         X,Y = torch.meshgrid(torch.linspace(-1.0, 1.0, img_shape[2]), torch.linspace(-1.0, 1.0, img_shape[3]))
#     else:
#         X,Y = torch.meshgrid(torch.linspace(1.0/img_shape[2]-1.0, 1.0-1.0/img_shape[2], img_shape[2]), \
#                     torch.linspace(1.0/img_shape[3]-1.0, 1.0-1.0/img_shape[3], img_shape[3]))

#     flat_grid = torch.stack([Y.flatten(), X.flatten(), torch.ones_like(X.flatten())], dim=0)
#     grid = torch.zeros(theta.shape[0],img_shape[2], img_shape[3], 2)
#     for i in range(theta.shape[0]):
#         theta_slice = theta[i,:,:]
#         grid_slice = torch.mm(theta_slice, flat_grid)
#         grid_slice_x = grid_slice[1,:].reshape(img_shape[2], img_shape[3])
#         grid_slice_y = grid_slice[0,:].reshape(img_shape[2], img_shape[3])
#         grid[i,:,:,:] = torch.stack([grid_slice_y, grid_slice_x], dim=2)

#     return grid



