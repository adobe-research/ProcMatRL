# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch
import numpy as np

def is_normal_inverted(normal_map):
    normal = normal_map.clone()
    normal_inverse = normal_map.clone()
    normal_inverse[:,1] = 1.0 - normal_inverse[:,1]
    original_curl = l_infinite_curl(normal)
    inverted_curl = l_infinite_curl(normal_inverse)
    if (original_curl / inverted_curl) <= 0.85:
        return False
    else:
        return True


def l_infinite_curl(normal_map):
    # central finite differences assuming tileble normal map
    normal = (normal_map * 2.0) - 1.0
    nx = normal[:, 0]
    nx_dy = torch.roll(nx, -1, dims=1) - torch.roll(nx, 1, dims=1)

    ny = normal[:,1]
    ny_dx = torch.roll(ny, 1, dims=2) - torch.roll(ny, -1, dims=2)

    return torch.abs(nx_dy - ny_dx).max()


def ggx_ndf(cos_h, alpha):
    mask = cos_h > 0.0
    c2 = cos_h ** 2
    t2 = (1 - c2) / (c2 + 1e-8)
    a2 = alpha ** 2
    denom = np.pi * c2**2 * (a2 + t2)**2 + 1e-8  # add 1e-8 to avoid zero-division
    return a2 * mask / denom

def brdf(n_dot_h, alpha, f0):
    D = ggx_ndf(n_dot_h, alpha)
    # return f0 * D / (4.0 * n_dot_h**2 + 1e-8) # add 1e-8 to avoid zero-division
    return f0 * D / 4.0 # set n_dot_h**2 as geometry function

def render_material(basecolor, normal, roughness, metallic, light_color=None, f0=0.04, size=30.0, camera=None, diffuse_only_flag=False, normal_format='gl', force_ogl_normal=False):

    if any(input_image.ndim != 4 for input_image in [basecolor, normal, roughness, metallic]):
        raise RuntimeError('Invalid shape for an input to the material renderer.')

    # default parameters
    if light_color is None:
        light_color = torch.tensor([3300.0, 3300.0, 3300.0], dtype=torch.float32, device=basecolor.device)
    light_color = light_color.view(1, 3, 1, 1)
    if camera is None:
        camera = torch.tensor([0, 0, 25.0], dtype=torch.float32, device=basecolor.device)
    camera = camera.view(1, 3, 1, 1)

    # Remove alpha channels and convert grayscale/color if necessary
    if basecolor.shape[1] == 4:
        basecolor = basecolor[:, :3, :, :]
    elif basecolor.shape[1] == 1: # basecolor should always have 3 channels
        basecolor = basecolor.expand(-1, 3, -1, -1)
    if normal.shape[1] == 4:
        normal = normal[:, :3, :, :]
    elif normal.shape[1] == 1: # normal should always have 3 channels
        normal = normal.expand(-1, 3, -1, -1)
    if roughness.shape[1] != 1: # roughness should always have 1 channel (also ignore the alpha channel)
        roughness = roughness[:, :3, :, :].mean(dim=1, keepdim=True)
    if metallic.shape[1] != 1: # metallic should always have 1 channel (also ignore the alpha channel)
        metallic = metallic[:, :3, :, :].mean(dim=1, keepdim=True)

    # if manually flip Y axis of normal
    if force_ogl_normal and is_normal_inverted(normal) or not force_ogl_normal and normal_format == 'dx':
        normal[:, 1] = 1.0 - normal[:, 1]

    # assume albedo in gamma space
    basecolor = basecolor ** 2.2

    axis = 1
    material_plane_size = normal.shape[-1]

    # from normal map to normals (range [0, 1] to [-1, 1] for each channel/axis)
    normal = normal * 2.0 - 1.0
    # normalize normals
    normal = normal / torch.clamp((normal**2).sum(dim=axis, keepdim=True).sqrt(), min=1e-8)

    # update albedo using metallic
    f0 = f0 + metallic * (basecolor - f0)
    basecolor = basecolor * (1.0 - metallic)

    # n points between [-size/2, size/2]
    x = torch.arange(material_plane_size, dtype=torch.float32, device=basecolor.device)
    x = (x + 0.5) / material_plane_size
    x = x*2.0 - 1.0
    x *= size / 2.0

    # surface positions
    y, x = torch.meshgrid((x, x), indexing='ij')
    z = torch.zeros_like(x)
    pos = torch.stack((x, -y, z), dim=0).unsqueeze(dim=0)

    # directions (omega_in = omega_out = half)
    omega = camera - pos
    dist_sq = (omega ** 2).sum(axis, keepdims=True)
    d = torch.sqrt(dist_sq)
    omega = omega / (torch.cat((d, d, d), axis) + 1e-8)

    # geometry term and brdf
    n_dot_h = (omega * normal).sum(axis, keepdims=True)
    geom = n_dot_h / (dist_sq + 1e-8)
    diffuse = geom * light_color * basecolor / np.pi

    # if nan presents, set value to 0
    diffuse[torch.isnan(diffuse)] = 0.0
    diffuse[torch.isinf(diffuse)] = 1.0
    diffuse = torch.clamp(diffuse, 0.0, 1.0)

    if diffuse_only_flag:
        rendering = diffuse
    else:
        specular = geom * brdf(n_dot_h, roughness ** 2, f0) * light_color
        specular[torch.isnan(specular)] = 0.0
        specular[torch.isinf(specular)] = 1.0
        specular = torch.clamp(specular, 0.0, 1.0)
        rendering = diffuse + specular

    rendering = torch.clamp(rendering, 1e-10, 1.0) ** (1/2.2)
    return rendering
