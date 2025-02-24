# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import shutil
import numbers
from collections import deque
import uuid
import xml.etree.ElementTree as ET
import platform

import numpy as np
import networkx as nx

from . import sbs_function_graph
from . import sbs_parameter
from . import sbs_graph_nodes
from . import sbs_graph
from .sbs_param_type import SBSParamType, param_type_to_tag, param_type_is_image, param_type_idx_to_name
from .sbs_utils import resolve_dependency_path, gen_unique_uids

# Format: node name in SBS xml -> (class name in sbs_graph_nodes.py, function name)
type_dict = {
    # Atomic nodes
    'blend':                    ('Blend', 'blend'),
    'blur':                     ('Blur', 'blur'),
    'shuffle':                  ('ChannelShuffle', 'channel_shuffle'),
    'curve':                    ('Curve', 'curve'),
    'dirmotionblur':            ('DBlur', 'd_blur'),
    'directionalwarp':          ('DWarp', 'd_warp'),
    'distance':                 ('Distance', 'distance'),
    'emboss':                   ('Emboss', 'emboss'),
    'gradient':                 ('GradientMap', 'gradient_map'),
    'dyngradient':              ('GradientMapDyn', 'gradient_map_dyn'),
    'grayscaleconversion':      ('C2G', 'c2g'),
    'hsl':                      ('HSL', 'hsl'),
    'levels':                   ('Levels', 'levels'),
    'normal':                   ('Normal', 'normal'),
    'sharpen':                  ('Sharpen', 'sharpen'),
    'transformation':           ('Transform2d', 'transform_2d'),
    'uniform':                  ('UniformColor', 'uniform_color'),
    'warp':                     ('Warp', 'warp'),

    # Non-atomic nodes (add as needed)
    'curvature':                  ('Curvature', 'curvature'),
    'invert_grayscale':           ('Invert', 'invert'),
    'invert':                     ('Invert', 'invert'),
    'histogram_scan':             ('HistogramScan', 'histogram_scan'),
    'histogram_range':            ('HistogramRange', 'histogram_range'),
    'histogram_select':           ('HistogramSelect', 'histogram_select'),
    'edge_detect':                ('EdgeDetect', 'edge_detect'),
    'safe_transform_grayscale':   ('SafeTransform', 'safe_transform'),
    'safe_transform':             ('SafeTransform', 'safe_transform'),
    'blur_hq_grayscale':          ('BlurHQ', 'blur_hq'),
    'blur_hq':                    ('BlurHQ', 'blur_hq'),
    'non_uniform_blur_grayscale': ('NonUniformBlur', 'non_uniform_blur'),
    'non_uniform_blur':           ('NonUniformBlur', 'non_uniform_blur'),
    'bevel':                      ('Bevel', 'bevel'),
    'slope_blur_grayscale_2':     ('SlopeBlur', 'slope_blur'),
    'slope_blur_grayscale':       ('SlopeBlur', 'slope_blur'),  # The difference from '_2' is unknown
    'slope_blur':                 ('SlopeBlur', 'slope_blur'),
    'mosaic_grayscale':           ('Mosaic', 'mosaic'),
    'mosaic':                     ('Mosaic', 'mosaic'),
    'auto_levels':                ('AutoLevels', 'auto_levels'),
    'ambient_occlusion_2':        ('AmbientOcclusion', 'ambient_occlusion'),
    'hbao':                       ('HBAO', 'hbao'),
    'highpass_grayscale':         ('Highpass', 'highpass'),
    'highpass':                   ('Highpass', 'highpass'),
    'normal_normalise':           ('NormalNormalize', 'normal_normalize'),
    'channel_mixer':              ('ChannelMixer', 'channel_mixer'),
    'normal_combine':             ('NormalCombine', 'normal_combine'),
    'multi_switch_grayscale':     ('MultiSwitch', 'multi_switch'),
    'multi_switch':               ('MultiSwitch', 'multi_switch'),
    'rgba_split':                 ('RGBASplit', 'rgba_split'),
    'rgba_merge':                 ('RGBAMerge', 'rgba_merge'),
    'basecolor_metallic_roughness_to_diffuse_specular_glossiness': ('PbrConverter', 'pbr_converter'),
    'height_to_normal_world_units_2': ('HeightToNormal', 'height_to_normal_world_units'),
    'normal_to_height':           ('NormalToHeight', 'normal_to_height'),
    'curvature_smooth':           ('CurvatureSmooth', 'curvature_smooth'),
    'make_it_tile_patch_grayscale': ('MakeItTilePatch', 'make_it_tile_patch'),
    'make_it_tile_patch':         ('MakeItTilePatch', 'make_it_tile_patch'),
    'make_it_tile_photo_grayscale': ('MakeItTilePhoto', 'make_it_tile_photo'),
    'make_it_tile_photo':         ('MakeItTilePhoto', 'make_it_tile_photo'),
    'rgb-a_split':                ('AlphaSplit', 'alpha_split'),
    'rgb-a_merge':                ('AlphaMerge', 'alpha_merge'),
    'switch_grayscale':           ('Switch', 'switch'),
    'switch':                     ('Switch', 'switch'),
    'normal_blend':               ('NormalBlend', 'normal_blend'),
    'mirror_grayscale':           ('Mirror', 'mirror'),
    'mirror':                     ('Mirror', 'mirror'),
    'vector_morph_grayscale':     ('VectorMorph', 'vector_morph'),
    'vector_morph':               ('VectorMorph', 'vector_morph'),
    'vector_warp_grayscale':      ('VectorWarp', 'vector_warp'),
    'vector_warp':                ('VectorWarp', 'vector_warp'),
    'passthrough':                ('Passthrough', 'passthrough'),
    'replace_color':              ('ReplaceColor', 'replace_color'),
    'normal_color':               ('NormalColor', 'normal_color'),
    'contrast_luminosity_grayscale': ('ContrastLuminosity', 'contrast_luminosity'),
    'contrast_luminosity':        ('ContrastLuminosity', 'contrast_luminosity'),
    'premult_to_straight':        ('P2S', 'p2s'),
    'straight_to_premult':        ('S2P', 's2p'),
    'clamp':                      ('Clamp', 'clamp'),
    'clamp_grayscale':            ('Clamp', 'clamp'),
    'pow':                        ('Pow', 'pow'),
    'pow_grayscale':              ('Pow', 'pow'),
    'quantize_grayscale':         ('Quantize', 'quantize'),
    'quantize':                   ('Quantize', 'quantize'),
    'anisotropic_blur_grayscale': ('AnisotropicBlur', 'anisotropic_blur'),
    'anisotropic_blur':           ('AnisotropicBlur', 'anisotropic_blur'),
    'glow_grayscale':             ('Glow', 'glow'),
    'glow':                       ('Glow', 'glow'),
    'cartesian_to_polar_grayscale': ('Car2Pol', 'car2pol'),
    'cartesian_to_polar':         ('Car2Pol', 'car2pol'),
    'polar_to_cartesian_grayscale': ('Pol2Car', 'pol2car'),
    'polar_to_cartesian':         ('Pol2Car', 'pol2car'),
}

# map of defintion paths from legacy generators to new generators
# (format: path:graph_name)
legacy_generator_map = {
    'sbs://alveolus.sbs:alveolus': 'sbs://pattern_alveolus.sbs:alveolus',
    'sbs://anisotropic_noise.sbs:anisotropic_noise': 'sbs://noise_anisotropic_noise.sbs:anisotropic_noise',
    'sbs://bnw_spots_1.sbs:bnw_spots_1': 'sbs://noise_bnw_spots_1.sbs:bnw_spots_1',
    'sbs://bnw_spots_2.sbs:bnw_spots_2': 'sbs://noise_bnw_spots_2.sbs:bnw_spots_2',
    'sbs://bnw_spots_3.sbs:bnw_spots_3': 'sbs://noise_bnw_spots_3.sbs:bnw_spots_3',
    'sbs://brick_1.sbs:brick_1': 'sbs://pattern_brick_1.sbs:brick_1',
    'sbs://brick_2.sbs:brick_2': 'sbs://pattern_brick_2.sbs:brick_2',
    'sbs://brick_generator.sbs:brick_generator': 'sbs://pattern_brick_generator.sbs:brick_generator',
    'sbs://cells_1.sbs:cells_1': 'sbs://noise_cells_1.sbs:cells_1',
    'sbs://cells_2.sbs:cells_2': 'sbs://noise_cells_2.sbs:cells_2',
    'sbs://cells_3.sbs:cells_3': 'sbs://noise_cells_3.sbs:cells_3',
    'sbs://cells_4.sbs:cells_4': 'sbs://noise_cells_4.sbs:cells_4',
    'sbs://checker_1.sbs:checker_1': 'sbs://pattern_checker_1.sbs:checker_1',
    'sbs://clouds_1.sbs:clouds_1': 'sbs://noise_clouds_1.sbs:clouds_1',
    'sbs://clouds_2.sbs:clouds_2': 'sbs://noise_clouds_2.sbs:clouds_2',
    'sbs://clouds_3.sbs:clouds_3': 'sbs://noise_clouds_3.sbs:clouds_3',
    'sbs://creased.sbs:creased': 'sbs://noise_creased.sbs:creased',
    'sbs://crystal_1.sbs:crystal_1': 'sbs://noise_crystal_1.sbs:crystal_1',
    'sbs://crystal_2.sbs:crystal_2': 'sbs://noise_crystal_2.sbs:crystal_2',
    'sbs://dirt_1.sbs:dirt_1': 'sbs://noise_dirt_1.sbs:dirt_1',
    'sbs://dirt_2.sbs:dirt_2': 'sbs://noise_dirt_2.sbs:dirt_2',
    'sbs://dirt_3.sbs:dirt_3': 'sbs://noise_dirt_3.sbs:dirt_3',
    'sbs://dirt_4.sbs:dirt_4': 'sbs://noise_dirt_4.sbs:dirt_4',
    'sbs://dirt_5.sbs:dirt_5': 'sbs://noise_dirt_5.sbs:dirt_5',
    'sbs://dirt_gradient.sbs:dirt_gradient': 'sbs://noise_dirt_gradient.sbs:dirt_gradient',
    'sbs://fibers_1.sbs:fibers_1': 'sbs://pattern_fibers_1.sbs:fibers_1',
    'sbs://fibers_2.sbs:fibers_2': 'sbs://pattern_fibers_2.sbs:fibers_2',
    'sbs://fluid.sbs:fluid': 'sbs://noise_fluid.sbs:fluid',
    'sbs://fractal_sum_1.sbs:fractal_sum_1': 'sbs://noise_fractal_sum_1.sbs:fractal_sum_1',
    'sbs://fractal_sum_2.sbs:fractal_sum_2': 'sbs://noise_fractal_sum_2.sbs:fractal_sum_2',
    'sbs://fractal_sum_3.sbs:fractal_sum_3': 'sbs://noise_fractal_sum_3.sbs:fractal_sum_3',
    'sbs://fractal_sum_4.sbs:fractal_sum_4': 'sbs://noise_fractal_sum_4.sbs:fractal_sum_4',
    'sbs://fractal_sum_base.sbs:fractal_sum_base': 'sbs://noise_fractal_sum_base.sbs:fractal_sum_base',
    'sbs://fur_1.sbs:fur_1': 'sbs://noise_fur_1.sbs:fur_1',
    'sbs://fur_2.sbs:fur_2': 'sbs://noise_fur_2.sbs:fur_2',
    'sbs://fur_3.sbs:fur_3': 'sbs://noise_fur_3.sbs:fur_3',
    'sbs://gaussian_1.sbs:gaussian_1': 'sbs://pattern_gaussian_1.sbs:gaussian_1',
    'sbs://gaussian_2.sbs:gaussian_2': 'sbs://pattern_gaussian_2.sbs:gaussian_2',
    'sbs://gaussian_spots_1.sbs:gaussian_spots_1': 'sbs://noise_gaussian_spots_1.sbs:gaussian_spots_1',
    'sbs://gaussian_spots_2.sbs:gaussian_spots_2': 'sbs://noise_gaussian_spots_2.sbs:gaussian_spots_2',
    'sbs://grunge_map_001.sbs:grunge_map_001': 'sbs://noise_grunge_map_001.sbs:grunge_map_001',
    'sbs://grunge_map_002.sbs:grunge_map_002': 'sbs://noise_grunge_map_002.sbs:grunge_map_002',
    'sbs://grunge_map_004.sbs:grunge_map_004': 'sbs://noise_grunge_map_004.sbs:grunge_map_004',
    'sbs://grunge_map_005.sbs:grunge_map_005': 'sbs://noise_grunge_map_005.sbs:grunge_map_005',
    'sbs://grunge_map_006.sbs:grunge_map_006': 'sbs://noise_grunge_map_006.sbs:grunge_map_006',
    'sbs://grunge_map_007.sbs:grunge_map_007': 'sbs://noise_grunge_map_007.sbs:grunge_map_007',
    'sbs://grunge_map_008.sbs:grunge_map_008': 'sbs://noise_grunge_map_008.sbs:grunge_map_008',
    'sbs://grunge_map_009.sbs:grunge_map_009': 'sbs://noise_grunge_map_009.sbs:grunge_map_009',
    'sbs://grunge_map_010.sbs:grunge_map_010': 'sbs://noise_grunge_map_010.sbs:grunge_map_010',
    'sbs://grunge_map_011.sbs:grunge_map_011': 'sbs://noise_grunge_map_011.sbs:grunge_map_011',
    'sbs://grunge_map_013.sbs:grunge_map_013': 'sbs://noise_grunge_map_013.sbs:grunge_map_013',
    'sbs://grunge_map_014.sbs:grunge_map_014': 'sbs://noise_grunge_map_014.sbs:grunge_map_014',
    'sbs://grunge_map_015.sbs:grunge_map_015': 'sbs://noise_grunge_map_015.sbs:grunge_map_015',
    'sbs://liquid.sbs:liquid': 'sbs://noise_liquid.sbs:liquid',
    'sbs://mesh_1.sbs:mesh_1': 'sbs://pattern_mesh_1.sbs:mesh_1',
    'sbs://mesh_2.sbs:mesh_2': 'sbs://pattern_mesh_2.sbs:mesh_2',
    'sbs://messy_fibers_1.sbs:messy_fibers_1': 'sbs://noise_messy_fibers_1.sbs:messy_fibers_1',
    'sbs://messy_fibers_2.sbs:messy_fibers_2': 'sbs://noise_messy_fibers_2.sbs:messy_fibers_2',
    'sbs://messy_fibers_3.sbs:messy_fibers_3': 'sbs://noise_messy_fibers_3.sbs:messy_fibers_3',
    'sbs://microscope_view.sbs:microscope_view': 'sbs://noise_microscope_view.sbs:microscope_view',
    'sbs://moisture_noise.sbs:moisture_noise': 'sbs://noise_moisture_noise.sbs:moisture_noise',
    'sbs://perlin_noise_1.sbs:perlin_noise_1': 'sbs://noise_perlin_noise.sbs:perlin_noise',
    'sbs://perlin_noise_2.sbs:perlin_noise_2': 'sbs://noise_perlin_noise.sbs:perlin_noise',
    'sbs://plasma.sbs:plasma': 'sbs://noise_plasma.sbs:plasma',
    'sbs://polygon_1.sbs:polygon_1': 'sbs://pattern_polygon_1.sbs:polygon_1',
    'sbs://polygon_2.sbs:polygon_2': 'sbs://pattern_polygon_2.sbs:polygon_2',
    'sbs://scratches_generator.sbs:scratches_generator': 'sbs://pattern_scratches_generator.sbs:scratches_generator',
    'sbs://shape.sbs:shape': 'sbs://pattern_shape.sbs:shape',
    'sbs://splatter_circular.sbs:splatter_circular': 'sbs://pattern_splatter_circular.sbs:splatter_circular',
    'sbs://stripes.sbs:stripes': 'sbs://pattern_stripes.sbs:stripes',
    'sbs://tile_generator.sbs:tile_generator': 'sbs://pattern_tile_generator.sbs:tile_generator',
    'sbs://tile_random.sbs:tile_random': 'sbs://pattern_tile_random.sbs:tile_random',
    'sbs://tile_sampler.sbs:tile_sampler': 'sbs://pattern_tile_sampler.sbs:tile_sampler',
    'sbs://weave_1.sbs:weave_1': 'sbs://pattern_weave_1.sbs:weave_1',
    'sbs://weave_2.sbs:weave_2': 'sbs://pattern_weave_2.sbs:weave_2',
    'sbs://weave_generator.sbs:weave_generator': 'sbs://pattern_weave_generator.sbs:weave_generator',
}

legacy_generator_output_name_map = {
    'sbs://alveolus.sbs:alveolus:Alveolus': 'output',
    'sbs://anisotropic_noise.sbs:anisotropic_noise:Anisotropic_Noise': 'Anisotropic_Noise',
    'sbs://bnw_spots_1.sbs:bnw_spots_1:BnW_Spots': 'output',
    'sbs://bnw_spots_2.sbs:bnw_spots_2:BnW_Spots_2': 'output',
    'sbs://bnw_spots_3.sbs:bnw_spots_3:BnW_Spots_3': 'output',
    'sbs://brick_1.sbs:brick_1:Brick': 'output',
    'sbs://brick_2.sbs:brick_2:Brick_2': 'output',
    'sbs://brick_generator.sbs:brick_generator:Bricks_Generator': 'Bricks_Generator',
    'sbs://cells_1.sbs:cells_1:Cells': 'output',
    'sbs://cells_2.sbs:cells_2:Cells_2': 'output',
    'sbs://cells_3.sbs:cells_3:Cells_3': 'output',
    'sbs://cells_4.sbs:cells_4:Cells_4': 'output',
    'sbs://checker_1.sbs:checker_1:Checker': 'output',
    'sbs://clouds_1.sbs:clouds_1:Clouds': 'output',
    'sbs://clouds_2.sbs:clouds_2:Clouds_2': 'output',
    'sbs://clouds_3.sbs:clouds_3:Clouds_3': 'output',
    'sbs://creased.sbs:creased:Creased': 'output',
    'sbs://crystal_1.sbs:crystal_1:Crystal': 'output',
    'sbs://crystal_2.sbs:crystal_2:Crystal_2': 'output',
    'sbs://dirt_1.sbs:dirt_1:Dirt': 'output',
    'sbs://dirt_2.sbs:dirt_2:Dirt_2': 'output',
    'sbs://dirt_3.sbs:dirt_3:Dirt_3': 'output',
    'sbs://dirt_4.sbs:dirt_4:Dirt_4': 'output',
    'sbs://dirt_5.sbs:dirt_5:Dirt_5': 'output',
    'sbs://dirt_gradient.sbs:dirt_gradient:Dirt_Gradient': 'output',
    'sbs://fibers_1.sbs:fibers_1:Rope': 'output',
    'sbs://fibers_2.sbs:fibers_2:Fiber': 'output',
    'sbs://fluid.sbs:fluid:Fluid': 'output',
    'sbs://fractal_sum_1.sbs:fractal_sum_1:Fractal_Sum': 'output',
    'sbs://fractal_sum_2.sbs:fractal_sum_2:Fractal_Sum_2': 'output',
    'sbs://fractal_sum_3.sbs:fractal_sum_3:Fractal_Sum_3': 'output',
    'sbs://fractal_sum_4.sbs:fractal_sum_4:Fractal_Sum_4': 'output',
    'sbs://fractal_sum_base.sbs:fractal_sum_base:Fractal_Sum_Base': 'output',
    'sbs://fur_1.sbs:fur_1:Fur': 'output',
    'sbs://fur_2.sbs:fur_2:Fur_2': 'output',
    'sbs://fur_3.sbs:fur_3:Fur_3': 'output',
    'sbs://gaussian_1.sbs:gaussian_1:Gaussian_Spots': 'output',
    'sbs://gaussian_2.sbs:gaussian_2:Gaussian_Spots_2': 'output',
    'sbs://gaussian_spots_1.sbs:gaussian_spots_1:Few_Gauss': 'output',
    'sbs://gaussian_spots_2.sbs:gaussian_spots_2:Gauss_Spots': 'output',
    'sbs://grunge_map_001.sbs:grunge_map_001:output': 'output',
    'sbs://grunge_map_002.sbs:grunge_map_002:output': 'output',
    'sbs://grunge_map_004.sbs:grunge_map_004:output': 'output',
    'sbs://grunge_map_005.sbs:grunge_map_005:output': 'output',
    'sbs://grunge_map_006.sbs:grunge_map_006:output': 'output',
    'sbs://grunge_map_007.sbs:grunge_map_007:output': 'output',
    'sbs://grunge_map_008.sbs:grunge_map_008:output': 'output',
    'sbs://grunge_map_009.sbs:grunge_map_009:output': 'output',
    'sbs://grunge_map_010.sbs:grunge_map_010:output': 'output',
    'sbs://grunge_map_011.sbs:grunge_map_011:output': 'output',
    'sbs://grunge_map_013.sbs:grunge_map_013:output': 'output',
    'sbs://grunge_map_014.sbs:grunge_map_014:output': 'output',
    'sbs://grunge_map_015.sbs:grunge_map_015:output': 'output',
    'sbs://liquid.sbs:liquid:Liquid_Thing': 'output',
    'sbs://mesh_1.sbs:mesh_1:Mesh': 'output',
    'sbs://mesh_2.sbs:mesh_2:Mesh_2': 'output',
    'sbs://messy_fibers_1.sbs:messy_fibers_1:Messy_Fibers': 'output',
    'sbs://messy_fibers_2.sbs:messy_fibers_2:Lines': 'output',
    'sbs://messy_fibers_3.sbs:messy_fibers_3:Spike_Fibers': 'output',
    'sbs://microscope_view.sbs:microscope_view:Microscope_View': 'output',
    'sbs://moisture_noise.sbs:moisture_noise:Moisture_Noise': 'output',
    'sbs://perlin_noise_1.sbs:perlin_noise_1:Noise': 'output',
    'sbs://perlin_noise_2.sbs:perlin_noise_2:Noise_2': 'output',
    'sbs://plasma.sbs:plasma:Plasma': 'output',
    'sbs://polygon_1.sbs:polygon_1:Polygon': 'output',
    'sbs://polygon_2.sbs:polygon_2:Polygon_2': 'output',
    'sbs://scratches_generator.sbs:scratches_generator:output': 'output',
    'sbs://shape.sbs:shape:Simple_Shape': 'output',
    'sbs://splatter_circular.sbs:splatter_circular:Splatter_Circular': 'Splatter_Circular',
    'sbs://stripes.sbs:stripes:Stripes': 'Stripes',
    'sbs://tile_generator.sbs:tile_generator:TileGenerator': 'output',
    'sbs://tile_random.sbs:tile_random:output': 'output',
    'sbs://tile_sampler.sbs:tile_sampler:Output': 'output',
    'sbs://weave_1.sbs:weave_1:Weave': 'output',
    'sbs://weave_2.sbs:weave_2:Weave_2': 'output',
}

# not legacy, but multiple different definitions (usually defined in multiple different custom sbs files that are not part of the standard SAT packages):
# marble_generator
# mstme0101
# mstme0102
# mstme0102_002
# mstme0103
# skin_generator

def param_type_is_image(dtype):
    return dtype in [
        SBSParamType.ENTRY_COLOR.value,
        SBSParamType.ENTRY_GRAYSCALE.value,
        SBSParamType.ENTRY_VARIANT.value,
        SBSParamType.ENTRY_COLOR_OPT.value,
        SBSParamType.ENTRY_GRAYSCALE_OPT.value,
        SBSParamType.ENTRY_VARIANT_OPT.value]

def load_sbs_graph(graph_name, filename, resource_dirs, use_alpha=False, res=None, use_abs_paths=True, explicit_default_params=False, skip_unsupported_params=False):

    graph, graph_xml, graph_inputs_by_uid, graph_outputs_by_uid, dependencies_by_uid = load_sbs_graph_signature(
        graph_name=graph_name, filename=filename, use_alpha=use_alpha, res=res, use_abs_paths=use_abs_paths)

    # parse graph nodes
    nodes_by_uid = {}
    node_outputs_by_uid = {}
    node_def_graph_signatures = {}
    for node_xml in graph_xml.iter('compNode'):
        node_uid = int(load_sbs_attribute_value(node_xml.find('uid')))
        node_imp_xml = node_xml.find('compImplementation')[0]

        # input node
        if node_imp_xml.tag == 'compInputBridge':
            input_uid = int(load_sbs_attribute_value(node_imp_xml.find('entry')))
            node_name = graph.gen_unique_node_name('Input')
            node = sbs_graph_nodes.SBSInputNode(name=node_name, graph_input=graph_inputs_by_uid[input_uid])

        # output node
        elif node_imp_xml.tag == 'compOutputBridge':
            output_uid = int(load_sbs_attribute_value(node_imp_xml.find('output')))
            node_name = graph.gen_unique_node_name('Output')
            node = sbs_graph_nodes.SBSOutputNode(name=node_name, graph_output=graph_outputs_by_uid[output_uid])

        # non-atomic node
        elif node_imp_xml.tag == 'compInstance':
            path_ = node_imp_xml.find('path')
            path = load_sbs_attribute_value(path_)
            if path is None:
                path = load_sbs_attribute_value(path_.find('value'))
            if not path.startswith('pkg:///'):
                raise RuntimeError(f'Unexpected dependency format, expected the follwing string to start with "pkg:///":\n{path}')
            instance_type = path[path.rfind('/')+1:path.rfind('?')]
            node_def_graph_name = path[len('pkg:///'):path.rfind('?')]
            dep_uid = int(path[path.rfind('?dependency=') + len('?dependency='):])
            if instance_type not in type_dict:
                # unsupported node
                node_name = graph.gen_unique_node_name('Unsupported')
                node = sbs_graph_nodes.SBSUnsupportedNode(
                    name=node_name, node_func=instance_type,
                    output_res=graph.res, use_alpha=graph.use_alpha,
                    definition=sbs_graph_nodes.SBSNodeDefinition(graph=node_def_graph_name, path=dependencies_by_uid[dep_uid]))
            else:
                # supported node
                node_class, node_type, _ = lookup_node_type(instance_type)
                node_name = graph.gen_unique_node_name(node_type)

                node = node_class(name=node_name, output_res=graph.res, use_alpha=graph.use_alpha)

        # atomic node
        elif node_imp_xml.tag == 'compFilter':
            filter_type = load_sbs_attribute_value(node_imp_xml.find('filter'))
            if filter_type not in type_dict: # and not node_xml.find('connections'):
                # unsupported node
                node_name = graph.gen_unique_node_name('Unsupported')
                node = sbs_graph_nodes.SBSUnsupportedNode(
                    name=node_name, node_func=filter_type,
                    output_res=graph.res, use_alpha=graph.use_alpha)
            else:
                # supported node
                node_class, node_type, _ = lookup_node_type(filter_type)
                node_name = graph.gen_unique_node_name(node_type)
                node = node_class(name=node_name, output_res=graph.res, use_alpha=graph.use_alpha)
        else:
            raise NotImplementedError(f'Unrecognized node type: {node_imp_xml.tag}')

        # add layout position of the node as user-defined parameter
        node_pos_xml = node_xml.find('GUILayout/gpos')
        if node_pos_xml is not None:
            node_pos = [float(i) for i in load_sbs_attribute_value(node_pos_xml).strip().split()]
            node.user_data['node_pos'] = node_pos

        # parse outputs of the node
        node_outputs_by_uid[node_uid] = {}
        if node.type != 'Output':

            node_output_map = {}
            for node_output in node.outputs:
                if isinstance(node_output.name_xml, list):
                    for output_name_xml in node_output.name_xml:
                        node_output_map[output_name_xml] = node_output
                else:
                    node_output_map[node_output.name_xml] = node_output

            node_outputs_xml = node_xml.findall('compOutputs/compOutput')
            node_output_bridges_xml = node_imp_xml.findall('outputBridgings/outputBridging')
            node_output_names_xml = {}
            if node_output_bridges_xml:
                for node_output_bridge_xml in node_output_bridges_xml:
                    node_output_names_xml[int(load_sbs_attribute_value(node_output_bridge_xml.find('uid')))] = load_sbs_attribute_value(node_output_bridge_xml.find('identifier'))
            for node_output_xml in node_outputs_xml:
                node_output_uid = int(load_sbs_attribute_value(node_output_xml.find('uid')))
                node_output_type = int(load_sbs_attribute_value(node_output_xml.find('comptype')))
                node_output_name_xml = node_output_names_xml[node_output_uid] if node_output_uid in node_output_names_xml else ''
                if node.type == 'Unsupported':
                    node_output = sbs_graph_nodes.SBSNodeOutput(name=node_output_name_xml, dtype=node_output_type, name_xml=node_output_name_xml)
                    node.add_output(node_output)
                else:
                    if node_output_name_xml not in node_output_map:
                        raise NotImplementedError(f'Node output {node_output_name_xml} is not supported for node type {node.type}.')
                    node_output = node_output_map[node_output_name_xml]
                    # TODO: check compatibility of the output type given in the xml with the pre-defined node output type
                    node_output.type = node_output_type
                node_outputs_by_uid[node_uid][node_output_uid] = node_output

        # parse parameters of the node
        if node.type != 'Passthrough':
            add_new_node_params = False

            # node parameters for unsupported node types must be created on the fly
            if node.type == 'Unsupported':
                # create the node parameters of an unsupported node from its definition graph
                if node.definition() is not None and node.definition().path.endswith('.sbs'):
                    def_graph_id = node.definition().uname()

                    # lazily load definition graph
                    if def_graph_id not in node_def_graph_signatures:
                        node_def_graph_signature, _, _, _, _ = load_sbs_graph_signature(
                            graph_name=node.definition().graph,
                            filename=resolve_dependency_path(path=node.definition().path, source_filename=filename, resource_dirs=resource_dirs),
                            use_alpha=use_alpha, res=res, use_abs_paths=use_abs_paths)
                        node_def_graph_signatures[def_graph_id] = node_def_graph_signature

                    node_def_graph_signature = node_def_graph_signatures[def_graph_id]
                    param_names = set(p.name for p in node.params)
                    for def_param in node_def_graph_signature.params:
                        if def_param.name not in param_names:
                            kwargs = {
                                'name': def_param.name,
                                'val': def_param.val,
                                'dtype': def_param.dtype,
                                'clamped': def_param.clamped,
                                'name_xml': def_param.name,
                            }
                            if isinstance(def_param.val_range, tuple):
                                val_min, val_max = def_param.val_range
                                kwargs.update({'val_min': val_min, 'val_max': val_max})
                            elif isinstance(def_param.val_range, list):
                                kwargs['val_range'] = def_param.val_range
                            node.add_param(sbs_graph_nodes.SBSNodeParameter(**kwargs))
                            param_names.add(def_param.name)

                # node definition cannot be read as a graph
                else:
                    add_new_node_params = True

            # load source node parameter values
            load_sbs_node_params(node_imp_xml=node_imp_xml, node=node, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths,
                                 skip_unsupported_params=skip_unsupported_params, add_new_node_params=add_new_node_params)

        # add input slots to unsupported nodes
        if node.type == 'Unsupported':
            if node.definition() is not None and node.definition().path.endswith('.sbs'):
                def_graph_id = node.definition().uname()

                # lazily load definition graph
                if def_graph_id not in node_def_graph_signatures:
                    node_def_graph_signature, _, _, _, _ = load_sbs_graph_signature(
                        graph_name=node.definition().graph,
                        filename=resolve_dependency_path(path=node.definition().path, source_filename=filename, resource_dirs=resource_dirs),
                        use_alpha=use_alpha, res=res, use_abs_paths=use_abs_paths)
                    node_def_graph_signatures[def_graph_id] = node_def_graph_signature

                node_def_graph_signature = node_def_graph_signatures[def_graph_id]
                for graph_input in node_def_graph_signature.inputs:
                    node.add_input(sbs_graph_nodes.SBSNodeInput(name=graph_input.name, dtype=graph_input.dtype, name_xml=graph_input.name))

        # add the node top the graph
        graph.add_node(node)
        nodes_by_uid[node_uid] = node

    # create graph edges (read input connections and update node outputs)
    for node_xml in graph_xml.iter('compNode'):
        uid = int(load_sbs_attribute_value(node_xml.find('uid')))
        node = nodes_by_uid[uid] # all nodes should have been parsed at this point already

        node_input_map = {}
        for node_input in node.inputs:
            if isinstance(node_input.name_xml, list):
                for node_input_name_xml in node_input.name_xml:
                    node_input_map[node_input_name_xml] = node_input
            else:
                node_input_map[node_input.name_xml] = node_input

        # iterate over input connections
        for conn_xml in node_xml.findall('connections/connection') + node_xml.findall('connexions/connexion'): # for some reason both of these labels are used
            conn_name_xml = load_sbs_attribute_value(conn_xml.find('identifier'))
            conn_ref = int(load_sbs_attribute_value(conn_xml.find('connRef')))
            conn_output_ref_xml = conn_xml.find('connRefOutput')
            conn_output_ref_uid = int(load_sbs_attribute_value(conn_xml.find('connRefOutput'))) if conn_output_ref_xml is not None else None

            if conn_name_xml in node_input_map:
                node_input = node_input_map[conn_name_xml]
            elif node.type == 'Unsupported' and (node.definition() is None or not node.definition().path.endswith('.sbs')):
                node_input = sbs_graph_nodes.SBSNodeInput(name=conn_name_xml, dtype=None, name_xml=conn_name_xml)
                node.add_input(node_input)
            else:
                raise NotImplementedError(f'Node input {conn_name_xml} is not supported for node type {node.func}.')

            # get the parent node and the parent node output
            # parent_node = graph.get_node_by_uid(conn_ref, error_on_miss=True)
            parent_node = nodes_by_uid[conn_ref]
            if conn_output_ref_uid is None:
                # no output reference was given, this means the connection references the single output of the node
                if len(parent_node.outputs) != 1:
                    raise RuntimeError('Found an ambiguous connection. The connection points to a node with multiple outputs and does not have an output reference.')
                parent_node_output = parent_node.outputs[0]
            else:
                # parent_node_output_uids = [o.uid for o in parent_node.outputs]
                # if conn_output_ref_uid not in parent_node_output_uids:
                #     raise RuntimeError(f'Found a dangling connection, an output with uid {conn_output_ref_uid} is missing in the parent node of the connection.')
                # parent_node_output = parent_node.outputs[parent_node_output_uids.index(conn_output_ref_uid)]
                if conn_output_ref_uid not in node_outputs_by_uid[conn_ref]:
                    raise RuntimeError(f'Found a dangling connection, an output with uid {conn_output_ref_uid} is missing in the parent node of the connection.')
                parent_node_output = node_outputs_by_uid[conn_ref][conn_output_ref_uid]

            node_input.connect(parent_output=parent_node_output)

            # The data type of the node input slot is unknown for unsupported atomic nodes.
            # (This information is not given in the .sbs file.)
            # In this case, set the data type of the input slot to match the output slot it is connected to.
            if node_input.dtype is None:
                node_input.dtype = parent_node_output.dtype

    return graph

def load_sbs_param_preset(xml):
    # parse input parameters and build the dictionary for those with default values
    input_name = load_sbs_attribute_value(xml.find('identifier'))
    input_uid = int(load_sbs_attribute_value(xml.find('uid')))
    input_dtype = load_sbs_attribute_value(xml.find('type'))
    if input_dtype is None:
        input_dtype = load_sbs_attribute_value(xml.find('type/value'))
    input_dtype = int(input_dtype)
    if input_dtype in (SBSParamType.ENTRY_COLOR.value, SBSParamType.ENTRY_GRAYSCALE.value):
        input_val = None
    else:
        input_val, dtype = sbs_parameter.SBSParameter.load_sbs_value(val_xml=xml.find('paramValue')[0], dependencies_by_uid=None, use_abs_paths=None, allow_fgraph=False)
        if dtype != input_dtype:
            raise RuntimeError('Bad sbs file, parameter type does not match parameter tag.')

    return sbs_parameter.SBSParameter(name=input_name, param_dtype=input_dtype, param_val=input_val, group=None, val_range=None), input_uid

def load_sbs_graph_signature(graph_name, filename, use_alpha, res, use_abs_paths):
    # load inputs, outputs and meta-info of the graph, but not the nodes

    graph = sbs_graph.SBSGraph(graph_name=graph_name, use_alpha=use_alpha, res=res)

    doc_xml = ET.parse(filename)

    dependencies_by_uid = load_sbs_dependencies(doc_xml, source_filename=filename, use_abs_paths=use_abs_paths)

    # get the root xml nodes of all graphs in the sbs file
    graph_xmls = doc_xml.getroot().findall('content/graph')
    graph_names = [load_sbs_attribute_value(r.find('identifier')) for r in graph_xmls]
    group_xmls = deque(doc_xml.getroot().findall('content/group'))
    group_names = deque([load_sbs_attribute_value(r.find('identifier')) for r in group_xmls])
    while len(group_xmls) > 0:
        parent_group_name = group_names.popleft()
        parent_group_xml = group_xmls.popleft()

        child_graph_xmls = parent_group_xml.findall('content/graph')
        graph_xmls.extend(child_graph_xmls)
        graph_names.extend([f'{parent_group_name}/{load_sbs_attribute_value(r.find("identifier"))}' for r in child_graph_xmls])

        child_group_xmls = parent_group_xml.findall('content/group')
        group_xmls.extend(child_group_xmls)
        group_names.extend([f'{parent_group_name}/{load_sbs_attribute_value(r.find("identifier"))}' for r in child_group_xmls])

    # pick the root xml node of the graph with the given name
    if graph_name not in graph_names:
        raise RuntimeError(f'A graph with name {graph_name} was not found in the given file {filename}.')
    graph_xml = graph_xmls[graph_names.index(graph_name)]

    # parse graph inputs and parameters and build the dictionary for those with default values
    graph_inputs_by_uid = {}
    params_xml = graph_xml.find('paraminputs')
    if params_xml is not None:
        for param_xml in params_xml.iter('paraminput'):
            graph_input, input_uid = sbs_parameter.SBSParameter.load_sbs(xml=param_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            # the only difference between graph parameters and inputs seems to be that inputs have image data types
            # previously I thought that the presence of 'isConnectable' distinguishes them, but it seems this is not always the case
            # (it might only be an old version or some error where it is not the case, since when I re-save with substance designer, isConnectable was added)
            if param_type_is_image(dtype=graph_input.dtype):
                graph.add_input(graph_input)
            else:
                graph.add_param(graph_input)
            # if param_xml.find('isConnectable') is not None:
            #     graph.add_input(graph_input)
            # else:
            #     graph.add_param(graph_input)
            graph_inputs_by_uid[input_uid] = graph_input

    # parse parameter presets
    presets_xml = graph_xml.find('sbspresets')
    if presets_xml is not None:
        for preset_xml in presets_xml.iter('sbspreset'):
            preset_name = load_sbs_attribute_value(preset_xml.find('label'))
            param_preset = sbs_graph.SBSGraphParameterPreset(name=preset_name)
            preset_inputs_xml = preset_xml.find('presetinputs')
            if preset_inputs_xml is not None:
                for preset_input_xml in preset_inputs_xml.iter('presetinput'):
                    preset_param, _ = load_sbs_param_preset(xml=preset_input_xml)
                    if isinstance(preset_param.val , sbs_function_graph.SBSFunctionGraph):
                        raise RuntimeError('Parameter presets may not be defined as function graphs.')
                    param_preset.add_param(param=preset_param)
            graph.add_param_preset(param_preset=param_preset)

    # get global resolution
    for param_ in graph_xml.find('baseParameters').iter('parameter'):
        if load_sbs_attribute_value(param_.find('name')) == 'outputsize':
            _, param_rel_to, param_val, param_dtype = load_sbs_node_param(param_, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            if param_dtype != SBSParamType.INTEGER2.value:
                raise RuntimeError('Data type for outputsize is not INTEGER2.')
            if isinstance(param_val, sbs_function_graph.SBSFunctionGraph):
                input_dict = {p.name: p.val for p in graph.params}
                input_dict.update({
                    '$size': [1 << graph.res[0], 1 << graph.res[1]],
                    '$sizelog2': graph.res,
                    '$normalformat': 0,
                    })
                param_val = param_val.eval(input_dict=input_dict)
            else:
                param_val = param_val
            graph.res = [graph.res[0] + param_val[0], graph.res[1] + param_val[1]] if param_rel_to else param_val

    # parse graph outputs
    graph_outputs_by_uid = {}
    for output_ in graph_xml.iter('graphoutput'):
        output_name = load_sbs_attribute_value(output_.find('identifier'))
        output_uid = int(load_sbs_attribute_value(output_.find('uid')))
        output_group_ = output_.find('group')
        output_group = load_sbs_attribute_value(output_group_) if output_group_ is not None else ''
        output_usage_ = output_.find('usages/usage')
        output_usage = load_sbs_attribute_value(output_usage_.find('name')) if output_usage_ else ''
        graph_output = sbs_graph.SBSGraphOutput(name=output_name, usage=output_usage, group=output_group)
        graph.add_output(graph_output)
        graph_outputs_by_uid[output_uid] = graph_output

    return graph, graph_xml, graph_inputs_by_uid, graph_outputs_by_uid, dependencies_by_uid

def load_sbs_dependencies(doc_xml, source_filename, use_abs_paths=True):
    dependencies_by_uid = {}
    for dependency_xml in doc_xml.getroot().find('dependencies').iter('dependency'):
        dep_uid = int(load_sbs_attribute_value(dependency_xml.find('uid')))
        dep_path = load_sbs_attribute_value(dependency_xml.find('filename'))
        if use_abs_paths:
            # make relative dependency paths absolute, to make them independent of the filename
            # also resolve symlinks since the sbs renderer/cooker seem to have difficulties with symlinks
            if dep_path is not None and '://' not in dep_path:
                dep_path = os.path.realpath(os.path.abspath(resolve_dependency_path(path=dep_path, source_filename=source_filename, resource_dirs={})))
        dependencies_by_uid[dep_uid] = dep_path

    return dependencies_by_uid

def load_sbs_node_params(node_imp_xml, node, dependencies_by_uid, use_abs_paths,
                         skip_unsupported_params=False, add_new_node_params=False):
    '''
    Analyze manually specified parameters and update their values in SBS nodes accordingly.
    '''

    param_names_xml = [p.name_xml for p in node.params]
    if len(param_names_xml) > 0:
        param_names_xml = np.hstack(param_names_xml).tolist()

    # parse parameters
    params_xml = node_imp_xml.find('parameters')
    if params_xml is not None:
        for param_xml in params_xml.iter('parameter'):
            param_name_xml, param_rel_to, param_val, param_dtype = load_sbs_node_param(param_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            if node.type != 'Unsupported' and param_name_xml not in param_names_xml:
                if skip_unsupported_params:
                    continue
                else:
                    raise NotImplementedError(f'A parameter with name {param_name_xml} is not supported in a node of type {node.type}.')
            if param_val is not None:
                source_param = sbs_graph_nodes.SBSNodeSourceParameter(name=param_name_xml, val=param_val, relative_to=param_rel_to, dtype=param_dtype)
                if node.type == 'Unsupported':
                    node.add_source_param(source_param, add_new_node_params=add_new_node_params)
                else:
                    node.add_source_param(source_param)

    # parse parameter arrays
    param_arrays_ = node_imp_xml.find('paramsArrays')
    if param_arrays_ is not None:
        for param_array_ in param_arrays_.iter('paramsArray'):
            param_array_name = load_sbs_attribute_value(param_array_.find('name'))

            if node.type != 'Unsupported' and param_array_name not in param_names_xml:
                if skip_unsupported_params:
                    continue
                else:
                    raise NotImplementedError(f'A parameter with name {param_array_name} is not supported in a node of type {node.type}.')

            # Extract parameter cells
            param_array = []
            param_array_dtypes = []
            for param_array_cell_ in param_array_.iter('paramsArrayCell'):
                param_cell = {}
                param_cell_dtypes = {}
                for param_xml in param_array_cell_.iter('parameter'):
                    param_name_xml, _, param_val, param_dtype = load_sbs_node_param(param_xml, dependencies_by_uid, use_abs_paths=use_abs_paths)
                    param_cell[param_name_xml] = param_val
                    param_cell_dtypes[param_name_xml] = param_dtype
                param_array.append(param_cell)
                param_array_dtypes.append(param_cell_dtypes)

            if len(param_array):
                source_param = sbs_graph_nodes.SBSNodeSourceParameter(name=param_array_name, val=param_array, relative_to=None, dtype=param_array_dtypes)
                if node.type == 'Unsupported':
                    node.add_source_param(source_param, add_new_node_params=add_new_node_params)
                else:
                    node.add_source_param(source_param)

def load_sbs_node_param(param_xml, dependencies_by_uid, use_abs_paths):
    '''
    Parse a single parameter in SBS.
    '''
    param_name = load_sbs_attribute_value(param_xml.find('name'))
    param_rel_to_xml = param_xml.find('relativeTo')
    param_rel_to = int(load_sbs_attribute_value(param_rel_to_xml)) if param_rel_to_xml is not None else 1 # default is relative to parent
    param_val_xml = param_xml.find('paramValue')[0]
    param_val, param_dtype = sbs_parameter.SBSParameter.load_sbs_value(val_xml=param_val_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
    return param_name, param_rel_to, param_val, param_dtype

def lookup_node_type(xml_type):
    '''
    Return the node information specified by its type.
    '''
    if xml_type not in type_dict:
        raise NotImplementedError(f'Node \'{xml_type}\' is not supported.')
    node_type, node_func = type_dict[xml_type]
    node_func = f'F.{node_func}'
    node_class = getattr(sbs_graph_nodes, f'SBS{node_type}Node')

    return node_class, node_type, node_func

def package_sbs_file_dependencies(input_graph_path, output_graph_path, package_dir, packaged_dependencies=None):
    if packaged_dependencies is None:
        packaged_dependencies = {}

    doc_xml = ET.parse(input_graph_path)
    dependencies_xml = doc_xml.getroot().find('dependencies')

    package_sbs_dependencies(
        dependencies_xml=dependencies_xml, input_graph_path=input_graph_path, output_graph_path=output_graph_path,
        package_dir=package_dir, packaged_dependencies=packaged_dependencies)

    # write graph with updated dependencies
    os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
    doc_xml.write(output_graph_path, encoding='utf-8', xml_declaration=True)

def package_sbs_dependencies(dependencies_xml, input_graph_path, output_graph_path, package_dir, packaged_dependencies=None):

    if packaged_dependencies is None:
        packaged_dependencies = {}

    for dependency_xml in dependencies_xml.iter('dependency'):
        src_path = load_sbs_attribute_value(dependency_xml.find('filename'))

        if src_path.startswith('/trainman-mount/trainman-k8s-storage-0ef661f6-a14e-46c8-9066-3f489d39f3c2'): # legacy
            src_path = src_path.replace('/trainman-mount/trainman-k8s-storage-0ef661f6-a14e-46c8-9066-3f489d39f3c2', '/mnt/session_space')

        if '://' not in src_path and src_path != '?himself':
            # dependency is to a file that is not in a known resource library

            src_path = os.path.join(os.path.dirname(input_graph_path), src_path) # dependency paths are absolute or relative to the path of the input graph

            if src_path in packaged_dependencies:
                # dependency has already been packaged
                packaged_path = packaged_dependencies[src_path]
            else:
                # dependency has not yet been packaged
                if not os.path.exists(src_path):
                    raise RuntimeError(f'Cannot package missing dependency:\n{src_path}')
                packaged_path = os.path.join(package_dir, os.path.basename(src_path))

                # rename the dependency file if a dependency file with the same target name (but from a different source path) has already been packaged
                packaged_path_candidate = packaged_path
                duplicate_index = 1
                while os.path.exists(packaged_path_candidate):
                    duplicate_index += 1
                    packaged_path_candidate = f'{os.path.splitext(packaged_path)[0]}_{duplicate_index}{os.path.splitext(packaged_path)[1]}'
                    if duplicate_index > 10000:
                        raise RuntimeError(f'Too many duplicates for dependency:\n{packaged_path}.')
                packaged_path = packaged_path_candidate

                # update the list of dependecies that have already been packaged
                packaged_dependencies[src_path] = packaged_path

                src_ext = os.path.splitext(src_path)[1]
                # recurse into dependencies of the dependency
                if src_ext == '.sbs':
                    package_sbs_file_dependencies(
                        input_graph_path=src_path,
                        output_graph_path=packaged_path,
                        package_dir=package_dir,
                        packaged_dependencies=packaged_dependencies)
                else:  # directly copy to target path
                    os.makedirs(os.path.dirname(packaged_path), exist_ok=True)
                    shutil.copyfile(src_path, packaged_path)

            # update the dependency path in the graph to point to the packaged location
            # the dependency path should be in Unix format, even on Windows
            dependency_path = os.path.relpath(packaged_path, start=os.path.dirname(output_graph_path))
            dependency_path = dependency_path.replace('\\', '/') if platform.system() == 'Windows' else dependency_path
            dependency_xml.find('filename').set('v', dependency_path)


def save_sbs_graph(graph, filename=None, resolve_resource_dirs=None, package_dependencies_dir=None, use_networkx=True):

    # root xml node and header
    package_xml = ET.Element('package')
    doc_xml = ET.ElementTree(package_xml)
    ET.SubElement(package_xml, 'identifier').set('v', 'Unsaved Package')
    ET.SubElement(package_xml, 'formatVersion').set('v', '1.1.0.202201')
    ET.SubElement(package_xml, 'updaterVersion').set('v', '1.1.0.202201')
    ET.SubElement(package_xml, 'fileUID').set('v', f'{{{uuid.uuid4().hex}}}')
    ET.SubElement(package_xml, 'versionUID').set('v', '0')

    graph.update_node_dtypes() # update input/output data types of all nodes by analyzing and solving their type signatures

    # dependencies
    deps = graph.dependencies()
    dep_paths = set(dep.path for dep in deps)
    dependency_uids = {dep_path: uid for dep_path, uid in zip(dep_paths, gen_unique_uids(count=len(dep_paths)))}
    dependencies_xml = ET.SubElement(package_xml, 'dependencies')
    # packaged_dependencies = {}
    for dep_path in dep_paths:
        dep_uid = dependency_uids[dep_path]
        if resolve_resource_dirs is not None:
            if '://' in dep_path:
                tokens = dep_path.split('://')
                if len(tokens) != 2:
                    raise RuntimeError(f'Cannot parse dependency path format:\n{dep_path}')
                prefix, _ = tokens[0], tokens[1]
                if prefix in resolve_resource_dirs:
                    dep_path = resolve_dependency_path(path=dep_path, source_filename=filename, resource_dirs=resolve_resource_dirs)

        # if package_dependencies_dir is not None and '://' not in dep_path:
        #     orig_path = resolve_dependency_path(path=dep_path, source_filename=filename, resource_dirs={})
        #     if orig_path in packaged_dependencies:
        #         dep_path = packaged_dependencies[orig_path]
        #     else:
        #         if not os.path.exists(orig_path):
        #             raise RuntimeError(f'Cannot package missing dependency:\n{orig_path}')
        #         dep_path = os.path.join(package_dependencies_dir, os.path.basename(orig_path)) # relative to sbs filename
        #         dep_path_candidate = dep_path
        #         duplicate_index = 1
        #         while os.path.exists(os.path.join(os.path.dirname(filename), dep_path_candidate)):
        #             duplicate_index += 1
        #             dep_path_candidate = f'{os.path.splitext(dep_path)[0]}_{duplicate_index}{os.path.splitext(dep_path)[1]}'
        #             if duplicate_index > 100000:
        #                 raise RuntimeError(f'Too many duplicates for dependency:\n{dep_path}.')
        #         dep_path = dep_path_candidate
        #         os.makedirs(os.path.dirname(os.path.join(os.path.dirname(filename), dep_path)), exist_ok=True)
        #         shutil.copyfile(src=orig_path, dst=os.path.join(os.path.dirname(filename), dep_path))
        #         packaged_dependencies[orig_path] = dep_path

        dependency_xml = ET.SubElement(dependencies_xml, 'dependency')
        ET.SubElement(dependency_xml, 'filename').set('v', dep_path)
        ET.SubElement(dependency_xml, 'uid').set('v', str(dep_uid))
        ET.SubElement(dependency_xml, 'type').set('v', 'package')
        ET.SubElement(dependency_xml, 'fileUID').set('v', '0')
        ET.SubElement(dependency_xml, 'versionUID').set('v', '0')

    # recursively package dependencies
    if package_dependencies_dir is not None:
        package_sbs_dependencies(
            dependencies_xml=dependencies_xml, input_graph_path=filename, output_graph_path=filename,
            package_dir=os.path.join(os.path.dirname(filename), package_dependencies_dir))

    content_xml = ET.SubElement(package_xml, 'content')
    if content_xml is None:
        raise RuntimeError('The document xml does not have a content element.')

    # save resources
    resource_uids = {resource.name: uid for resource, uid in zip(graph.resources, gen_unique_uids(count=len(graph.resources)))}
    for resource in graph.resources:
        resource_xml = ET.SubElement(content_xml, 'resource')
        ET.SubElement(resource_xml, 'identifier').set('v', resource.name)
        ET.SubElement(resource_xml, 'uid').set('v', str(resource_uids[resource.name]))
        ET.SubElement(resource_xml, 'type').set('v', resource.type)
        for param in resource.params:
            ET.SubElement(resource_xml, param.name).set('v', str(param.val))

    # graph identifier and uid
    graph_xml = ET.SubElement(content_xml, 'graph')
    ET.SubElement(graph_xml, 'identifier').set('v', graph.name)
    ET.SubElement(graph_xml, 'uid').set('v', str(1000000000)) # constant uid since we only write a single graph into the sbs file

    # graph attributes
    attributes_xml = ET.SubElement(graph_xml, 'attributes')
    # ET.SubElement(attributes_xml, 'category').set('v', '')
    ET.SubElement(attributes_xml, 'label').set('v', graph.name)
    # ET.SubElement(attributes_xml, 'author').set('v', '')
    # ET.SubElement(attributes_xml, 'tags').set('v', '')

    # graph inputs and parameters
    graph_input_uids = {}
    if len(graph.inputs) > 0:
        graph_input_uids = {graph_input.name: uid for graph_input, uid in zip(graph.inputs, gen_unique_uids(count=len(graph.inputs)))}
        graph_inputs_xml = ET.SubElement(graph_xml, 'paraminputs')
        for graph_input in graph.inputs:
            graph_input_xml = ET.SubElement(graph_inputs_xml, 'paraminput')
            ET.SubElement(graph_input_xml, 'identifier').set('v', graph_input.name)
            ET.SubElement(graph_input_xml, 'uid').set('v', str(graph_input_uids[graph_input.name]))

            graph_input_attributes_xml = ET.SubElement(graph_input_xml, 'attributes')
            ET.SubElement(graph_input_attributes_xml, 'label').set('v', graph_input.name)

            ET.SubElement(graph_input_xml, 'type').set('v', str(graph_input.dtype))

            if param_type_is_image(graph_input.dtype):
                ET.SubElement(graph_input_xml, 'isConnectable').set('v', str(1))
            else:
                graph_input_val_xml = ET.SubElement(graph_input_xml, 'defaultValue')
                save_sbs_constant_value(parent_xml=graph_input_val_xml, val=graph_input.val, dtype=graph_input.dtype)

            graph_input_widget_xml = ET.SubElement(graph_input_xml, 'defaultWidget')
            ET.SubElement(graph_input_widget_xml, 'name').set('v', '')
            ET.SubElement(graph_input_widget_xml, 'options')

    # graph outputs
    graph_output_uids = {graph_output.name: uid for graph_output, uid in zip(graph.outputs, gen_unique_uids(count=len(graph.outputs)))}
    graph_outputs_xml = ET.SubElement(graph_xml, 'graphOutputs')
    for graph_output in graph.outputs:
        graph_output_xml = ET.SubElement(graph_outputs_xml, 'graphoutput')
        ET.SubElement(graph_output_xml, 'identifier').set('v', graph_output.name)
        ET.SubElement(graph_output_xml, 'uid').set('v', str(graph_output_uids[graph_output.name]))

        graph_output_attributes_xml = ET.SubElement(graph_output_xml, 'attributes')
        ET.SubElement(graph_output_attributes_xml, 'label').set('v', graph_output.name)

        if graph_output.usage is not None:
            graph_output_usages_xml = ET.SubElement(graph_output_xml, 'usages')
            graph_output_usage_xml = ET.SubElement(graph_output_usages_xml, 'usage')
            ET.SubElement(graph_output_usage_xml, 'components').set('v', 'RGBA') # hard-coded to RGBA
            ET.SubElement(graph_output_usage_xml, 'name').set('v', graph_output.usage)

        ET.SubElement(graph_output_xml, 'group').set('v', graph_output.group)
        # is visibleIf necessary?

    # nodes
    if use_networkx:
        nxgraph = graph.save_networkx()
        networkx_node_layout = nx.nx_agraph.graphviz_layout(nxgraph, prog='dot') # requires pygraphviz. To install: sudo apt-get install graphviz graphviz-dev; pip/conda install pygraphviz (may need conda-forge channel)
        for node_name, gpos in networkx_node_layout.items():
            networkx_node_layout[node_name] = [-gpos[1]*3, gpos[0]*3, 0]
    else:
        networkx_node_layout = None
    # node_layout = nx.kamada_kawai_layout(nxgraph) # requires pygraphviz. To install: sudo apt-get install graphviz graphviz-dev; pip/conda install pygraphviz (may need conda-forge channel)
    # for node_name, gpos in node_layout.items():
    #     node_layout[node_name] = [gpos[0]*1000, gpos[1]*1000, 0]
    # node_layout = nx.spring_layout(nxgraph) # requires pygraphviz. To install: sudo apt-get install graphviz graphviz-dev; pip/conda install pygraphviz (may need conda-forge channel)
    # for node_name, gpos in node_layout.items():
    #     node_layout[node_name] = [gpos[0]*1000, gpos[1]*1000, 0]
    existing_source_param_uids = []
    guiobj_uids = []
    node_uids = {node.name: uid for node, uid in zip(graph.nodes, gen_unique_uids(count=len(graph.nodes)))}
    node_outputs = [node_output for node in graph.nodes for node_output in node.outputs]
    node_output_uids = {node_output.uname(): uid for node_output, uid in zip(node_outputs, gen_unique_uids(count=len(node_outputs)))}
    # source_param_uids = {}
    nodes_xml = ET.SubElement(graph_xml, 'compNodes')
    guiobjs_xml = ET.SubElement(graph_xml, 'GUIObjects')
    # gpos = [0, 0, 0]
    for node in graph.nodes:
        node_xml, comment_xml = save_sbs_node(
            node=node,
            node_uids = node_uids,
            node_output_uids = node_output_uids,
            graph_input_uids = graph_input_uids,
            graph_output_uids = graph_output_uids,
            dependency_uids = dependency_uids,
            existing_source_param_uids=existing_source_param_uids,
            guiobj_uids = guiobj_uids,
            graph_res = graph.res,
            # node_signature=node_signatures[node.name] if node.name in node_signatures else None,
            gpos=networkx_node_layout[node.name] if networkx_node_layout is not None else None,
        )
        nodes_xml.append(node_xml)
        guiobjs_xml.append(comment_xml)
        # gpos = [gpos[0] + 300, gpos[1], gpos[2]]

    # base parameters of the graph, put the resolution here
    base_params_xml = ET.SubElement(graph_xml, 'baseParameters')
    base_param_xml = ET.SubElement(base_params_xml, 'parameter')
    ET.SubElement(base_param_xml, 'name').set('v', 'outputsize')
    ET.SubElement(base_param_xml, 'relativeTo').set('v', str(0))
    base_param_val_xml = ET.SubElement(base_param_xml, 'paramValue')
    save_sbs_constant_value(parent_xml=base_param_val_xml, val=graph.res, dtype=SBSParamType.INTEGER2.value)
    # ET.SubElement(ET.SubElement(base_param_xml, 'paramValue'), 'constantValueInt2').set('v', f'{graph.res[0]} {graph.res[1]}')

    # root and root outputs (?)
    root_xml = ET.SubElement(graph_xml, 'root')
    root_outputs_xml = ET.SubElement(root_xml, 'rootOutputs')
    for graph_output in graph.outputs:
        root_output_xml = ET.SubElement(root_outputs_xml, 'rootOutput')
        ET.SubElement(root_output_xml, 'output').set('v', str(graph_output_uids[graph_output.name]))
        ET.SubElement(root_output_xml, 'format').set('v', str(0))
        ET.SubElement(root_output_xml, 'usertag').set('v', '')

    if filename is not None:
        doc_xml.write(filename, encoding='utf-8', xml_declaration=True)

    return graph_xml, doc_xml

def save_sbs_node(node, node_uids, node_output_uids, graph_input_uids, graph_output_uids, dependency_uids, existing_source_param_uids, guiobj_uids, graph_res, gpos=None):
    node_xml = ET.Element('compNode')

    # comment with the node name (just for convenience)
    comment_xml = ET.Element('GUIObject')
    ET.SubElement(comment_xml, 'type').set('v', 'COMMENT')
    ET.SubElement(comment_xml, 'GUIDependency').set('v', f'NODE?{node_uids[node.name]}')
    comment_layout_xml = ET.SubElement(comment_xml, 'GUILayout')
    ET.SubElement(comment_layout_xml, 'gpos').set('v', ' '.join(str(p) for p in [-80, 55, 0]))
    ET.SubElement(comment_layout_xml, 'size').set('v', ' '.join(str(p) for p in [200, 40]))
    ET.SubElement(comment_xml, 'GUIName').set('v', node.name)
    comment_uid = gen_unique_uids(count=1, existing_uids=guiobj_uids)[0]
    guiobj_uids.append(comment_uid)
    ET.SubElement(comment_xml, 'uid').set('v', str(comment_uid))
    ET.SubElement(comment_xml, 'frameColor').set('v', ' '.join(str(p) for p in [0, 0, 0, 1]))

    # uid and giu layout
    node_pos = node.user_data['node_pos'] if 'node_pos' in node.user_data else gpos
    ET.SubElement(node_xml, 'uid').set('v', str(node_uids[node.name]))
    ET.SubElement(ET.SubElement(node_xml, 'GUILayout'), 'gpos').set('v', ' '.join(str(p) for p in ([0, 0, 0] if node_pos is None else node_pos)))

    # outputs are specified as follows in the sbs format:
    # atomic nodes -> only compOutputs->compOutput
    # non-atomic nodes -> compOutputs->compOutput + compImplementation->compInstance->outputBridgings->outputBridging
    # SBSOutputNode -> only compImplementation->compOutputBridge
    # SBSInputNode -> only compOutputs->compOutput

    # node outputs
    if node.type != 'Output':
        node_outputs_xml = ET.SubElement(node_xml, 'compOutputs')
        for node_output in node.outputs:
            node_output_xml = ET.SubElement(node_outputs_xml, 'compOutput')
            ET.SubElement(node_output_xml, 'uid').set('v', str(node_output_uids[node_output.uname()]))
            ET.SubElement(node_output_xml, 'comptype').set('v', str(node_output.dtype) if node_output.dtype is not None else str(SBSParamType.ENTRY_VARIANT.value))

    # node inputs
    if len(node.get_connected_inputs()) > 0:
        connections_xml = ET.SubElement(node_xml, 'connections')
        for node_input in node.get_connected_inputs():
            node_input_xml = ET.SubElement(connections_xml, 'connection')
            ET.SubElement(node_input_xml, 'identifier').set('v', node_input.name_xml[0] if isinstance(node_input.name_xml, list) else node_input.name_xml)
            ET.SubElement(node_input_xml, 'connRef').set('v', str(node_uids[node_input.parent.name]))
            ET.SubElement(node_input_xml, 'connRefOutput').set('v', str(node_output_uids[node_input.parent_output.uname()]))

    # implementation
    impl_xml = ET.SubElement(node_xml, 'compImplementation')
    if node.type == 'Input':
        filter_or_instance_xml = ET.SubElement(impl_xml, 'compInputBridge')
        ET.SubElement(filter_or_instance_xml, 'entry').set('v', str(graph_input_uids[node.graph_input.name]))
    elif node.type == 'Output':
        filter_or_instance_xml = ET.SubElement(impl_xml, 'compOutputBridge')
        ET.SubElement(filter_or_instance_xml, 'output').set('v', str(graph_output_uids[node.graph_output.name]))
    else:
        # node_type_xml = None
        # if node.type == 'Unsupported':
        #     node_type_xml = node.func
        # else:
        #     if node.definition() is None:

        #     else:
        #         node_type_xml = node.definition().graph

        #     node_type_xml = sorted([node_type_xml for node_type_xml, (node_type, node_func) in type_dict.items() if node_type == node.type])
        #     if len(node_type_xml) == 1:
        #         # the node type has exactly one corresponding sbs node type
        #         node_type_xml = node_type_xml[0]
        #     else:
        #         # the node type has multiple corresponding sbs node types, choose an sbs node type based on the data type of the node output
        #         # node_output_signature = node_signature[1]
        #         if len(node.outputs) != 1:
        #             raise RuntimeError('Cannot choose between grayscale and color nodes if the node does not have exactly one output.')
        #             # Would need to check which output determines if the node is the color or grayscale version
        #         # node_output_dtype = list(node_output_signature.values())[0]
        #         if len(node_type_xml) == 2 and node_type_xml[1].endswith('_grayscale') and node_type_xml[0] == node_type_xml[1][:-len('_grayscale')]:
        #             if node.outputs[0].dtype == SBSParamType.ENTRY_COLOR.value:
        #                 node_type_xml = node_type_xml[0]
        #             elif node.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
        #                 node_type_xml = node_type_xml[1]
        #             else:
        #                 node_type_xml = node_type_xml[0]
        #                 print(f'WARNING: Unexpected node output data type: {param_type_idx_to_name(node.outputs[0].dtype)}')
        #         elif node_type_xml == ['slope_blur', 'slope_blur_grayscale', 'slope_blur_grayscale_2']:
        #             if node.outputs[0].dtype == SBSParamType.ENTRY_COLOR.value:
        #                 node_type_xml = node_type_xml[0]
        #             elif node.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
        #                 node_type_xml = node_type_xml[2] # slope_blur_grayscale_2 seems to be the non-legacy version
        #             else:
        #                 node_type_xml = node_type_xml[0]
        #                 print(f'WARNING: Unexpected node output data type: {param_type_idx_to_name(node.outputs[0].dtype)}')
        #         else:
        #             raise RuntimeError(f'Unrecognized xml node type options {node_type_xml}')
        #         # if node.type in ['MakeItTilePatch', 'MakeItTilePhoto']:
        #         #     # special case: MakeItTilePatch/MakeItTilePhoto are defined in two different files, one for grayscale and one for color
        #         #     # pick the right definition file here, depending on the output type (color or grayscale)
        #         #     node.definition().path = f'sbs://{node_type_xml}.sbs'
        #         # if node.definition() is not None:
        #         #     node.definition().graph = node_type_xml

        # node definition
        filter_or_instance_xml = None
        if node.definition() is None:
            # atomic node
            if node.type == 'Unsupported':
                node_type_xml = node.func
            else:
                node_type_xml = sorted([node_type_xml for node_type_xml, (node_type, _) in type_dict.items() if node_type == node.type])
                if len(node_type_xml) != 1:
                    raise RuntimeError('Atomic nodes should have exactly one corresponding sbs node type.')
                node_type_xml = node_type_xml[0]

            filter_or_instance_xml = ET.SubElement(impl_xml, 'compFilter')
            ET.SubElement(filter_or_instance_xml, 'filter').set('v', node_type_xml)
        else:
            # non-atomic node
            filter_or_instance_xml = ET.SubElement(impl_xml, 'compInstance')
            ET.SubElement(filter_or_instance_xml, 'path').set('v', f'pkg:///{node.definition().graph}?dependency={dependency_uids[node.definition().path]}')

    # node parameters (not for output nodes)
    if node.type not in ['Output']:
        params_xml = ET.SubElement(filter_or_instance_xml, 'parameters')
        param_arrays_xml = None
        source_params = node.uncondition_params()
        for source_param in source_params:
            if source_param.dtype is None:
                raise RuntimeError(f'Parameter {source_param.name} of unsupported node {node.name} (type {node.func}) has no data type.')

            # special case: for some reason the three substance graphs that use st_sand as subgraph use an additional parameter 'Sand_Color'
            # for the st_sand node that is not defined as paramter in the subgraph. Not sure why this does not cause an error when loading the sbs file in Substance Designer.
            # TODO: in load_sbs, check that paramters that are defined for an unsupported non-atomic node actually exist in the node (using load_sbs_graph_signature)
            if node.func == 'st_sand' and source_param.name == 'Sand_Color':
                print(f'WARNING: skipping invalid parameter {source_param.name} of node with function {node.func}')
                continue

            if isinstance(source_param.dtype, list):
                # parameter array
                if len(source_param.val) != len(source_param.dtype):
                    raise RuntimeError('Size of parameter values and parameter data types for a parameter array do not match.')
                if param_arrays_xml is None:
                    param_arrays_xml = ET.SubElement(filter_or_instance_xml, 'paramsArrays')
                param_array_uid = gen_unique_uids(count=1, existing_uids=existing_source_param_uids)[0]
                existing_source_param_uids.append(param_array_uid)
                param_array_xml = ET.SubElement(param_arrays_xml, 'paramsArray')
                param_cells_xml = ET.SubElement(param_array_xml, 'paramsArrayCells')
                ET.SubElement(param_array_xml, 'name').set('v', source_param.name)
                ET.SubElement(param_array_xml, 'uid').set('v', str(param_array_uid))

                for param_cell_dtypes, param_cell_vals in zip(source_param.dtype, source_param.val):
                    param_cell_uid = gen_unique_uids(count=1, existing_uids=existing_source_param_uids)[0]
                    existing_source_param_uids.append(param_cell_uid)
                    param_cell_xml = ET.SubElement(param_cells_xml, 'paramsArrayCell')
                    ET.SubElement(param_cell_xml, 'uid').set('v', str(param_cell_uid))
                    param_cell_params_xml = ET.SubElement(param_cell_xml, 'parameters')

                    for param_name, param_val in param_cell_vals.items():
                        param_dtype = param_cell_dtypes[param_name]
                        param_cell_param_xml = ET.SubElement(param_cell_params_xml, 'parameter')
                        ET.SubElement(param_cell_param_xml, 'name').set('v', param_name)
                        param_cell_param_val_xml = ET.SubElement(param_cell_param_xml, 'paramValue')
                        save_sbs_constant_value(
                            parent_xml=param_cell_param_val_xml,
                            val=param_val,
                            dtype=param_dtype,
                            use_int32_tag=True)
            else:
                # non-array parameter
                param_xml = ET.SubElement(params_xml, 'parameter')
                ET.SubElement(param_xml, 'name').set('v', source_param.name)

                if node.type=='Unsupported' and node.func == 'bitmap' and source_param.name == 'bitmapresourcepath':
                    # special case: bitmap resource path, add link to the sbs document itself, since the resource is defined in the sbs document
                    if '?himself' not in dependency_uids:
                        raise RuntimeError('Dependencies must contain the sbs document itself (denoted by "?himself") as dependency')
                    source_param.val = f'pkg:///{source_param.val}?dependency={str(dependency_uids["?himself"])}'
                elif source_param.name == 'outputsize':
                    # special case: set node resolution relative to parent if it is currently absolute
                    if source_param.relative_to in [None, 0]:
                        source_param.val = [source_param.val[0] - graph_res[0], source_param.val[1] - graph_res[1]]
                        source_param.relative_to = 1

                if source_param.relative_to is not None:
                    ET.SubElement(param_xml, 'relativeTo').set('v', str(source_param.relative_to))

                param_val_xml = ET.SubElement(param_xml, 'paramValue')
                save_sbs_constant_value(
                    parent_xml=param_val_xml,
                    val=source_param.val,
                    dtype=source_param.dtype,
                    use_int32_tag=True)

    # node output bridges (only for non-atomic nodes)
    if node.type not in ['Input', 'Output']:
        if node.definition() is not None:
            output_bridgings_xml = ET.SubElement(filter_or_instance_xml, 'outputBridgings')
            for node_output in node.outputs:
                output_bridging_xml = ET.SubElement(output_bridgings_xml, 'outputBridging')
                ET.SubElement(output_bridging_xml, 'uid').set('v', str(node_output_uids[node_output.uname()]))
                if isinstance(node_output.name_xml, list):
                    if node.type == 'Invert':
                        # special case for the invert node: it has a differnt output name for the color and grayscale version
                        if node_output.dtype == SBSParamType.ENTRY_GRAYSCALE.value:
                            node_output_name = node_output.name_xml[0]
                        elif node_output.dtype == SBSParamType.ENTRY_COLOR.value:
                            node_output_name = node_output.name_xml[1]
                        else:
                            node_output_name = node_output.name_xml[0]
                            print(f'WARNING: Unexpected node output data type: {param_type_idx_to_name(node_output.dtype)}')
                    else:
                        raise RuntimeError(f'Unexpected overloaded sbs output name for node {node.name} (type {node.type}).')
                else:
                    node_output_name = node_output.name_xml
                ET.SubElement(output_bridging_xml, 'identifier').set('v', node_output_name)

    return node_xml, comment_xml

def save_sbs_constant_value(parent_xml, val, dtype, use_int32_tag=False):
    if dtype not in param_type_to_tag:
        raise RuntimeError(f'Unrecognized parameter value type: {dtype}.')
    if SBSParamType(dtype) == SBSParamType.BOOLEAN:
        param_val_str = '1' if val else '0'
    elif isinstance(val, numbers.Number):
        param_val_str = str(val)
    elif isinstance(val, str):
        param_val_str = val
    elif isinstance(val, list):
        if not all(isinstance(x, numbers.Number) for x in val):
            raise RuntimeError('Unknown parameter type.')
        param_val_str = ' '.join(str(x) for x in val)
    else:
        raise RuntimeError('Unknown parameter type.')

    if use_int32_tag and dtype == SBSParamType.INTEGER1.value:
        param_tag = 'constantValueInt32'
    else:
        param_tag = param_type_to_tag[dtype]
    ET.SubElement(parent_xml, param_tag).set('v', param_val_str)

def load_sbs_attribute_value(attr_xml):
    v = attr_xml.get('v')
    if v is None: # an attribute value can be represented in two different formats: <attribute_name v=...> or <attribute_name><value v=...></attribute_name>
        v = attr_xml.find('value').get('v')
    return v

def load_sbs_function_graph(function_name=None, filename=None, fgraph_xml=None, dependencies_by_uid=None, use_abs_paths=True):

    if fgraph_xml is None:
        # fgraph xml and dependencies are defined in an sbs file (possibly along with multiple other function graphs)

        if function_name is None or filename is None:
            raise RuntimeError('Need to provide either (function name, filename, and resource_dirs) or (fgraph xml and dependencies) .')

        doc_xml = ET.parse(filename)

        dependencies_by_uid = load_sbs_dependencies(doc_xml, source_filename=filename, use_abs_paths=use_abs_paths)

        function_xmls = doc_xml.getroot().findall('content/function')

        # handle content group hierarchies
        group_xmls = deque(doc_xml.getroot().findall('content/group'))
        while len(group_xmls) > 0:
            group_root = group_xmls.popleft()
            function_xmls.extend(group_root.findall('content/function'))
            group_xmls.extend(group_root.findall('content/group'))

        function_names = [load_sbs_attribute_value(function_xml.find('identifier')) for function_xml in function_xmls]

        if function_name not in function_names:
            raise RuntimeError(f'A function with name {function_name} was not found in the given file or xml tree.')
        function_xml = function_xmls[function_names.index(function_name)]

        fgraph_xml = function_xml.find('paramValue/dynamicValue')
        if fgraph_xml is None:
            raise RuntimeError('Missing dynamic value in a function.')
        if isinstance(fgraph_xml, list) and len(fgraph_xml) != 1:
            raise RuntimeError('Need exactly one dynamicValue node in a function.')
    else:
        # fgraph xml and dependencies are given explicitly
        if fgraph_xml is None or dependencies_by_uid is None:
            raise RuntimeError('Need to provide either (function name, filename, and resource_dirs) or (fgraph xml and dependencies) .')

        function_xml = None
        function_name = None

    fgraph = sbs_function_graph.SBSFunctionGraph(name=function_name)

    # parse inputs
    if function_xml is not None:
        for param_xml in function_xml.iter('paraminput'):
            input_param, _ = sbs_parameter.SBSParameter.load_sbs(xml=param_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            fgraph.inputs.append(input_param)

    # parse nodes
    nodes_by_uid = {}
    for node_xml in fgraph_xml.iter('paramNode'):

        # Basic node information
        node_uid = int(load_sbs_attribute_value(node_xml.find('uid')))
        node_type = load_sbs_attribute_value(node_xml.find('function'))
        node_dtype = int(load_sbs_attribute_value(node_xml.find('type')))
        if node_dtype is None:
            node_dtype = load_sbs_attribute_value(node_xml.find('type/value'))
        node_dtype = int(node_dtype)

        # Build function node (constant or operation)
        if node_type == 'instance':
            # non-atomic function node

            # parse node definition
            dep_xml = node_xml.find('funcDatas/funcData/constantValue/constantValueString')
            if dep_xml is None:
                raise RuntimeError('Could not parse dependency.')
            dep_path = load_sbs_attribute_value(dep_xml)
            if '?' not in dep_path:
                raise NotImplementedError('Dependencies without ? are currently not supported.')
            node_type = dep_path[dep_path.rfind('/') + 1:dep_path.rfind('?')]
            dep_uid = int(dep_path[dep_path.rfind('?dependency=') + len('?dependency='):])
            node_def = sbs_function_graph.SBSFunctionNodeDefinition(graph=node_type, path=dependencies_by_uid[dep_uid])
            # create node
            if node_type not in fgraph.op_dict:
                # unsupported non-atomic node (needs to be expanded before being used)
                node = sbs_function_graph.SBSFunctionUnsupported(dtype=node_dtype, node_type=node_type, definition=node_def)
            else:
                # supported non-atomic node
                node_class, node_args = fgraph.lookup_node_type(node_type, node_dtype)
                node = node_class(dtype=node_dtype, node_type=node_type, definition=node_def, **node_args)
        else:
            # atomic function node

            node_class, node_args = fgraph.lookup_node_type(node_type, node_dtype)
            node = node_class(dtype=node_dtype, node_type=node_type, definition=None, **node_args)

        # Resolve function data
        func_data = []
        for func_data_ in node_xml.findall('funcDatas/funcData'):
            func_data_name = load_sbs_attribute_value(func_data_.find('name'))
            func_data_val_ = func_data_.find('constantValue')[0]
            func_data_val = fgraph.parse_param_val(func_data_val_)
            func_data.append((func_data_name, func_data_val))

        if len(func_data) > 1:
            raise RuntimeError('Unexpected sbs structure: a parameter node has more than one function data object.')
        elif len(func_data) == 1:
            node.update_data(func_data[0][1])
        else:
            pass

        fgraph.add_node(node)
        nodes_by_uid[node_uid] = node

    # Obtain root node reference
    root_node_xml = fgraph_xml.find('rootnode')
    if root_node_xml is not None:
        root_node_uid = int(load_sbs_attribute_value(root_node_xml))
        fgraph.root_node = nodes_by_uid[root_node_uid]
    else:
        raise RuntimeError('Function graph does not have a root node.')

    # Scan graph connectivity
    for node_xml in fgraph_xml.iter('paramNode'):
        node_uid = int(load_sbs_attribute_value(node_xml.find('uid')))
        node = nodes_by_uid[node_uid]
        for conn_xml in node_xml.findall('connections/connection'):
            conn_name = load_sbs_attribute_value(conn_xml.find('identifier'))
            conn_ref = int(load_sbs_attribute_value(conn_xml.find('connRef')))
            fnode_input = sbs_function_graph.SBSFunctionNodeInput(name=conn_name)
            node.add_input(fnode_input=fnode_input)
            fnode_input.connect(parent=nodes_by_uid[conn_ref])

    return fgraph
