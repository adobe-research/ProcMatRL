# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from . import sbs_function_graph
from .sbs_param_type import SBSParamType, param_tag_to_type
from . import io_sbs

class SBSParameter:
    '''
    A parameter used for both graphs and function graphs.
    '''
    def __init__(self, name, param_dtype=1, param_val=None, parent=None, group=None, val_range=None, clamped=False):
        # Basic information
        self.name = name
        self.dtype = param_dtype
        self.val = param_val
        self.parent = parent # SBSInputNode for this parameter (typically set if the parameter is an input image, otherwise None)
        self.group = group
        self.val_range = val_range # range of valid values (if the value is a vector, then the range for each component)
        self.clamped = clamped # whether the value is clamped to the valid range

    def is_trainable(self):
        # Image and bool inputs are treated as untrainable
        return self.dtype >= 256

    @staticmethod
    def load_sbs_value(val_xml, dependencies_by_uid, use_abs_paths, round_float=6, allow_fgraph=True):
        '''
        Parse the default value of a graph parameter according to its type ID.
        '''
        param_tag = val_xml.tag
        if param_tag in ['constantValueInt32', 'constantValueInt1']:
            param_val = int(io_sbs.load_sbs_attribute_value(val_xml))
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag in ['constantValueInt2', 'constantValueInt3', 'constantValueInt4']:
            param_val = [int(i) for i in io_sbs.load_sbs_attribute_value(val_xml).strip().split()]
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag == 'constantValueFloat1':
            param_val = round(float(io_sbs.load_sbs_attribute_value(val_xml)), round_float)
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag in ['constantValueFloat2', 'constantValueFloat3', 'constantValueFloat4']:
            param_val = [round(float(i), round_float) for i in io_sbs.load_sbs_attribute_value(val_xml).strip().split()]
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag == 'constantValueBool':
            param_val = bool(int(io_sbs.load_sbs_attribute_value(val_xml)))
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag == 'constantValueString':
            param_val = io_sbs.load_sbs_attribute_value(val_xml)
            param_dtype = param_tag_to_type[param_tag]
        elif param_tag == 'dynamicValue':
            if not allow_fgraph:
                raise RuntimeError('Function graph not allowed as parameter value.')
            param_val = sbs_function_graph.SBSFunctionGraph.load_sbs(fgraph_xml=val_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            param_dtype = param_val.root_node.dtype
        else:
            raise TypeError('Unknown parameter type')
        return param_val, param_dtype

    @staticmethod
    def load_sbs(xml, dependencies_by_uid, use_abs_paths):
        # parse input parameters and build the dictionary for those with default values
        input_name = io_sbs.load_sbs_attribute_value(xml.find('identifier'))
        input_uid = int(io_sbs.load_sbs_attribute_value(xml.find('uid')))
        input_dtype = io_sbs.load_sbs_attribute_value(xml.find('type'))
        if input_dtype is None:
            input_dtype = io_sbs.load_sbs_attribute_value(xml.find('type/value'))
        input_dtype = int(input_dtype)
        if input_dtype in (SBSParamType.ENTRY_COLOR.value, SBSParamType.ENTRY_GRAYSCALE.value):
            input_val = None
        else:
            input_val, dtype = SBSParameter.load_sbs_value(val_xml=xml.find('defaultValue')[0], dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)
            if dtype != input_dtype:
                raise RuntimeError('Bad sbs file, parameter type does not match parameter tag.')
        input_group = io_sbs.load_sbs_attribute_value(xml.find('group')) if xml.find('group') is not None else None

        # parse default parameter range according to widget type
        input_widget_name = io_sbs.load_sbs_attribute_value(xml.find('defaultWidget/name'))
        input_widget_options = xml.findall('defaultWidget/options/option')
        input_widget_option_dict = {io_sbs.load_sbs_attribute_value(o.find('name')): io_sbs.load_sbs_attribute_value(o.find('value')) for o in input_widget_options}
        input_range, input_clamped = None, False

        # categorical parameters
        if input_widget_name == 'dropdownlist':
            input_values = input_widget_option_dict['parameters']
            input_range = [int(c) for c in input_values.strip().split(';')[1::2]]
            input_clamped = True

        # slider-based parameters
        elif input_dtype & (SBSParamType.INTEGER.value | SBSParamType.FLOAT.value):
            input_min = input_widget_option_dict.get('min')
            input_max = input_widget_option_dict.get('max')
            if input_min and input_max:
                if input_dtype & SBSParamType.INTEGER.value:
                    input_range = int(input_min), int(input_max)
                else:
                    input_range = float(input_min), float(input_max)
            input_clamped = int(input_widget_option_dict.get('clamped', '0')) > 0

        return SBSParameter(name=input_name, param_dtype=input_dtype, param_val=input_val, group=input_group, val_range=input_range, clamped=input_clamped), input_uid
