# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from enum import Enum

class SBSParamType(Enum):
    DUMMY_TYPE          = 0
    ENTRY_COLOR         = 1
    ENTRY_GRAYSCALE     = 2
    ENTRY_VARIANT       = ENTRY_COLOR | ENTRY_GRAYSCALE         # grayscale or color entry
    BOOLEAN             = 4
    INTEGER1            = 16
    INTEGER2            = 32
    INTEGER3            = 64
    INTEGER4            = 128
    INTEGER5            = INTEGER3 | INTEGER2
    INTEGER6            = INTEGER4 | INTEGER2
    FLOAT1              = 256
    FLOAT2              = 512
    FLOAT3              = 1024
    FLOAT4              = 2048
    FLOAT5              = FLOAT3 | FLOAT2                       # for anchors in gradient map nodes
    FLOAT6              = FLOAT4 | FLOAT2                       # for anchors in gradient map and curve nodes
    FLOAT_VARIANT       = FLOAT4 | FLOAT1                       # grayscale or color value
    ENTRY_COLOR_OPT     = ENTRY_COLOR | FLOAT4                  # optional color entry
    ENTRY_GRAYSCALE_OPT = ENTRY_GRAYSCALE | FLOAT1              # optional grayscale entry
    ENTRY_VARIANT_OPT   = ENTRY_COLOR_OPT | ENTRY_GRAYSCALE_OPT
    ENTRY_PARAMETER     = 4096                                  # parameter graph entry (used for fx-map)
    COMPLEX             = 8192
    STRING              = 16384
    PATH                = 32768
    VOID_TYPE           = 65536                                 # no return type, function only
    TEMPLATE1           = 131072                                # Template type 1, function only
    TEMPLATE2           = 262144                                # Template type 2, function only

    # type masks
    INTEGER             = INTEGER1 | INTEGER2 | INTEGER3 | INTEGER4
    FLOAT               = FLOAT1 | FLOAT2 | FLOAT3 | FLOAT4
    FUNCTION_ALL        = FLOAT | BOOLEAN | INTEGER | VOID_TYPE | STRING | PATH # All functions types

def param_type_is_image(dtype):
    return dtype in [
        SBSParamType.ENTRY_COLOR.value,
        SBSParamType.ENTRY_GRAYSCALE.value,
        SBSParamType.ENTRY_VARIANT.value,
        SBSParamType.ENTRY_COLOR_OPT.value,
        SBSParamType.ENTRY_GRAYSCALE_OPT.value,
        SBSParamType.ENTRY_VARIANT_OPT.value]

param_type_defaults = {
    SBSParamType.INTEGER1.value: 0,
    SBSParamType.INTEGER2.value: [0, 0],
    SBSParamType.INTEGER3.value: [0, 0, 0],
    SBSParamType.INTEGER4.value: [0, 0, 0, 0],
    SBSParamType.FLOAT1.value: 0.0,
    SBSParamType.FLOAT2.value: [0.0, 0.0],
    SBSParamType.FLOAT3.value: [0.0, 0.0, 0.0],
    SBSParamType.FLOAT4.value: [0.0, 0.0, 0.0, 0.0],
    SBSParamType.BOOLEAN.value: False,
    SBSParamType.STRING.value: '',
    SBSParamType.FUNCTION_ALL.value: 0.0 # float seems safest here, not sure what to choose
}

# dictionary: data type integer -> constant parameter value tag
param_type_to_tag = {
    SBSParamType.BOOLEAN.value: 'constantValueBool',
    SBSParamType.INTEGER1.value: 'constantValueInt1',
    SBSParamType.INTEGER2.value: 'constantValueInt2',
    SBSParamType.INTEGER3.value: 'constantValueInt3',
    SBSParamType.INTEGER4.value: 'constantValueInt4',
    SBSParamType.FLOAT1.value: 'constantValueFloat1',
    SBSParamType.FLOAT2.value: 'constantValueFloat2',
    SBSParamType.FLOAT3.value: 'constantValueFloat3',
    SBSParamType.FLOAT4.value: 'constantValueFloat4',
    SBSParamType.STRING.value: 'constantValueString'
}

param_tag_to_type = {
    'constantValueBool': SBSParamType.BOOLEAN.value,
    'constantValueInt32': SBSParamType.INTEGER1.value,
    'constantValueInt1': SBSParamType.INTEGER1.value,
    'constantValueInt2': SBSParamType.INTEGER2.value,
    'constantValueInt3': SBSParamType.INTEGER3.value,
    'constantValueInt4': SBSParamType.INTEGER4.value,
    'constantValueFloat1': SBSParamType.FLOAT1.value,
    'constantValueFloat2': SBSParamType.FLOAT2.value,
    'constantValueFloat3': SBSParamType.FLOAT3.value,
    'constantValueFloat4': SBSParamType.FLOAT4.value,
    'constantValueString': SBSParamType.STRING.value
}

param_short_to_type = {
    'integer1': SBSParamType.INTEGER1.value,
    'integer2': SBSParamType.INTEGER2.value,
    'integer3': SBSParamType.INTEGER3.value,
    'integer4': SBSParamType.INTEGER4.value,
    'int1': SBSParamType.INTEGER1.value,
    'int2': SBSParamType.INTEGER2.value,
    'int3': SBSParamType.INTEGER3.value,
    'int4': SBSParamType.INTEGER4.value,
    'float1': SBSParamType.FLOAT1.value,
    'float2': SBSParamType.FLOAT2.value,
    'float3': SBSParamType.FLOAT3.value,
    'float4': SBSParamType.FLOAT4.value,
    'bool': SBSParamType.BOOLEAN.value,
    'string': SBSParamType.STRING.value,
}

def param_type_idx_to_name(param_type_idx):
    if isinstance(param_type_idx, list):
        param_type_name = []
        for t in param_type_idx:
            param_type_name.append(param_type_idx_to_name(t))
    else:
        param_type_name = SBSParamType(param_type_idx).name

    if isinstance(param_type_name, list):
        if len(set(param_type_name)) != 1:
            raise RuntimeError('Heterogeneous parameter arrays are not supported.')
        if not any(param_type_name[0].startswith(prefix) for prefix in ['FLOAT', 'INTEGER', 'BOOLEAN']):
            raise RuntimeError(f'Unsupported parameter array type {param_type_name[0]}.')
        param_type_name = f'{param_type_name[0]}_ARRAY_{len(param_type_name)}'

    return param_type_name

def param_type_name_to_idx(param_type_name):

    array_len = None
    if '_ARRAY_' in param_type_name:
        array_len = int(param_type_name[param_type_name.find('_ARRAY_')+len('_ARRAY_'):])
        param_type_name = param_type_name[:param_type_name.find('_ARRAY_')]

    param_type_idx = SBSParamType[param_type_name].value
    if array_len is not None:
        param_type_idx = [param_type_idx]*array_len

    return param_type_idx

def param_val_to_type(val, noarrays=False):

    valtype = None
    if isinstance(val, list):
        if all(isinstance(v, (float, int)) for v in val) and any(isinstance(v, float) for v in val):
            valtype = SBSParamType[f'FLOAT{len(val)}'].value
        elif all(isinstance(v, int) for v in val):
            valtype = SBSParamType[f'INTEGER{len(val)}'].value
        elif all(isinstance(v, dict) for v in val):
            # source parameter array (list of cells, each cell has a dict of parameters)
            if noarrays:
                raise RuntimeError('Unknown parameter value data type.')
            valtype = []
            for cell in val:
                valtype.append({})
                for name, v in cell.items():
                    valtype[-1][name] = param_val_to_type(v, noarrays=True)
        elif any(isinstance(v, list) for v in val):
            # parameter array
            if noarrays:
                raise RuntimeError('Unknown parameter value data type.')
            valtype = []
            for v in val:
                valtype.append(param_val_to_type(v, noarrays=True)) # prevent nested parameter arrays
        else:
            raise RuntimeError('Unknown parameter value data type.')

    elif isinstance(val, float):
        valtype = SBSParamType.FLOAT1.value

    elif isinstance(val, bool): # needs to be checked before int, because bools are also ints
        valtype = SBSParamType.BOOLEAN.value

    elif isinstance(val, int):
        valtype = SBSParamType.INTEGER1.value

    elif isinstance(val, str):
        valtype = SBSParamType.STRING.value

    else:
        raise RuntimeError('Unknown parameter value data type.')

    return valtype
