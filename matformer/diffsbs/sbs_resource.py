# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from .sbs_param_type import SBSParamType

class SBSResourceParameter():
    '''
    A parameter of a resource.
    '''
    def __init__(self, name, val, dtype):
        self.name = name
        self.val = val
        self.dtype = dtype

class SBSResource:
    '''
    A resource of a substance graph.
    '''
    def __init__(self, name, type):
        self.name = name
        self.type = type

        self.params = []

    def add_param(self, param):
        if param.name in [p.name for p in self.params]:
            raise RuntimeError('A parameter with this name already exists.')

        self.params.append(param)

    def get_param_by_name(self, name):
        for param in self.params:
            if param.name == name:
                return param
        return None

class SBSBitmapResource(SBSResource):
    '''
    A bitmap resource of a substance graph.
    '''
    def __init__(self, name):
        super().__init__(name=name, type='bitmap')

        self.add_param(SBSResourceParameter(name='colorSpace', val='[use_embedded_profile]', dtype=SBSParamType.STRING))
        self.add_param(SBSResourceParameter(name='format', val='hdr', dtype=SBSParamType.STRING))
        self.add_param(SBSResourceParameter(name='filepath', val='', dtype=SBSParamType.STRING))

class SBSResourceDefinition():
    '''
    Information about the data that defines the resource
    '''
    def __init__(self, resource, path):
        self.resource = resource # name of the resource
        self.path = path # path to the file that contains the definition graph
