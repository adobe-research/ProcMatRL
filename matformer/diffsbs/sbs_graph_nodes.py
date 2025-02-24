# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import numpy as np

from .sbs_function_graph import SBSFunctionGraph
from .sbs_param_type import SBSParamType, param_type_defaults, param_val_to_type
from . import sbs_functional as F
from .sbs_utils import to_zero_one, from_zero_one, dtype_castable

class SBSNodeInput:
    '''
    An input slot of a node.
    '''
    def __init__(self, name, dtype=None, parent=None, parent_output=None, name_xml=None):
        self.name = name
        self.dtype = dtype
        self.parent = parent # parent node (None if the slot is not connected)
        self.parent_output = parent_output # output slot of the parent node (None if the slot is not connected)
        self.node = None
        self.name_xml = name_xml

    def connect(self, parent_output):
        if self.node is None:
            raise RuntimeError('The given node input is not attached to any node.')

        if self.parent_output == parent_output:
            # unchanged
            return

        # remove node from children of old parent output
        if self.parent_output is not None:
            if self in self.parent_output.children:
                self.parent_output.children.remove(self)

        # update the reference to the parent output
        self.parent_output = parent_output
        self.parent = parent_output.node if parent_output is not None else None

        # add node to children of new referenced output
        if self.parent_output is not None:
            if self not in self.parent_output.children:
                self.parent_output.children.append(self)

        return self

    def disconnect(self):
        return self.connect(parent_output=None)

class SBSNodeOutput:
    '''
    An output slot of a node.
    '''
    def __init__(self, name, dtype=None, name_xml=None):
        self.name = name # output slot name (node-unique)
        self.dtype = dtype
        self.children = [] # list of child node inputs
        self.node = None
        self.name_xml = name_xml

    # get graph-unique name for this output slot
    def uname(self):
        if self.node is None:
            raise RuntimeError('Cannot get variable name of a node output that is not attached to any node.')
        return f'{self.node.name}_{self.name}'

    def connect(self, child_input):
        if child_input is None:
            raise RuntimeError('Need to provide a valid node input as target for the connection (use disconnect to remove all connections from the node output).')
        child_input.connect(parent_output=self)
        return self

    def disconnect(self):
        # disconnect all child node inputs
        for child_node_input in list(self.children): # copy since we will be mutating the list
            child_node_input.disconnect()
        return self

class SBSNodeSourceParameter:
    def __init__(self, name, val, relative_to, dtype):
        self.name = name
        self.val = val # original parameter value, which may be a function graph
        self.relative_to = relative_to
        self.dtype = dtype
        self.eval_val = None # cached evaluated parameter value (e.g. the output value of a function graph)

    def get_eval_val(self, input_dict=None):
        if self.eval_val is None:
            if isinstance(self.val, SBSFunctionGraph):
                if input_dict is None:
                    raise RuntimeError('Must provide input dictionary to evaluate a function graph.')
                self.eval_val = self.val.eval(input_dict=input_dict)
            else:
                self.eval_val = copy.deepcopy(self.val)
        return self.eval_val

    def clear_eval_val(self):
        self.eval_val = None

class SBSNodeParameter:
    '''
    A parameter of a node.
    '''
    def __init__(self, name, val, dtype=None, trainable=False, val_min=None, val_max=None, val_range=None, clamped=False, convert_func=None, name_xml=None):
        self.name = name
        self.dtype = dtype # currently not always defined
        self.trainable = trainable # needs to be set before val
        self.val_min = val_min # minimal value of a ranged parameter
        self.val_max = val_max # maximal value of a ranged parameter
        self.val_range = val_range # valid values of a categorical parameter
        self.clamped = clamped # whether a ranged parameter is clamped to its default range

        self.val = val() if callable(val) else val
        self.node = None

        self.convert_func = convert_func
        self.name_xml = name_xml

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        if self.node is not None:
            raise RuntimeError('A parameter cannot be renamed once it has been added to a node.')
            # would need to update the key in the torch.nn.Module._parameters list
        self._name = v

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, v):
        if self.trainable:
            if not isinstance(v, torch.nn.Parameter):
                v = torch.nn.Parameter(torch.as_tensor(v, device=self.node.device if self.node is not None else torch.device('cpu')))
            if self.node is not None:
                if v.device != self.node.device:
                    requires_grad = v.requires_grad
                    with torch.no_grad():
                        v = v.to(self.node.device)
                    v = torch.nn.Parameter(v, requires_grad=requires_grad) # moving to a different device creates a copy of the tensor, which is no longer a parameter
                self.node.register_parameter(name=self.name, param=v) # overwrites the existing parameter with the same name if necessary
        self._val = v

    @property
    def node(self):
        try:
            return self._node
        except AttributeError:
            # not yet set
            return None

    @node.setter
    def node(self, v):
        # unregister parameter from the old node
        if self.trainable:
            if self.node is not None and self.node != v:
                del self.node._parameters[self.name] # there is currently no explicit method to unregister a parameter in PyTorch unfortunately, need to access the protected list

        # change node
        self._node = v

        # register parameter to the new node
        if self.trainable:
            if self.node is not None:
                self.val = self.val # registers param with ndoe and moves param to node's device

class SBSNodeDefinition():
    '''
    Information about the graph that defines a node
    '''
    def __init__(self, graph, path):
        self.graph = graph # name of the graph that defines the node
        self.path = path # path to the file that contains the definition graph

    def uname(self):
        return f'{self.path}:{self.graph}'

class SBSNode(torch.nn.Module, ABC):
    '''
    The base class of a generic SBS node.
    '''

    default_output_format = 0
    def __init__(self, name, node_type, node_func, output_res=None, use_alpha=False):
        super().__init__()

        # basic information
        self.name = name
        self.type = node_type
        self.func = node_func
        self.res = [9, 9] if output_res is None else output_res

        # alpha channel switch
        self.use_alpha = use_alpha

        # source parameters without conditioning (without conversion to the format needed for our implementation).
        # These are necessary to re-evaluate parameters that are defined as functions.
        self.source_params = []

        self.device = torch.device('cpu')

        self.inputs = [] # node input slots
        self.outputs = [] # node output slots
        self.params = [] # node parameters

        self.user_data = {} # dictionary of optional user-defined properties

        # add base parameters https://docs.substance3d.com/sddoc/graph-parameters-102400069.html
        # these are present in all nodes except output nodes
        if self.type != 'Output':
            self.add_param(SBSNodeParameter(name='outputsize', val=self.res, dtype=SBSParamType.INTEGER2.value, name_xml='outputsize'))
            self.add_param(SBSNodeParameter(name='format', val=SBSNode.default_output_format, dtype=SBSParamType.INTEGER1.value, name_xml='format'))
            self.add_param(SBSNodeParameter(name='pixelsize', val=[1.0, 1.0], dtype=SBSParamType.FLOAT2.value, name_xml='pixelsize'))
            self.add_param(SBSNodeParameter(name='pixelratio', val=0, dtype=SBSParamType.INTEGER1.value, name_xml='pixelratio'))
            self.add_param(SBSNodeParameter(name='tile_mode', val=3, dtype=SBSParamType.INTEGER1.value, name_xml='tiling'))
            self.add_param(SBSNodeParameter(name='seed', val=0, dtype=SBSParamType.INTEGER1.value, name_xml='randomseed'))

    # from typing import Union, Optional, TypeVar
    # T = TypeVar('T', bound='torch.nn.Module')
    # def to(self: T, device: Optional[Union[int, torch.device]] = ..., dtype: Optional[Union[torch.dtype, str]] = ..., non_blocking: bool = ...) -> T:
    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    # definition of the node or None if the node is atomic
    @abstractmethod
    def definition(self):
        return None

    # valid combinations of data types for inputs and outputs with the current parameter configuration
    @abstractmethod
    def signatures(self):
        return None

    @ staticmethod
    def find_matching_signatures(query_signature, signatures, first_match_only=False):
        matching_sigs = []
        for signature in signatures:
            is_matching = True
            for input_name, input_val in query_signature[0].items():
                if input_name not in signature[0] or (input_val is not None and not dtype_castable(input_val, signature[0][input_name])):
                    is_matching = False
                    break
            if is_matching:
                for output_name, output_val in query_signature[1].items():
                    if output_name not in signature[1] or (output_val is not None and not dtype_castable(output_val, signature[1][output_name])):
                        is_matching = False
                        break
            if is_matching:
                matching_sigs.append(signature)
                if first_match_only:
                    return matching_sigs

        return matching_sigs

    # the given signature matches a node signature if the node signature has all input and output names of the given signature and all non-None data types in the given signature match
    def matching_signatures(self, signature, first_match_only=False):
        return SBSNode.find_matching_signatures(query_signature=signature, signatures=self.signatures(), first_match_only=first_match_only)

    @staticmethod
    def get_default_params(node_type, res, use_alpha):
        node_class = globals()[f'SBS{node_type}Node']
        if node_type == 'Unsupported':
            raise RuntimeError('Cant get default parameters for an unsupported node.')
        default_node = node_class(name='default', output_res=res, use_alpha=use_alpha)
        return default_node.params

    def add_param(self, param):
        if param.name in [p.name for p in self.params]:
            raise RuntimeError('An parameter with the given name already exists in this node, remove it first.')

        param.node = self # registers the node's value as pytorch parameter
        self.params.append(param)

    def remove_param(self, param):
        if param not in self.params:
            raise RuntimeError('The given parameter is not in the node.')

        param.node = None # unregisters the node's value from the pytorch parameters
        self.params.remove(param)

    def add_input(self, node_input):
        if node_input.name in [i.name for i in self.inputs]:
            raise RuntimeError('An input with the given name already exists in this node, remove it first.')
        if node_input.parent is not None or node_input.parent_output is not None:
            raise RuntimeError('The given node input has a connection, remove the connection first.')

        node_input.node = self
        self.inputs.append(node_input)

    def add_output(self, node_output):
        if node_output.name in [o.name for o in self.outputs]:
            raise RuntimeError('An output with the given name already exists in this node, remove it first.')
        if len(node_output.children) > 0:
            raise RuntimeError('The given node output has outgoing connections, remove them first.')

        node_output.node = self
        self.outputs.append(node_output)

    def add_source_param(self, source_param):
        if not isinstance(source_param, SBSNodeSourceParameter):
            raise RuntimeError('Invalid source parameter.')
        self.source_params.append(source_param)

    def get_connected_inputs(self):
        return [node_input for node_input in self.inputs if node_input.parent is not None]

    def get_connected_child_inputs(self):
        return [child_input for output in self.outputs for child_input in output.children]

    def get_child_nodes(self):
        return [child_input.node for output in self.outputs for child_input in output.children if child_input.node is not None]

    def get_input_by_name(self, name):
        for node_input in self.inputs:
            if node_input.name == name:
                return node_input
        return None

    def get_output_by_name(self, name):
        for node_output in self.outputs:
            if node_output.name == name:
                return node_output
        return None

    def get_param_by_name(self, name):
        for param in self.params:
            if param.name == name:
                return param
        return None

    def get_unused_source_params(self):
        return [source_param for source_param in self.source_params if source_param.eval_val is None]

    # evaluate all source parameters to get the actual parameters for the node
    def condition_params(self, source_params, input_dict, clamp_params=True):

        # clear any previously evaluated source parameter values,
        # since some other parameters this value depends on might have changed
        # (like some function graph parameters)
        for source_param in source_params:
            source_param.clear_eval_val()

        # update the input dictionary with the default values
        input_dict = copy.copy(input_dict)
        input_dict.update({
            '$size': [1 << self.res[0], 1 << self.res[1]],
            '$sizelog2': self.res,
            '$normalformat': 0,
            })

        all_xml_param_names = set()
        for param in self.params:
            if isinstance(param.name_xml, list):
                all_xml_param_names.update(param.name_xml)
            else:
                all_xml_param_names.add(param.name_xml)

        source_params = {source_param.name: source_param for source_param in source_params}

        # evaluate all source parameters, the results are cached in the variable 'eval_val' of the source parameter
        for source_param in source_params.values():
            # if source_param.name not in all_xml_param_names:
            #     type_label = self.type if self.type != 'Unsupported' else self.func
            #     print(f'WARNING: Unexpected source parameter: {source_param.name} for node {self.name} of type {type_label}')
            source_param.get_eval_val(input_dict=input_dict)

        for param in self.params:
            param_name_xml_list = param.name_xml if isinstance(param.name_xml, list) else [param.name_xml]
            related_source_params = OrderedDict([(name_xml, source_params[name_xml]) for name_xml in param_name_xml_list if name_xml in source_params])
            if len(related_source_params) > 0:
                param_val = self.condition_param_val(param=param, source_params=related_source_params)
                if param_val is not None:
                    param.val = param_val
                    # make sure trainable parameters are tensors and have correct data types
                    if param.trainable:
                        if not isinstance(param.val, torch.Tensor):
                            raise RuntimeError('Trainable parameters needs to be pytorch tensors.')
                        if param.val.dtype not in [torch.float32, torch.float64]:
                            raise RuntimeError('Data type of trainable parameters needs to be float32.')
                    # make sure parameters are withing their valid value ranges, if they are known
                    if clamp_params and (param.val_min is not None or param.val_max is not None):
                        with torch.no_grad():
                            if ((param.val < param.val_min) | (param.val > param.val_max)).any():
                                print(f'Warning: parameter {param.name} of node {self.name} has value {param.val} outside its valid range [{param.val_min},{param.val_max}], clamping it.')
                                param.val.clamp_(min=param.val_min, max=param.val_max)
            param.dtype = param_val_to_type(param.val.tolist() if isinstance(param.val, torch.Tensor) else param.val)

    def condition_param_val(self, param, source_params):
        '''
        source_params: given in the same order as the parameter's name_xml
        returns None if the parameter value should not be changed
        '''
        val = None
        if param.name == 'outputsize':
            source_param = source_params[param.name_xml]
            val = copy.deepcopy(source_param.get_eval_val())
            if source_param.relative_to == 2:
                # relative to input
                # print('WARNING: Output size relative to input is not supported, using global resolution instead.')
                val = [self.res[0], self.res[1]]
            elif source_param.relative_to == 1:
                # relative to parent (=node)
                val = [self.res[0] + val[0], self.res[1] + val[1]]
            elif source_param.relative_to == 0:
                # absolute
                pass
            else:
                raise RuntimeError(f'Unknown relative mode {source_param.relative_to} for parameter outputsize of node {self.name}.')
        else:
            # default: directly copy value from source parameter
            # if first_valid(source_params) is not None:
            val = copy.deepcopy(first_valid(source_params).get_eval_val())

        return val

    def uncondition_params(self):

        params_by_source_name = {}
        for param in self.params:
            if isinstance(param.name_xml, list):
                for name_xml in param.name_xml:
                    if name_xml not in params_by_source_name:
                        params_by_source_name[name_xml] = {}
                    params_by_source_name[name_xml][param.name] = param
            else:
                if param.name_xml not in params_by_source_name:
                    params_by_source_name[param.name_xml] = {}
                params_by_source_name[param.name_xml][param.name] = param

        source_params = []
        for source_param_name, related_params in params_by_source_name.items():
            source_param = SBSNodeSourceParameter(
                name=source_param_name, val=None, relative_to=None, dtype=None)
            source_param_val, source_param_relative_to = self.uncondition_param_val(source_param=source_param, params=related_params)
            if source_param_val is not None:
                source_param.val = source_param_val
                source_param.relative_to = 0 if source_param_relative_to is None else source_param_relative_to
                source_param.dtype = param_val_to_type(source_param.val)
                source_params.append(source_param)

        return source_params

    def uncondition_param_val(self, source_param, params):
        '''
        params: given as dict param_name->param
        returns None if the given source parameter should not be added to the list of source parameters
        '''
        val = None
        relative_to = None
        if source_param.name == 'outputsize':
            param = params['outputsize']
            val = param.val.tolist() if isinstance(param.val, torch.Tensor) else copy.deepcopy(param.val)
            relative_to = 0
        else:
            if self.type != 'Unsupported' and len(params) != 1:
                raise RuntimeError('The default node parameter unconditioning expects constructing the source parameter from only a single related non-source parameter.')
            param = next(iter(params.values()))
            assert param.convert_func is None, 'Unconditioning a parameter that has a convert function with the default unconditioning function.'
            val = param.val.tolist() if isinstance(param.val, torch.Tensor) else copy.deepcopy(param.val)

        return val, relative_to

    def forward(self, node_inputs, validate_signatures=False):

        # get input signature and resize node inputs to match the output size if necessary
        output_size = torch.Size([2**self.res[0], 2**self.res[1]])
        input_signature = {}
        for input_name, node_input in node_inputs.items():

            if validate_signatures:
                # update input signature
                if node_input is None:
                    input_signature[input_name] = None
                elif node_input.shape[1] == 1:
                    input_signature[input_name] = SBSParamType.ENTRY_GRAYSCALE.value
                elif node_input.shape[1] == 3 or node_input.shape[1] == 4:
                    input_signature[input_name] = SBSParamType.ENTRY_COLOR.value
                else:
                    raise RuntimeError(f'Unexpected number of channels for node input: {node_input.shape[1]}')

            # resize node input to match the output size if necessary
            if node_input is not None and node_input.shape[2:4] != output_size:
                node_inputs[input_name] = torch.nn.functional.interpolate(
                    node_input, size=output_size, mode='bilinear')

        if validate_signatures:
            # validate input signature
            node_signatures = self.matching_signatures((input_signature, {}))
            if len(node_signatures) == 0:
                err_str = f'Signature Error: Node {self.name} ({self.func}) does not have a signature matching the given inputs:\n'
                err_str += f'  {input_signature}\n'
                err_str += 'Expected one of the following input signatures:\n'
                for sig in self.signatures():
                    err_str += f'  {sig[0]}\n'
                raise RuntimeError(err_str)

        # create a list of node function arguments that includes the node inputs and the node parameters
        node_kwargs = node_inputs
        for param in self.params:
            assert param.name is not None
            if param.trainable and (param.val_min is not None or param.val_max is not None):
                param_val = torch.clamp(param.val, min=param.val_min, max=param.val_max)
            else:
                param_val = param.val
            node_kwargs[param.name] = param_val

        # pass device as additional argument
        node_kwargs['device'] = self.device

        # run node function
        if not self.func.startswith('F.'):
            raise RuntimeError('Unknown function type.')
        node_function = getattr(F, self.func[2:])
        node_outputs = node_function(**node_kwargs)

        # get node node function outputs
        if len(self.outputs) == 1:
            node_outputs = (node_outputs,) # function does not return a tuple if there is only a single output

        if validate_signatures:
            # get output signature
            output_signature = {}
            for output_index, node_output_slot in enumerate(self.outputs):
                output_name = node_output_slot.name
                node_output = node_outputs[output_index]

                if node_output is None:
                    output_signature[output_name] = None
                elif node_output.shape[1] == 1:
                    output_signature[output_name] = SBSParamType.ENTRY_GRAYSCALE.value
                elif node_output.shape[1] == 3 or node_output.shape[1] == 4:
                    output_signature[output_name] = SBSParamType.ENTRY_COLOR.value
                else:
                    raise RuntimeError(f'Unexpected number of channels for node output: {node_output.shape[1]}')

            # validate output signature
            if len(SBSNode.find_matching_signatures(query_signature=({}, output_signature), signatures=node_signatures, first_match_only=True)) == 0:
                err_str = f'Signature Error: Node {self.name} ({self.func}) has unexpected output signature:\n'
                err_str += f'  {output_signature}\n'
                err_str += 'Expected one of the following output signatures:\n'
                for sig in node_signatures:
                    err_str += f'  {sig[1]}\n'
                raise RuntimeError(err_str)

        return node_outputs

### Special nodes

class SBSInputNode(SBSNode):
    '''
    SBS input node (only the reference of an input; no implementation).
    '''
    def __init__(self, name, graph_input, output_res=None, use_alpha=None):
        self.graph_input = graph_input
        if graph_input is not None:
            graph_input.parent = self
        super().__init__(name=name, node_type='Input', node_func=None, output_res=output_res, use_alpha=use_alpha)

        self.add_output(SBSNodeOutput(name='', dtype=self.graph_input.dtype if self.graph_input is not None else SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None
    
    def signatures(self):
        return [
            ({}, {'': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({}, {'': SBSParamType.ENTRY_COLOR.value})]

class SBSOutputNode(SBSNode):
    '''
    SBS output node (only the reference of an output; no implementation).
    '''
    def __init__(self, name, graph_output, output_res=None, use_alpha=None):
        self.graph_output = graph_output
        graph_output.parent = self
        super().__init__(name, node_type='Output', node_func=None, output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='input', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='inputNodeOutput'))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'input': SBSParamType.ENTRY_GRAYSCALE.value}, {}),
            ({'input': SBSParamType.ENTRY_COLOR.value}, {})]

    # Get the variable reference of the output
    def get_variable_name(self):
        assert len(self.inputs) ==  1

        if self.inputs[0].parent is None:
            # output node is not connected, this output is undefined
            return None
        else:
            return self.inputs[0].parent_output.uname()

    def get_dtype(self):
        assert len(self.inputs) ==  1

        if self.inputs[0].parent is None:
            # output node is not connected, this output is undefined
            return None
        else:
            return self.inputs[0].parent_output.dtype


class SBSUnsupportedNode(SBSNode):
    '''
    Node that is currently not directly implemented as differentiable function,
    this node needs to be expanded into its definition to be usable.
    '''
    def __init__(self, name, node_func, output_res=None, use_alpha=False, definition=None):
        super().__init__(name=name, node_type='Unsupported', node_func=node_func, output_res=output_res, use_alpha=use_alpha)
        self.defin = definition

    def definition(self):
        return self.defin

    def signatures(self):
        # assume that a node that is defined by a graph can only have a single fixed data type for each input and output slot
        return [
            ({inp.name: inp.dtype for inp in self.inputs}, {output.name: output.dtype for output in self.outputs})]

    def add_source_param(self, source_param, add_new_node_params=False): # param_name_xml, param_val, param_rel_to, param_dtype):
        if add_new_node_params and source_param.name not in [p.name for p in self.params]:
            if source_param.dtype not in param_type_defaults:
                raise RuntimeError(f'Could not find default value for parameter type {source_param.dtype}.')
            self.add_param(SBSNodeParameter(name=source_param.name, val=param_type_defaults[source_param.dtype], dtype=source_param.dtype, name_xml=source_param.name))
        super().add_source_param(source_param=source_param)


### Helpers for conversion of ranged parameters

def intensity_helper(max_intensity):
    return lambda intensity: min(intensity / max_intensity, 0.5)

def max_intensity_helper(max_intensity):
    return lambda intensity: max(max_intensity, intensity * 2.0)

def intensity_helper_zero_one(max_intensity):
    return lambda intensity: to_zero_one(intensity / max(max_intensity, abs(intensity * 2.0)))

def max_intensity_helper_zero_one(max_intensity):
    return lambda intensity: max(max_intensity, abs(intensity * 2.0))

def intensity_helper_getitem(max_intensity, index):
    return lambda p: to_zero_one(p[index] / max(max_intensity, abs(p[index] * 2.0)))

def max_intensity_helper_getitem(max_intensity, index):
    return lambda p: max(max_intensity, abs(p[index] * 2.0))


def first_valid(val_odict):
    assert isinstance(val_odict, OrderedDict)
    return next((x for x in val_odict.values() if x is not None), None)

# def first_match(val_list, other_iter):
#     if not isinstance(val_list, list):
#         val_list = [val_list]
#     return next((x for x in val_list if x in other_iter), None)

def intensity_cond(intensity, default_max_intensity):
    max_intensity = max_intensity_cond(intensity, default_max_intensity)
    return intensity / max_intensity

def intensity_uncond(intensity, max_intensity):
    return intensity * max_intensity

def max_intensity_cond(intensity, default_max_intensity):
    return max(default_max_intensity, intensity * 2.0)

def intensity_cond_zero_one(intensity, default_max_intensity):
    max_intensity = max_intensity_cond_zero_one(intensity, default_max_intensity)
    return to_zero_one(intensity / max_intensity)

def intensity_uncond_zero_one(intensity, max_intensity):
    return from_zero_one(intensity) * max_intensity

def max_intensity_cond_zero_one(intensity, default_max_intensity):
    return max(default_max_intensity, abs(intensity * 2.0))

# def intensity_cond_getitem(intensity_array, max_intensity, index):
#     return to_zero_one(intensity_array[index] / max(max_intensity, abs(intensity_array[index] * 2.0)))

# def max_intensity_cond_getitem(intensity_array, max_intensity, index):
#     return max(max_intensity, abs(intensity_array[index] * 2.0))


### Filter nodes

class SBSBlendNode(SBSNode):
    '''
    SBS blend node.

    Format for each parameter: `(sbs_name, trans_name, type, default_value=None, convert_func=None)`
      - `sbs_name`: name or a list of names as appeared in the `*.sbs` file.
      - `trans_name`: name used in the node implementation in functional.py.
      - `type`: parameter type which can be `'input'`, `'output'`, or `'other'`.
      - `default_value` (optional): default parameter value from `*.sbs` file.
      - `convert_func` (optional): the function for converting values in `*.sbs` to our implementation.
        It takes two arguments - new parameter and the existing parameter. By default, the parameter is
        copied directly without conversion.

    Notes:
      1. [CRUCIAL] Write all parameters according to their orders in functional.py.
      2. [CRUCIAL] Make sure that sbs_name and trans_name match.
      3. trans_name can be arbitrary for `'output'` parameters.
      4. sbs_name is not unique for extra parameters in our implementation, e.g. max_intensity. However,
         make sure that it does not have any conflict with other parameters.
      5. convert_func can be any callable object which takes only one positional argument. Two additional
         arguments is required if the parameter is related to multiple parameters in SBS.
      6. For atomic nodes that do not have output identifiers, set sbs_name as '' by default.
    '''
    blending_mode_list = [
        'copy', 'add', 'subtract', 'multiply', 'add_sub',
        'max', 'min', 'switch', 'divide', 'overlay', 'screen', 'soft_light']
    alphablending_list = ['use_source', 'ignore', 'straight', 'premultiplied']

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Blend', node_func='F.blend', output_res=output_res, use_alpha=use_alpha)

        mode_func = lambda p, a, b: self.blending_mode_list[p]
        alpha_func = lambda p: self.alphablending_list[p]

        self.add_input(SBSNodeInput(name='img_fg', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=['source', 'Foreground']))
        self.add_input(SBSNodeInput(name='img_bg', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=['destination', 'Background']))
        self.add_input(SBSNodeInput(name='blend_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml=['opacity', 'Opacity']))

        self.add_param(SBSNodeParameter(name='blending_mode', val='copy', dtype=SBSParamType.STRING.value, trainable=False, val_min=None, val_max=None, convert_func=mode_func, name_xml=['blendingmode', 'Blending_Mode']))
        self.add_param(SBSNodeParameter(name='cropping', val=[0.0, 1.0, 0.0, 1.0], dtype=SBSParamType.FLOAT4.value, trainable=False, val_min=None, val_max=None, name_xml='maskrectangle'))
        self.add_param(SBSNodeParameter(name='opacity', val=1.0, dtype=SBSParamType.FLOAT1.value, trainable=True, val_min=0.0, val_max=1.0, name_xml=['opacitymult', 'Opacity']))
        self.add_param(SBSNodeParameter(name='alphablending', val='use_source', dtype=SBSParamType.STRING.value, trainable=False, val_min=None, val_max=None, convert_func=alpha_func, name_xml='colorblending'))
        # self.add_param(SBSNodeParameter(name='colorblending', val=0, dtype=SBSParamType.INTEGER1.value, trainable=False, val_min=None, val_max=None, name_xml='colorblending'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=['', 'Blend']))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_fg': SBSParamType.ENTRY_COLOR.value, 'img_bg': SBSParamType.ENTRY_COLOR.value, 'blend_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_COLOR.value}),
            ({'img_fg': SBSParamType.ENTRY_GRAYSCALE.value, 'img_bg': SBSParamType.ENTRY_GRAYSCALE.value, 'blend_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'blending_mode':
            return self.blending_mode_list[first_valid(source_params).get_eval_val()]
        elif param.name == 'alphablending':
            return self.alphablending_list[source_params['colorblending'].get_eval_val()]
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name in ['Blending_Mode', 'Opacity']:
            return None, None # probably an older version of the parameter names
        elif source_param.name == 'blendingmode':
            return self.blending_mode_list.index(params['blending_mode'].val), None
        elif source_param.name == 'colorblending':
            return self.alphablending_list.index(params['alphablending'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)


class SBSBlurNode(SBSNode):
    '''
    SBS blur node.
    '''
    default_intensity = 0.5
    default_max_intensity = 20.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Blur', node_func='F.blur', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=SBSParamType.FLOAT1.value, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=SBSParamType.FLOAT1.value, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSChannelShuffleNode(SBSNode):
    '''
    SBS channel shuffle node.
    '''
    channel_dict = {'channelred': 0, 'channelgreen': 1, 'channelblue': 2, 'channelalpha': 3}

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='ChannelShuffle', node_func='F.channel_shuffle', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))
        self.add_input(SBSNodeInput(name='img_in_aux', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input2'))

        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=SBSParamType.BOOLEAN.value, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='shuffle_idx', val=[0, 1, 2, 3], dtype=SBSParamType.INTEGER4.value, trainable=False, val_min=None, val_max=None, convert_func=self.update_shuffle_indices, name_xml=['channelred', 'channelgreen', 'channelblue', 'channelalpha']))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_COLOR.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_in_aux': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_in_aux': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'img_in_aux': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'img_in_aux': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'shuffle_idx':
            val = copy.deepcopy(param.val)
            for source_param in source_params.values():
                if source_param is not None:
                    val[self.channel_dict[source_param.name]] = source_param.get_eval_val()
            return val
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name in ['channelred', 'channelgreen', 'channelblue', 'channelalpha']:
            return params['shuffle_idx'].val[self.channel_dict[source_param.name]], None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

    def update_shuffle_indices(self, new_val, exist_val, sbs_name):
        channel_dict = {'channelred': 0, 'channelgreen': 1, 'channelblue': 2, 'channelalpha': 3}
        exist_val[channel_dict[sbs_name]] = new_val# - (new_val > 3)
        return exist_val

class SBSCurveNode(SBSNode):
    '''
    SBS curve node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Curve', node_func='F.curve', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='anchors', val=self.get_default_anchors, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.get_anchors, name_xml=['curveluminance', 'curveblue']))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'anchors':
            source_val = first_valid(source_params).get_eval_val()
            assert len(source_val) >= 2
            anchors = []
            for cell in source_val:
                pos = cell['position']
                cpl = cell['position'] if cell['isLeftBroken'] else cell['left']
                cpr = cell['position'] if cell['isRightBroken'] else cell['right']
                anchors.append(pos + cpl + cpr)
            return anchors
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'curveblue':
            return None, None # probably an older version of the parameter names (?)
        elif source_param.name == 'curveluminance':
            anchors = params['anchors'].val.tolist()
            source_val = []
            for anchor in anchors:
                source_val.append({'position': anchor[:2], 'left': anchor[2:4], 'isLeftBroken': int(anchor[2] >= anchor[0]), 'right': anchor[4:6], 'isRightBroken': int(anchor[4] <= anchor[0])})
            return source_val, None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

    def get_anchors(self, new_val, exist_val, sbs_name):
        assert len(new_val) >= 2
        anchors = []
        for cell in new_val:
            pos = cell['position']
            cpl = cell['position'] if cell['isLeftBroken'] else cell['left']
            cpr = cell['position'] if cell['isRightBroken'] else cell['right']
            anchors.append(pos + cpl + cpr)
        return anchors

    def get_default_anchors(self):
        return np.linspace(0.0, 1.0, 2).reshape(-1, 1).repeat(6, axis=1).tolist()

class SBSDBlurNode(SBSNode):
    '''
    SBS directional blur node.
    '''
    default_intensity = 0.5
    default_max_intensity = 20.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='DBlur', node_func='F.d_blur', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: np.remainder(p, 1.0), name_xml='mblurangle'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'angle':
            return np.remainder(source_params['mblurangle'].get_eval_val(), 1.0)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        elif source_param.name == 'mblurangle':
            return params['angle'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSDWarpNode(SBSNode):
    '''
    SBS directional warp node.
    '''
    default_intensity = 0.5
    default_max_intensity = 20.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='DWarp', node_func='F.d_warp', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))
        self.add_input(SBSNodeInput(name='intensity_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='inputintensity'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: np.remainder(p, 1.0), name_xml='warpangle'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'intensity_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'intensity_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'angle':
            return np.remainder(source_params['warpangle'].get_eval_val(), 1.0)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        elif source_param.name == 'warpangle':
            return params['angle'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSDistanceNode(SBSNode):
    '''
    SBS distance node.
    '''
    default_distance = 10.0 / 256.0
    default_max_distance = 256.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Distance', node_func='F.distance', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='mask'))
        self.add_input(SBSNodeInput(name='img_source', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='source'))

        self.add_param(SBSNodeParameter(name='mode', val='gray', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'color' if p else 'gray', name_xml='colorswitch'))
        self.add_param(SBSNodeParameter(name='combine', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='combinedistance'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='dist', val=self.default_distance, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_distance), name_xml='distance'))
        self.add_param(SBSNodeParameter(name='max_dist', val=self.default_max_distance, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_distance), name_xml='distance'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_mask': SBSParamType.ENTRY_GRAYSCALE.value, 'img_source': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_mask': SBSParamType.ENTRY_GRAYSCALE.value, 'img_source': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'mode':
            return 'color' if source_params['colorswitch'].get_eval_val() else 'gray'
        elif param.name == 'dist':
            return intensity_cond(source_params['distance'].get_eval_val(), self.default_max_distance)
        elif param.name == 'max_dist':
            return max_intensity_cond(source_params['distance'].get_eval_val(), self.default_max_distance)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'colorswitch':
            return int(params['mode'].val == 'color'), None
        elif source_param.name == 'distance':
            return intensity_uncond(params['dist'].val.tolist(), params['max_dist'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSEmbossNode(SBSNode):
    '''
    SBS emboss node.
    '''
    default_intensity = 0.5
    default_max_intensity = 10.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Emboss', node_func='F.emboss', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))
        self.add_input(SBSNodeInput(name='height_map', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='inputgradient'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='light_angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: np.remainder(p, 1.0), name_xml='lightangle'))
        self.add_param(SBSNodeParameter(name='highlight_color', val=[1.0, 1.0, 1.0, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='highlightcolor'))
        self.add_param(SBSNodeParameter(name='shadow_color', val=[0.0, 0.0, 0.0, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='shadowcolor'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'height_map': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'height_map': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'light_angle':
            return np.remainder(source_params['lightangle'].get_eval_val(), 1.0)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        elif source_param.name == 'lightangle':
            return params['light_angle'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSGradientMapNode(SBSNode):
    '''
    SBS gradient map node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='GradientMap', node_func='F.gradient_map', output_res=output_res, use_alpha=use_alpha)

        mode_func = lambda p: 'color' if p else 'gray'

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='interpolate', val=True, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=self.update_interpolate_flag, name_xml='gradientrgba'))
        self.add_param(SBSNodeParameter(name='mode', val='color', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=mode_func, name_xml='colorswitch'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='interpmode', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='interpmode'))
        self.add_param(SBSNodeParameter(name='anchors', val=self.get_default_anchors, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.get_anchors, name_xml=['colorswitch', 'gradientrgba']))
        self.add_param(SBSNodeParameter(name='addressingrepeat', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='addressingrepeat'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        if self.get_param_by_name(name='mode').val == 'color':
            return [({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value})]
        elif self.get_param_by_name(name='mode').val == 'gray':
            return [({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]
        else:
            raise RuntimeError('Unexpected value for mode parameter of Gradient Map node.')

    def condition_param_val(self, param, source_params):
        if param.name == 'interpolate':
            return any([cell['midpoint'] < 0 for cell in source_params['gradientrgba'].get_eval_val()])
        elif param.name == 'mode':
            return 'color' if source_params['colorswitch'].get_eval_val() else 'gray'
        elif param.name == 'anchors':
            if 'colorswitch' in source_params:
                mode = ('color' if source_params['colorswitch'].get_eval_val() else 'gray')
            else:
                mode = self.get_param_by_name('mode').val

            if 'gradientrgba' in source_params:
                anchors = torch.tensor([[cell['position']] + cell['value'] for cell in source_params['gradientrgba'].get_eval_val()])
            else:
                anchors = param.val

            # convert anchors to the given color mode if necessary
            if mode == 'color':
                if anchors.shape[1] == 2:
                    anchors = anchors[:, [0, 1, 1, 1, 1] if self.use_alpha else [0, 1, 1, 1]]
                    if self.use_alpha:
                        anchors[:, 4] = 1
                elif anchors.shape[1] == 4 and self.use_alpha:
                    anchors = torch.cat([anchors, torch.ones(size=[anchors.shape[0], 1], device=anchors.device, dtype=anchors.dtype)], dim=1)
                elif anchors.shape[1] == 5 and not self.use_alpha:
                    anchors = anchors[:, [0, 1, 2, 3]]
            else:
                if anchors.shape[1] > 2:
                    anchors = torch.cat([anchors[:, [0]], anchors[:, 1:4].mean(dim=1, keepdims=True)], dim=1)

            return sorted(anchors.tolist(), key=lambda p: p[0])
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'colorswitch':
            return params['mode'].val == 'color', None
        elif source_param.name == 'gradientrgba':
            anchors = params['anchors'].val.tolist()
            source_val = []
            for anchor in anchors:
                anchor_val = [anchor[1]]*3+[1.0] if len(anchor) == 2 else (anchor[1:] if self.use_alpha else anchor[1:]+[1.0])
                assert len(anchor_val) == 4, 'The unconditioned anchor should always be RGBA (even if the mode is grayscale)'
                source_val.append({'position': anchor[0], 'midpoint': -1.0 if params['interpolate'].val else 0.0, 'value': anchor_val})
            return source_val, None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

    def get_anchors(self, new_val, exist_val, sbs_name):
        if sbs_name == 'colorswitch':
            # return self.get_default_anchors()

            # color mode may have changed, check if the anchors still have the right format
            # note that a conversions from color to gray looses some information.
            # To be sure that the color mode and anchors can be parsed in any order without loosing information,
            # make sure that the default color mode is 'color' (which it should currently be)
            new_color_mode = ('color' if new_val else 'gray')
            anchors = self.get_param_by_name('anchors').val
            if new_color_mode == 'color' and anchors.shape[1] != 4 + self.use_alpha:
                # color mode changed from gray to color, convert the anchors to the new color mode
                anchors = anchors[:, [0, 1, 1, 1, 1] if self.use_alpha else [0, 1, 1, 1]]
                if self.use_alpha:
                    anchors[:, 4] = 1

            elif new_color_mode == 'gray' and anchors.shape[1] != 2:
                # color mode changed from color to gray, convert the anchors to the new color mode
                anchors = torch.cat([
                    anchors[:, [0]],
                    anchors[:, 1:4].mean(dim=1, keepdim=True)], dim=1)
        elif sbs_name == 'gradientrgba':
            if self.get_param_by_name('mode').val == 'gray':
                anchors = torch.tensor([[cell['position'], (cell['value'][0]+cell['value'][1]+cell['value'][2])/3.0] for cell in new_val])
            elif self.use_alpha:
                anchors = torch.tensor([[cell['position']] + cell['value'] for cell in new_val])
            else:
                anchors = torch.tensor([[cell['position']] + cell['value'][:3] for cell in new_val])
        else:
            raise RuntimeError('Unknown type of update for anchors.')

        return sorted(anchors.tolist(), key=lambda p: p[0])

    def get_default_anchors(self):
        num_cols = 2 if self.get_param_by_name('mode').val == 'gray' else 4 + self.use_alpha
        anchors = np.linspace(0.0, 1.0, 2).reshape(-1, 1).repeat(num_cols, axis=1)
        # anchors = [[0.0] * num_cols, [1.0] * num_cols]
        if num_cols > 4 and self.use_alpha:
            anchors[:, 4] = 1.0
        anchors = anchors.tolist()
        return anchors

    def update_interpolate_flag(self, new_val):
        return any([cell['midpoint'] < 0 for cell in new_val])

class SBSGradientMapDynNode(SBSNode):
    '''
    SBS gradient map node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='GradientMapDyn', node_func='F.gradient_map_dyn', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input1'))
        self.add_input(SBSNodeInput(name='img_gradient', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input2'))

        self.add_param(SBSNodeParameter(name='orientation', val='horizontal', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'vertical' if p else 'horizontal', name_xml='uvselector'))
        self.add_param(SBSNodeParameter(name='addressingrepeat', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='addressingrepeat'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='position', val=0.0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='coordinate'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_gradient': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_gradient': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'orientation':
            return 'vertical' if source_params['uvselector'].get_eval_val() else 'horizontal'
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'uvselector':
            return int(params['orientation'].val == 'vertical'), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSC2GNode(SBSNode):
    '''
    SBS grayscale conversion node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='C2G', node_func='F.c2g', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='flatten_alpha', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='alphamult'))
        self.add_param(SBSNodeParameter(name='rgba_weights', val=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='channelsweights'))
        self.add_param(SBSNodeParameter(name='bg', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='mattelevel'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSHSLNode(SBSNode):
    '''
    SBS hsl node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HSL', node_func='F.hsl', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='hue', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='hue'))
        self.add_param(SBSNodeParameter(name='saturation', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='saturation'))
        self.add_param(SBSNodeParameter(name='lightness', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='luminosity'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_COLOR.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value})]

class SBSLevelsNode(SBSNode):
    '''
    SBS levels node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Levels', node_func='F.levels', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='in_low', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.update_levels_anchors, name_xml='levelinlow'))
        self.add_param(SBSNodeParameter(name='in_mid', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.update_levels_anchors, name_xml='levelinmid'))
        self.add_param(SBSNodeParameter(name='in_high', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.update_levels_anchors, name_xml='levelinhigh'))
        self.add_param(SBSNodeParameter(name='out_low', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.update_levels_anchors, name_xml='leveloutlow'))
        self.add_param(SBSNodeParameter(name='out_high', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=self.update_levels_anchors, name_xml='levelouthigh'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name in ['in_low', 'in_mid', 'in_high', 'out_low', 'out_high']:
            source_val = next(iter(source_params.values())).get_eval_val()
            if isinstance(source_val, (int, float)):
                return source_val
            # elif source_val[0] == source_val[1] and source_val[0] == source_val[2]:
                # return a single value list
            #     return source_val[0]
            elif not self.use_alpha:
                # return a length 3 vector
                return source_val[:3]
            else:
                # return the full length (4) vector
                return source_val
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name in ['levelinlow', 'levelinmid', 'levelinhigh', 'leveloutlow', 'levelouthigh']:
            val = next(iter(params.values())).val.tolist()
            if isinstance(val, (int, float)):
                return [val, val, val, 1], None
            elif len(val) == 3:
                return val + [1], None
            elif len(val) == 4:
                return val, None
            else:
                raise RuntimeError(f'Unexpected vector dimension for levels node parameter {source_param.name}: it is a {len(val)} dimensional vector, but expected a 1, 3, or 4 dimensional vector.')
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

    def update_levels_anchors(self, new_val):
        if isinstance(new_val, (int, float)):
            return new_val
        # elif new_val[0] == new_val[1] and new_val[0] == new_val[2]:
            # return a single value list
        #     return new_val[0]
        elif not self.use_alpha:
            # return a length 3 vector
            return new_val[:3]
        else:
            # return the full length (4) vector
            return new_val

class SBSNormalNode(SBSNode):
    '''
    SBS normal node.
    '''
    default_max_intensity = 3.0
    default_intensity = 1.0 / default_max_intensity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Normal', node_func='F.normal', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='mode', val='tangent_space', dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='inversedy'))
        self.add_param(SBSNodeParameter(name='use_input_alpha', val=True, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: True if p else False, name_xml='input2alpha'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_COLOR.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['inversedy'].get_eval_val() else 'dx'
        elif param.name == 'use_input_alpha':
            return True if source_params['input2alpha'].get_eval_val() else False
        elif param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'inversedy':
            return params['normal_format'].val == 'gl', None
        elif source_param.name == 'input2alpha':
            return params['use_input_alpha'].val, None
        elif source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSSharpenNode(SBSNode):
    '''
    SBS sharpen node.
    '''
    default_max_intensity = 3.0
    default_intensity = 1.0 / default_max_intensity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Sharpen', node_func='F.sharpen', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity'))
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSTransform2dNode(SBSNode):
    '''
    SBS transform 2d node.
    '''
    default_max_intensity = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Transform2d', node_func='F.transform_2d', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))

        self.add_param(SBSNodeParameter(name='sample_mode', val='bilinear', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'nearest' if p else 'bilinear', name_xml='filtering'))
        self.add_param(SBSNodeParameter(name='mipmap_mode', val='auto', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'manual' if p else 'auto', name_xml='mipmapmode'))
        self.add_param(SBSNodeParameter(name='mipmap_level', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='manualmiplevel'))
        self.add_param(SBSNodeParameter(name='mattecolor', val=[0.0, 0.0, 0.0, 0.0], dtype=None, trainable=False, val_min=None, val_max=None, name_xml='mattecolor'))
        self.add_param(SBSNodeParameter(name='x1', val=to_zero_one(1.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 0), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='x1_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 0), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='x2', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 1), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='x2_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 1), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='x_offset', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 0), name_xml='offset'))
        self.add_param(SBSNodeParameter(name='x_offset_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 0), name_xml='offset'))
        self.add_param(SBSNodeParameter(name='y1', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 2), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='y1_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 2), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='y2', val=to_zero_one(1.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 3), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='y2_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 3), name_xml='matrix22'))
        self.add_param(SBSNodeParameter(name='y_offset', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_getitem(self.default_max_intensity, 1), name_xml='offset'))
        self.add_param(SBSNodeParameter(name='y_offset_max', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_getitem(self.default_max_intensity, 1), name_xml='offset'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'sample_mode':
            return 'nearest' if source_params['filtering'].get_eval_val() else 'bilinear'
        elif param.name == 'mipmap_mode':
            return 'manual' if source_params['mipmapmode'].get_eval_val() else 'auto'
        elif param.name in ['x1', 'x2', 'y1', 'y2']:
            elm_idx = {'x1': 0, 'x2': 1, 'y1': 2, 'y2': 3}
            return intensity_cond_zero_one(source_params['matrix22'].get_eval_val()[elm_idx[param.name]], self.default_max_intensity)
        elif param.name in ['x1_max', 'x2_max', 'y1_max', 'y2_max']:
            elm_idx = {'x1_max': 0, 'x2_max': 1, 'y1_max': 2, 'y2_max': 3}
            return max_intensity_cond_zero_one(source_params['matrix22'].get_eval_val()[elm_idx[param.name]], self.default_max_intensity)
        elif param.name in ['x_offset', 'y_offset']:
            elm_idx = {'x_offset': 0, 'y_offset': 1}
            return intensity_cond_zero_one(source_params['offset'].get_eval_val()[elm_idx[param.name]], self.default_max_intensity)
        elif param.name in ['x_offset_max', 'y_offset_max']:
            elm_idx = {'x_offset_max': 0, 'y_offset_max': 1}
            return max_intensity_cond_zero_one(source_params['offset'].get_eval_val()[elm_idx[param.name]], self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'filtering':
            return int(params['sample_mode'].val == 'nearest'), None
        elif source_param.name == 'mipmapmode':
            return int(params['mipmap_mode'].val == 'manual'), None
        elif source_param.name == 'matrix22':
            return [
                intensity_uncond_zero_one(params['x1'].val.tolist(), params['x1_max'].val),
                intensity_uncond_zero_one(params['x2'].val.tolist(), params['x2_max'].val),
                intensity_uncond_zero_one(params['y1'].val.tolist(), params['y1_max'].val),
                intensity_uncond_zero_one(params['y2'].val.tolist(), params['y2_max'].val),
                ], None
        elif source_param.name == 'offset':
            return [
                intensity_uncond_zero_one(params['x_offset'].val.tolist(), params['x_offset_max'].val),
                intensity_uncond_zero_one(params['y_offset'].val.tolist(), params['y_offset_max'].val),
                ], None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSUniformColorNode(SBSNode):
    '''
    SBS uniform color node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='UniformColor', node_func='F.uniform_color', output_res=output_res, use_alpha=use_alpha)

        color_func = lambda p: p if isinstance(p, list) else [p, p, p, 1.0]

        self.add_param(SBSNodeParameter(name='mode', val='color', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'color' if p else 'gray', name_xml='colorswitch')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='num_imgs', val=1, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='rgba', val=[0.0, 0.0, 0.0, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=color_func, name_xml='outputcolor')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        if self.get_param_by_name('mode').val == 'color':
            return [({}, {'': SBSParamType.ENTRY_COLOR.value})]
        elif self.get_param_by_name('mode').val == 'gray':
            return [({}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]
        else:
            raise RuntimeError('Unexpected value for mode parameter of Uniform Color node.')

    def condition_param_val(self, param, source_params):
        if param.name == 'mode':
            return 'color' if source_params['colorswitch'].get_eval_val() else 'gray'
        elif param.name == 'rgba':
            source_val = source_params['outputcolor'].get_eval_val()
            return source_val if isinstance(source_val, list) else [source_val, source_val, source_val, 1.0]
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'colorswitch':
            return int(params['mode'].val == 'color'), None
        elif source_param.name == 'outputcolor':
            return params['rgba'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSWarpNode(SBSNode):
    '''
    SBS warp node.
    '''
    default_max_intensity = 2.0
    default_intensity = 1.0 / default_max_intensity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Warp', node_func='F.warp', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input1'))
        self.add_input(SBSNodeInput(name='intensity_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='inputgradient'))

        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'intensity_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'intensity_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSPassthroughNode(SBSNode):
    '''
    SBS passthrough node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Passthrough', node_func='F.passthrough', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_output(SBSNodeOutput(name='', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=''))

    def definition(self):
        return None

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'': SBSParamType.ENTRY_COLOR.value}),
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSCurvatureNode(SBSNode):
    '''
    SBS curvature node.
    '''
    default_intensity = 0.1
    default_max_intensity = 10.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Curvature', node_func='F.curvature', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='normal_format')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='emboss_intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='emboss_max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Output'))

    def definition(self):
        return SBSNodeDefinition(graph='curvature', path='sbs://curvature.sbs')

    def signatures(self):
        return [
            ({'normal': SBSParamType.ENTRY_COLOR.value}, {'Output': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        elif param.name == 'emboss_intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'emboss_max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        elif source_param.name == 'intensity':
            return intensity_uncond(params['emboss_intensity'].val.tolist(), params['emboss_max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSInvertNode(SBSNode):
    '''
    SBS invert node (color or grayscale).
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Invert', node_func='F.invert', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='invert_switch', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='invert'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=['Invert_Grayscale', 'Invert_Color']))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='invert_grayscale', path='sbs://invert.sbs')
        else:
            return SBSNodeDefinition(graph='invert', path='sbs://invert.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSHistogramScanNode(SBSNode):
    '''
    SBS histogram scan node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HistogramScan', node_func='F.histogram_scan', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Input_1'))

        self.add_param(SBSNodeParameter(name='invert_position', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Invert_Position'))
        self.add_param(SBSNodeParameter(name='position', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Position'))
        self.add_param(SBSNodeParameter(name='contrast', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Contrast'))

        self.add_output(SBSNodeOutput(name='Output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Output'))

    def definition(self):
        return SBSNodeDefinition(graph='histogram_scan', path='sbs://histogram_scan.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Output': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSHistogramRangeNode(SBSNode):
    '''
    SBS histogram range node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HistogramRange', node_func='F.histogram_range', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='ranges', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='range'))
        self.add_param(SBSNodeParameter(name='position', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='position'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='histogram_range', path='sbs://histogram_range.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSHistogramSelectNode(SBSNode):
    '''
    SBS histogram select node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HistogramSelect', node_func='F.histogram_select', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='position', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='position'))
        self.add_param(SBSNodeParameter(name='ranges', val=0.25, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='range'))
        self.add_param(SBSNodeParameter(name='contrast', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='constrast'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='histogram_select', path='sbs://histogram_select.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSEdgeDetectNode(SBSNode):
    '''
    SBS edge detect node.
    '''
    default_max_width = 16.0
    default_width = 2.0/default_max_width
    default_max_roundness = 16.0
    default_roundness = 4.0/default_max_roundness

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='EdgeDetect', node_func='F.edge_detect', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='invert_flag', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='invert'))
        self.add_param(SBSNodeParameter(name='edge_width', val=self.default_width, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_width), name_xml='edge_width')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_edge_width', val=self.default_max_width, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_width), name_xml='edge_width')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='edge_roundness', val=self.default_roundness, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_roundness), name_xml='edge_roundness')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_edge_roundness', val=self.default_max_roundness, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_roundness), name_xml='edge_roundness')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='tolerance', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='tolerance'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='edge_detect', path='sbs://edge_detect.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name in 'edge_width':
            return intensity_cond(source_params['edge_width'].get_eval_val(), self.default_max_width)
        elif param.name == 'max_edge_width':
            return max_intensity_cond(source_params['edge_width'].get_eval_val(), self.default_max_width)
        elif param.name in 'edge_roundness':
            return intensity_cond(source_params['edge_roundness'].get_eval_val(), self.default_max_roundness)
        elif param.name == 'max_edge_roundness':
            return max_intensity_cond(source_params['edge_roundness'].get_eval_val(), self.default_max_roundness)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'edge_width':
            return intensity_uncond(params['edge_width'].val.tolist(), params['max_edge_width'].val), None
        elif source_param.name == 'edge_roundness':
            return intensity_uncond(params['edge_roundness'].val.tolist(), params['max_edge_roundness'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSSafeTransformNode(SBSNode):
    '''
    SBS safe transform node (color or grayscale).
    '''
    symmetry_vals = ['none', 'X', 'Y', 'X+Y']
    offset_modes = ['manual', 'random']

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='SafeTransform', node_func='F.safe_transform', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='tile', val=1, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='tile'))
        self.add_param(SBSNodeParameter(name='tile_safe_rot', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='tile_safe_rotation'))
        self.add_param(SBSNodeParameter(name='symmetry', val='none', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: self.symmetry_vals[p], name_xml='symmetry')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mipmap_mode', val='auto', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'manual' if p else 'auto', name_xml='mipmapmode')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mipmap_level', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='manualmiplevel'))
        self.add_param(SBSNodeParameter(name='offset_mode', val='manual', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: self.offset_modes[p], name_xml='offset_mode')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='offset_x', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: p[0], name_xml='offset')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='offset_y', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: p[1], name_xml='offset')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: np.remainder(p, 1.0), name_xml='rotation')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='safe_transform_grayscale', path='sbs://safe_transform.sbs')
        else:
            return SBSNodeDefinition(graph='safe_transform', path='sbs://safe_transform.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name in 'symmetry':
            return self.symmetry_vals[source_params['symmetry'].get_eval_val()]
        elif param.name in 'mipmap_mode':
            return 'manual' if source_params['mipmapmode'].get_eval_val() else 'auto'
        elif param.name in 'offset_mode':
            return self.offset_modes[source_params['offset_mode'].get_eval_val()]
        elif param.name in 'offset_x':
            return source_params['offset'].get_eval_val()[0]
        elif param.name in 'offset_y':
            return source_params['offset'].get_eval_val()[1]
        elif param.name in 'angle':
            return np.remainder(source_params['rotation'].get_eval_val(), 1.0)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'symmetry':
            return self.symmetry_vals.index(params['symmetry'].val), None
        elif source_param.name == 'mipmapmode':
            return int(params['mipmap_mode'].val == 'manual'), None
        elif source_param.name == 'offset_mode':
            return self.offset_modes.index(params['offset_mode'].val), None
        elif source_param.name == 'offset':
            return [params['offset_x'].val.tolist(), params['offset_y'].val.tolist()], None
        elif source_param.name == 'rotation':
            return params['angle'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSBlurHQNode(SBSNode):
    '''
    SBS blur hq node (color or grayscale).
    '''
    default_max_intensity = 16.0
    default_intensity = 10.0 / default_max_intensity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='BlurHQ', node_func='F.blur_hq', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='high_quality', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Quality'))
        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Blur_HQ', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Blur_HQ'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='blur_hq_grayscale', path='sbs://blur_hq.sbs')
        else:
            return SBSNodeDefinition(graph='blur_hq', path='sbs://blur_hq.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Blur_HQ': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Blur_HQ': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSNonUniformBlurNode(SBSNode):
    '''
    SBS non-uniform blur node (color or grayscale).
    '''
    default_intensity = 0.2
    default_max_intensity = 50.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NonUniformBlur', node_func='F.non_uniform_blur', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))
        self.add_input(SBSNodeInput(name='img_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Effect'))

        self.add_param(SBSNodeParameter(name='samples', val=4, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Samples'))
        self.add_param(SBSNodeParameter(name='blades', val=5, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Blades'))
        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='anisotropy', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Anisotropy'))
        self.add_param(SBSNodeParameter(name='asymmetry', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Asymmetry'))
        self.add_param(SBSNodeParameter(name='angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: np.remainder(p, 1.0), name_xml='Angle')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Non_Uniform_Blur', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Non_Uniform_Blur'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='non_uniform_blur_grayscale', path='sbs://non_uniform_blur.sbs')
        else:
            return SBSNodeDefinition(graph='non_uniform_blur', path='sbs://non_uniform_blur.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Non_Uniform_Blur': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Non_Uniform_Blur': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name in 'angle':
            return np.remainder(source_params['Angle'].get_eval_val(), 1.0)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        elif source_param.name == 'Angle':
            return params['angle'].val.tolist(), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSBevelNode(SBSNode):
    '''
    SBS bevel node.
    '''
    corner_types = ['Round', 'Angular']

    default_smoothing = 0.0
    default_max_smoothing = 5.0
    default_normal_intensity = 0.2
    default_max_normal_intensity = 50.0
    default_max_dist = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Bevel', node_func='F.bevel', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))
        self.add_input(SBSNodeInput(name='custom_curve', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='custom_curve'))

        self.add_param(SBSNodeParameter(name='non_uniform_blur_flag', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='non_uniform_blur'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='corner_type', val='Round', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda type_idx: self.corner_types[type_idx], name_xml='bevel_mode')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='use_custom_curve', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Use_Custom_Curve'))
        self.add_param(SBSNodeParameter(name='dist', val=to_zero_one(0.5), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_zero_one(self.default_max_dist), name_xml='distance')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_dist', val=self.default_max_dist, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_zero_one(self.default_max_dist), name_xml='distance')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='smoothing', val=self.default_smoothing, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_smoothing), name_xml='smoothing')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_smoothing', val=self.default_max_smoothing, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_smoothing), name_xml='smoothing')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='normal_intensity', val=self.default_normal_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_normal_intensity), name_xml='normal_intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_normal_intensity', val=self.default_max_normal_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_normal_intensity), name_xml='normal_intensity')) # <- REPLACE convert_func with actual convert_func
        # changed convert_func in line above from intensity_helper to max_intensity_helper (might have been an error in the original match code)

        self.add_output(SBSNodeOutput(name='height', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='height'))
        self.add_output(SBSNodeOutput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='normal'))

    def definition(self):
        return SBSNodeDefinition(graph='bevel', path='sbs://bevel.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'custom_curve': SBSParamType.ENTRY_GRAYSCALE.value}, {'height': SBSParamType.ENTRY_GRAYSCALE.value, 'normal': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'corner_type':
            return self.corner_types[source_params['bevel_mode'].get_eval_val()]
        elif param.name in 'dist':
            return intensity_cond_zero_one(source_params['distance'].get_eval_val(), self.default_max_dist)
        elif param.name == 'max_dist':
            return max_intensity_cond_zero_one(source_params['distance'].get_eval_val(), self.default_max_dist)
        elif param.name in 'smoothing':
            return intensity_cond(source_params['smoothing'].get_eval_val(), self.default_max_smoothing)
        elif param.name == 'max_smoothing':
            return max_intensity_cond(source_params['smoothing'].get_eval_val(), self.default_max_smoothing)
        elif param.name in 'normal_intensity':
            return intensity_cond(source_params['normal_intensity'].get_eval_val(), self.default_max_normal_intensity)
        elif param.name == 'max_normal_intensity':
            return max_intensity_cond(source_params['normal_intensity'].get_eval_val(), self.default_max_normal_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'bevel_mode':
            return self.corner_types.index(params['corner_type'].val), None
        elif source_param.name == 'distance':
            return intensity_uncond_zero_one(params['dist'].val.tolist(), params['max_dist'].val), None
        elif source_param.name == 'smoothing':
            return intensity_uncond(params['smoothing'].val.tolist(), params['max_smoothing'].val), None
        elif source_param.name == 'normal_intensity':
            return intensity_uncond(params['normal_intensity'].val.tolist(), params['max_normal_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSSlopeBlurNode(SBSNode):
    '''
    SBS slope blur node (color or grayscale).
    '''
    default_max_intensity = 16.0
    default_intensity = 10.0 / default_max_intensity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='SlopeBlur', node_func='F.slope_blur', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))
        self.add_input(SBSNodeInput(name='img_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Effect'))

        self.add_param(SBSNodeParameter(name='samples', val=8, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Samples'))
        self.add_param(SBSNodeParameter(name='mode', val='blur', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'min' if p == 6 else 'max', name_xml='mode')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Slope_Blur', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Slope_Blur'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='slope_blur_grayscale_2', path='sbs://slope_blur.sbs')
        else:
            return SBSNodeDefinition(graph='slope_blur', path='sbs://slope_blur.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Slope_Blur': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Slope_Blur': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'mode':
            return 'min' if source_params['mode'].get_eval_val() == 6 else 'max'
        elif param.name == 'intensity':
            return intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'mode':
            return 6 if params['mode'].val == 'min' else 0, None # TODO: check if 0 is correct here or what other value the 'mode' parameter can take on
        elif source_param.name == 'Intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSMosaicNode(SBSNode):
    '''
    SBS mosaic node (color or grayscale).
    '''
    default_intensity = 0.5
    default_max_intensity = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Mosaic', node_func='F.mosaic', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))
        self.add_input(SBSNodeInput(name='img_mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Effect'))

        self.add_param(SBSNodeParameter(name='samples', val=8, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Samples'))
        self.add_param(SBSNodeParameter(name='intensity', val=self.default_intensity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Mosaic', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Mosaic'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='mosaic_grayscale', path='sbs://mosaic.sbs')
        else:
            return SBSNodeDefinition(graph='mosaic', path='sbs://mosaic.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Mosaic': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'img_mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Mosaic': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'intensity':
            return intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSAutoLevelsNode(SBSNode):
    '''
    SBS auto levels node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='AutoLevels', node_func='F.auto_levels', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='quality', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Quality'))

        self.add_output(SBSNodeOutput(name='Output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Output'))

    def definition(self):
        return SBSNodeDefinition(graph='auto_levels', path='sbs://auto_levels.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Output': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSAmbientOcclusionNode(SBSNode):
    '''
    SBS ambient occlusion node.
    '''
    default_max_spreading = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='AmbientOcclusion', node_func='F.ambient_occlusion', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='spreading', val=0.15, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_spreading), name_xml='spreading')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_spreading', val=self.default_max_spreading, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_spreading), name_xml='spreading')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='equalizer', val=[0.0, 0.0, 0.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='equalizer'))
        self.add_param(SBSNodeParameter(name='levels_param', val=[0.0, 0.5, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='levels'))

        self.add_output(SBSNodeOutput(name='ambient_occlusion', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='ambient_occlusion'))

    def definition(self):
        return SBSNodeDefinition(graph='ambient_occlusion_2', path='sbs://ambient_occlusion_2.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'ambient_occlusion': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'spreading':
            return intensity_cond(source_params['spreading'].get_eval_val(), self.default_max_spreading)
        elif param.name == 'max_spreading':
            return max_intensity_cond(source_params['spreading'].get_eval_val(), self.default_max_spreading)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'spreading':
            return intensity_uncond(params['spreading'].val.tolist(), params['max_spreading'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSHBAONode(SBSNode):
    '''
    SBS hbao node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HBAO', node_func='F.hbao', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='quality', val=8, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='samples'))
        self.add_param(SBSNodeParameter(name='depth', val=0.1, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='height_depth'))
        self.add_param(SBSNodeParameter(name='height_scale', val=1.0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='height_depth_cm'))
        self.add_param(SBSNodeParameter(name='surface_size', val=1.0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='surface_size'))
        self.add_param(SBSNodeParameter(name='radius', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='radius'))
        self.add_param(SBSNodeParameter(name='gpu_optim', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='gpu_optim'))
        self.add_param(SBSNodeParameter(name='use_world_units', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='use_world_units'))
        self.add_param(SBSNodeParameter(name='non_square', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='non_square'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='hbao', path='sbs://hbao.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSHighpassNode(SBSNode):
    '''
    SBS highpass node.
    '''
    default_max_radius = 64.0
    default_radius = 6.0 / default_max_radius

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Highpass', node_func='F.highpass', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='radius', val=self.default_radius, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_radius), name_xml='Radius')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_radius', val=self.default_max_radius, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_radius), name_xml='Radius')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Highpass', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Highpass'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='highpass_grayscale', path='sbs://highpass.sbs')
        else:
            return SBSNodeDefinition(graph='highpass', path='sbs://highpass.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Highpass': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Highpass': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'radius':
            return intensity_cond(source_params['Radius'].get_eval_val(), self.default_max_radius)
        elif param.name == 'max_radius':
            return max_intensity_cond(source_params['Radius'].get_eval_val(), self.default_max_radius)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Radius':
            return intensity_uncond(params['radius'].val.tolist(), params['max_radius'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSNormalNormalizeNode(SBSNode):
    '''
    SBS normal combine node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalNormalize', node_func='F.normal_normalize', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normal'))

        self.add_output(SBSNodeOutput(name='Normalise', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normalise'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_normalise', path='sbs://normal_normalise.sbs')

    def signatures(self):
        return [
            ({'normal': SBSParamType.ENTRY_COLOR.value}, {'Normalise': SBSParamType.ENTRY_COLOR.value})]

class SBSNormalCombineNode(SBSNode):
    '''
    SBS normal combine node.
    '''
    combine_mode = ['whiteout', 'channel_mixer', 'detail_oriented']

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalCombine', node_func='F.normal_combine', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='normal_one', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input'))
        self.add_input(SBSNodeInput(name='normal_two', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input_1'))

        self.add_param(SBSNodeParameter(name='mode', val='whiteout', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: self.combine_mode[p], name_xml='blend_quality')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='normal'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_combine', path='sbs://normal_combine.sbs')

    def signatures(self):
        return [
            ({'normal_one': SBSParamType.ENTRY_COLOR.value, 'normal_two': SBSParamType.ENTRY_COLOR.value}, {'normal': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'mode':
            return self.combine_mode[source_params['blend_quality'].get_eval_val()]
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'blend_quality':
            return self.combine_mode.index(params['mode'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSChannelMixerNode(SBSNode):
    '''
    SBS normal combine node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='ChannelMixer', node_func='F.channel_mixer', output_res=output_res, use_alpha=use_alpha)

        scale_func = lambda p: [i/400.0+0.5 for i in p]

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='monochrome', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='monochrome'))
        self.add_param(SBSNodeParameter(name='red', val=[0.75, 0.5, 0.5, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=scale_func, name_xml='red_channel')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='green', val=[0.5, 0.75, 0.5, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=scale_func, name_xml='green_channel')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='blue', val=[0.5, 0.5, 0.75, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=scale_func, name_xml='blue_channel')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Output'))

    def definition(self):
        return SBSNodeDefinition(graph='channel_mixer', path='sbs://channel_mixer.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name in ['red', 'green', 'blue']:
            return [i/400.0+0.5 for i in next(iter(source_params.values())).get_eval_val()]
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name in ['red_channel', 'green_channel', 'blue_channel']:
            return [(i-0.5)*400.0 for i in next(iter(params.values())).val.tolist()], None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSRGBASplitNode(SBSNode):
    '''
    SBS rgba_split node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='RGBASplit', node_func='F.rgba_split', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='rgba', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGBA'))

        self.add_output(SBSNodeOutput(name='R', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='R'))
        self.add_output(SBSNodeOutput(name='G', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='G'))
        self.add_output(SBSNodeOutput(name='B', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='B'))
        self.add_output(SBSNodeOutput(name='A', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='A'))

    def definition(self):
        return SBSNodeDefinition(graph='rgba_split', path='sbs://rgba_split.sbs')

    def signatures(self):
        return [
            ({'rgba': SBSParamType.ENTRY_COLOR.value}, {'R': SBSParamType.ENTRY_GRAYSCALE.value, 'G': SBSParamType.ENTRY_GRAYSCALE.value, 'B': SBSParamType.ENTRY_GRAYSCALE.value, 'A': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSRGBAMergeNode(SBSNode):
    '''
    SBS rgba_merge node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='RGBAMerge', node_func='F.rgba_merge', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='r', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='R'))
        self.add_input(SBSNodeInput(name='g', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='G'))
        self.add_input(SBSNodeInput(name='b', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='B'))
        self.add_input(SBSNodeInput(name='a', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='A'))

        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))

        self.add_output(SBSNodeOutput(name='RGBA_Merge', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGBA_Merge'))

    def definition(self):
        return SBSNodeDefinition(graph='rgba_merge', path='sbs://rgba_merge.sbs')

    def signatures(self):
        return [
            ({'r': SBSParamType.ENTRY_GRAYSCALE.value, 'g': SBSParamType.ENTRY_GRAYSCALE.value, 'b': SBSParamType.ENTRY_GRAYSCALE.value, 'a': SBSParamType.ENTRY_GRAYSCALE.value}, {'RGBA_Merge': SBSParamType.ENTRY_COLOR.value})]

class SBSMultiSwitchNode(SBSNode):
    '''
    SBS multi switch node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='MultiSwitch', node_func='F.multi_switch', output_res=output_res, use_alpha=use_alpha)

        for i in range(20):
            self.add_input(SBSNodeInput(name=f'input_{i+1}', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml=f'input_{i+1}'))

        self.add_param(SBSNodeParameter(name='input_number', val=2, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='input_number'))
        self.add_param(SBSNodeParameter(name='input_selection', val=1, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='input_selection'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='multi_switch_grayscale', path='sbs://blend_switch.sbs')
        else:
            return SBSNodeDefinition(graph='multi_switch', path='sbs://blend_switch.sbs')

    def signatures(self):
        return [
            ({f'input_{i+1}': SBSParamType.ENTRY_GRAYSCALE.value for i in range(20)}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({f'input_{i+1}': SBSParamType.ENTRY_COLOR.value for i in range(20)}, {'output': SBSParamType.ENTRY_COLOR.value})]


# TODO: this is basecolor_metallic_roughness_to_diffuse_specular_glossiness in pbr_converter, which is used most often (6213 sbs files), also add basecolor_metallic_roughness_converter, which is used less often (82 sbs files)
class SBSPbrConverterNode(SBSNode):
    '''
    SBS pbr_converter node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='PbrConverter', node_func='F.pbr_converter', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='base_color', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='basecolor'))
        self.add_input(SBSNodeInput(name='metallic', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='metallic'))
        self.add_input(SBSNodeInput(name='roughness', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='roughness'))

        self.add_output(SBSNodeOutput(name='diffuse', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='diffuse'))
        self.add_output(SBSNodeOutput(name='specular', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='specular'))
        self.add_output(SBSNodeOutput(name='glossiness', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='glossiness'))

    def definition(self):
        return SBSNodeDefinition(graph='basecolor_metallic_roughness_to_diffuse_specular_glossiness', path='sbs://pbr_converter.sbs')

    def signatures(self):
        return [
            ({'base_color': SBSParamType.ENTRY_COLOR.value, 'metallic': SBSParamType.ENTRY_GRAYSCALE.value, 'roughness': SBSParamType.ENTRY_GRAYSCALE.value}, {'diffuse': SBSParamType.ENTRY_COLOR.value, 'specular': SBSParamType.ENTRY_COLOR.value, 'glossiness': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSAlphaSplitNode(SBSNode):
    '''
    SBS alpha_split node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='AlphaSplit', node_func='F.alpha_split', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='rgba', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGBA'))

        self.add_output(SBSNodeOutput(name='RGB', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGB'))
        self.add_output(SBSNodeOutput(name='A', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='A'))

    def definition(self):
        return SBSNodeDefinition(graph='rgb-a_split', path='sbs://rgb-a_split.sbs')

    def signatures(self):
        return [
            ({'rgba': SBSParamType.ENTRY_COLOR.value}, {'RGB': SBSParamType.ENTRY_COLOR.value, 'A': SBSParamType.ENTRY_GRAYSCALE.value})]

class SBSAlphaMergeNode(SBSNode):
    '''
    SBS alpha_blend node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='AlphaMerge', node_func='F.alpha_merge', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='rgb', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGB'))
        self.add_input(SBSNodeInput(name='a', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='A'))

        self.add_output(SBSNodeOutput(name='RGB-A_Merge', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='RGB-A_Merge'))

    def definition(self):
        return SBSNodeDefinition(graph='rgb-a_merge', path='sbs://rgb-a_merge.sbs')

    def signatures(self):
        return [
            ({'rgb': SBSParamType.ENTRY_COLOR.value, 'a': SBSParamType.ENTRY_GRAYSCALE.value}, {'RGB-A_Merge': SBSParamType.ENTRY_COLOR.value})]

class SBSSwitchNode(SBSNode):
    '''
    SBS switch node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Switch', node_func='F.switch', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_1', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input_1'))
        self.add_input(SBSNodeInput(name='img_2', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input_2'))

        self.add_param(SBSNodeParameter(name='flag', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='switch'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='switch_grayscale', path='sbs://blend_switch.sbs')
        else:
            return SBSNodeDefinition(graph='switch', path='sbs://blend_switch.sbs')

    def signatures(self):
        return [
            ({'img_1': SBSParamType.ENTRY_GRAYSCALE.value, 'img_2': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_1': SBSParamType.ENTRY_COLOR.value, 'img_2': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSNormalBlendNode(SBSNode):
    '''
    SBS normal blend node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalBlend', node_func='F.normal_blend', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='normal_fg', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='NormalFG'))
        self.add_input(SBSNodeInput(name='normal_bg', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='NormalBG'))
        self.add_input(SBSNodeInput(name='mask', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='Mask'))

        self.add_param(SBSNodeParameter(name='use_mask', val=True, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: True if int(p) else False, name_xml='Use_Mask')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='opacity', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Opacity'))

        self.add_output(SBSNodeOutput(name='Normal_Blend', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normal_Blend'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_blend', path='sbs://normal_blend.sbs')

    def signatures(self):
        return [
            ({'normal_fg': SBSParamType.ENTRY_COLOR.value, 'normal_bg': SBSParamType.ENTRY_COLOR.value, 'mask': SBSParamType.ENTRY_GRAYSCALE.value}, {'Normal_Blend': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'use_mask':
            return True if int(source_params['Use_Mask'].get_eval_val()) else False
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Use_Mask':
            return int(params['use_mask'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSMirrorNode(SBSNode):
    '''
    SBS normal blend node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Mirror', node_func='F.mirror', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='mirror_axis', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='mirror_type'))
        self.add_param(SBSNodeParameter(name='offset', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml=['axis_x', 'axis_y']))
        self.add_param(SBSNodeParameter(name='invert_axis', val=False, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=['invert_x', 'invert_y']))
        self.add_param(SBSNodeParameter(name='corner_type', val=0, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='corner_type'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='mirror_grayscale', path='sbs://mirror.sbs')
        else:
            return SBSNodeDefinition(graph='mirror', path='sbs://mirror.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSHeightToNormalNode(SBSNode):
    '''
    SBS height to normal world units node.
    '''
    default_max_surface_size = 1000.0
    default_surface_size = 300.0 / default_max_surface_size
    default_max_height_depth = 100.0
    default_height_depth = 16.0 / default_max_height_depth

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='HeightToNormal', node_func='F.height_to_normal_world_units', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='normal_format')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='sampling_mode', val='standard', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'sobel' if p else 'standard', name_xml='sampling')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='surface_size', val=self.default_surface_size, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_surface_size), name_xml='surface_size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_surface_size', val=self.default_max_surface_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_surface_size), name_xml='surface_size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='height_depth', val=self.default_height_depth, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_height_depth), name_xml='height_depth')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_height_depth', val=self.default_max_height_depth, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_height_depth), name_xml='height_depth')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='height_to_normal_world_units_2', path='sbs://height_to_normal_world_units.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        elif param.name == 'sampling_mode':
            return 'sobel' if source_params['sampling'].get_eval_val() else 'standard'
        elif param.name == 'surface_size':
            return intensity_cond(source_params['surface_size'].get_eval_val(), self.default_max_surface_size)
        elif param.name == 'max_surface_size':
            return max_intensity_cond(source_params['surface_size'].get_eval_val(), self.default_max_surface_size)
        elif param.name == 'height_depth':
            return intensity_cond(source_params['height_depth'].get_eval_val(), self.default_max_height_depth)
        elif param.name == 'max_height_depth':
            return max_intensity_cond(source_params['height_depth'].get_eval_val(), self.default_max_height_depth)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        elif source_param.name == 'sampling':
            return int(params['sampling_mode'].val == 'sobel'), None
        elif source_param.name == 'surface_size':
            return intensity_uncond(params['surface_size'].val.tolist(), params['max_surface_size'].val), None
        elif source_param.name == 'height_depth':
            return intensity_uncond(params['height_depth'].val.tolist(), params['max_height_depth'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSNormalToHeightNode(SBSNode):
    '''
    SBS normal to height node.
    '''
    default_max_opacity = 1.0
    default_opacity = 0.36 / default_max_opacity

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalToHeight', node_func='F.normal_to_height', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='normal_format')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='relief_balance', val=[0.5, 0.5, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Relief_Balance'))
        self.add_param(SBSNodeParameter(name='opacity', val=self.default_opacity, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_opacity), name_xml='global_opacity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_opacity', val=self.default_max_opacity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_opacity), name_xml='global_opacity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='height', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='height'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_to_height', path='sbs://normal_to_height_2.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'height': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        elif param.name == 'opacity':
            return intensity_cond(source_params['global_opacity'].get_eval_val(), self.default_max_opacity)
        elif param.name == 'max_opacity':
            return max_intensity_cond(source_params['global_opacity'].get_eval_val(), self.default_max_opacity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        elif source_param.name == 'global_opacity':
            return intensity_uncond(params['opacity'].val.tolist(), params['max_opacity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSCurvatureSmoothNode(SBSNode):
    '''
    SBS curvature smooth node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='CurvatureSmooth', node_func='F.curvature_smooth', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='normal_format')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='height', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='height'))

    def definition(self):
        return SBSNodeDefinition(graph='curvature_smooth', path='sbs://curvature_smooth.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'height': SBSParamType.ENTRY_GRAYSCALE.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSMakeItTilePatchNode(SBSNode):
    '''
    SBS make it tile patch node.
    '''
    default_max_mask_size = 1.0
    default_max_mask_precision = 1.0
    default_max_mask_warping = 100.0
    default_max_pattern_size = 1000.0
    default_max_disorder = 1.0
    default_max_size_variation = 100.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='MakeItTilePatch', node_func='F.make_it_tile_patch', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='octave', val=3, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='Octave'))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='mask_size', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_size), name_xml='Mask_Size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_size', val=self.default_max_mask_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_size), name_xml='Mask_Size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_precision', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_precision', val=self.default_max_mask_precision, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_warping', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_warping', val=self.default_max_mask_warping, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='pattern_width', val=0.2, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_pattern_size), name_xml='Pattern_size_width')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_pattern_width', val=self.default_max_pattern_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_pattern_size), name_xml='Pattern_size_width')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='pattern_height', val=0.2, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_pattern_size), name_xml='Pattern_size_height')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_pattern_height', val=self.default_max_pattern_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_pattern_size), name_xml='Pattern_size_height')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='disorder', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_disorder), name_xml='Disorder')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_disorder', val=self.default_max_disorder, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_disorder), name_xml='Disorder')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='size_variation', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_size_variation), name_xml='Size_Variation')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_size_variation', val=self.default_max_size_variation, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_size_variation), name_xml='Size_Variation')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='rotation', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: to_zero_one(p / 360), name_xml='Rotation')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='rotation_variation', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda p: p / 360, name_xml='Rotation_Variation')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='background_color', val=[0.0, 0.0, 0.0, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Background_Color'))
        self.add_param(SBSNodeParameter(name='color_variation', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=lambda x, y, s: x, name_xml=['Color_Variation', 'Luminosity_Variation'])) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Make_It_Tile_Patch', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Make_It_Tile_Patch'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='make_it_tile_patch_grayscale', path='sbs://make_it_tile_patch_grayscale.sbs')
        else:
            return SBSNodeDefinition(graph='make_it_tile_patch', path='sbs://make_it_tile_patch.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Make_It_Tile_Patch': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Make_It_Tile_Patch': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        elif param.name == 'mask_size':
            return intensity_cond(source_params['Mask_Size'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'max_mask_size':
            return max_intensity_cond(source_params['Mask_Size'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'mask_precision':
            return intensity_cond(source_params['Mask_Precision'].get_eval_val(), self.default_max_mask_precision)
        elif param.name == 'max_mask_precision':
            return max_intensity_cond(source_params['Mask_Precision'].get_eval_val(), self.default_max_mask_precision)
        elif param.name == 'mask_warping':
            return intensity_cond_zero_one(source_params['Mask_Warping'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'max_mask_warping':
            return max_intensity_cond_zero_one(source_params['Mask_Warping'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'pattern_width':
            return intensity_cond(source_params['Pattern_size_width'].get_eval_val(), self.default_max_pattern_size)
        elif param.name == 'max_pattern_width':
            return max_intensity_cond(source_params['Pattern_size_width'].get_eval_val(), self.default_max_pattern_size)
        elif param.name == 'pattern_height':
            return intensity_cond(source_params['Pattern_size_height'].get_eval_val(), self.default_max_pattern_size)
        elif param.name == 'max_pattern_height':
            return max_intensity_cond(source_params['Pattern_size_height'].get_eval_val(), self.default_max_pattern_size)
        elif param.name == 'disorder':
            return intensity_cond(source_params['Disorder'].get_eval_val(), self.default_max_disorder)
        elif param.name == 'max_disorder':
            return max_intensity_cond(source_params['Disorder'].get_eval_val(), self.default_max_disorder)
        elif param.name == 'size_variation':
            return intensity_cond(source_params['Size_Variation'].get_eval_val(), self.default_max_size_variation)
        elif param.name == 'max_size_variation':
            return max_intensity_cond(source_params['Size_Variation'].get_eval_val(), self.default_max_size_variation)
        elif param.name == 'rotation':
            return to_zero_one(source_params['Rotation'].get_eval_val() / 360)
        elif param.name == 'rotation_variation':
            return source_params['Rotation_Variation'].get_eval_val() / 360
        elif param.name == 'color_variation':
            return first_valid(source_params).get_eval_val()
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        elif source_param.name == 'Mask_Size':
            return intensity_uncond(params['mask_size'].val.tolist(), params['max_mask_size'].val), None
        elif source_param.name == 'Mask_Precision':
            return intensity_uncond(params['mask_precision'].val.tolist(), params['max_mask_precision'].val), None
        elif source_param.name == 'Mask_Warping':
            return intensity_uncond_zero_one(params['mask_warping'].val.tolist(), params['max_mask_warping'].val), None
        elif source_param.name == 'Pattern_size_width':
            return intensity_uncond(params['pattern_width'].val.tolist(), params['max_pattern_width'].val), None
        elif source_param.name == 'Pattern_size_height':
            return intensity_uncond(params['pattern_height'].val.tolist(), params['max_pattern_height'].val), None
        elif source_param.name == 'Disorder':
            return intensity_uncond(params['disorder'].val.tolist(), params['max_disorder'].val), None
        elif source_param.name == 'Size_Variation':
            return intensity_uncond(params['size_variation'].val.tolist(), params['max_size_variation'].val), None
        elif source_param.name == 'Rotation':
            return from_zero_one(params['rotation'].val.tolist()) * 360, None
        elif source_param.name == 'Rotation_Variation':
            return params['rotation_variation'].val.tolist() * 360, None
        elif source_param.name == 'Color_Variation':
            return params['color_variation'].val.tolist(), None
        elif source_param.name == 'Luminosity_Variation':
            return None, None # TODO: probably not used? or does it depend on the input image type (grayscale or color)?
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSMakeItTilePhotoNode(SBSNode):
    '''
    SBS make it tile photo node.
    '''
    default_max_mask_warping = 100.0
    default_max_mask_size = 1.0
    default_max_mask_precision = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='MakeItTilePhoto', node_func='F.make_it_tile_photo', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='mask_warping_x', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_warping_x', val=self.default_max_mask_warping, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_warping_y', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping_V')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_warping_y', val=self.default_max_mask_warping, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper_zero_one(self.default_max_mask_warping), name_xml='Mask_Warping_V')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_size_x', val=0.1, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_size), name_xml='Mask_Size_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_size_x', val=self.default_max_mask_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_size), name_xml='Mask_Size_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_size_y', val=0.1, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_size), name_xml='Mask_Size_V')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_size_y', val=self.default_max_mask_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_size), name_xml='Mask_Size_V')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_precision_x', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_precision_x', val=self.default_max_mask_precision, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision_H')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='mask_precision_y', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision_V')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_mask_precision_y', val=self.default_max_mask_precision, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_mask_precision), name_xml='Mask_Precision_V')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Make_It_Tile_Photo', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Make_It_Tile_Photo'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='make_it_tile_photo_grayscale', path='sbs://make_it_tile_photo_grayscale.sbs')
        else:
            return SBSNodeDefinition(graph='make_it_tile_photo', path='sbs://make_it_tile_photo.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Make_It_Tile_Photo': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Make_It_Tile_Photo': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'mask_warping_x':
            return intensity_cond_zero_one(source_params['Mask_Warping_H'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'max_mask_warping_x':
            return max_intensity_cond_zero_one(source_params['Mask_Warping_H'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'mask_warping_y':
            return intensity_cond_zero_one(source_params['Mask_Warping_V'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'max_mask_warping_y':
            return max_intensity_cond_zero_one(source_params['Mask_Warping_V'].get_eval_val(), self.default_max_mask_warping)
        elif param.name == 'mask_size_x':
            return intensity_cond(source_params['Mask_Size_H'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'max_mask_size_x':
            return max_intensity_cond(source_params['Mask_Size_H'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'mask_size_y':
            return intensity_cond(source_params['Mask_Size_V'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'max_mask_size_y':
            return max_intensity_cond(source_params['Mask_Size_V'].get_eval_val(), self.default_max_mask_size)
        elif param.name == 'mask_precision_x':
            return intensity_cond(source_params['Mask_Precision_H'].get_eval_val(), self.default_max_mask_precision)
        elif param.name == 'max_mask_precision_x':
            return max_intensity_cond(source_params['Mask_Precision_H'].get_eval_val(), self.default_max_mask_precision)
        elif param.name == 'mask_precision_y':
            return intensity_cond(source_params['Mask_Precision_V'].get_eval_val(), self.default_max_mask_precision)
        elif param.name == 'max_mask_precision_y':
            return max_intensity_cond(source_params['Mask_Precision_V'].get_eval_val(), self.default_max_mask_precision)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Mask_Warping_H':
            return intensity_uncond_zero_one(params['mask_warping_x'].val.tolist(), params['max_mask_warping_x'].val), None
        elif source_param.name == 'Mask_Warping_V':
            return intensity_uncond_zero_one(params['mask_warping_y'].val.tolist(), params['max_mask_warping_y'].val), None
        elif source_param.name == 'Mask_Size_H':
            return intensity_uncond(params['mask_size_x'].val.tolist(), params['max_mask_size_x'].val), None
        elif source_param.name == 'Mask_Size_V':
            return intensity_uncond(params['mask_size_y'].val.tolist(), params['max_mask_size_y'].val), None
        elif source_param.name == 'Mask_Precision_H':
            return intensity_uncond(params['mask_precision_x'].val.tolist(), params['max_mask_precision_x'].val), None
        elif source_param.name == 'Mask_Precision_V':
            return intensity_uncond(params['mask_precision_y'].val.tolist(), params['max_mask_precision_y'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSReplaceColorNode(SBSNode):
    '''
    SBS replace color
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='ReplaceColor', node_func='F.replace_color', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='source_color', val=[0.5, 0.5, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='SourceColor'))
        self.add_param(SBSNodeParameter(name='target_color', val=[0.5, 0.5, 0.5], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='TargetColor'))

        self.add_output(SBSNodeOutput(name='ToTargetColor', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='ToTargetColor'))

    def definition(self):
        return SBSNodeDefinition(graph='replace_color', path='sbs://replace_color.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'ToTargetColor': SBSParamType.ENTRY_COLOR.value})]

class SBSNormalColorNode(SBSNode):
    '''
    SBS normal color node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalColor', node_func='F.normal_color', output_res=output_res, use_alpha=use_alpha)

        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='Invert_Y')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='num_imgs', val=1, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='use_alpha', val=self.use_alpha, dtype=None, trainable=False, val_min=None, val_max=None, name_xml=[]))
        self.add_param(SBSNodeParameter(name='direction', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Direction'))
        self.add_param(SBSNodeParameter(name='slope_angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Slope_Angle'))

        self.add_output(SBSNodeOutput(name='Normal_Color', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normal_Color'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_color', path='sbs://normal_color.sbs')

    def signatures(self):
        return [
            ({}, {'Normal_Color': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['Invert_Y'].get_eval_val() else 'dx'
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Invert_Y':
            return int(params['normal_format'].val == 'gl'), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSNormalVectorRotationNode(SBSNode):
    '''
    SBS normal vector rotation node.
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='NormalVectorRotation', node_func='F.normal_vector_rotation', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normal'))
        self.add_input(SBSNodeInput(name='rotation_map', dtype=SBSParamType.ENTRY_GRAYSCALE.value, name_xml='rotation_map'))
        
        self.add_param(SBSNodeParameter(name='normal_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='normal_format'))
        self.add_param(SBSNodeParameter(name='rotation_angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='rotation_angle'))

        self.add_output(SBSNodeOutput(name='Normal', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='Normal'))

    def definition(self):
        return SBSNodeDefinition(graph='normal_vector_rotation', path='sbs://normal_vector_rotation.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'rotation_map': SBSParamType.ENTRY_GRAYSCALE.value}, {'Normal': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'normal_format':
            return 'gl' if source_params['normal_format'].get_eval_val() else 'dx'
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'normal_format':
            return int(params['normal_format'].val == 'gl'), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSVectorMorphNode(SBSNode):
    '''
    SBS vector morph node.
    '''
    default_max_amount = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='VectorMorph', node_func='F.vector_morph', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))
        self.add_input(SBSNodeInput(name='vector_field', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='vector_field'))

        self.add_param(SBSNodeParameter(name='amount', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_amount), name_xml='amount')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_amount', val=self.default_max_amount, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_amount), name_xml='amount')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='vector_morph_grayscale', path='sbs://vector_morph.sbs')
        else:
            return SBSNodeDefinition(graph='vector_morph', path='sbs://vector_morph.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'vector_field': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'vector_field': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'amount':
            return intensity_cond(source_params['amount'].get_eval_val(), self.default_max_amount)
        elif param.name == 'max_amount':
            return max_intensity_cond(source_params['amount'].get_eval_val(), self.default_max_amount)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'amount':
            return intensity_uncond(params['amount'].val.tolist(), params['max_amount'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSVectorWarpNode(SBSNode):
    '''
    SBS vector warp node.
    '''
    default_max_intensity = 1.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='VectorWarp', node_func='F.vector_warp', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))
        self.add_input(SBSNodeInput(name='vector_map', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='vector_map'))

        self.add_param(SBSNodeParameter(name='vector_format', val='dx', dtype=None, trainable=False, val_min=None, val_max=None, convert_func=lambda p: 'gl' if p else 'dx', name_xml='vector_format')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='intensity', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='intensity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='vector_warp_grayscale', path='sbs://vector_warp.sbs')
        else:
            return SBSNodeDefinition(graph='vector_warp', path='sbs://vector_warp.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value, 'vector_map': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value, 'vector_map': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'vector_format':
            return 'gl' if source_params['vector_format'].get_eval_val() else 'dx'
        elif param.name == 'intensity':
            return intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'vector_format':
            return int(params['vector_format'].val == 'gl'), None
        elif source_param.name == 'intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSContrastLuminosityNode(SBSNode):
    '''
    SBS contrast luminosity node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='ContrastLuminosity', node_func='F.contrast_luminosity', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='contrast', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=to_zero_one, name_xml='Contrast')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='luminosity', val=to_zero_one(0.0), dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=to_zero_one, name_xml='Luminosity')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Contrast_Luminosity', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Contrast_Luminosity'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='contrast_luminosity_grayscale', path='sbs://contrast_luminosity.sbs')
        else:
            return SBSNodeDefinition(graph='contrast_luminosity', path='sbs://contrast_luminosity.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Contrast_Luminosity': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Contrast_Luminosity': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'contrast':
            return to_zero_one(source_params['Contrast'].get_eval_val())
        elif param.name == 'luminosity':
            return to_zero_one(source_params['Luminosity'].get_eval_val())
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Contrast':
            return from_zero_one(params['contrast'].val.tolist()), None
        elif source_param.name == 'Luminosity':
            return from_zero_one(params['luminosity'].val.tolist()), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSP2SNode(SBSNode):
    '''
    SBS pre-multiplied to straight node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='P2S', node_func='F.p2s', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='input'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='premult_to_straight', path='sbs://pre_multiply_straight.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSS2PNode(SBSNode):
    '''
    SBS straight to pre-multiplied node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='S2P', node_func='F.s2p', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='input'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_COLOR.value, name_xml='output'))

    def definition(self):
        return SBSNodeDefinition(graph='straight_to_premult', path='sbs://pre_multiply_straight.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSClampNode(SBSNode):
    '''
    SBS clamp node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Clamp', node_func='F.clamp', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='clamp_alpha', val=True, dtype=None, trainable=False, val_min=None, val_max=None, name_xml='apply_to_alpha'))
        self.add_param(SBSNodeParameter(name='low', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='min'))
        self.add_param(SBSNodeParameter(name='high', val=1.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='max'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='clamp_grayscale', path='sbs://clamp.sbs')
        else:
            return SBSNodeDefinition(graph='clamp', path='sbs://clamp.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSPowNode(SBSNode):
    '''
    SBS pow node
    '''
    default_max_exponent = 10.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Pow', node_func='F.pow', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_param(SBSNodeParameter(name='exponent', val=0.4, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_exponent), name_xml='exponent')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_exponent', val=self.default_max_exponent, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_exponent), name_xml='exponent')) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='pow_grayscale', path='sbs://pow.sbs')
        else:
            return SBSNodeDefinition(graph='pow', path='sbs://pow.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'exponent':
            return intensity_cond(source_params['exponent'].get_eval_val(), self.default_max_exponent)
        elif param.name == 'max_exponent':
            return max_intensity_cond(source_params['exponent'].get_eval_val(), self.default_max_exponent)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'exponent':
            return intensity_uncond(params['exponent'].val.tolist(), params['max_exponent'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSQuantizeNode(SBSNode):
    '''
    SBS quantize node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Quantize', node_func='F.quantize', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Input'))

        self.add_param(SBSNodeParameter(name='quantize_number', val=3, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=self.update_quantize_number, name_xml=['Quantize', 'Quantize_R', 'Quantize_G', 'Quantize_B', 'Quantize_A'])) # <- REPLACE convert_func with actual convert_func

        self.add_output(SBSNodeOutput(name='Quantize', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Quantize'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='quantize_grayscale', path='sbs://quantize.sbs')
        else:
            return SBSNodeDefinition(graph='quantize', path='sbs://quantize.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Quantize': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Quantize': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'quantize_number':
            if len(source_params) == 4:
                return [
                    source_params['Quantize_R'].get_eval_val(),
                    source_params['Quantize_G'].get_eval_val(),
                    source_params['Quantize_B'].get_eval_val(),
                    source_params['Quantize_A'].get_eval_val()]
            elif len(source_params) == 3:
                return [
                    source_params['Quantize_R'].get_eval_val(),
                    source_params['Quantize_G'].get_eval_val(),
                    source_params['Quantize_B'].get_eval_val()]
            elif len(source_params) == 1:
                return source_params['Quantize'].get_eval_val()
            else:
                raise RuntimeError('Unexpected number of source parameters for a quantize node.')
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Quantize':
            if not isinstance(params['quantize_number'].val, list):
                return params['quantize_number'].val, None
            else:
                return None, None
        elif source_param.name in ['Quantize_R', 'Quantize_G', 'Quantize_B']:
            if isinstance(params['quantize_number'].val, list):
                channel_dict = {'Quantize_R': 0, 'Quantize_G': 1, 'Quantize_B': 2}
                return params['quantize_number'].val[channel_dict[source_param.name]], None
            else:
                return None, None
        elif source_param.name == 'Quantize_A':
            if isinstance(params['quantize_number'].val, list) and len(params['quantize_number'].val) >= 4:
                return params['quantize_number'].val[3], None
            else:
                return None, None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

    def update_quantize_number(self, new_val, exist_val, sbs_name):
        channel_dict = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
        if sbs_name == 'Quantize':
            return new_val
        else:
            idx = channel_dict[sbs_name[-1]]

            if not isinstance(exist_val, list):
                exist_val = [exist_val] * 4 if idx == 3 else [exist_val] * 3
                exist_val[idx] = new_val
            elif len(exist_val) == 3 and idx == 3:
                exist_val.append(new_val)
            else:
                exist_val[idx] = new_val

            return exist_val

class SBSAnisotropicBlurNode(SBSNode):
    '''
    SBS anisotropic blur node
    '''
    default_max_intensity = 16.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='AnisotropicBlur', node_func='F.anisotropic_blur', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='high_quality', val=False, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=bool, name_xml='Quality')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='intensity', val=0.625, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_intensity', val=self.default_max_intensity, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_intensity), name_xml='Intensity')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='anisotropy', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Anisotropy'))
        self.add_param(SBSNodeParameter(name='angle', val=0.0, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Angle'))

        self.add_output(SBSNodeOutput(name='Anisotropic_Blur', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Anisotropic_Blur'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='anisotropic_blur_grayscale', path='sbs://anisotropic_blur.sbs')
        else:
            return SBSNodeDefinition(graph='anisotropic_blur', path='sbs://anisotropic_blur.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Anisotropic_Blur': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Anisotropic_Blur': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'high_quality':
            return bool(source_params['Quality'].get_eval_val())
        elif param.name == 'intensity':
            return intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        elif param.name == 'max_intensity':
            return max_intensity_cond(source_params['Intensity'].get_eval_val(), self.default_max_intensity)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Quality':
            return int(params['high_quality'].val), None
        elif source_param.name == 'Intensity':
            return intensity_uncond(params['intensity'].val.tolist(), params['max_intensity'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSGlowNode(SBSNode):
    '''
    SBS glow node
    '''
    default_max_size = 20.0

    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Glow', node_func='F.glow', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Source'))

        self.add_param(SBSNodeParameter(name='glow_amount', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Glow_Amount'))
        self.add_param(SBSNodeParameter(name='clear_amount', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Clear_Amount'))
        self.add_param(SBSNodeParameter(name='size', val=0.5, dtype=None, trainable=True, val_min=0.0, val_max=1.0, convert_func=intensity_helper(self.default_max_size), name_xml='Glow_Size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='max_size', val=self.default_max_size, dtype=None, trainable=False, val_min=None, val_max=None, convert_func=max_intensity_helper(self.default_max_size), name_xml='Glow_Size')) # <- REPLACE convert_func with actual convert_func
        self.add_param(SBSNodeParameter(name='color', val=[1.0, 1.0, 1.0, 1.0], dtype=None, trainable=True, val_min=0.0, val_max=1.0, name_xml='Glow_Color'))

        self.add_output(SBSNodeOutput(name='Glow', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='Glow'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='glow_grayscale', path='sbs://glow.sbs')
        else:
            return SBSNodeDefinition(graph='glow', path='sbs://glow.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'Glow': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'Glow': SBSParamType.ENTRY_COLOR.value})]

    def condition_param_val(self, param, source_params):
        if param.name == 'size':
            return intensity_cond(source_params['Glow_Size'].get_eval_val(), self.default_max_size)
        elif param.name == 'max_size':
            return max_intensity_cond(source_params['Glow_Size'].get_eval_val(), self.default_max_size)
        else:
            return super().condition_param_val(param=param, source_params=source_params)

    def uncondition_param_val(self, source_param, params):
        if source_param.name == 'Glow_Size':
            return intensity_uncond(params['size'].val.tolist(), params['max_size'].val), None
        else:
            return super().uncondition_param_val(source_param=source_param, params=params)

class SBSCar2PolNode(SBSNode):
    '''
    SBS cartesian to polar node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Car2Pol', node_func='F.car2pol', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='cartesian_to_polar_grayscale', path='sbs://cartesian_polar_transformation.sbs')
        else:
            return SBSNodeDefinition(graph='cartesian_to_polar', path='sbs://cartesian_polar_transformation.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]

class SBSPol2CarNode(SBSNode):
    '''
    SBS polar to cartesian node
    '''
    def __init__(self, name, output_res=None, use_alpha=False):
        super().__init__(name=name, node_type='Pol2Car', node_func='F.pol2car', output_res=output_res, use_alpha=use_alpha)

        self.add_input(SBSNodeInput(name='img_in', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='input'))

        self.add_output(SBSNodeOutput(name='output', dtype=SBSParamType.ENTRY_VARIANT.value, name_xml='output'))

    def definition(self):
        if self.outputs[0].dtype == SBSParamType.ENTRY_GRAYSCALE.value:
            return SBSNodeDefinition(graph='polar_to_cartesian_grayscale', path='sbs://cartesian_polar_transformation.sbs')
        else:
            return SBSNodeDefinition(graph='polar_to_cartesian', path='sbs://cartesian_polar_transformation.sbs')

    def signatures(self):
        return [
            ({'img_in': SBSParamType.ENTRY_GRAYSCALE.value}, {'output': SBSParamType.ENTRY_GRAYSCALE.value}),
            ({'img_in': SBSParamType.ENTRY_COLOR.value}, {'output': SBSParamType.ENTRY_COLOR.value})]
