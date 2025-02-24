# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import abc
import operator
import math
import copy

import numpy as np

from .sbs_param_type import SBSParamType, param_type_defaults, param_short_to_type#, param_type_to_tag
from .sbs_utils import resolve_dependency_path, UnsupportedNodeExpansionError
from . import io_sbs

class SBSFunctionGraph:
    '''
    The wrapper class of a SBS function graph parser for a dynamic node parameter.
    '''
    def __init__(self, name):

        self.name = name

        # nodes
        self.nodes = []
        self.root_node = None

        # inputs
        self.inputs = []

        # Initialize dictionaries for constants and functions
        self.op_dict = None
        self.init_dicts()

    @staticmethod
    def load_sbs(function_name=None, filename=None, fgraph_xml=None, dependencies_by_uid=None, use_abs_paths=True):

        fgraph = io_sbs.load_sbs_function_graph(
            function_name=function_name, filename=filename, fgraph_xml=fgraph_xml, dependencies_by_uid=dependencies_by_uid, use_abs_paths=use_abs_paths)

        return fgraph

    def set_root_node(self, node):
        if node not in self.nodes:
            raise RuntimeError('The root node must be chosen from among the nodes of the parameter.')
        self.root_node = node

    def add_node(self, node):
        if node in self.nodes:
            raise RuntimeError('The given node already exists in this function graph.')
        self.nodes.append(node)

    # this does not remove inputs that reference the node
    # (would need to store a list of referencing nodes, like the children list of the SBSNodeOutput,
    # or go through all nodes, which is expensive)
    def remove_node(self, node):
        if node == self.root_node:
            raise RuntimeError('Cannot remove the root node, need to change to a different root node first.')
        index = [ni for ni, n in enumerate(self.nodes) if n == node]
        if len(index) == 0:
            raise RuntimeError('The given node is not in this function graph.')
        elif len(index) > 1:
            raise RuntimeError('This node has duplicates in the function graph.')
        index = index[0]
        del self.nodes[index]

    # get a list of unique dependencies
    def dependencies(self):
        deps = set()
        for node in self.nodes:
            if node.definition() is not None:
                deps.add(node.definition())
        deps = list(deps)

        return deps

    # def update_node_definition_paths(self, old_subpath, new_subpath):
    #     for node in self.nodes:
    #         if node.definition() is not None and node.definition().path is not None:
    #             if old_subpath in node.definition().path:
    #                 node.definition().path.replace(old_subpath, new_subpath)

    def init_dicts(self):
        '''
        Initialize the dictionaries for type lookup.
        '''
        # Default value dictionary for constant nodes
        # Format: tag -> default value

        # Helpers for function data
        def swizzle(x, d):
            d = np.array(d)
            if not isinstance(x, np.ndarray):
                x = np.array([x])
            ret = np.where(d < len(x), x[np.minimum(d, len(x) - 1)], 0.0)
            return ret if ret.size > 1 else ret.item()

        vector = lambda x, y: np.hstack((x, y))
        array_cast_float = lambda x: x.astype(np.float64)
        array_cast_int = lambda x: x.astype(np.int64)

        swizzle_data = lambda dim: {'data': list(range(dim)) if dim > 1 else 0, 'func': swizzle, 'input_seq': ['vector']}
        vector_data = {'data':None, 'func': vector, 'input_seq': ['componentsin', 'componentslast']}
        lerp_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['a', 'b', 'x']}
        ifelse_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['condition', 'ifpath', 'elsepath']}
        unary_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['a']}
        binary_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['a', 'b']}
        cast_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['value']}
        passthrough_op_data = lambda func: {'data': None, 'func': func, 'input_seq': ['input']}

        # Default function data dictionary for math operation nodes (add as needed)
        # Format: tag -> (default data, default func, input name sequence)
        #      or tag -> {dtype1: (...), dtype2: (...)}
        self.op_dict = {}

        # Swizzle op
        for i in (1, 2, 3, 4):
            self.op_dict[f'swizzle{i}'] = swizzle_data(i)
            self.op_dict[f'iswizzle{i}'] = swizzle_data(i)

        # Vector op
        for i in (2, 3, 4):
            self.op_dict[f'vector{i}'] = vector_data
            self.op_dict[f'ivector{i}'] = vector_data

        # Cast op
        self.op_dict['tofloat'] = cast_op_data(float)
        self.op_dict['tointeger'] = cast_op_data(int)
        self.op_dict['toint1'] = cast_op_data(int)
        for i in (2, 3, 4):
            self.op_dict[f'tofloat{i}'] = cast_op_data(array_cast_float)
            self.op_dict[f'tointeger{i}'] = cast_op_data(array_cast_int)
            self.op_dict[f'toint{i}'] = cast_op_data(array_cast_int)

        # lerp
        lerp = lambda a, b, x: a + (b-a)*x
        ifelse = lambda condition, ifpath, elsepath: ifpath if condition else elsepath
        dotprod = lambda x, y: np.dot(x, y)
        pow2 = lambda x: 2**x
        passthrough = lambda x: x
        mulscalar = lambda a, b: a * b

        def rand(x):
            print('Warning: random function in node parameter.')
            return np.random.uniform(high=x)

        # Other ops
        # (for full list see https://docs.substance3d.com/sddoc/function-nodes-overview-102400052.html)
        self.op_dict.update({
            'add': binary_op_data(operator.add),
            'sub': binary_op_data(operator.sub),
            'mul': binary_op_data(operator.mul),
            'mulscalar': binary_op_data(mulscalar),
            'div': {SBSParamType.INTEGER1.value: binary_op_data(operator.floordiv),
                    SBSParamType.INTEGER2.value: binary_op_data(operator.floordiv),
                    SBSParamType.INTEGER3.value: binary_op_data(operator.floordiv),
                    SBSParamType.INTEGER4.value: binary_op_data(operator.floordiv),
                    SBSParamType.FLOAT1.value: binary_op_data(operator.truediv),
                    SBSParamType.FLOAT2.value: binary_op_data(operator.truediv),
                    SBSParamType.FLOAT3.value: binary_op_data(operator.truediv),
                    SBSParamType.FLOAT4.value: binary_op_data(operator.truediv)},
            'neg': unary_op_data(operator.neg),
            'mod': binary_op_data(operator.mod),
            'dot': binary_op_data(dotprod),

            'and': binary_op_data(operator.and_),
            'or': binary_op_data(operator.or_),
            'not': unary_op_data(operator.not_),

            'eq': binary_op_data(operator.eq),
            'noteq': binary_op_data(operator.ne),
            'gt': binary_op_data(operator.gt),
            'gteq': binary_op_data(operator.ge),
            'lr': binary_op_data(operator.lt),
            'lreq': binary_op_data(operator.le),

            'abs': unary_op_data(abs),
            'floor': unary_op_data(math.floor),
            'ceil': unary_op_data(math.ceil),
            'cos': unary_op_data(math.cos),
            'sin': unary_op_data(math.sin),
            'tan': unary_op_data(math.tan),
            'atan': unary_op_data(math.atan),
            'atan2': unary_op_data(math.atan2),
            'sqrt': unary_op_data(math.sqrt),
            'log': unary_op_data(math.log),
            'exp': unary_op_data(math.exp),
            'pow2': unary_op_data(pow2),
            'lerp': lerp_op_data(lerp),
            'min': binary_op_data(min),
            'max': binary_op_data(max),

            'rand': unary_op_data(rand),
            # missing: sequence
            'ifelse': ifelse_op_data(ifelse),

            'passthrough': passthrough_op_data(passthrough),
        })

    def lookup_node_type(self, node_type, dtype):
        '''
        Obtain class name, default data, and function via node tag
        '''
        if node_type.startswith('const_'):
            node_class = SBSFunctionConstant
            node_args = {'data': param_type_defaults[param_short_to_type[node_type[len('const_'):]]]}
        elif node_type.startswith('get_'):
            node_class = SBSFunctionInput
            node_args = {'data': param_type_defaults[param_short_to_type[node_type[len('get_'):]]]}
        else:
            if node_type not in self.op_dict:
                raise NotImplementedError(f'Function node \'{node_type}\' is not supported.')

            node_class = SBSFunctionOp
            ret = self.op_dict[node_type]
            node_args = ret[dtype] if node_type =='div' else ret
        return node_class, node_args

    def parse_param_val(self, param_value_node):
        '''
        Extract the default value of a parameter input according to its type ID.
        '''
        param_value_ = param_value_node
        param_tag = param_value_.tag
        if param_tag in ['constantValueInt32', 'constantValueInt1']:
            param_val = int(io_sbs.load_sbs_attribute_value(param_value_))
        elif param_tag in ['constantValueInt2', 'constantValueInt3', 'constantValueInt4']:
            param_val = [int(i) for i in io_sbs.load_sbs_attribute_value(param_value_).strip().split()]
        elif param_tag == 'constantValueFloat1':
            param_val = float(io_sbs.load_sbs_attribute_value(param_value_))
        elif param_tag in ['constantValueFloat2', 'constantValueFloat3', 'constantValueFloat4']:
            param_val = [float(i) for i in io_sbs.load_sbs_attribute_value(param_value_).strip().split()]
        elif param_tag == 'constantValueBool':
            param_val = bool(int(io_sbs.load_sbs_attribute_value(param_value_)))
        elif param_tag == 'constantValueString':
            param_val = io_sbs.load_sbs_attribute_value(param_value_)
        else:
            raise TypeError('Unknown parameter type')
        return param_val

    def param_val_str(self, val, dtype):
        if dtype in [
                SBSParamType.FLOAT2.value,
                SBSParamType.FLOAT3.value,
                SBSParamType.FLOAT4.value,
                SBSParamType.INTEGER2.value,
                SBSParamType.INTEGER3.value,
                SBSParamType.INTEGER4.value]:
            return ' '.join([str(x) for x in val])
        elif dtype in [
                SBSParamType.INTEGER1.value,
                SBSParamType.FLOAT1.value]:
            return str(val)
        elif dtype in [
                SBSParamType.BOOLEAN.value]:
            return str(int(val))
        elif dtype in [
                SBSParamType.STRING.value]:
            return val
        else:
            raise RuntimeError(f'Unsupporte parameter value type: {SBSParamType(dtype).name}')

    def expand_input_node(self, ex_node, new_val):

        if ex_node not in self.nodes:
            raise RuntimeError('The given function node is not in the function graph.')

        if not isinstance(ex_node, SBSFunctionInput):
            raise RuntimeError('The expanded node must be an SBSFunctionInput')

        if isinstance(new_val, SBSFunctionGraph):
            # the new value is a function graph, expand the function input node with the function graph

            new_val = copy.deepcopy(new_val)
            new_val_root_node = new_val.root_node
            new_val_nodes = list(new_val.nodes)
        else:
            # the new value is a constant, convert the function input node to a constant node

            if not ex_node.type.startswith('get_'):
                raise RuntimeError(f'Found function input node with an unexpected node type: {ex_node.type}.')

            # generate new parameters
            node_type = f'const_{ex_node.type[len("get_"):]}'
            node_dtype = ex_node.dtype

            # create new constant value node
            new_val_root_node = SBSFunctionConstant(dtype=node_dtype, node_type=node_type, definition=None, data=new_val)
            new_val_nodes = [new_val_root_node]

        # if a connection in the parameter of the node definition graph pointed at the function input node, point it at the root node instead
        for other_node in self.nodes:
            for other_node_input in list(other_node.inputs): # copy the list of inputs since I will be mutating it
                if other_node_input.parent == ex_node:
                    other_node_input.connect(parent=new_val_root_node)

        # update function nodes of the parameter in the node definition graph
        # (remove the input function nodes, and add all function nodes in the parameter of the expanded node)
        for new_val_node in new_val_nodes:
            self.add_node(new_val_node)
        if self.root_node == ex_node:
            self.set_root_node(new_val_root_node)
        self.remove_node(ex_node)

    def expand_node(self, ex_node, filename, resource_dirs, use_abs_paths):

        if ex_node not in self.nodes:
            raise RuntimeError('The given function node is not in the function graph.')

        # print(f'expanding function node of type {ex_node.definition().graph}')

        # load function graph that defines the function node type
        if ex_node.definition() is None:
            raise RuntimeError('Can''t expand an atomic function node.')
        expanded_node_dep_path = resolve_dependency_path(path=ex_node.definition().path, source_filename=filename, resource_dirs=resource_dirs)

        try:
            if os.path.splitext(expanded_node_dep_path)[1] == '.sbsar':
                raise NotImplementedError('Cannot expand a function node that is defined by a compiled substance graph (a file ending in .sbsar).')
            node_def_fgraph = SBSFunctionGraph.load_sbs(function_name=ex_node.definition().graph, filename=expanded_node_dep_path, use_abs_paths=use_abs_paths)
        except NotImplementedError as err:
            raise UnsupportedNodeExpansionError(f'Error while loading the definition graph of a function graph node with type: {ex_node.type}):\n  {str(err)}') from err

        # ex_node.inputs
        ex_node_input_names = [inp.name for inp in ex_node.inputs]

        # set input nodes in the definition fgraph that don't have a corresponding input in the expanded node to their default values
        # skip input nodes that start with $, these refer to global parameters that  will only be resolved when the graph is run
        for node in list(node_def_fgraph.nodes): # copy list since we will be mutating it
            if isinstance(node, SBSFunctionInput) and not node.data.startswith('$'):
                if node.data not in ex_node_input_names:
                    # input node in the definition fgraph that doesn't have a corresponding input in the expanded node
                    node_def_fgraph_input_dict = {p.name: p.val for p in node_def_fgraph.inputs}
                    if node.data not in node_def_fgraph_input_dict:
                        raise RuntimeError(f'Could not find default value for parameter {node.data} in graph {node_def_fgraph.name}.')
                    else:
                        default_val = node_def_fgraph_input_dict[node.data]
                    node_def_fgraph.expand_input_node(ex_node=node, new_val=default_val)

        # change all inputs pointing to the expanded node
        # to point to the root of the node_def_fgraph instead
        outer_nodes_connected_to_ndg = []
        for other_node in self.nodes:
            for other_node_input in list(other_node.inputs): # copy the list of inputs since I will be mutating it
                if other_node_input.parent == ex_node:
                    other_node_input.connect(parent=node_def_fgraph.root_node)
                    outer_nodes_connected_to_ndg.append(other_node)

        # change all inputs pointing to any remaining input nodes in the ndg
        # to point to the targets of the corresponding inputs of the expanded node (all should match now)
        for node in list(node_def_fgraph.nodes) + outer_nodes_connected_to_ndg:
            for node_input in list(node.inputs): # copy the list of inputs since I will be mutating it
                if isinstance(node_input.parent, SBSFunctionInput) and not node_input.parent.data.startswith('$') and node_input.parent in node_def_fgraph.nodes:
                    # node in the definition fgraph with connection to a function node
                    if node_input.parent.data not in ex_node_input_names:
                        raise RuntimeError(f'Could not find corresponding connection in expanded node for input {node_input.parent.data} in the definition graph.')
                        # this should not happen because all input nodes that do not have a corresponding connection should already have been set to their default values

                    new_parent = ex_node.inputs[ex_node_input_names.index(node_input.parent.data)].parent
                    node_input.connect(parent=new_parent)

        # remove all input nodes from the node_def_fgraph,
        # all inputs to them have been changed to point to the corresponding connection in the expanded node
        # or they have been relaced by their defaults
        for node in list(node_def_fgraph.nodes): # copy list since we will be mutating it
            if isinstance(node, SBSFunctionInput) and not node.data.startswith('$'):
                node_def_fgraph.remove_node(node)

        # add all nodes in the node definition graph
        for node in node_def_fgraph.nodes:
            self.add_node(node)
        if self.root_node == ex_node:
            self.set_root_node(node_def_fgraph.root_node)
        self.remove_node(ex_node)

    def eval(self, input_dict):
        '''
        Evaluate the result of the function graph via recursion.
        '''
        # # Fill in missing node parameters in 'get' nodes
        # for _, node in self.nodes.items():
        #     if node.tag.startswith('get_') and callable(node.data):
        #         node.data = node_param_dict[node.data()]

        # Evalulate function graph
        return param_type_cast(self.root_node.eval(input_dict), self.root_node.dtype)

class SBSFunctionNodeInput:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.fnode = None

    def disconnect(self):
        return self.connect(parent=None)

    def connect(self, parent):
        if self.fnode is None:
            raise RuntimeError('The given function node input is not attached to any function node.')

        if self.parent == parent:
            # unchanged
            return

        self.parent = parent

        return self

class SBSFunctionNodeDefinition():
    '''
    Information about the function graph that defines a function node
    '''
    def __init__(self, graph, path):
        self.graph = graph # name of the graph that defines the node
        self.path = path # path to the file that contains the definition graph

class SBSFunctionNode(abc.ABC):
    '''
    The wrapper class of a function graph node.
    '''
    def __init__(self, dtype, node_type, definition, data):

        self.dtype = dtype
        self.type = node_type
        self.data = data

        # function node inputs
        self.inputs = []

        self.defin = definition

    def definition(self):
        return self.defin
    
    def add_input(self, fnode_input):
        if fnode_input.name in [i.name for i in self.inputs]:
            raise RuntimeError('An input with the given name already exists in this function node, remove it first.')
        if fnode_input.parent is not None:
            raise RuntimeError('The given function node input has a connection, remove it first.')

        fnode_input.fnode = self
        self.inputs.append(fnode_input)

    # Obtain an input input indexed by name
    def get_input_by_name(self, name):
        for fnode_input in self.inputs:
            if fnode_input.name == name:
                return fnode_input
        raise RuntimeError(f"Input '{name}' is not found in a function node of type {self.dtype}")

    # Update function data
    def update_data(self, data):
        if isinstance(self.data, dict):
            for key, val in data:
                self.data[key] = val
        else:
            self.data = data

    @abc.abstractmethod
    def eval(self):
        '''
        Evaluate the result at the current node.
        '''
        return None

class SBSFunctionConstant(SBSFunctionNode):
    '''
    A constant value from a 'const_' function.
    '''
    def eval(self, input_dict):
        return np.array(self.data) if isinstance(self.data, list) else self.data

class SBSFunctionInput(SBSFunctionNode):
    '''
    A value from a 'get_' function, which refers to a value in the input dict.
    '''
    def eval(self, input_dict):
        if self.data not in input_dict:
            # print('WARNING: Node parameter references non-existent graph parameter, using default value instead.')
            val = param_type_defaults[param_short_to_type[self.type[len('get_'):]]]
        else:
            val = input_dict[self.data]
            if isinstance(val, SBSFunctionGraph):
                # input value is itself a function graph, evaluate it
                val = val.eval(input_dict=input_dict)
            val = param_type_cast(val, self.dtype)
        return np.array(val) if isinstance(val, list) else val

class SBSFunctionOp(SBSFunctionNode):
    '''
    An operation on one or more inputs.
    '''
    def __init__(self, dtype, node_type, definition, data, func, input_seq):
        super().__init__(dtype=dtype, node_type=node_type, definition=definition, data=data)
        self.func = func
        self.input_seq = input_seq

    def eval(self, input_dict):
        args = []
        input_names = [conn.name for conn in self.inputs]
        for name in self.input_seq:
            if name in input_names:
                args.append(self.get_input_by_name(name).parent.eval(input_dict))
            else:
                print('WARNING: Missing input in function graph, using default value.')
                args.append(param_type_defaults[self.dtype])

        if self.data is not None:
            args.append(self.data)
        return self.func(*args)

class SBSFunctionUnsupported(SBSFunctionNode):
    '''
    An function node that is currently not directly implemented,
    this function node needs to be expanded into its definition to be usable.
    '''
    def __init__(self, dtype, node_type, definition):
        super().__init__(dtype=dtype, node_type=node_type, definition=definition, data=None)

    def eval(self, input_dict):
        raise RuntimeError(f'Unsupported node: {self.definition().graph if self.definition() is not None else "<no definition>"}')

### Helper functions

def param_type_cast(val, dtype):
    '''
    Type checking and casting.
    '''
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if dtype == SBSParamType.BOOLEAN.value and not isinstance(val, bool):
        val = bool(val)
    elif dtype == SBSParamType.INTEGER1.value and not isinstance(val, int):
        val = int(val)
    elif dtype in (SBSParamType.INTEGER2.value, SBSParamType.INTEGER3.value, SBSParamType.INTEGER4.value) and isinstance(val, list) and not isinstance(val[0], int):
        val = [int(i) for i in val]
    elif dtype == SBSParamType.FLOAT1.value and not isinstance(val, float):
        val = float(val)
    elif dtype in (SBSParamType.FLOAT2.value, SBSParamType.FLOAT3.value, SBSParamType.FLOAT4.value) and isinstance(val, list) and not isinstance(val[0], float):
        val = [float(i) for i in val]
    elif dtype == SBSParamType.STRING.value and not isinstance(val, str):
        val = str(val)
    return val
