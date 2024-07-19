# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import copy
import inspect

import onnx
import torch
import torch.onnx.symbolic_helper as sym_help

from . import trt_weight_converter
from . import trt_attribute_converter


class MetaONNXNode(torch.nn.Module):

    def __init__(self, orig_module,
                 weights_as_attributes, weights_dtype,
                 weight_converter=trt_weight_converter.DefaultWeightConverter(),
                 attribute_converter=trt_attribute_converter.DefaultAttributeConverter()):
        super().__init__()
        # Used to store the original module so it can be used for tracing.
        self._module = orig_module
        # Used to store the original input and output shapes which
        # is required for ONNX export.
        self._input_shape = None
        self._output_shape = None
        # Used to toggle between trace and meta mode.
        self._trace_mode = True
        # Used to store the name of the custom ONNX op.
        self._custom_onnx_name = self._module._get_name()
        self._custom_onnx_name = "metaonnx::" + self._custom_onnx_name
        self._module_meta = None
        self._module_members = self.trace_other_members(self._module,
                                                        filter_dict_by_signature=True)

        if attribute_converter is not None:
            self._module_members = attribute_converter.convert(
                copy.deepcopy(self._module),
                copy.deepcopy(self._module_members),
            )

        # If there is a custom weight converter, we use it to get weights and names.
        # Else we will just use all named_parameters
        if weight_converter is None:
            weight_converter = trt_weight_converter.DefaultWeightConverter()
        self.weight_converter = weight_converter

        if attribute_converter is not None:
            attribute_converter = trt_attribute_converter.DefaultAttributeConverter()
        self.attribute_converter = attribute_converter

        self.init_weights_done = False

        self._weights_as_attributes = weights_as_attributes
        self._weights_dtype = weights_dtype

    def init_weights_from_input(self, example_inputs):
        self._module_names, self._module_parameters = self.weight_converter.convert(
            copy.deepcopy(self._module),
            copy.deepcopy(self._module_members),
            example_inputs
        )

        # Really ugly hack so that the names of the inputs in the
        # onnx graph are still nice. This is kind a important for debugging...
        updated_attrs = []
        for name, param in zip(self._module_names, self._module_parameters):
            name = name.replace(".", "_")
            # if not isinstance(param, torch.nn.Parameter):
            param = torch.nn.Parameter(param, requires_grad=False)
            self.register_parameter(name, param)
            attr = getattr(self, name)
            updated_attrs.append(attr)
        self._module_parameters = updated_attrs

        if self._weights_as_attributes:
            for name, param in zip(self._module_names, self._module_parameters):
                dtype = self._weights_dtype
                param = param.to(dtype=dtype).detach()
                self._module_members[name + "_t"] = param

            module_type_int = 0 if dtype == torch.float32 else 1
            self._module_members["type_id_i"] = module_type_int

        self.init_weights_done = True

    def trace_other_members(self, module, filter_dict_by_signature=False):
        # This function traces all members of a module for interesting
        # attributes like int, float, bool, str, list, tuple, torch.Tensor.
        # It will return a dict with the name of the member and the value.
        # If the value is a torch.Tensor, it will be traced as well.
        # The key is adjusted for onnx export.
        members_dict = {}

        for m in dir(module):
            r = self.handle_member(module, m)
            if r is None:
                continue
            if isinstance(r, dict):
                r = {f"{m}_{k}": v for k, v in r.items() if v is not None}
                members_dict.update(r)
            else:
                name, value = r
                members_dict[name] = value

        if filter_dict_by_signature:
            signature_keys = inspect.signature(module.__init__).parameters.keys()
            new_dict = {}
            for k, v in members_dict.items():
                # check if k is partially in signature_keys
                found = False
                for s in signature_keys:
                    if s in k:
                        found = True
                        break
                if found:
                    k_strip = k.split('_')[-1]
                    new_dict[f'{s}_{k_strip}'] = v
            members_dict = new_dict

        return members_dict

    def handle_member(self, module, m, member=None):
        if m.startswith('_'):
            return None
        if member is None:
            member = getattr(module, m)
        else:
            member = module
        return_value = None
        # We want to find all other members of submodules.
        # Only types like int, float, bool, str, lists are supported.
        if isinstance(member, torch.nn.Module):
            return_value = self.trace_other_members(member)
        elif isinstance(member, torch.Tensor):
            return_value = self.trace_other_members(m)
        elif isinstance(member, bool):
            return_value = (f'{m}_i', int(member))
        elif isinstance(member, int):
            return_value = (f'{m}_i', member)
        elif isinstance(member, float):
            return_value = (f'{m}_f', member)
        elif isinstance(member, str):
            return_value = (f'{m}_s', member)
        elif isinstance(member, torch._C.Value):
            return_value = (f'{m}_v', member)
        elif isinstance(member, list) or isinstance(member, tuple):
            if len(member) > 0:
                name, _ = self.handle_member(member[0], m, member=member[0])
                return_value = (name, member)
        return return_value

    def set_trace_mode(self):
        # Set to trace mode to store the input and output shapes
        # of the original modules.
        self._trace_mode = True

    def set_meta_mode(self):
        # Set to meta mode to replace the original modules with custom ONNX ops
        self._trace_mode = False

        # Create the custom ONNX node class.
        # This is done here because the input and output shapes are only known
        # after tracing the model.
        class MetaONNXNodeFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input, *args):
                assert input.shape == self._input_shape
                return torch.randn(self._output_shape)

            @staticmethod
            def symbolic(g, input, *args):
                output = g.op(self._custom_onnx_name, input, *args, **self._module_members)

                # Determine the output shape based on the traced output shape
                # and the current input shape.
                # All axis of input which are None are treated as
                # dynamic axes in the output shape.
                input_shape = sym_help._get_tensor_sizes(input)
                target_output_shape = []
                for i, v in enumerate(input_shape):
                    if v is None:
                        target_output_shape.append(None)
                    else:
                        target_output_shape.append(self._output_shape[i])
                output_type = input.type().with_sizes(target_output_shape)

                output.setType(output_type)
                return output

            # This is a workaround for a pytorch bug in pytorch 2.0.0.
            # The bug was fixed in 2.1.0. See:
            # https://github.com/pytorch/pytorch/issues/104700
            # https://github.com/pytorch/pytorch/pull/104785/commits/a7964ada39bf1642f6fd3889df9a2b98f8a7cb9f
            # As a workaround we need to implement this function.
            # However we probabbly do wan't to ignore this for pytorch>=2.1.0.
            def to_function_proto():
                return onnx.FunctionProto(name=self._custom_onnx_name, domain='metaonnx')

        self._module_meta = MetaONNXNodeFunction.apply
        # Register the custom ONNX op with the symbolic function.
        # This makes torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        # obsolete and helps tracing dynamic shapes.
        torch.onnx.register_custom_op_symbolic(self._custom_onnx_name, MetaONNXNodeFunction, 1)

    def forward(self, x):

        if not self.init_weights_done:
            self.init_weights_from_input(x)
        # In trace mode, we just forward the input to the original module
        # and save the input and output shapes.
        if self._trace_mode:
            self._input_shape = x.shape
            x = self._module(x)
            self._output_shape = x.shape
        # In meta mode, we forward the input to the custom ONNX node.
        # This node only outputs random data but has the correct shape.
        # Because it's a auto grad function it will be it's own node in the
        # ONNX graph.
        else:
            if self._weights_as_attributes:
                x = self._module_meta(x)
            else:
                x = self._module_meta(x, *self._module_parameters)
        return x


def custom_model_patch(model, example_inputs, custom_blocks,
                       weights_as_attributes=True, weights_dtype=torch.float32,
                       weight_converters=[], attribute_converters=[]):
    # Traces a model, and replaces blocks with custom implementations if provided.
    # model: The model to be traced and patched.
    # example_inputs: The example inputs for tracing the model.
    # custom_blocks: A list of custom blocks. Each block in the list must be a torch.nn.Module.
    #                The custom blocks will be replaced with a custom ONNX op in the ONNX graph
    #                instead of the original block.
    # weights_as_attributes: If True, weights are added as attributes to the ONNX graph.
    #                        If False, weights are added as inputs to the ONNX graph.
    #                        This is required for some models, e.g. Swin Transformer
    #                        parse the weights from the attributes because they
    #                        preallocate the memory for the forward pass.
    # weights_dtype: The data type of the weights. This is required if weights_as_attributes
    #                is True, because the weights are added as attributes to the ONNX graph
    #                and the data type of the attributes is fixed (TRT can't change it after).
    # weight_converters: A list of weight converters. A weight converter is a class which
    #                    implements an interface for computing additional weights/inputs
    #                    which are required for the custom ONNX op in TRT.
    if example_inputs is not None:
        model.eval()
        model.requires_grad = False
        # Test if inference works
        _ = model(example_inputs)

    # Check if model is already a custom block
    if isinstance(model, tuple(custom_blocks)):

        # Check if converter needs to be applied
        weight_converter = None
        for converter in weight_converters:
            if isinstance(model, converter.get_converter_type()):
                weight_converter = converter()

        attribute_converter = None
        for converter in attribute_converters:
            if isinstance(model, converter.get_converter_type()):
                attribute_converter = converter()

        model = MetaONNXNode(model,
                             weights_as_attributes=weights_as_attributes,
                             weights_dtype=weights_dtype,
                             weight_converter=weight_converter,
                             attribute_converter=attribute_converter)
    else:
        # Iterate over torch model and replace every block with a custom block
        # if the block is in the custom_blocks list.
        # Search for the blocks in the model and replace them with tracer.
        for name, module in model.named_children():
            if isinstance(module, tuple(custom_blocks)):
                # Check if converter needs to be applied
                weight_converter = None
                # Apply converter if available
                for converter in weight_converters:
                    if isinstance(module, converter.get_converter_type()):
                        weight_converter = converter()

                attribute_converter = None
                for converter in attribute_converters:
                    if isinstance(module, converter.get_converter_type()):
                        attribute_converter = converter()

                # Create a meta module which will replace the original module.
                # It will not do the original computation, but instead it will
                # just return random data with the correct shape.
                # The meta module will have a trace and a meta mode.
                # The trace mode is used to store the input and output shapes
                # of the original module. The meta mode is used to replace the
                # original module with a custom ONNX op.
                # The meta module will also store the weights of the original
                # module as attributes or inputs (so its available in the ONNX graph).
                meta_module = MetaONNXNode(module,
                                           weights_as_attributes=weights_as_attributes,
                                           weights_dtype=weights_dtype,
                                           weight_converter=weight_converter,
                                           attribute_converter=attribute_converter)
                # Replace the module with the meta module
                setattr(model, name, meta_module)
            else:
                # If the module is not a custom block, we recursively call the
                # function again to check if there are custom blocks in the
                # submodules.
                module_updated = custom_model_patch(module, None, custom_blocks,
                                                    weights_as_attributes=weights_as_attributes,
                                                    weights_dtype=weights_dtype,
                                                    weight_converters=weight_converters,
                                                    attribute_converters=attribute_converters)
                setattr(model, name, module_updated)

    if example_inputs is not None:
        # Trace the model to get the input and output shapes
        _ = model(example_inputs)
        # Set to meta mode to replace the original modules with custom ONNX ops
        for module in model.modules():
            if isinstance(module, MetaONNXNode):
                module.set_meta_mode()
    return model
