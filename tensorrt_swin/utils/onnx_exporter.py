# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import copy
import os

import torch
import torch.onnx
import torchvision

from . import onnx_meta, trt_weight_converter, trt_attribute_converter
from . import module_wrapper
from .utils import check_trt_has_layer_norm_builtin
from .utils import check_trt_has_layer_norm_plugin


def export_to_onnx(model, input_data, onnx_name, use_swin_extension=False,
                   meta_input_data=None, **kwargs):

    # Determine the input for the meta model
    if meta_input_data is None:
        meta_input_data = input_data

    # Copy the model to avoid modifying the original one
    meta_model = copy.deepcopy(model)

    # In some cases we need to wrap certain nn.Modules to make them compatible
    # with the ONNX tracer or the TensorRT plugin. In this dict, we define which
    # wrappers should be used for which modules.
    module_wrappers = {

    }

    # In the following, we modify the model to make it compatible with TRT: We
    # replace some modules with a custom ONNX op (own block in the graph).
    # Additionally, we convert the weights to a format which is compatible
    # with the TRT plugin.

    # These is the full list of modules, which will be replaced to a ONNX
    # custom op.
    replace_blocks = []
    if use_swin_extension:
        replace_blocks.extend([
            torchvision.models.swin_transformer.SwinTransformerBlock,
            torchvision.models.swin_transformer.PatchMerging,
            torchvision.models.swin_transformer.SwinTransformerBlockV2,
            torchvision.models.swin_transformer.PatchMergingV2,
        ])

    # The Swin TRT plugin needs the weights to be converted to a specific
    # format, which is done by the following converters.
    trt_weight_converters = []
    if use_swin_extension:
        trt_weight_converters.extend([
            trt_weight_converter.TRTSwinTransformerBlockWeightConverter,
            trt_weight_converter.TRTSwinTransformerBlockV2WeightConverter
        ])

    trt_attribute_converters = []
    if use_swin_extension:
        trt_attribute_converters.extend([
            trt_attribute_converter.TRTSwinTransformerBlockAttributeConverter,
            trt_attribute_converter.TRTSwinTransformerBlockV2AttributeConverter,
            trt_attribute_converter.LayerNormAttributeConverter
        ])

    # Moreover, we need to handle the LayerNorm operation in a special way,
    # based on the TensorRT version:
    # - TensorRT < 8.5: LayerNorm is not supported at all
    # - TensorRT 8.5.x: LayerNorm is supported via plugin, but only works
    #   correctly for tensors with ndim == 3
    # - TensorRT >= 8.6: LayerNorm is supported as builtin operation
    if not check_trt_has_layer_norm_builtin():
        # verify that the TensorRT plugin is available
        if not check_trt_has_layer_norm_plugin():
            raise RuntimeError(
                "TensorRT has no builtin LayerNorm and no plugin available. "
                "Please consider using a newer version of TensorRT or install "
                "the LayerNorm plugin. For further details see:\n"
                "https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/layerNormPlugin"
            )

        # To match the TensorRT plugin, we need to replace the LayerNorm module
        replace_blocks.append(torch.nn.LayerNorm)

        # LayerNorm plugin is available, make sure that ndim == 3
        module_wrappers[torch.nn.LayerNorm] = \
            module_wrapper.LayerNormReshapeWrapper

    # In some situations, the ONNX tracer cannot identify the shapes for
    # the AdaptiveAvgPool2d module. To fix this, we need to wrap the module
    # and replace it with a static version.
    if not use_swin_extension:
        module_wrappers[torch.nn.AdaptiveAvgPool2d] = \
            module_wrapper.AdaptiveAvgPool2dWrapper

    module_wrapper._replace_modules_with_wrappers(
        meta_model, module_wrappers
    )

    # Patch the model - note that this model is not usable for inference
    # in PyTorch anymore.

    # For tracing we always use bs 1
    patched_model = onnx_meta.custom_model_patch(meta_model,
                                                 meta_input_data,
                                                 replace_blocks,
                                                 weights_as_attributes=False,
                                                 weight_converters=trt_weight_converters,
                                                 attribute_converters=trt_attribute_converters)

    # Determine the dynamic axes
    if 'dynamic_axes' not in kwargs:
        dynamic_axes = {}
        if 'input_names' in kwargs:
            for input_name in kwargs['input_names']:
                dynamic_axes[input_name] = {0: 'batch_size'}
        if 'output_names' in kwargs:
            for output_name in kwargs['output_names']:
                dynamic_axes[output_name] = {0: 'batch_size'}
        kwargs['dynamic_axes'] = dynamic_axes

    # Set the onnx opset version to at least 17 for LayerNorm support
    if 'opset_version' in kwargs:
        if kwargs['opset_version'] < 17:
            raise RuntimeError("The opset version must be at least 17 for "
                               "LayerNorm support.")
    else:
        kwargs['opset_version'] = 17

    # If blocks were replace with custom ops, we need to set the custom opset
    custom_opsets = {}
    if len(replace_blocks) > 0:
        custom_opsets['metaonnx'] = 1

    torch.onnx.export(
        patched_model, input_data, onnx_name,
        verbose=False, do_constant_folding=True,
        custom_opsets=custom_opsets,
        **kwargs
    )
    assert os.path.isfile(onnx_name)
