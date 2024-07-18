# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torchvision


class DefaultAttributeConverter():

    @staticmethod
    def get_converter_type():
        return torch.nn.Module

    def convert(self, module, module_attributes):
        return module_attributes


class LayerNormAttributeConverter(DefaultAttributeConverter):

    @staticmethod
    def get_converter_type():
        return torch.nn.LayerNorm

    def convert(self, module, module_attributes):
        # name epsilon parameter
        module_attributes['epsilon_f'] = module_attributes.pop('eps_f')
        # default value, we do not know the axis
        module_attributes['axis_i'] = -1
        return module_attributes


class TRTSwinTransformerBlockAttributeConverter(DefaultAttributeConverter):
    @staticmethod
    def get_converter_type():
        return torchvision.models.swin_transformer.SwinTransformerBlock

    def convert(self, module, module_attributes):
        # We compute some of the attributes which are required for the TRT
        # plugin. The default should be fine for most of the attributes,
        # but it will raise a warning if the attribute is not set.
        # In addition we are more flexible with that.
        mlp_ratio = module.mlp[0].out_features//module.mlp[0].in_features
        module_attributes['mlp_ratio_i'] = mlp_ratio

        qkv_bias = module.attn.qkv.bias is not None
        module_attributes['qkv_bias_i'] = qkv_bias

        # Swin implementation from microsoft has a optional parameter
        # qk_scale. However torchvision does not have this parameter.
        # The swin implementation of FasterTransformer expects this
        # parameter to be set to 1.0 if not specified.
        module_attributes['qk_scale_f'] = 1.0

        return module_attributes


class TRTSwinTransformerBlockV2AttributeConverter(TRTSwinTransformerBlockAttributeConverter):
    @staticmethod
    def get_converter_type():
        return torchvision.models.swin_transformer.SwinTransformerBlockV2
