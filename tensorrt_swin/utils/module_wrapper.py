# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch


def _replace_modules_with_wrappers(module, module_wrappers):
    for name, child in module.named_children():
        if isinstance(child, tuple(module_wrappers.keys())):
            wrapper_to_use = module_wrappers[type(child)]
            setattr(module, name, wrapper_to_use(child))
        else:
            _replace_modules_with_wrappers(child, module_wrappers)


class LayerNormReshapeWrapper(torch.nn.Module):
    def __init__(self, layernorm):
        super().__init__()

        assert isinstance(layernorm, torch.nn.LayerNorm)
        assert len(layernorm.normalized_shape) == 1
        self.layernorm = layernorm

    def forward(self, x):
        # reshape to 3D tensor, if input is 4D
        if x.dim() == 4:
            x_ = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        else:
            x_ = x

        # apply layernorm
        x_ = self.layernorm(x_)

        # reshape back to 4D tensor, if input was 4D
        if x.dim() == 4:
            x_ = x_.view(x.shape)

        return x_


class AdaptiveAvgPool2dWrapper(torch.nn.Module):
    def __init__(self, avgpool):
        super().__init__()

        assert isinstance(avgpool, torch.nn.AdaptiveAvgPool2d)
        self.avgpool = avgpool

        self._replaced = False

    def forward(self, x):
        # apply pooling
        y = self.avgpool(x)

        if not self._replaced:
            # compute pooling parameters (assumes BCHW) to match
            # AdaptiveAvgPool2d
            stride_hw = [x.shape[i] // y.shape[i] for i in (-2, -1)]
            kernel_size = [x.shape[i] - (y.shape[i] - 1) * stride_hw[k]
                           for k, i in enumerate((-2, -1))]

            # replace the avgpool with a static avgpool
            self.avgpool = torch.nn.AvgPool2d(kernel_size, stride=stride_hw)

            self._replaced = True

        return y
