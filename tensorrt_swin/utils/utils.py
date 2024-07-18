# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import ctypes
from packaging.version import Version
import operator
import os

import tensorrt as trt
import torch


def load_trt_plugin(path: str):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    handle = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    if not handle:
        raise RuntimeError(f"Fail to load plugin library: {path}")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def get_trt_version():
    return Version(trt.__version__)


def check_trt_version(version: str, compare_operator: str):
    operator_map = {
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
        '==': operator.eq
    }
    assert compare_operator in operator_map, f"Operator {compare_operator} is not supported."
    trt_version = get_trt_version()
    compare_version = Version(version)

    return operator_map[compare_operator](trt_version, compare_version)


def check_trt_has_layer_norm_builtin():
    # TensorRT supports LayerNorm from 8.6.x.x onwards as builtin op (export
    # requires ONNX opset 17)
    # Note: TensorRT 8.5.x also support LayerNorm via plugin; however, this
    # plugin only works correctly with ndim == 3
    return check_trt_version('8.6.0', '>=')


def check_trt_has_layer_norm_plugin():
    # init plugin registry
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()

    # check registered plugins for LayerNorm
    creator_list = registry.plugin_creator_list
    for creator in creator_list:
        if creator.name == 'LayerNorm':
            return True

    return False


def load_torch_plugin(path: str):
    torch.classes.load_library(path)


def get_faster_transformer_path():
    # Find the path of the plugin library
    faster_transformer_path = os.environ.get('FASTER_TRANSFORMER_PATH', None)
    if faster_transformer_path is None or not os.path.isdir(faster_transformer_path):
        # Assume that the script is in the examples/swin_imagenet directory
        faster_transformer_path = os.path.abspath(
            os.path.join(__file__, '..', '..', 'FasterTransformer')
        )
        assert os.path.isdir(
            faster_transformer_path), f'FASTER_TRANSFORMER_PATH={faster_transformer_path} is not a directory'
    return faster_transformer_path


def get_trt_swin_plugin_path():
    faster_transformer_path = get_faster_transformer_path()
    plugin_swin = f"{faster_transformer_path}/build/lib/libswinTransformer_plugin.so"
    return plugin_swin


def get_torch_swin_plugin_path():
    faster_transformer_path = get_faster_transformer_path()
    th_path = f"{faster_transformer_path}/build/lib/libpyt_swintransformer.so"
    return th_path


def get_trt_swin_gemm_tool_path():
    faster_transformer_path = get_faster_transformer_path()
    swin_gemm_path = f"{faster_transformer_path}/build/bin/swin_gemm"
    return swin_gemm_path


TRT_PLUGINS_LOADED = False


def load_trt_plugins():
    global TRT_PLUGINS_LOADED
    if TRT_PLUGINS_LOADED:
        return
    plugin_swin = get_trt_swin_plugin_path()
    load_trt_plugin(plugin_swin)
    TRT_PLUGINS_LOADED = True


def assert_trt_plugins_loaded():
    global TRT_PLUGINS_LOADED
    if not TRT_PLUGINS_LOADED:
        raise RuntimeError(
            "TRT plugins are not loaded. Call load_trt_plugins() first."
        )


TORCH_PLUGINS_LOADED = False


def load_torch_plugins():
    global TORCH_PLUGINS_LOADED
    if TORCH_PLUGINS_LOADED:
        return
    th_path = get_torch_swin_plugin_path()
    load_trt_plugin(th_path)
    TORCH_PLUGINS_LOADED = True


def assert_torch_plugins_loaded():
    global TORCH_PLUGINS_LOADED
    if not TORCH_PLUGINS_LOADED:
        raise RuntimeError(
            "Torch plugins are not loaded. Call load_torch_plugins() first."
        )
