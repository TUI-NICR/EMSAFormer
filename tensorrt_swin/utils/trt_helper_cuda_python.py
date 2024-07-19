# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import List, Dict, Optional, Union

import tensorrt as trt
import torch

from .utils import assert_trt_plugins_loaded

# common_runtime.py is copied from the TensorRT samples and unmodified:
# https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.1/samples/python/common_runtime.py
from .common_runtime import do_inference
from .common_runtime import allocate_buffers


class TRTModel:
    def __init__(self, engine_path, profile_idx: Optional[int] = None):
        assert_trt_plugins_loaded()
        self._engine_path = engine_path
        self._profile_idx = profile_idx

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self._engine_path)
        self.context = self.engine.create_execution_context()

        # get input/output names/shapes
        self._input_specs = []
        self._output_specs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)

            if self._profile_idx is None:
                shape = self.engine.get_tensor_shape(name)
            else:
                shape = self._engine.get_tensor_profile_shape(
                    name, self._profile_idx
                )[-1]   # max shape

            # convert to simple tuple
            shape = tuple(shape)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_specs.append({'name': name, 'shape': shape})
            else:
                self._output_specs.append({'name': name, 'shape': shape})

        # allocate buffers
        # note: compared to Polygraphy this is one only once
        self._input_buffers, self._output_buffers, self._bindings, self._stream = \
            allocate_buffers(self.engine)

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def __call__(
        self,
        x_input: Union[torch.Tensor,
                       List[torch.Tensor],
                       Dict[str, torch.Tensor]]
    ):
        if isinstance(x_input, torch.Tensor):
            input_list = [x_input]
        elif isinstance(x_input, list):
            input_list = x_input
        elif isinstance(x_input, dict):
            # assume that for each input tensor a key exists in x_input
            input_list = [x_input[spec['name']] for spec in self._input_specs]

        # sanity check
        assert len(input_list) == len(self._input_buffers)

        # copy input tensors to host buffers
        for buf, x in zip(self._input_buffers, input_list):
            buf.host = x.cpu().numpy()

        # run inference
        # notes:
        # - do_inference will also copy input host buffers to input device
        #   buffers and output device buffers to output host buffers
        # - do_inference is returning a list of output host buffers
        trt_outputs = do_inference(
            self.context,
            engine=self.engine,
            bindings=self._bindings,
            inputs=self._input_buffers,
            outputs=self._output_buffers,
            stream=self._stream,
        )

        # sanity check
        assert len(trt_outputs) == len(self._output_specs)

        # prepare output, assume we want return a dict (most applications)
        # reshape/view is required because the output buffers are flat
        outputs = {
            spec['name']: torch.from_numpy(output).view(spec['shape'])
            for spec, output in zip(self._output_specs, trt_outputs)
        }

        if isinstance(x_input, list):
            # we assume that the output should be a list as well
            # note, python 3.7+ guarantees that the order of dict items is
            # preserved
            return list(outputs.values())
        elif isinstance(x_input, torch.Tensor):
            # we assume that the output should be a tensor
            assert len(outputs) == 1
            return list(outputs.values())[0]

        return outputs
