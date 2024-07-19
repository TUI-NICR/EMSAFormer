# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Notes:
- matching inputs/outputs of the onnx model to pass them to the
  postprocessors is not quite stable (just a fast proof-of-concept
  implementation)
- postprocessing is always done using PyTorch (on GPU if available) and not
  much optimized so far (many operations could be done using ONNX) and, thus,
  should not be part of a timing comparison
"""
import os
import re
import subprocess
import time
import warnings

from nicr_mt_scene_analysis.data import CollateIgnoredDict
from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
import numpy as np
import torch

from emsaformer.model import EMSAFormer
from emsaformer.args import ArgParserEMSAFormer
from emsaformer.data import mt_collate
from emsaformer.data import get_datahelper
from emsaformer.data import get_dataset
from emsaformer.preprocessing import get_preprocessor
from emsaformer.visualization import visualize
from emsaformer.weights import load_weights


def _parse_args():
    parser = ArgParserEMSAFormer()
    group = parser.add_argument_group('Inference Timing')
    # add arguments
    # general
    group.add_argument(
        '--model-onnx-filepath',
        type=str,
        default=None,
        help="Path to ONNX model file when `model` is 'onnx'."
    )

    # input
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--inference-batch-size',
        type=int,
        default=1,
        help="Batch size to use for inference."
    )

    # runs
    group.add_argument(
        '--n-runs',
        type=int,
        default=100,
        help="Number of runs the inference time will be measured."
    )
    group.add_argument(
        '--n-runs-warmup',
        type=int,
        default=10,
        help="Number of forward passes through the model before the inference "
             "time measurements starts. This is necessary as the first runs "
             "are slower."
    )

    # timings
    group.add_argument(
        '--no-time-pytorch',
        action='store_true',
        default=False,
        help="Do not measure inference time using PyTorch."
    )
    group.add_argument(
        '--no-time-tensorrt',
        action='store_true',
        default=False,
        help="Do not measure inference time using TensorRT."
    )
    group.add_argument(
        '--with-postprocessing',
        action='store_true',
        default=False,
        help="Include postprocessing in timing."
    )

    # export
    group.add_argument(
        '--export-outputs',
        action='store_true',
        default=False,
        help="Whether to export the outputs of the model."
    )

    # tensorrt
    group.add_argument(
        '--trt-floatx',
        type=int,
        choices=(16, 32),
        default=32,
        help="Whether to measure with float16 or float32."
    )
    group.add_argument(
        '--trt-onnx-opset-version',
        type=int,
        default=17,
        help="Opset version to use for export."
    )
    group.add_argument(
        '--trt-do-not-force-rebuild',
        dest='trt_force_rebuild',
        action='store_false',
        default=True,
        help="Reuse existing TensorRT engine."
    )
    group.add_argument(
        '--trt-enable-dynamic-batch-axis',
        action='store_true',
        default=False,
        help="Enable dynamic axes."
    )
    group.add_argument(
        '--trt-onnx-export-only',
        action='store_true',
        default=False,
        help="Export ONNX model for TensorRT only. To measure inference time, "
             "use '--model-onnx-filepath ./model_tensorrt.onnx' in a second "
             "run."
    )
    group.add_argument(
        '--trt-use-python',
        action='store_true',
        default=False,
        help="Use python bindings instead of trtexec to use the engine, which "
             "might be slightly slower but is required to do inference with "
             "real samples."
    )
    parser.add_argument('--trt-do-not-use-extension', action='store_true',
                        help='do not use the FasterTransformer extension')

    args = parser.parse_args()
    args.trt_use_extension = not args.trt_do_not_use_extension
    return args


def create_batch(data, start_idx, batch_size):
    batch = [data[i % len(data)]
             for i in range(start_idx, start_idx + batch_size)]

    return mt_collate(batch, type_blacklist=(np.ndarray,
                                             CollateIgnoredDict,
                                             OrientationDict,
                                             SampleIdentifier))


def sample_batches(data, batch_size, n_batches):
    for i in range(n_batches):
        yield create_batch(data, i*batch_size, batch_size)


def create_engine(onnx_filepath,
                  engine_filepath,
                  floatx=16,
                  batch_size=1,
                  use_extension=True,
                  inputs=None,
                  input_names=None,
                  force_rebuild=True):

    if os.path.exists(engine_filepath) and not force_rebuild:
        # engine already exists
        return

    # note, we use trtexec to convert ONNX files to TensorRT engines
    print("Building engine using trtexec ...")
    if 32 == floatx:
        print("\t... this may take a while")
    else:
        print("\t... this may take -> AGES <-")

    if use_extension:
        # build command to generate the gemm config file
        swin_gemm_tool_path = utils.get_trt_swin_gemm_tool_path()

        # EMSAFormer uses SwinV2 with a unmodified window size of 8
        print('Building GEMM config file ...')
        patch_size = 8
        input_height = inputs[0]['rgb'].shape[-2]
        input_width = inputs[0]['rgb'].shape[-1]
        fp16 = floatx == 16
        gemm_cmd = (
            f'{swin_gemm_tool_path} {batch_size}'
            f' {input_height} {input_width}'
            f' {patch_size} 4 32 {int(fp16)}'
        )
        # execute command and hide output. The tool writes a 'gemm_config.in'
        # as config file, which will be used in the plugin.
        out = subprocess.run(gemm_cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        assert out.returncode == 0

    # build command for trtexec to create engine
    trt_swin_plugin_path = utils.get_trt_swin_plugin_path()
    cmd = (
        f'trtexec'
        f' --plugins={trt_swin_plugin_path}'
        f' --onnx={onnx_filepath}'
        f' --saveEngine={engine_filepath}'
    )
    if 16 == floatx:
        cmd += ' --fp16'

    # add input shapes to command
    shape_str = ''
    input_names_for_shape_str = input_names

    # if an RGB-D encoder is used, we still need to get the shapes from rgb
    # and depth separately
    if len(input_names_for_shape_str) == 1:
        if 'rgbd' == input_names_for_shape_str[0]:
            input_names_for_shape_str = ['rgb', 'depth']

    for name in input_names_for_shape_str:
        shape = inputs[0][name].shape
        if 4 == len(shape):
            _, c, h, w = shape
        else:
            c, h, w = shape
        shape_str += f'{name}:{batch_size}x{c}x{h}x{w},'
    shape_format = (
        f' --minShapes={shape_str[:-1]}'
        f' --optShapes={shape_str[:-1]}'
        f' --maxShapes={shape_str[:-1]}'
    )
    cmd += shape_format

    # Execute command
    print('Building engine ...')
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    if out.returncode != 0:
        print(out.stdout.decode('utf-8'))
    assert out.returncode == 0


def time_inference_tensorrt_trtexec(onnx_filepath,
                                    inputs,
                                    input_names,
                                    floatx=16,
                                    batch_size=1,
                                    use_extension=True,
                                    n_runs=100,
                                    n_runs_warmup=10,
                                    force_engine_rebuild=True,
                                    postprocessors=None,
                                    postprocessors_device='cpu',
                                    store_data=False):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'
    create_engine(onnx_filepath, trt_filepath,
                  floatx=floatx, batch_size=batch_size,
                  use_extension=use_extension,
                  inputs=inputs, input_names=input_names,
                  force_rebuild=force_engine_rebuild)

    # build execution command for TensorRT
    N_WARMUP_TIME = 10000    # = 10 seconds (we do not use n_runs_warmup here)
    trt_swin_plugin_path = utils.get_trt_swin_plugin_path()
    cmd = (
        f'trtexec'
        f' --plugins={trt_swin_plugin_path}'
        f' --loadEngine={trt_filepath}'
        f' --useSpinWait'
        f' --separateProfileRun'
        f' --warmUp={N_WARMUP_TIME}'
        f' --iterations={n_runs}'
    )
    if 16 == floatx:
        cmd += ' --fp16 '
    print('Running inference ...')
    print(cmd)

    # execute command and parse output
    outs = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    output = outs.stdout.decode('utf-8')
    # get qps from output: Throughput: ([0-9.]+) qps
    res = re.findall(r'Throughput: ([0-9.]+) qps', output)
    assert len(res) == 1

    # return outputs that match the output of the remaining functions, i.e.,
    # convert qps to a single timing, and return an empty list for inputs &
    # outputs (we do not have them)
    return np.array([1/float(res[0])]), []


def time_inference_tensorrt_python(onnx_filepath,
                                   inputs,
                                   input_names,
                                   floatx=16,
                                   batch_size=1,
                                   use_extension=True,
                                   n_runs=100,
                                   n_runs_warmup=10,
                                   force_engine_rebuild=True,
                                   postprocessors=None,
                                   postprocessors_device='cpu',
                                   store_data=False):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'
    create_engine(onnx_filepath, trt_filepath,
                  floatx=floatx, batch_size=batch_size,
                  use_extension=use_extension,
                  inputs=inputs, input_names=input_names,
                  force_rebuild=force_engine_rebuild)

    # load engine
    trt_model = TRTModel(trt_filepath)

    # time inference
    timings = []
    outs = []
    for i, input_ in enumerate(sample_batches(inputs,
                                              batch_size,
                                              n_runs+n_runs_warmup)):
        start_time = time.time()

        # get model output
        output = trt_model(input_)

        if postprocessors is None:
            out_trt = output
        else:
            out_trt = {}
            for name, post in postprocessors.items():
                # create input
                # bit hacky, this works as the keys are ordered
                in_post = [
                    output.cpu()
                    for k, output in output.items()
                    if name in k
                ]

                if 'cpu' != postprocessors_device:
                    # copy back to GPU (not smart)
                    in_post = [t.to(postprocessors_device) for t in in_post]

                    # we also need some inputs on gpu for the postprocessing
                    input_post = {
                        k: v.to(postprocessors_device)
                        for k, v in input_.items()
                        if ('rgb' in k or 'depth' in k) and torch.is_tensor(v)   # includes fullres
                    }
                else:
                    # simply we use the whole input batch for postprocessing
                    input_post = input_

                in_post_side = None
                if 1 == len(in_post):
                    # single input to postprocessor
                    in_post = in_post[0]
                else:
                    # multiple inputs to postprocessor (instance / panoptic)
                    in_post = tuple(in_post)

                if 'panoptic_helper' == name:
                    # this is not quite smart but works for now
                    # first element is semantic, the remaining instance
                    in_post = (in_post[0], in_post[1:])
                    in_post_side = None, None

                out_trt.update(
                    post.postprocess(data=(in_post, in_post_side),
                                     batch=input_post,
                                     is_training=False)
                )

            # copy back to cpu
            if 'cpu' != postprocessors_device:
                out_trt = move_batch_to_device(out_trt, 'cpu')

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)

        if store_data:
            outs.append((input_, out_trt))

    return np.array(timings), outs


def time_inference_pytorch(model,
                           inputs,
                           device,
                           n_runs=100,
                           n_runs_warmup=5,
                           batch_size=1,
                           with_postprocessing=False,
                           store_data=False):
    timings = []
    with torch.no_grad():
        outs = []
        for i, input_ in enumerate(sample_batches(inputs,
                                                  batch_size,
                                                  n_runs+n_runs_warmup)):
            # use PyTorch to time events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # copy to gpu
            inputs_gpu = {
                k: v.to(device)
                for k, v in input_.items()
                if ('rgb' in k or 'depth' in k) and torch.is_tensor(v)   # includes fullres
            }

            # model forward pass
            out_pytorch = model(inputs_gpu,
                                do_postprocessing=with_postprocessing)

            # copy back to cpu
            if not with_postprocessing:
                out_pytorch_cpu = []
                # output is tuple (outputs, side_output)
                for outputs, _ in out_pytorch:    # ignore side outputs
                    for output in outputs:
                        if isinstance(output, tuple):
                            # panoptic helper is again a tuple
                            out_pytorch_cpu.extend([o.cpu() for o in output])
                        else:
                            out_pytorch_cpu.append(output.cpu())
            else:
                # output is a dict
                out_pytorch_cpu = move_batch_to_device(out_pytorch, 'cpu')

            end.record()
            torch.cuda.synchronize()

            if i >= n_runs_warmup:
                timings.append(start.elapsed_time(end) / 1e3)

            if store_data:
                outs.append((input_, out_pytorch_cpu))

    return np.array(timings), outs


def get_fps_from_timings(timings, batch_size):
    return np.mean(1 / timings) * batch_size


def main(args):
    # prepare inputs -----------------------------------------------------------
    n_samples = 49
    args.batch_size = 1    # force bs 1 for collecting samples
    args.validation_batch_size = 1    # force bs 1 for collecting samples
    args.n_workers = 0    # no threads in torch dataloaders, use main thread

    data_helper = get_datahelper(args)

    inputs = []
    if args.dataset_path is not None:
        if not args.trt_use_python:
            raise ValueError("Please set '--trt-use-python' to use "
                             "real samples for inference timing.")

        # simply use first dataset (they all share the same properties)
        dataset = data_helper.datasets_valid[0]

        # get preprocessed samples of the given dataset
        data_helper.set_valid_preprocessor(
            get_preprocessor(
                args,
                dataset=dataset,
                phase='test',
                multiscale_downscales=None
            )
        )

        # disable memory pinning as it currently (pytorch 2.3.1) handles types
        # derived from tuple (e.g., our SampleIdentifier, see mt_collate usage)
        # in a wrong way, see:
        # https://github.com/pytorch/pytorch/blob/v2.3.1/torch/utils/data/_utils/pin_memory.py#L79
        data_helper.valid_dataloaders[0].pin_memory = False

        for sample in data_helper.valid_dataloaders[0]:
            inputs.append(
                {k: v[0] for k, v in sample.items()}    # remove batch axis
            )

            if (n_samples) == len(inputs):
                # enough samples collected
                break
    else:
        if args.with_postprocessing:
            # postprocessing random inputs does not really make sense
            # moreover, we need more fullres keys
            raise ValueError("Please set `--dataset-path` to enable "
                             "inference with meaningful inputs.")

        # the dataset's config is used later on for model building
        dataset = get_dataset(args, split=args.validation_split)

        # we do not have access to the data of dataset, simply collect random
        # inputs
        rgb_images = []
        depth_images = []
        for _ in range(n_samples):
            img_rgb = np.random.randint(
                low=0,
                high=255,
                size=(args.input_height, args.input_width, 3),
                dtype='uint8'
            )
            img_depth = np.random.randint(
                low=0,
                high=40000,
                size=(args.input_height, args.input_width),
                dtype='uint16'
            )
            # preprocess
            img_rgb = (img_rgb / 255).astype('float32').transpose(2, 0, 1)
            img_depth = (img_depth.astype('float32') / 20000)[None]
            img_rgb = np.ascontiguousarray(img_rgb)
            img_depth = np.ascontiguousarray(img_depth)
            rgb_images.append(torch.tensor(img_rgb))
            depth_images.append(torch.tensor(img_depth))

        # convert to input format (see BatchType)
        if 2 == len(args.input_modalities):
            inputs = [{'rgb': rgb_images[i], 'depth': depth_images[i]}
                      for i in range(len(rgb_images))]
        elif 'rgb' in args.input_modalities:
            inputs = [{'rgb': rgb_images[i]}
                      for i in range(len(rgb_images))]
        elif 'depth' in args.input_modalities:
            inputs = [{'depth': depth_images[i]}
                      for i in range(len(rgb_images))]
        elif 'rgbd' in args.input_modalities:
            inputs = [{'rgb': rgb_images[i], 'depth': depth_images[i]}
                      for i in range(len(rgb_images))]

    # create model ------------------------------------------------------------
    if args.model_onnx_filepath is not None:
        warnings.warn(
            "PyTorch inference timing disabled since onnx model is given."
        )
        args.no_time_pytorch = True

    # create model
    args.no_pretrained_backbone = True
    model = EMSAFormer(args=args, dataset_config=dataset.config)

    # load weights
    if args.weights_filepath is not None:
        checkpoint = torch.load(args.weights_filepath,
                                map_location=lambda storage, loc: storage)
        print(f"Loading checkpoint: '{args.weights_filepath}'.")
        if 'epoch' in checkpoint:
            print(f"-> Epoch: {checkpoint['epoch']}")
        load_weights(args, model, checkpoint['state_dict'])
    else:
        # Make all parameters (weights and biases) completely random
        # because else TensorRT can fail to build the engine.
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data = torch.randn(param.size())

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    model.eval()

    # define dummy input for export
    dummy_input = (create_batch(inputs, start_idx=0, batch_size=1),
                   {'do_postprocessing': False})

    # When using real data there will be many more keys in the input dict
    # which are not required for the model. For onnx export we filter them.
    if args.dataset_path is not None:
        keys_to_keep = ['rgb', 'depth']
        dummy_input_dict = {
            k: v for k, v in dummy_input[0].items() if k in keys_to_keep
        }
        dummy_input = (dummy_input_dict, dummy_input[1])

    # define names for input and output graph nodes
    # note, meaningful names are required to match postprocessors and
    # to set up dynamic_axes dict correctly
    input_names = [k for k in dummy_input[0].keys()]

    # time inference using PyTorch --------------------------------------------
    if not args.no_time_pytorch:
        # move model to gpu
        model.to(device)

        timings_pytorch, ios_pytorch = time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs=args.n_runs,
            n_runs_warmup=args.n_runs_warmup,
            batch_size=args.inference_batch_size,
            with_postprocessing=args.with_postprocessing,
            store_data=args.export_outputs
        )
        mean_fps = get_fps_from_timings(
            timings_pytorch,
            batch_size=args.inference_batch_size
        )
        print(f'fps pytorch: {mean_fps:0.4f}')

        # move model back to cpu (required for further steps)
        model.to('cpu')

    # time inference using TensorRT -------------------------------------------
    if not args.no_time_tensorrt:
        if args.model_onnx_filepath is None:
            # we have to export the model to onnx

            # determine output structure in order to derive names
            outputs = model(dummy_input[0], **dummy_input[1])
            assert len(outputs) == len(model.decoders)
            # encode output structure to output names (note, this is parsed
            # later to assign the outputs to the postprocessors if the model
            # is loaded from pure onnx)
            output_names = []
            for (outs, _), decoder_name in zip(outputs, model.decoders):
                if not isinstance(outs, tuple):
                    # semantic (single tensor)
                    outs = tuple(outs)

                if 'panoptic_helper' == decoder_name:
                    # this is not quite smart but works for now
                    # join semantic (single tensor) and instance outputs
                    outs = (outs[0], ) + outs[1]

                for j, _ in enumerate(outs):
                    # format output name
                    output_names.append(f'{decoder_name}_{j}')

            onnx_filepath = './model_tensorrt.onnx'

            # determine the dynamic axes
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: 'batch_size'}

            for output_name in output_names:
                dynamic_axes[output_name] = {0: 'batch_size'}

            if not args.trt_do_not_use_extension:
                # Load required plugin for exporting the model
                utils.load_torch_plugins()

            # export model to ONNX
            export_to_onnx(model, dummy_input, onnx_filepath,
                           use_swin_extension=args.trt_use_extension,
                           # meta_input_data is required as the model
                           # takes optional kwargs, which are not
                           # supported by the internal tracer.
                           meta_input_data=dummy_input[0],
                           input_names=input_names, output_names=output_names,
                           dynamic_axes=dynamic_axes,
                           opset_version=args.trt_onnx_opset_version)

            print(f"ONNX file (opset {args.trt_onnx_opset_version}) written "
                  f"to '{onnx_filepath}'.")

            if args.trt_onnx_export_only:
                # stop here
                exit(0)
        else:
            onnx_filepath = args.model_onnx_filepath

        # extract postprocessors
        if args.with_postprocessing:
            postprocessors = {
                k: v.postprocessing for k, v in model.decoders.items()
            }
        else:
            postprocessors = None

        if args.trt_use_python:
            # Load the FasterTransformer plugin
            utils.load_trt_plugins()
            timings_tensorrt, ios_tensorrt = time_inference_tensorrt_python(
                onnx_filepath,
                inputs,
                input_names,
                floatx=args.trt_floatx,
                batch_size=args.inference_batch_size,
                use_extension=args.trt_use_extension,
                n_runs=args.n_runs,
                n_runs_warmup=args.n_runs_warmup,
                force_engine_rebuild=args.trt_force_rebuild,
                postprocessors=postprocessors,
                postprocessors_device=device,
                store_data=args.export_outputs
            )
            mean_fps = get_fps_from_timings(
                timings_tensorrt,
                batch_size=args.inference_batch_size
            )
            print(f'fps tensorrt (python): {mean_fps:0.4f}')
        else:
            timings_tensorrt, ios_tensorrt = time_inference_tensorrt_trtexec(
                onnx_filepath,
                inputs,
                input_names,
                floatx=args.trt_floatx,
                batch_size=args.inference_batch_size,
                use_extension=args.trt_use_extension,
                n_runs=args.n_runs,
                n_runs_warmup=args.n_runs_warmup,
                force_engine_rebuild=args.trt_force_rebuild,
                postprocessors=postprocessors,
                postprocessors_device=device,
                store_data=args.export_outputs
            )
            mean_fps = get_fps_from_timings(
                timings_tensorrt,
                batch_size=args.inference_batch_size
            )
            print(f'fps tensorrt (trtexec): {mean_fps:0.4f}')

    if args.export_outputs:
        assert args.with_postprocessing, "Re-run with `--with-postprocessing`"

        results_path = os.path.join(os.path.dirname(__file__),
                                    f'inference_results',
                                    args.dataset)

        os.makedirs(results_path, exist_ok=True)

        if 'ios_pytorch' in locals():
            for inp, out in ios_pytorch:
                visualize(
                    output_path=os.path.join(results_path, 'pytorch'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )

        if 'ios_tensorrt' in locals():
            for inp, out in ios_tensorrt:
                visualize(
                    output_path=os.path.join(results_path,
                                             f'tensorrt_{args.trt_floatx}'),
                    batch=inp,
                    predictions=out,
                    dataset_config=dataset.config
                )


if __name__ == '__main__':
    # parse args
    args = _parse_args()

    print('PyTorch version:', torch.__version__)

    if not args.no_time_tensorrt:
        # to enable execution without TensorRT, we import relevant modules here
        import tensorrt as trt

        from tensorrt_swin.utils.onnx_exporter import export_to_onnx
        from tensorrt_swin.utils import utils
        from tensorrt_swin.utils.trt_helper_cuda_python import TRTModel

        print('TensorRT version:', trt.__version__)

    main(args)
