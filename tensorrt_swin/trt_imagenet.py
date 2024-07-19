# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import os
import tempfile

import torch
import torchvision
from tqdm import tqdm

from utils import trt_helper_cuda_python as trt_helper
from utils import utils
from utils.onnx_exporter import export_to_onnx

try:
    import vision.references.classification.presets as torchvision_presets
    import vision.references.classification.utils as torchvision_utils
except ImportError:
    raise ImportError(
        "torchvision repository is not found. "
        "Please follow the instruction in README.md to install it."
    )


# Models which cane be used for testing.
model_dict = {
    # Test swin v1 even though it's not part of the paper.
    'swin_t': torchvision.models.swin_t,
    # Test swin v2
    'swin_t_v2': torchvision.models.swin_v2_t,
    'swin_s_v2': torchvision.models.swin_v2_s,
    'swin_b_v2': torchvision.models.swin_v2_b,
}


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(
        description='Swin Transformer TRT Inference'
    )
    parser.add_argument('--model', default='swin_t_v2',
                        type=str, help='model name', choices=model_dict.keys())
    # Batch sizes to test
    parser.add_argument('--batch-sizes', default=[1, 16, 32, 128], nargs='+',
                        type=int, help='batch sizes to test')
    parser.add_argument('--fp16', action='store_true',
                        help='use fp16')
    parser.add_argument('--data-path', required=True,
                        type=str, help='path to imagenet dataset')
    parser.add_argument('--tmp-path', default='/tmp',
                        type=str, help='path to store temporary files')
    parser.add_argument('--do-not-use-extension', action='store_true',
                        help='do not use the FasterTransformer extension')
    parser.add_argument('--do-pytorch-inference', action='store_true',
                        help='do inference with PyTorch instead of TRT')
    args = parser.parse_args(args_list)
    return args


def load_data(path, size, batch_size):
    from torchvision.transforms.functional import InterpolationMode
    interpolation = InterpolationMode('bicubic')
    preprocessing = torchvision_presets.ClassificationPresetEval(
        crop_size=size, resize_size=size,
        interpolation=interpolation
    )
    dataset_test = torchvision.datasets.ImageFolder(
        path,
        preprocessing,
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=os.cpu_count()//2,
        pin_memory=True,
        persistent_workers=True,
    )
    return data_loader_test


def onnx_to_engine(onnx_path, engine_path, batch_size,
                   input_size, fp16=False, use_extension=True):
    if fp16:
        # Our plugin uses an environment variable to decide whether to
        # use fp16 or not which enforces the fp16 mode.
        add_args = "--fp16"
    else:
        add_args = ""

    if use_extension:
        # The FasterTransformer extension wants to have a gemm config file
        # which is created in the following step.
        swin_gemm_tool_path = utils.get_trt_swin_gemm_tool_path()
        window_size = 8 if input_size == 256 else 7
        gemm_cmd = (
            f'{swin_gemm_tool_path} {batch_size} {input_size} {input_size} '
            f'{window_size} 3 32 {int(fp16)}'
        )
        print(gemm_cmd)
        return_code = os.system(gemm_cmd)
        assert return_code == 0

    # Create engine
    shape = f"input:{batch_size}x{3}x{input_size}x{input_size}"
    shape_cmd = f"--minShapes={shape} --optShapes={shape} --maxShapes={shape}"
    return_code = os.system(
        f'trtexec --onnx={onnx_path} '
        f'--plugins={utils.get_trt_swin_plugin_path()} '
        f'--saveEngine={engine_path} {shape_cmd} {add_args} '
        '--useSpinWait --separateProfileRun --warmUp=10000 --iterations=200'
    )
    assert return_code == 0
    # Ensure that the engine was created
    assert os.path.isfile(engine_path)


def main(args):
    # Load and put the original model to CPU to avoid memory issues
    model = model_dict[args.model](weights='DEFAULT').cpu()
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_path:
        os.makedirs(tmp_path, exist_ok=True)

        # Export to ONNX
        input_size = 256 if 'v2' in args.model else 224
        onnx_name = f'{args.model}_{input_size}_{input_size}.onnx'
        onnx_name = os.path.join(tmp_path, onnx_name)
        use_extension = not args.do_not_use_extension
        tracer_input_data = torch.randn(1, 3, input_size, input_size)

        # The model dosn't need to be exported to ONNX if PyTorch inference is used
        if not args.do_pytorch_inference:
            if not args.do_not_use_extension:
                utils.load_torch_plugins()
            export_to_onnx(
                model,
                tracer_input_data,
                onnx_name,
                use_swin_extension=use_extension,
                input_names=['input'], output_names=['output'],
            )

        # Can be used for putting the data to GPU
        device = torch.device('cuda')
        for bs in args.batch_sizes:
            # Load ImageNet dataset for testing
            dataloader = load_data(args.data_path, input_size, bs)

            # Load the TRT engine and do inference
            if not args.do_pytorch_inference:
                # Convert the ONNX model to TRT engine with trtexec.
                # Note that this is done for every batch size, because the
                # it will be optimized for the specific batch size.
                engine_name = onnx_name.replace('.onnx', f'_bs{bs}.engine')
                onnx_to_engine(onnx_name, engine_name, bs,
                               input_size, args.fp16, use_extension)
                # Load the FasterTransformer plugin
                utils.load_trt_plugins()
                inference_model = trt_helper.TRTModel(engine_path=engine_name)
            else:
                # Use the PyTorch model for inference
                inference_model = model.to(device)
            metric_logger = torchvision_utils.MetricLogger(delimiter="  ")

            # do some dry runs to warm up the engine
            n_dry_runs = 30
            for i, (image, _) in enumerate(dataloader):
                if not args.do_pytorch_inference:
                    _ = inference_model(image)
                else:
                    _ = inference_model(image.to(device))
                if i >= n_dry_runs:
                    break

            # run inference
            tqdm_obj = tqdm(dataloader)
            for image, target in tqdm_obj:
                # run inference
                # if pytorch inference is used, the data needs to be put explicitly
                # to GPU. The TRT engine does this internally.
                if not args.do_pytorch_inference:
                    model_output = inference_model(image)
                else:
                    model_output = inference_model(image.to(device)).cpu()

                # measure accuracy
                model_output = model_output[:target.size(0)]    # handle (smaller) last batch
                acc1_trt, acc5_trt = torchvision_utils.accuracy(
                    model_output, target, topk=(1, 5)
                )
                metric_logger.meters["acc1"].update(acc1_trt.item(), n=bs)
                metric_logger.meters["acc5"].update(acc5_trt.item(), n=bs)

            # Compute and print the average accuracy
            metric_logger.synchronize_between_processes()
            print(f"TRT: Acc@1 {metric_logger.acc1.global_avg:.3f} "
                  f"Acc@5 {metric_logger.acc5.global_avg:.3f}")

            # For saving memory we delete the TRT engine
            del inference_model
            # Sanity check: Compare the output of the TRT engine with the
            # output of the original model.
            out_torch = model.to(device)(image.to(device)).cpu()
            abs = torch.abs(model_output - out_torch)
            print("tqdm time: ", tqdm_obj.format_dict['elapsed'])
            print("torch_output vs plugin_output:")
            print("mean diff : ", abs.mean().item())
            print("max diff : ", abs.max().item())
            assert abs.mean() < 0.01, "[ERROR] SWIN PLUGIN TEST FAIL !"


if __name__ == '__main__':
    args = parse_args()
    main(args)
