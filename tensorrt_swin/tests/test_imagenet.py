import pytest

from trt_imagenet import parse_args
from trt_imagenet import main
from trt_imagenet import model_dict

IMAGENET_PATH = "/datasets_nas/classification/ImageNet/val"


@pytest.mark.parametrize("model_name", model_dict.keys())
@pytest.mark.parametrize("batch_sizes", [1, 16])
@pytest.mark.parametrize("fp16", [False, True])
def test_imagenet_models(model_name, batch_sizes, fp16):

    # Prepare args as array of strings
    args_str = f"--model {model_name}"
    args_str += f" --batch-sizes {batch_sizes}"
    args_str += f" --data-path {IMAGENET_PATH}"
    if fp16:
        args_str += " --fp16"
    args_list = args_str.split(" ")
    args = parse_args(args_list)
    main(args)
