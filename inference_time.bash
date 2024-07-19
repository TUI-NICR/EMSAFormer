#!/bin/bash
set -o xtrace

ARGS_EMSAFORMER_COMMON='--dataset nyuv2 --input-modalities rgbd --tasks instance semantic orientation scene --enable-panoptic --rgbd-encoder-backbone swin-multi-t-v2-128 --encoder-normalization ln --instance-decoder emsanet --instance-encoder-decoder-fusion swin-ln-add'

# Args specific to each model
EMSAFORMER_EMSANET_DECODER_WEIGHTS='./trained_models/nyuv2/nyuv2_swin_multi_t_v2_128_emsanet_decoder.pth'
ARGS_EMSAFORMER_EMSANET_DECODER='--semantic-decoder emsanet --semantic-encoder-decoder-fusion swin-ln-add --semantic-decoder-upsampling learned-3x3-zeropad --semantic-decoder-n-channels 512 256 128 --weights-filepath'

EMSAFORMER_SEGFORMER_DECODER_WEIGHTS='./trained_models/nyuv2/nyuv2_swin_multi_t_v2_128_segformermlp_decoder.pth'
ARGS_EMSAFORMER_SEGFORMER_DECODER='--semantic-decoder segformermlp --semantic-encoder-decoder-fusion swin-ln-select --semantic-decoder-n-channels 256 128 64 64 --semantic-decoder-upsampling bilinear --weights-filepath'

# Build list for full model commands
MODELS_TO_TIME=(
    "$ARGS_EMSAFORMER_COMMON $ARGS_EMSAFORMER_EMSANET_DECODER $EMSAFORMER_EMSANET_DECODER_WEIGHTS"
    "$ARGS_EMSAFORMER_COMMON $ARGS_EMSAFORMER_SEGFORMER_DECODER $EMSAFORMER_SEGFORMER_DECODER_WEIGHTS"
)


# Args to use for timing
ARGS_PYTORCH='--no-time-tensorrt'
ARGS_EXPORT_ONNX_TRT='--no-time-pytorch --trt-onnx-export-only --trt-enable-dynamic-batch-axis'
ARGS_TIME_TRT32='--model-onnx-filepath ./model_tensorrt.onnx --n-runs-warmup 20 --n-runs 80 --no-time-pytorch'
ARGS_TIME_TRT16=$ARGS_TIME_TRT32' --trt-floatx 16'

SED_PYTORCH="sed -n 's/.*fps pytorch: \([0-9.]*\).*$/\1/p'"
SED_TRT="sed -n 's/.*fps tensorrt (trtexec): \([0-9.]*\).*$/\1/p'"
SED_TRT_USE_PYTHON="sed -n 's/.*fps tensorrt (python): \([0-9.]*\).*$/\1/p'"

# ------------------------------------------------------------------------------
# Verify that both weights files exist
if [ ! -f $EMSAFORMER_EMSANET_DECODER_WEIGHTS ] || [ ! -f $EMSAFORMER_SEGFORMER_DECODER_WEIGHTS ];
then
    echo "Weights files not found. Please download the NYUv2 weights files from the README."
    exit 1
fi

# ------------------------------------------------------------------------------
RESULTS_FILE='./nyuv2_timings.csv'

# Iterate over each model
for MODEL_ARGS in "${MODELS_TO_TIME[@]}"
do
    echo -n "${MODEL_ARGS}," >> $RESULTS_FILE

    # time pytorch
    python3 inference_time_whole_model.py $MODEL_ARGS $ARGS_PYTORCH | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
    echo -n "," >> $RESULTS_FILE

    # export onnx model first and time in second call -> saves resources
    python3 inference_time_whole_model.py $MODEL_ARGS $ARGS_EXPORT_ONNX_TRT

    # time tensorrt float32
    python3 inference_time_whole_model.py $MODEL_ARGS $ARGS_TIME_TRT32 | eval $SED_TRT | xargs echo -n >> $RESULTS_FILE
    echo -n "," >> $RESULTS_FILE

    # time tensorrt float16
    python3 inference_time_whole_model.py $MODEL_ARGS $ARGS_TIME_TRT16 | eval $SED_TRT | xargs echo -n >> $RESULTS_FILE
    echo "," >> $RESULTS_FILE
done
