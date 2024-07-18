# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch
import torchvision

from .utils import assert_torch_plugins_loaded


class DefaultWeightConverter():

    @staticmethod
    def get_converter_type():
        return torch.nn.Module

    def convert(self, module, module_attributes, example_inputs):
        module_weights = dict(module.named_parameters())
        module_parameters = [v for _, v in module_weights.items()]
        module_names = [k for k, _ in module_weights.items()]
        return module_names, module_parameters


class TRTSwinTransformerBlockWeightConverter(DefaultWeightConverter):

    def __init__(self) -> None:
        super().__init__()
        assert_torch_plugins_loaded()
        self.transform_trt_mask = torch.ops.fastertransformer.transform_trt_mask

        weight_prefix = '_module.'
        weight_prefix = ''
        self.qkv_key_weight = f'{weight_prefix}attn.qkv.weight'
        self.qkv_bias_name = f'{weight_prefix}attn.qkv.bias'

        self.relative_position_bias_table_name = f'{weight_prefix}attn.relative_position_bias_table'
        self.trt_relative_position_bias_table_name = f'{weight_prefix}trt_relative_position_bias'

        self.attn_mask_name = f'{weight_prefix}attn_mask'
        self.trt_attn_mask_name = f"{weight_prefix}trt_attn_mask"

    @staticmethod
    def get_converter_type():
        return torchvision.models.swin_transformer.SwinTransformerBlock

    def fuse_weights(self, weights):
        # There seems to be a bug or "undocumented feature" in TensorRT which
        # limits the input of a node to a maximum of 14.
        # Starting from 15 (experimentally tested) the fp16 optimization is
        # not done anymore.
        # The following github issue seems to be related:
        # https://github.com/NVIDIA/TensorRT/issues/2246
        # For SwinV2 we have 21 inputs (1 for actual input, 20 for weights).
        # To avoid this issue we fuse weights and bias together and do some
        # pointer arithmetic in the plugin to unfuse them again.
        # For fusing we first check for every weight if it has a bias.
        # If so we check if the dimensions match and if so we fuse them.

        # Copy the weights to avoid modifying the original ones
        weights = weights.copy()

        # Search for weights with bias
        fused_weights = {}
        for name, weight in weights.items():
            # We only support 1d and 2d weights
            if len(weight.shape) > 2:
                continue
            if not name.endswith('weight'):
                continue
            bias_name = name.replace('weight', 'bias')
            if bias_name not in weights:
                continue
            bias = weights[bias_name]
            # We only support 1d bias
            if len(bias.shape) > 1:
                continue

            # Figure out which dim has the same size
            same_size_idx = -1
            for i in range(len(weight.shape)):
                if weight.shape[i] == bias.shape[0]:
                    same_size_idx = i
                    break

            if same_size_idx == -1:
                # If the dimensions don't match we can't fuse them
                continue

            flat_weights = weight.flatten()
            flat_bias = bias.flatten()
            fused_weight = torch.cat((flat_weights, flat_bias), dim=0)

            if same_size_idx == 0:
                fused_weight = fused_weight.reshape(weight.shape[0], -1)
            elif same_size_idx == 1:
                fused_weight = fused_weight.reshape(-1, weight.shape[1])
            else:
                raise RuntimeError("This should never happen")

            # Do the actual concatenation, we indicate with _first and _last
            # if the weight was stacked in the first or second dimension.
            if same_size_idx == 0:
                fused_name = name.replace('.weight', '_first')
                assert (fused_weight.flatten()[:(fused_weight.shape[1]-1)*fused_weight.shape[0]] == flat_weights).all()
                assert (fused_weight.flatten()[(fused_weight.shape[1]-1)*fused_weight.shape[0]:] == flat_bias).all()
            else:
                fused_name = name.replace('.weight', '_last')
                assert (fused_weight.flatten()[:(fused_weight.shape[0]-1)*fused_weight.shape[1]] == flat_weights).all()
                assert (fused_weight.flatten()[(fused_weight.shape[0]-1)*fused_weight.shape[1]:] == flat_bias).all()

            fused_weights[fused_name] = fused_weight

        # Remove the fused weights from the output weights
        for name in fused_weights.keys():
            name_old = name.replace('_first', '').replace('_last', '')
            del weights[name_old + '.weight']
            del weights[name_old + '.bias']
        # Add the fused weights to the output weights
        weights.update(fused_weights)
        # Sort the weights by name
        weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[0])}
        return weights

    def get_attn_mask(self, window_size, shift_size, example_input):
        # The following code is taken from the torchvision swin implementation
        # and modified to only generate the mask.
        # See: https://github.com/pytorch/vision/blob/1d6f78755bafa412ab19d0fa12654444b02d362b/torchvision/models/swin_transformer.py#L116
        _, H, W, _ = example_input.shape

        # pad feature maps to multiples of window size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = torch.nn.functional.pad(example_input, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape
        # If window size is larger than feature size, there is no need to shift window
        if window_size[0] >= pad_H:
            shift_size[0] = 0
        if window_size[1] >= pad_W:
            shift_size[1] = 0

        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])

        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]:h[1], w[0]:w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def convert(self, module, module_attributes, example_inputs):
        module_weights = dict(module.named_parameters())
        output_weights = module_weights

        # Read the required attributes
        num_heads = module_attributes['num_heads_i']
        shift_size = module_attributes['shift_size_i']
        window_size = module_attributes['window_size_i']

        # In the following the 3 is always because of query, key, value
        # Reshape the qkv weight:
        # In: [3*num_heads*in_dim_per_head, qkv_in_dim]
        # Out:[qkv_in_dim, num_heads*3*in_dim_per_head, ]
        qkv_weight = module_weights[self.qkv_key_weight]
        qkv_out_dim, qkv_in_dim = qkv_weight.shape
        in_dim_per_head = qkv_in_dim // num_heads
        assert qkv_out_dim == 3 * num_heads * in_dim_per_head
        qkv_weight = qkv_weight.reshape(3, num_heads, in_dim_per_head, qkv_in_dim)
        qkv_weight = qkv_weight.permute(3, 1, 0, 2).reshape(qkv_in_dim, qkv_out_dim).clone()
        output_weights[self.qkv_key_weight] = qkv_weight

        # Reshape the qkv bias:
        # In: [3*num_heads*in_dim_per_head]
        # Out:[num_heads*3*in_dim_per_head]
        qkv_bias = module_weights[self.qkv_bias_name]
        qkv_bias = qkv_bias.reshape(3, num_heads, in_dim_per_head)
        qkv_bias = qkv_bias.permute(1, 0, 2).reshape(3 * num_heads * in_dim_per_head).clone()
        output_weights[self.qkv_bias_name] = qkv_bias
        # For training the relative_position_bias_table consists of a table
        # and a index which can be fused together.
        # In the FasterTransformer Swin implementation the 'gen_relative_pos_bias'
        # function was used to fuse both together.
        # In the torchvision implementation there is a function 'get_relative_position_bias'
        # in the window attention which is equivalent.
        relativ_position_bias_table = module.attn.get_relative_position_bias()
        output_weights[self.relative_position_bias_table_name] = relativ_position_bias_table

        # In FasterTransfromer also a special table for trt is generated.
        # However under certain conditions it's just a tensor.
        # The special table seems to be only required for FP16 inference,
        # as there are already fused - optimized - kernels for computation.
        # See here: https://github.com/NVIDIA/FasterTransformer/tree/main/3rdparty/trt_fused_multihead_attention
        # Because its only used for fp16 we already set it to a half, to safe some memory.
        trt_relative_position_bias = torch.Tensor().half()
        if relativ_position_bias_table.shape[2] <= 256 and in_dim_per_head == 32:
            trt_relative_position_bias = self.transform_trt_mask(relativ_position_bias_table.half().cuda(),
                                                                 relativ_position_bias_table.shape[1],
                                                                 relativ_position_bias_table.shape[2],
                                                                 # We dont use int8 -> False
                                                                 False)
        output_weights[self.trt_relative_position_bias_table_name] = \
            trt_relative_position_bias.to(relativ_position_bias_table.device)

        # Also generate the attn_mask which is required for the shifted window attention
        attn_mask = self.get_attn_mask(window_size, shift_size, example_inputs)
        zero = torch.zeros(1).to(relativ_position_bias_table.device)
        # We only need the mask for shifted attentions, else we can just store
        # zeros to safe some memory
        if sum(shift_size) > 0:
            output_weights[self.attn_mask_name] = attn_mask.clone()
        else:
            output_weights[self.attn_mask_name] = zero.clone()

        trt_attn_mask = self.transform_trt_mask(attn_mask.half().cuda(),
                                                attn_mask.shape[0],
                                                attn_mask.shape[1],
                                                False)
        if sum(shift_size) > 0:
            output_weights[self.trt_attn_mask_name] = \
                trt_attn_mask.to(attn_mask.device).clone()
        else:
            # If we dont shift windows the masks are not required so we
            # just write some zeros to safe memory.
            output_weights[self.trt_attn_mask_name] = zero.clone()

        output_weights = self.fuse_weights(output_weights)

        output_names = [k for k, _ in output_weights.items()]
        output_parameters = [v for _, v in output_weights.items()]
        return output_names, output_parameters


class TRTSwinTransformerBlockV2WeightConverter(TRTSwinTransformerBlockWeightConverter):

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale_name = 'attn.logit_scale'

    @staticmethod
    def get_converter_type():
        return torchvision.models.swin_transformer.SwinTransformerBlockV2

    def convert(self, module, module_attributes, example_inputs):
        module_weights = dict(module.named_parameters())
        output_weights = module_weights

        # Read the required attributes
        num_heads = module_attributes['num_heads_i']
        shift_size = module_attributes['shift_size_i']
        window_size = module_attributes['window_size_i']

        # In the following the 3 is always because of query, key, value
        # Reshape the qkv weight:
        # In: [3*num_heads*in_dim_per_head, qkv_in_dim]
        # Out:[qkv_in_dim, num_heads*3*in_dim_per_head, ]
        qkv_weight = module_weights[self.qkv_key_weight]
        qkv_out_dim, qkv_in_dim = qkv_weight.shape
        in_dim_per_head = qkv_in_dim // num_heads
        assert qkv_out_dim == 3 * num_heads * in_dim_per_head
        qkv_weight = qkv_weight.reshape(3, num_heads, in_dim_per_head, qkv_in_dim)
        qkv_weight = qkv_weight.permute(3, 1, 0, 2).reshape(qkv_in_dim, qkv_out_dim).clone()
        output_weights[self.qkv_key_weight] = qkv_weight

        # Reshape the qkv bias:
        # In: [3*num_heads*in_dim_per_head]
        # Out:[num_heads*3*in_dim_per_head]
        # Note: This is a little different compared how its done in the
        # original FasterTransfomer implementation. SwinV2 only has bias for
        # query and value and not for key. Because of this microsoft didn't
        # implement a qkv bias but else a separate q and v bias.
        # For inference they concatenate and fille the middle (for value)
        # with zeros.
        # In torchvision implementation its already qkv bias and in forward
        # the bias elements for value are set to 0.
        qkv_bias = module_weights[self.qkv_bias_name].clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length:2 * length] = 0
        qkv_bias = qkv_bias.reshape(3, num_heads, in_dim_per_head)
        qkv_bias = qkv_bias.permute(1, 0, 2).reshape(3 * num_heads * in_dim_per_head).clone()
        output_weights[self.qkv_bias_name] = qkv_bias

        # Check documentation of parent class for some explanation.
        # Note that we dont need to do any further calculation with the
        # continuous position bias (cpb) because its already handle by the
        # used function.
        relativ_position_bias_table = module.attn.get_relative_position_bias()
        output_weights[self.relative_position_bias_table_name] = relativ_position_bias_table

        # In FasterTransfromer also a special table for trt is generated.
        # However under certain conditions it's just a tensor.
        # The special table seems to be only required for FP16 inference,
        # as there are already fused - optimized - kernels for computation.
        # See here: https://github.com/NVIDIA/FasterTransformer/tree/main/3rdparty/trt_fused_multihead_attention
        # Because its only used for fp16 we already set it to a half, to safe some memory.
        trt_relative_position_bias = torch.Tensor().half()
        if relativ_position_bias_table.shape[2] <= 256 and in_dim_per_head == 32:
            trt_relative_position_bias = self.transform_trt_mask(relativ_position_bias_table.half().cuda(),
                                                                 relativ_position_bias_table.shape[1],
                                                                 relativ_position_bias_table.shape[2],
                                                                 # We dont use int8 -> False
                                                                 False)
        output_weights[self.trt_relative_position_bias_table_name] = \
            trt_relative_position_bias.to(relativ_position_bias_table.device)

        # Also generate the attn_mask which is required for the shifted window attention
        attn_mask = self.get_attn_mask(window_size, shift_size, example_inputs)
        output_weights[self.attn_mask_name] = attn_mask
        trt_attn_mask = self.transform_trt_mask(attn_mask.half().cuda(),
                                                attn_mask.shape[0],
                                                attn_mask.shape[1],
                                                False)
        output_weights[self.trt_attn_mask_name] = \
            trt_attn_mask.to(attn_mask.device)

        # Update logit scale
        logit_scale = module_weights[self.logit_scale_name]
        # Logit scale will stay constant because we arn't in training and the
        # following calulation is done in forward. We precompute it so
        # we don't need it in TensorRT anymore.
        logit_scale = torch.clamp(logit_scale,
                                  max=torch.log(torch.tensor(100.0))).exp()
        module_weights[self.logit_scale_name] = logit_scale

        # See description in the function why we fuse weights and bias.
        # For Swinv2 this reduces 21 inputs to 14.
        output_weights = self.fuse_weights(output_weights)

        # In addition to the fusion we can remove every weight which stats with
        # 'attn.cpb_' as they are not required for inference.
        # This reduces the number of inputs to 11.
        output_weights = {k: v for k, v in output_weights.items() if not k.startswith('attn.cpb_')}

        output_names = [k for k, _ in output_weights.items()]
        output_parameters = [v for _, v in output_weights.items()]

        return output_names, output_parameters
