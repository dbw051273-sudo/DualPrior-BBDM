# import pdb
# from abc import abstractmethod
# from functools import partial
# import math
# from typing import Iterable

# import numpy as np
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from model.BrownianBridge.base.modules.diffusionmodules.util import (
#     checkpoint,
#     conv_nd,
#     linear,
#     avg_pool_nd,
#     zero_module,
#     normalization,
#     timestep_embedding,
# )
# from model.BrownianBridge.base.modules.attention import SpatialTransformer


# # dummy replace
# def convert_module_to_f16(x):
#     pass

# def convert_module_to_f32(x):
#     pass


# ## go
# class AttentionPool2d(nn.Module):
#     """
#     Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
#     """

#     def __init__(
#         self,
#         spacial_dim: int,
#         embed_dim: int,
#         num_heads_channels: int,
#         output_dim: int = None,
#     ):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
#         self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
#         self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
#         self.num_heads = embed_dim // num_heads_channels
#         self.attention = QKVAttention(self.num_heads)

#     def forward(self, x):
#         b, c, *_spatial = x.shape
#         x = x.reshape(b, c, -1)  # NC(HW)
#         x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
#         x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
#         x = self.qkv_proj(x)
#         x = self.attention(x)
#         x = self.c_proj(x)
#         return x[:, :, 0]


# class TimestepBlock(nn.Module):
#     """
#     Any module where forward() takes timestep embeddings as a second argument.
#     """

#     @abstractmethod
#     def forward(self, x, emb):
#         """
#         Apply the module to `x` given `emb` timestep embeddings.
#         """


# class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
#     """
#     A sequential module that passes timestep embeddings to the children that
#     support it as an extra input.
#     """

#     def forward(self, x, emb, context=None):
#         # pdb.set_trace()
#         for layer in self:
#             if isinstance(layer, TimestepBlock):
#                 x = layer(x, emb)
#             elif isinstance(layer, SpatialTransformer):
#                 x = layer(x, context)
#             else:
#                 x = layer(x)
#         return x


# class Upsample(nn.Module):
#     """
#     An upsampling layer with an optional convolution.
#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  upsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         if use_conv:
#             self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         if self.dims == 3:
#             x = F.interpolate(
#                 x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
#             )
#         else:
#             x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if self.use_conv:
#             x = self.conv(x)
#         return x


# class TransposedUpsample(nn.Module):
#     'Learned 2x upsampling without padding'
#     def __init__(self, channels, out_channels=None, ks=5):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels

#         self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

#     def forward(self,x):
#         return self.up(x)


# class Downsample(nn.Module):
#     """
#     A downsampling layer with an optional convolution.
#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  downsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         stride = 2 if dims != 3 else (1, 2, 2)
#         if use_conv:
#             self.op = conv_nd(
#                 dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
#             )
#         else:
#             assert self.channels == self.out_channels
#             self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         return self.op(x)


# class ResBlock(TimestepBlock):
#     """
#     A residual block that can optionally change the number of channels.
#     :param channels: the number of input channels.
#     :param emb_channels: the number of timestep embedding channels.
#     :param dropout: the rate of dropout.
#     :param out_channels: if specified, the number of out channels.
#     :param use_conv: if True and out_channels is specified, use a spatial
#         convolution instead of a smaller 1x1 convolution to change the
#         channels in the skip connection.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param use_checkpoint: if True, use gradient checkpointing on this module.
#     :param up: if True, use this block for upsampling.
#     :param down: if True, use this block for downsampling.
#     """

#     def __init__(
#         self,
#         channels,
#         emb_channels,
#         dropout,
#         out_channels=None,
#         use_conv=False,
#         use_scale_shift_norm=False,
#         dims=2,
#         use_checkpoint=False,
#         up=False,
#         down=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.use_checkpoint = use_checkpoint
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.in_layers = nn.Sequential(
#             normalization(channels),
#             nn.SiLU(),
#             conv_nd(dims, channels, self.out_channels, 3, padding=1),
#         )

#         self.updown = up or down

#         if up:
#             self.h_upd = Upsample(channels, False, dims)
#             self.x_upd = Upsample(channels, False, dims)
#         elif down:
#             self.h_upd = Downsample(channels, False, dims)
#             self.x_upd = Downsample(channels, False, dims)
#         else:
#             self.h_upd = self.x_upd = nn.Identity()

#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             linear(
#                 emb_channels,
#                 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
#             ),
#         )
#         self.out_layers = nn.Sequential(
#             normalization(self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             zero_module(
#                 conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
#             ),
#         )

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         elif use_conv:
#             self.skip_connection = conv_nd(
#                 dims, channels, self.out_channels, 3, padding=1
#             )
#         else:
#             self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

#     def forward(self, x, emb):
#         """
#         Apply the block to a Tensor, conditioned on a timestep embedding.
#         :param x: an [N x C x ...] Tensor of features.
#         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         return checkpoint(
#             self._forward, (x, emb), self.parameters(), self.use_checkpoint
#         )


#     def _forward(self, x, emb):
#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             x = self.x_upd(x)
#             h = in_conv(h)
#         else:
#             h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = th.chunk(emb_out, 2, dim=1)
#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
#         return self.skip_connection(x) + h


# class AttentionBlock(nn.Module):
#     """
#     An attention block that allows spatial positions to attend to each other.
#     Originally ported from here, but adapted to the N-d case.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
#     """

#     def __init__(
#         self,
#         channels,
#         num_heads=1,
#         num_head_channels=-1,
#         use_checkpoint=False,
#         use_new_attention_order=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         if num_head_channels == -1:
#             self.num_heads = num_heads
#         else:
#             assert (
#                 channels % num_head_channels == 0
#             ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
#             self.num_heads = channels // num_head_channels
#         self.use_checkpoint = use_checkpoint
#         self.norm = normalization(channels)
#         self.qkv = conv_nd(1, channels, channels * 3, 1)
#         if use_new_attention_order:
#             # split qkv before split heads
#             self.attention = QKVAttention(self.num_heads)
#         else:
#             # split heads before split qkv
#             self.attention = QKVAttentionLegacy(self.num_heads)

#         self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

#     def forward(self, x):
#         return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
#         #return pt_checkpoint(self._forward, x)  # pytorch

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)


# def count_flops_attn(model, _x, y):
#     """
#     A counter for the `thop` package to count the operations in an
#     attention operation.
#     Meant to be used like:
#         macs, params = thop.profile(
#             model,
#             inputs=(inputs, timestamps),
#             custom_ops={QKVAttention: QKVAttention.count_flops},
#         )
#     """
#     b, c, *spatial = y[0].shape
#     num_spatial = int(np.prod(spatial))
#     # We perform two matmuls with the same number of ops.
#     # The first computes the weight matrix, the second computes
#     # the combination of the value vectors.
#     matmul_ops = 2 * b * (num_spatial ** 2) * c
#     model.total_ops += th.DoubleTensor([matmul_ops])


# class QKVAttentionLegacy(nn.Module):
#     """
#     A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts", q * scale, k * scale
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v)
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention and splits in a different order.
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.chunk(3, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts",
#             (q * scale).view(bs * self.n_heads, ch, length),
#             (k * scale).view(bs * self.n_heads, ch, length),
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)


# class UNetModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     :param num_heads_channels: if specified, ignore num_heads and instead use
#                                a fixed channel width per attention head.
#     :param num_heads_upsample: works with num_heads to set a different number
#                                of heads for upsampling. Deprecated.
#     :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
#     :param resblock_updown: use residual blocks for up/downsampling.
#     :param use_new_attention_order: use a different attention pattern for potentially
#                                     increased efficiency.
#     """

#     def __init__(
#         self,
#         image_size,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         num_classes=None,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=-1,
#         num_head_channels=-1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         resblock_updown=False,
#         use_new_attention_order=False,
#         use_spatial_transformer=False,    # custom transformer support
#         transformer_depth=1,              # custom transformer support
#         context_dim=None,                 # custom transformer support
#         n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
#         legacy=True,
#         condition_key="concat",
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.num_classes = num_classes
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None
#         self.condition_key = condition_key

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_embed_dim)

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         #num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     layers.append(
#                         AttentionBlock(
#                             ch,
#                             use_checkpoint=use_checkpoint,
#                             num_heads=num_heads,
#                             num_head_channels=dim_head,
#                             use_new_attention_order=use_new_attention_order,
#                         ) if not use_spatial_transformer else SpatialTransformer(
#                             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
#                         )
#                     )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             #num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(
#                             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
#                         ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self._feature_size += ch

#         self.output_blocks = nn.ModuleList([])
#         for level, mult in list(enumerate(channel_mult))[::-1]:
#             for i in range(num_res_blocks + 1):
#                 ich = input_block_chans.pop()
#                 layers = [
#                     ResBlock(
#                         ch + ich,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels * mult,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = model_channels * mult
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         #num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     layers.append(
#                         AttentionBlock(
#                             ch,
#                             use_checkpoint=use_checkpoint,
#                             num_heads=num_heads_upsample,
#                             num_head_channels=dim_head,
#                             use_new_attention_order=use_new_attention_order,
#                         ) if not use_spatial_transformer else SpatialTransformer(
#                             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
#                         )
#                     )
#                 if level and i == num_res_blocks:
#                     out_ch = ch
#                     layers.append(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             up=True,
#                         )
#                         if resblock_updown
#                         else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
#                     )
#                     ds //= 2
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch

#         self.out = nn.Sequential(
#             normalization(ch),
#             nn.SiLU(),
#             zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
#         )
#         if self.predict_codebook_ids:
#             self.id_predictor = nn.Sequential(
#             normalization(ch),
#             conv_nd(dims, model_channels, n_embed, 1),
#             #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
#         )

#         def get_parameter_number(model):
#             total_num = sum(p.numel() for p in model.parameters())
#             trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#             print("Total Number of parameter: %.2fM" % (total_num / 1e6))
#             print("Trainable Number of parameter: %.2fM" % (total_num / 1e6))

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         self.output_blocks.apply(convert_module_to_f32)

#     def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
#         """
#         Apply the model to an input batch.
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param context: conditioning plugged in via crossattn
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         assert (y is not None) == (
#             self.num_classes is not None
#         ), "must specify y if and only if the model is class-conditional"
#         hs = []
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)

#         # if self.condition_key != 'nocond':
#         #     x = th.cat([x, context], dim=1)
#         if self.condition_key == "first_stage":
#             x = th.cat([x, context], dim=1)  # 拼接输入
#         elif self.condition_key == "SpatialRescaler":
#             # 不拼接，context单独传入
#             pass
#         elif self.condition_key == "nocond":
#             context = None
#         else:
#             # 其他情况
#             pass
#         h = x.type(self.dtype)
#         for module in self.input_blocks:
#             h = module(h, emb, context)
#             hs.append(h)
#         h = self.middle_block(h, emb, context)

#         for module in self.output_blocks:
#             hspop = hs.pop()
#             h = th.cat([h, hspop], dim=1)
#             h = module(h, emb, context)
#         h = h.type(x.dtype)

#         if self.predict_codebook_ids:
#             return self.id_predictor(h)
#         else:
#             return self.out(h)


# class EncoderUNetModel(nn.Module):
#     """
#     The half UNet model with attention and timestep embedding.
#     For usage, see UNet.
#     """

#     def __init__(
#         self,
#         image_size,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=1,
#         num_head_channels=-1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         resblock_updown=False,
#         use_new_attention_order=False,
#         pool="adaptive",
#         *args,
#         **kwargs
#     ):
#         super().__init__()

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     layers.append(
#                         AttentionBlock(
#                             ch,
#                             use_checkpoint=use_checkpoint,
#                             num_heads=num_heads,
#                             num_head_channels=num_head_channels,
#                             use_new_attention_order=use_new_attention_order,
#                         )
#                     )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch

#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=num_head_channels,
#                 use_new_attention_order=use_new_attention_order,
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self._feature_size += ch
#         self.pool = pool
#         if pool == "adaptive":
#             self.out = nn.Sequential(
#                 normalization(ch),
#                 nn.SiLU(),
#                 nn.AdaptiveAvgPool2d((1, 1)),
#                 zero_module(conv_nd(dims, ch, out_channels, 1)),
#                 nn.Flatten(),
#             )
#         elif pool == "attention":
#             assert num_head_channels != -1
#             self.out = nn.Sequential(
#                 normalization(ch),
#                 nn.SiLU(),
#                 AttentionPool2d(
#                     (image_size // ds), ch, num_head_channels, out_channels
#                 ),
#             )
#         elif pool == "spatial":
#             self.out = nn.Sequential(
#                 nn.Linear(self._feature_size, 2048),
#                 nn.ReLU(),
#                 nn.Linear(2048, self.out_channels),
#             )
#         elif pool == "spatial_v2":
#             self.out = nn.Sequential(
#                 nn.Linear(self._feature_size, 2048),
#                 normalization(2048),
#                 nn.SiLU(),
#                 nn.Linear(2048, self.out_channels),
#             )
#         else:
#             raise NotImplementedError(f"Unexpected {pool} pooling")

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)

#     def forward(self, x, timesteps):
#         """
#         Apply the model to an input batch.
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :return: an [N x K] Tensor of outputs.
#         """
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

#         results = []
#         h = x.type(self.dtype)
#         for module in self.input_blocks:
#             h = module(h, emb)
#             if self.pool.startswith("spatial"):
#                 results.append(h.type(x.dtype).mean(dim=(2, 3)))
#         h = self.middle_block(h, emb)
#         if self.pool.startswith("spatial"):
#             results.append(h.type(x.dtype).mean(dim=(2, 3)))
#             h = th.cat(results, axis=-1)
#             return self.out(h)
#         else:
#             h = h.type(x.dtype)
#             return self.out(h)
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from Register import Register
from .dual_injection import SpatialInjection, TextureCrossAttention


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MoEFFNGating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])

    def forward(self, x):
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinMoEFFN(nn.Module):
    """轻量 MoE FFN: 3 个结构相同的前馈专家 + softmax 全软路由。
    仅替换解码器第 1/2 层 SwinTransformerBlock 的 self.mlp,接口与 Mlp 一致([B,L,C]→[B,L,C])。
    aux_loss 仅在 training 模式计算并挂到 self.aux_loss;每次 forward 先清 None,
    防止 eval/采样期残留旧批次计算图引用。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., num_experts=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Mlp(in_features=in_features, hidden_features=hidden_features,
                out_features=out_features, act_layer=act_layer, drop=drop)
            for _ in range(num_experts)])
        self.router = nn.Linear(in_features, num_experts)
        self.aux_loss = None
        self.last_mean_prob = None

    def forward(self, x):
        self.aux_loss = None  # 每次前向先清空,eval/推理不残留计算图引用
        weights = torch.softmax(self.router(x), dim=-1)  # [B, L, E]
        out = 0.
        for i, expert in enumerate(self.experts):
            out = out + weights[..., i:i + 1] * expert(x)
        with torch.no_grad():
            self.last_mean_prob = weights.mean(dim=(0, 1)).detach()  # 路由健康度统计(无梯度)
        if self.training:
            mean_prob = weights.mean(dim=(0, 1))
            self.aux_loss = self.num_experts * torch.sum(
                (mean_prob - 1.0 / self.num_experts) ** 2)
        return out


class CrossAttention(nn.Module):
    """标准 Transformer cross-attention(Q 来自 Swin, K/V 来自 UNI)。
    用于把 UNI 的病理语义注入 BBDM 主干。

    query_dim:   Swin token 维度(该 stage 的 dim)
    context_dim: UNI token 维度(1024)
    heads:       多头数
    head_dim:    每头维度(默认 query_dim // heads)
    """
    def __init__(self, query_dim, context_dim, heads=8, head_dim=None, drop=0.):
        super().__init__()
        self.heads = heads
        inner_dim = head_dim * heads if head_dim else query_dim
        self.scale = (inner_dim // heads) ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(drop)
        )
        # 零初始化输出层 → 初始时不改变主干输出(无伤接入,稳定训练)
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x, context):
        """
        x:       [B, N_q, query_dim]   Swin token(Q)
        context: [B, N_kv, context_dim] UNI token(K/V 源)
        return:  [B, N_q, query_dim]   注入后的 token(残差已在 block 里加)
        """
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 分头: [B, N, inner] → [B, h, N, inner/h]
        q = q.view(q.shape[0], q.shape[1], h, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], h, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], h, -1).transpose(1, 2)

        # attention: [B, h, N_q, N_kv]
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # [B, h, N_q, inner/h] → [B, N_q, inner]
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_cross_attn=False, uni_context_dim=1024, cross_attn_heads=None,
                 use_spatial_injection=False, use_texture_attn=False, style_dim=512,
                 use_moe=False, num_experts=3):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_cross_attn = use_cross_attn
        self.use_spatial_injection = use_spatial_injection
        self.use_texture_attn = use_texture_attn
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_moe:
            self.mlp = SwinMoEFFN(in_features=dim, hidden_features=mlp_hidden_dim,
                                  act_layer=act_layer, drop=drop, num_experts=num_experts)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # ===== UNI cross-attention 子层(可选) =====
        if use_cross_attn:
            self.norm_cross = norm_layer(dim)
            ca_heads = cross_attn_heads or min(8, num_heads)
            self.cross_attn = CrossAttention(
                query_dim=dim,
                context_dim=uni_context_dim,
                heads=ca_heads,
                drop=drop,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None

        # ===== Re-BBDM 双通路注入（新增）=====
        # 1. SpatialInjection（Zero-Conv 逐像素相加，W-MSA 之前）
        if use_spatial_injection:
            self.spatial_injection = SpatialInjection(uni_dim=uni_context_dim, swin_dim=dim)
        else:
            self.spatial_injection = None

        # 2. TextureCrossAttention（W-MSA 之后）
        if use_texture_attn:
            self.norm_texture = norm_layer(dim)
            self.texture_cross_attn = TextureCrossAttention(query_dim=dim, style_dim=style_dim, heads=min(8, num_heads))
        else:
            self.norm_texture = None
            self.texture_cross_attn = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, uni_feat=None, f_spatial=None, f_style=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # ===== 1. SpatialInjection (W-MSA 之前): 零卷积逐像素相加,锁死空间位置 =====
        if self.spatial_injection is not None and f_spatial is not None:
            x = self.spatial_injection(x, f_spatial, self.input_resolution)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # ===== 2. W-MSA / SW-MSA 自注意力 (原封不动) =====
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # W-MSA 残差
        x = shortcut + self.drop_path(x)

        # ===== 3a. 旧版 UNI cross-attention (向后兼容,可选) =====
        if self.use_cross_attn and uni_feat is not None:
            x = x + self.cross_attn(self.norm_cross(x), uni_feat)

        # ===== 3b. TextureCrossAttention (W-MSA 之后): 动态查询纹理风格 =====
        if self.texture_cross_attn is not None and f_style is not None:
            x = x + self.texture_cross_attn(self.norm_texture(x), f_style)

        # ===== 4. FFN (原封不动) =====
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 use_cross_attn=False, uni_context_dim=1024,
                 use_spatial_injection=False, use_texture_attn=False, style_dim=512,
                 use_moe=False, num_experts=3):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_cross_attn=use_cross_attn,
                                 uni_context_dim=uni_context_dim,
                                 use_spatial_injection=use_spatial_injection,
                                 use_texture_attn=use_texture_attn,
                                 style_dim=style_dim,
                                 use_moe=use_moe,
                                 num_experts=num_experts)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, uni_feat=None, f_spatial=None, f_style=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, uni_feat, f_spatial, f_style)
            else:
                x = blk(x, uni_feat, f_spatial, f_style)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class UNetModel(nn.Module):
    r""" Swin Transformer UNet for BBDM
        Fully aligned with your YAML configuration
    """

    def __init__(self,
                 image_size=64,
                 in_channels=6,
                 num_classes=3,
                 model_channels=96,
                 latent_mode=False,
                 context_dim=3,
                 patch_size=4,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 final_upsample="expand_first",
                 use_uni_cross_attn=False,      # UNI cross-attention 总开关(旧版,向后兼容)
                 uni_context_dim=1024,          # UNI token 维度(UNI 是 1024)
                 uni_cross_attn_layers=None,    # 哪些解码层启用 cross-attn
                 use_dual_injection=False,      # Re-BBDM 双通路总开关
                 dual_injection_layers=None,    # 哪些解码层启用双通路注入, 默认 [1, 2]
                 style_dim=512,                 # F_style 维度(CONCH 是 512)
                 use_moe=False,                 # 解码器 FFN MoE 总开关(默认关=零行为变化)
                 moe_layers=None,               # 哪些解码层启用 MoE, 默认 [1, 2]
                 num_experts=3,                 # 专家数(肾/肝/脑)
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_moe = use_moe
        self.moe_layers = moe_layers if moe_layers is not None else [1, 2]
        self.num_moe_experts = num_experts
        self.num_layers = len(depths)
        self.embed_dim = model_channels
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.is_latent = latent_mode
        self.patch_size = patch_size
        self.use_uni_cross_attn = use_uni_cross_attn
        if uni_cross_attn_layers is None:
            uni_cross_attn_layers = [0, 1]
        self.uni_cross_attn_layers = uni_cross_attn_layers
        # Re-BBDM 双通路注入配置
        self.use_dual_injection = use_dual_injection
        if dual_injection_layers is None:
            dual_injection_layers = [1, 2]  # 默认解码层 1 和 2
        self.dual_injection_layers = dual_injection_layers
        self.style_dim = style_dim

        # 时间嵌入层
        time_embed_dim = self.embed_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.embed_dim),
        )

        # 补丁嵌入层
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 潜在空间上下文投影
        if self.is_latent:
            self.latent_context_proj = nn.Sequential(
                nn.Conv2d(context_dim, self.embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
            )
            self.norm_latent_inject = nn.LayerNorm(self.embed_dim)
        else:
            # 占位符，防止加载 Checkpoint 时报 KeyError
            self.latent_context_proj = nn.Identity()
            self.norm_latent_inject = nn.Identity()
        # ==================== 编码器 ====================
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(self.embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2)
            else:
                # 该解码层是否启用 UNI cross-attention
                this_use_ca = use_uni_cross_attn and (i_layer in self.uni_cross_attn_layers)
                layer_up = BasicLayer_up(dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):
                                                       sum(depths[:(self.num_layers - 1 - i_layer + 1)])],
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_cross_attn=this_use_ca,
                                         uni_context_dim=uni_context_dim,
                                         use_spatial_injection=use_dual_injection and (i_layer in self.dual_injection_layers),
                                         use_texture_attn=use_dual_injection and (i_layer in self.dual_injection_layers),
                                         style_dim=style_dim,
                                         use_moe=use_moe and (i_layer in self.moe_layers),
                                         num_experts=num_experts)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # 归一化层
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # 最终上采样和输出
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(image_size // patch_size, image_size // patch_size),
                dim_scale=patch_size,
                dim=self.embed_dim,
                norm_layer=norm_layer
            )
            self.output = nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                bias=False
            )
        print("编码器层数:", [len(l.blocks) for l in self.layers])
        print("解码器层数:", [len(l.blocks) if hasattr(l, "blocks") else 0 for l in self.layers_up])
        # 权重初始化
        self.apply(self._init_weights)
        # MoE router 零初始化(须在 apply 之后): 初始路由精确均匀 1/3,
        # 避免随机小权重经 LayerNorm 大幅输入放大出初始偏置
        if self.use_moe:
            for m in self.modules():
                if isinstance(m, SwinMoEFFN):
                    nn.init.zeros_(m.router.weight)
                    nn.init.zeros_(m.router.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_aux_loss(self):
        """汇总所有 SwinMoEFFN 的负载均衡损失;eval/推理期(或未启用 MoE)返回 0。"""
        total = None
        for m in self.modules():
            if isinstance(m, SwinMoEFFN) and m.aux_loss is not None:
                total = m.aux_loss if total is None else total + m.aux_loss
        if total is None:
            return torch.tensor(0.0)
        return total

    def get_moe_stats(self):
        """各 SwinMoEFFN 平均的三专家激活概率(监控路由是否塌陷)。"""
        probs, n = None, 0
        for m in self.modules():
            if isinstance(m, SwinMoEFFN) and m.last_mean_prob is not None:
                probs = m.last_mean_prob if probs is None else probs + m.last_mean_prob
                n += 1
        if n == 0:
            return [0.0, 0.0, 0.0]
        return [round(p, 4) for p in (probs / n).tolist()]

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        if self.final_upsample == "expand_first":
            x = self.up(x)
            ps = self.patch_embed.patch_size
            x = x.view(B, ps[0] * H, ps[1] * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x, timesteps, context=None, uni_feat=None, f_spatial=None, f_style=None):
        B, C, H, W = x.shape
        # 输入处理 (像素模式拼接，潜在模式注入)
        if not self.is_latent:
            # 🟢 像素模式：保持原版拼接
            if context is not None:
                x = torch.cat([x, context], dim=1)
            x = self.patch_embed(x)
        else:
            # 🔵 潜在模式：特征 Add 注入
            x = self.patch_embed(x)  # (B, N, C)
            if context is not None:
                # 动态空间对齐：将 context 插值到与 Patch 后的特征图空间一致
                target_size = (H // self.patch_size, W // self.patch_size)
                if context.shape[2:] != target_size:
                    context = F.interpolate(context, size=target_size, mode='bilinear', align_corners=False)

                # 特征映射与 Add
                c_feat = self.latent_context_proj(context)  # (B, C, H_p, W_p)
                c_feat = c_feat.view(B, self.embed_dim, -1).permute(0, 2, 1)  # (B, N, C)
                x = self.norm_latent_inject(x + c_feat)  # 稳定注入

        # 时间步注入
        t_emb = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        x = x + t_emb.unsqueeze(1)

        # 位置嵌入和Dropout
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 编码器前向
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)

        # 解码器前向(uni_feat/f_spatial/f_style 下发给 BasicLayer_up; PatchExpand 不接收)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)  # PatchExpand, 无 blocks
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x, uni_feat, f_spatial, f_style)  # 透传所有条件特征
        x = self.norm_up(x)

        # 最终上采样输出
        x = self.up_x4(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops