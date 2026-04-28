# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import pdb
import glob
import json
import math
import os
import types
from einops import rearrange
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from ..utils import cfg_skip
from .attention_utils import attention
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter

from .wan_transformer3d import (
    WanRMSNorm,
    rope_apply_qk,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_params,
    MLPProj,
    WanSelfAttention,
    sinusoidal_embedding_1d
)

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")
# create_block_mask = torch.compile(create_block_mask, dynamic=False, mode="max-autotune")

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 cross_view = False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.cross_view = cross_view
        self.max_attention_size = 5632 if window_size[0] == -1 else window_size[0] * 1560

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, dtype=torch.bfloat16, t=0, kv_cache = None, current_start=0, cache_start=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        #pdb.set_trace()
        q, k, v = qkv_fn(x)

        # pdb.set_trace() #todo: len(kv_cache) = 4 ???
        if kv_cache is None:
            q, k = rope_apply_qk(q, k, grid_sizes, freqs, self.cross_view)
            # @Zhenya: change to flex attention for customized mask
            x = flex_attention(
                    query=q.transpose(2, 1),
                    key=k.transpose(2, 1),
                    value=v.transpose(2, 1),
                    block_mask=block_mask,
                    kernel_options={
                                    "BLOCK_M": 64,
                                    "BLOCK_N": 64,
                                    "BLOCK_M1": 32,
                                    "BLOCK_N1": 64,
                                    "BLOCK_M2": 64,
                                    "BLOCK_N2": 32,
                                }
                ).transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            num_new_tokens = roped_query.shape[1]
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_cross_view = False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.use_cross_view = use_cross_view

        if self.use_cross_view:
            self.norm4 = WanLayerNorm(dim, eps)
            self.cross_view_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, cross_view = True)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask, 
        dtype=torch.bfloat16,
        t=0,
        mv_freqs = None,
        kv_cache = None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # pdb.set_trace()

        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2) #a tuple contains 6 elements, each has shape-[6, 1, 1, 1536]
            e = [e.squeeze(2) for e in e] #a list contains 6 elements, each has shape-[6, 1, 1536]
        else:        
            e = (self.modulation + e).chunk(6, dim=1)
        
        ##pdb.set_trace()

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, block_mask, dtype, t, kv_cache, current_start, cache_start)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context, context_lens, dtype, t=t)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3] #todo: a little bit different
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5] #todo: a little bit different
            return x

        # pdb.set_trace()
        x = cross_attn_ffn(x, context, context_lens, e)

        # print(x.shape) #infer: torch.Size([12, 704, 1536]) train: torch.Size([6, 5632, 1536])

        if self.use_cross_view:
            # multi-view attention
            x = x.unflatten(0, (-1, 6)) # B V S D # group each 6 views into a group (for CFG)
            bs = x.shape[0]
            x = rearrange(x, 'b v (t h w) d -> (b t h) (v w) d', h=16, w=32) #todo: support multi-card infrence
            mv_seq_lens = torch.tensor([x.shape[1]]*x.shape[0], dtype=seq_lens.dtype, device=seq_lens.device)
            mv_grid_sizes = None
            y = self.cross_view_attn(self.norm4(x), mv_seq_lens, mv_grid_sizes, mv_freqs, dtype, t=t)
            x = x + self.cross_norm(y, x)
            #reshape to original shape
            x = rearrange(x, '(b t h) (v w) d -> b v (t h w) d', b=bs, h=16, w=32).flatten(0, 1)

        return x
    
    def cross_norm(self, y, x, scale=0.2):
        # Inspired by the Cross Norm proposed in ControlNext
        ###pdb.set_trace()
        mean_latents, std_latents = torch.mean(x, dim=(1, 2), keepdim=True), torch.std(x, dim=(1, 2), keepdim=True)
        mean_control, std_control = torch.mean(y, dim=(1, 2), keepdim=True), torch.std(y, dim=(1, 2), keepdim=True)
        y = (y - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
        return y * scale


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x



class CausalWanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, use_cross_view=True)
            if i%3==0 or i==num_layers-1 else
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, use_cross_view=False)
            for i in range(num_layers)
        ])

        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim

        ##pdb.set_trace()

        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        self.mv_freqs = rope_params(max_seq_len = 32*6, dim=d)
        self.block_mask = None

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # For normal model.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

        # For vace model.
        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, 
        num_frames: int = 21,
        frame_seqlen: int = 1560, 
        num_frame_per_block=1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """

        # adapt from 5120 to 5632
        # pdb.set_trace()

        total_length = (num_frames + 1) * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        frame_indices = torch.cat([torch.tensor([0], device=device), frame_indices + frame_seqlen])

        for idx, tmp in enumerate(frame_indices):
            if idx ==0:
                ends[tmp:tmp + frame_seqlen] = tmp + frame_seqlen
            else:
                ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=False, device=device)

        # pdb.set_trace()

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @cfg_skip()
    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        kv_cache = None,
        crossattn_cache = None,
        current_start = None,
        current_end = None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Wan2.2 don't need a clip.
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params

        # import pdb
        ##pdb.set_trace()

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)
            self.mv_freqs = self.mv_freqs.to(device)

        if self.block_mask is None:
            #pdb.set_trace() # e.shape = [6, 5632, 1536] e0.shape = [6, 5632, 6, 1536]
            # optimized block_mask implementation
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                                        device, 
                                        num_frames=x.shape[2], 
                                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                                        num_frame_per_block= 2
                                )
            # dense_mask = self.block_mask.to_dense()
            # self.block_mask = dense_mask.unsqueeze(4).unsqueeze(3).repeat(1, 1, 1, 128, 1, 128).flatten(4, 5).flatten(2, 3)
            # extend_size = 512
            # extended_block_mask = torch.zeros((1, 1, self.block_mask.shape[2] + extend_size, self.block_mask.shape[3] + extend_size), device=self.block_mask.device, dtype=self.block_mask.dtype)
            # extended_block_mask[:, :, extend_size:, extend_size:] = self.block_mask
            # extended_block_mask[:, :, :, :extend_size] = 1
            # self.block_mask = extended_block_mask
            
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] #[1, 1536, 10, 16, 32]
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        #pdb.set_trace() #[1, 1536, 10, 16, 32]

        x = [u.flatten(2).transpose(1, 2) for u in x] # list of [1, 5120, 1536]
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)] # 5632 = 11 * 16 * 32

            if t.dim() != 1 and t.size(1) < seq_len: 
                # pad_size = seq_len - t.size(1)
                # last_elements = t[:, -1].unsqueeze(1)
                # padding = last_elements.repeat(1, pad_size)
                # t = torch.cat([padding, t], dim=1)

                #adapt to Diffusion Forcing: 
                #step1: [6, 10] --> [6, 5120]
                t = t.unsqueeze(2).repeat(1, 1, full_ref.shape[1]).flatten(1)
                #step2: [6, 5120] --> [6, 5632]
                t_ref = torch.zeros((t.shape[0], full_ref.shape[1]), device=t.device, dtype=t.dtype)
                t = torch.cat([t_ref, t], dim=1)

        #@zhenya: the repeated t.shape = [6, 5632]
        if subject_ref is not None:
            subject_ref_frames = subject_ref.size(2)
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += subject_ref.size(1)
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for _subject_ref, u in zip(subject_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        #pdb.set_trace()
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    exit(-1) #should not step in here
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            # #pdb.set_trace()
            # print(context_clip.shape, context.shape) #torch.Size([12, 257, 1536]) torch.Size([2, 512, 1536])
            # exit(-1)

            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        
                        # import pdb
                        # ##pdb.set_trace()

                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            self.mv_freqs,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            mv_freqs = self.mv_freqs,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                    ##pdb.set_trace()
                    
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        self.block_mask,
                        dtype,
                        t,
                        self.mv_freqs,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        block_mask = self.block_mask,
                        dtype=dtype,
                        t=t,
                        mv_freqs = self.mv_freqs  
                    )
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            # pdb.set_trace()
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x.to(dtype), e.unsqueeze(2).to(dtype), **ckpt_kwargs)
        else:
            x = self.head(x.to(dtype), e.unsqueeze(2).to(dtype))

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


    @cfg_skip()
    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        kv_cache = None,
        crossattn_cache = None,
        current_start = 0,
        cache_start = 0,
        ref_input = False
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Wan2.2 don't need a clip.
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params

        # import pdb; pdb.set_trace()
        # print(t, t.shape) #shape = [6, 1]

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)
            self.mv_freqs = self.mv_freqs.to(device)
            
        if ref_input:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2) #[6, 512, 1536]
            grid_sizes = torch.stack([torch.tensor([1, 16, 32], dtype=torch.long) for _ in x]).to(full_ref.device) # ugly hard-code 
            x = full_ref
            seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long) #todo: recheck
        else:
            assert y is not None #concat control_latents to the noise latents
            # pdb.set_trace()
            # print(x.shape, y.shape)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x] #[1, 1536, 10, 16, 32]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x] # list of [1, 5120, 1536]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            x = torch.cat(x)

        # @zhenya: the below is not needed for inference
        # if self.ref_conv is not None and full_ref is not None:
            # full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            # grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            # seq_len += full_ref.size(1)
            # x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)] # 5632 = 11 * 16 * 32

            # if t.dim() != 1 and t.size(1) < seq_len: 
            #     t = t.unsqueeze(2).repeat(1, 1, full_ref.shape[1]).flatten(1)
            #     t_ref = torch.zeros((t.shape[0], full_ref.shape[1]), device=t.device, dtype=t.dtype)
            #     t = torch.cat([t_ref, t], dim=1)

        
        # seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len

        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])

        # time embeddings #todo: need to be re-implemented for casual inference
        # pdb.set_trace()

        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)) #[6, 1536]
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape) #[6, 1, 6, 1536]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        

        # pdb.set_trace()

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        
                        # import pdb
                        # ##pdb.set_trace()

                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            self.mv_freqs,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            mv_freqs = self.mv_freqs,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block_index, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                    # kwargs = {
                    #             "kv_cache": kv_cache[block_index],
                    #             "current_start": current_start,
                    #             "cache_start": cache_start
                    #         }
                    
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        self.block_mask,
                        dtype,
                        t,
                        self.mv_freqs,
                        kv_cache[block_index],
                        current_start,
                        cache_start,
                        **ckpt_kwargs,
                    )

                    #todo: re-implementation
                    # kwargs = dict(
                    #     e=e0,
                    #     seq_lens=seq_lens,
                    #     grid_sizes=grid_sizes,
                    #     freqs=self.freqs,
                    #     context=context,
                    #     context_lens=context_lens,
                    #     block_mask = self.block_mask,
                    #     dtype=dtype,
                    #     t=t,
                    #     mv_freqs = self.mv_freqs  
                    # )

                    # kwargs.update(
                    #     {
                    #         "kv_cache": kv_cache[block_index],
                    #         "current_start": current_start,
                    #         "cache_start": cache_start
                    #     }
                    # )

                    # x = torch.utils.checkpoint.checkpoint(
                    #     create_custom_forward(block),
                    #     x, **kwargs,
                    #     use_reentrant=False,
                    # )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        block_mask = self.block_mask,
                        dtype=dtype,
                        t=t,
                        mv_freqs = self.mv_freqs  
                    )

                    kwargs.update(
                        {
                            "kv_cache": kv_cache[block_index],
                            "current_start": current_start,
                            "cache_start": cache_start
                        }
                    )
                    
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            # pdb.set_trace()
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x.to(dtype), e.unsqueeze(1).unsqueeze(2).to(dtype), **ckpt_kwargs)
        else:
            x = self.head(x.to(dtype), e.unsqueeze(1).unsqueeze(2).to(dtype))

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # todo: ar inference may not need to remove the full ref
        # if self.ref_conv is not None and full_ref is not None:
        #     full_ref_length = full_ref.size(1)
        #     x = x[:, full_ref_length:]
        #     grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # if subject_ref is not None:
        #     subject_ref_length = subject_ref.size(1)
        #     x = x[:, :-subject_ref_length]
        #     grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
    
    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
        
    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
