# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from collections import defaultdict, namedtuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.multiview_dit import MultiViewDiT
from imaginaire.utils.graph import create_cuda_graph
from cosmos_predict2.models.text2image_dit import (
    Attention,
    Block,
    SACConfig,
    VideoRopePosition3DEmb,
)
from cosmos_predict2.module.attention2 import MultiViewCrossAttentionSDPA

from megatron.core import parallel_state
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])

@contextmanager
def temporarily_remove_hooks(root: torch.nn.Module):
    saved = []  # [(m, f_pre_dict, f_dict, b_dict)]
    for m in root.modules():
        fpre = dict(m._forward_pre_hooks)
        fwd  = dict(m._forward_hooks)
        bwd  = dict(m._backward_hooks)
        if fpre or fwd or bwd:
            saved.append((m, fpre, fwd, bwd))
            m._forward_pre_hooks.clear()
            m._forward_hooks.clear()
            m._backward_hooks.clear()
    try:
        yield
    finally:
        for m, fpre, fwd, bwd in saved:
            for _, fn in fpre.items():
                m.register_forward_pre_hook(fn)
            for _, fn in fwd.items():
                m.register_forward_hook(fn)
            for _, fn in bwd.items():
                m.register_full_backward_hook(fn)

class FramePackMemory(nn.Module):
    def __init__(self, in_ch, D,
                 kernels=((1,2,2),(1,4,4),(1,8,8)),
                 lam=2.0, max_tokens=1024,
                 output_order="oldest_first",
                 per_view_budget=True):
        super().__init__()
        self.lam = lam
        self.max_tokens = max_tokens
        self.output_order = output_order
        self.per_view_budget = per_view_budget

        self.blocks = nn.ModuleList([
            nn.Sequential(nn.AvgPool3d(k, k), nn.Conv3d(in_ch, D, 1))
            for k in kernels
        ])

    def _level_for_t(self, t, T):
        dist = (T - 1) - t
        lvl = 0 if dist <= 0 else int(math.floor(math.log(dist + 1, self.lam)))
        return min(lvl, len(self.blocks) - 1)

    def forward(self, hist):
        if hist is None: return None
        B, C, V, T, H, W = hist.shape
        if T == 0: return None

        levels = [self._level_for_t(t, T) for t in range(T)]
        per_frame = [None] * T  # each: (B,V,HpWp,D)

        x_bv = hist.permute(0,2,1,3,4,5).contiguous().view(B*V, C, T, H, W)

        for lvl, block in enumerate(self.blocks):
            sel = [t for t,l in enumerate(levels) if l == lvl]
            if not sel: continue
            x = x_bv[:, :, sel]                 # (B*V,C,Tk,H,W)
            y = block(x)                         # (B*V,D,Tk,H',W')
            y = y.flatten(3).permute(0,2,3,1)    # (B*V,Tk,HpWp,D)
            Tk, HpWp, D_ = len(sel), y.shape[2], y.shape[3]
            y = y.view(B, V, Tk, HpWp, D_)       # (B,V,Tk,HpWp,D)

            for j,t in enumerate(sel):
                per_frame[t] = y[:, :, j]        # (B,V,HpWp,D)
        frame_order = range(T-1, -1, -1) if self.output_order == "recent_first" else range(T)

        if self.per_view_budget:
            max_per_view = max(1, self.max_tokens // V)
            # list of tokens per view: each element is list[(B,HpWp,D)]
            toks_v = [[] for _ in range(V)]
            cur_v = [0 for _ in range(V)]

            for t in frame_order:
                tok = per_frame[t]
                if tok is None: 
                    continue
                # tok: (B,V,HpWp,D) -> split per view
                for v in range(V):
                    if cur_v[v] >= max_per_view:
                        continue
                    tv = tok[:, v]  # (B,HpWp,D)
                    remain = max_per_view - cur_v[v]
                    if tv.shape[1] > remain:
                        toks_v[v].append(tv[:, :remain])
                        cur_v[v] += remain
                    else:
                        toks_v[v].append(tv)
                        cur_v[v] += tv.shape[1]

            # concat per view -> (B, V, N_token, D)
            mem_per_view = []
            for v in range(V):
                if len(toks_v[v]) == 0:
                    mem_v = torch.zeros(B, 0, self.blocks[0][1].out_channels, device=hist.device, dtype=hist.dtype)
                else:
                    mem_v = torch.cat(toks_v[v], dim=1)  # (B,Nv,D)
                mem_per_view.append(mem_v)

            Nmax = max(m.shape[1] for m in mem_per_view)
            D = mem_per_view[0].shape[-1]
            out = torch.zeros(B, V, Nmax, D, device=hist.device, dtype=mem_per_view[0].dtype)
            for v in range(V):
                Nv = mem_per_view[v].shape[1]
                if Nv > 0:
                    out[:, v, :Nv] = mem_per_view[v]
            return out

        else:
            toks = []
            cur = 0
            for t in frame_order:
                tok = per_frame[t]
                if tok is None: 
                    continue
                tok = tok.reshape(B, -1, tok.shape[-1])  # (B,V*HpWp,D)
                if cur + tok.shape[1] > self.max_tokens:
                    remain = self.max_tokens - cur
                    if remain > 0:
                        toks.append(tok[:, :remain])
                    break
                toks.append(tok)
                cur += tok.shape[1]
                if cur >= self.max_tokens:
                    break
            if not toks: 
                return None
            mem = torch.cat(toks, dim=1)  # (B,<=max_tokens,D)

            max_per_view = max(1, mem.shape[1] // V)
            mem = mem[:, :max_per_view * V]
            mem = mem.view(B, V, max_per_view, -1)
            return mem

def _fourier_encode(x: torch.Tensor, num_bands: int, max_freq: float = 10.0) -> torch.Tensor:
    """
    x: (..., d)
    returns: (..., d * (2*num_bands + 1)) with [x, sin(2^k*pi*x), cos(2^k*pi*x)]
    """
    # frequencies: 1, 2, 4, ... scaled up to max_freq-ish
    device = x.device
    d = x.shape[-1]
    # geometric progression
    freqs = torch.logspace(0, math.log10(max_freq), steps=num_bands, base=10.0, device=device)  # (B,)
    freqs = freqs * math.pi
    # (..., 1, d) * (num_bands,) -> (..., num_bands, d)
    xb = x.unsqueeze(-2) * freqs.view(*([1] * (x.dim() - 1)), num_bands, 1)
    sin = torch.sin(xb)
    cos = torch.cos(xb)
    # concat: (..., d) + (..., num_bands, d)*2 -> (..., d*(2*num_bands) + d)
    out = torch.cat([x, sin.flatten(-2), cos.flatten(-2)], dim=-1)
    return out

def _make_2d_sincos(H: int, W: int, dim: int, device) -> torch.Tensor:
    """
    returns: (H*W, dim) 2D sin-cos positional embedding
    """
    assert dim % 4 == 0, "2D sincos dim should be divisible by 4"
    dim_each = dim // 2
    assert dim_each % 2 == 0
    half = dim_each // 2

    y = torch.linspace(-1.0, 1.0, H, device=device)
    x = torch.linspace(-1.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)

    # freqs
    omega = torch.arange(half, device=device).float()
    omega = 1.0 / (10000 ** (omega / half))  # (half,)

    # (H,W,half)
    out_x = xx[..., None] * omega
    out_y = yy[..., None] * omega
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # (H,W,2*half)
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # (H,W,2*half)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # (H,W,dim)
    return pe.view(H * W, dim)

class BlurPool2d(nn.Module):
    """Fixed blur then stride-2 downsample (anti-aliased)."""
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        assert stride in (2, 4), "keep it simple"
        # binomial [1,2,1] outer-product -> 3x3
        kernel = torch.tensor([1., 2., 1.])
        kernel2d = (kernel[:, None] * kernel[None, :])
        kernel2d = kernel2d / kernel2d.sum()
        self.register_buffer("kernel", kernel2d[None, None, :, :])  # (1,1,3,3)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        # depthwise conv with blur kernel
        k = self.kernel.repeat(self.channels, 1, 1, 1)  # (C,1,3,3)
        x = F.conv2d(x, k, stride=1, padding=1, groups=self.channels)
        x = x[:, :, ::self.stride, ::self.stride]
        return x


class PoseAwareHistoryTokenizer(nn.Module):
    def __init__(
        self,
        in_ch: int = 16,
        D: int = 1024,
        spatial_downsample: int = 2,
        downsample_mode: str = "conv",   # "avg" | "conv" | "blurconv"
        stem_hidden: int = 128,          # used for conv stem
        pe_2d_dim: int = 128,
        pose_pe_dim: int = 256,
        pose_fourier_bands: int = 8,
        pose_max_freq: float = 10.0,
        rot_scale: float = 1.0,
        trans_scale: float = 1.0,
        view_embed: bool = False,
        max_views: int = 16,
    ):
        super().__init__()
        self.D = D
        self.spatial_downsample = spatial_downsample
        self.downsample_mode = downsample_mode
        self.pe_2d_dim = pe_2d_dim

        # ---- learnable downsample + projection stem ----
        # goal: (BVT,16,32,32) -> (BVT,D,Hp,Wp)
        if spatial_downsample == 1:
            # no downsample: still allow a small stem if you want
            if downsample_mode == "avg":
                self.down = nn.Identity()
                self.latent_proj = nn.Conv2d(in_ch, D, 1)
            else:
                self.down = nn.Identity()
                self.latent_proj = nn.Sequential(
                    nn.Conv2d(in_ch, stem_hidden, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(stem_hidden, D, 1),
                )
        else:
            assert spatial_downsample in (2, 4), "common choices"
            if downsample_mode == "avg":
                self.down = nn.AvgPool2d(kernel_size=spatial_downsample, stride=spatial_downsample)
                self.latent_proj = nn.Conv2d(in_ch, D, 1)

            elif downsample_mode == "conv":
                # stride conv does both downsample + feature mixing
                # if ds=4, do two stride-2 blocks
                blocks = []
                c_in = in_ch
                ds_left = spatial_downsample
                while ds_left > 1:
                    blocks += [
                        nn.Conv2d(c_in, stem_hidden, 3, stride=2, padding=1),
                        nn.SiLU(),
                    ]
                    c_in = stem_hidden
                    ds_left //= 2
                blocks += [nn.Conv2d(stem_hidden, D, 1)]
                self.down = nn.Sequential(*blocks)
                self.latent_proj = nn.Identity()

            elif downsample_mode == "blurconv":
                # blurpool (fixed) + conv to D
                # ds=4 -> blur+stride2 twice
                ops = []
                c_in = in_ch
                ds_left = spatial_downsample
                while ds_left > 1:
                    ops += [
                        BlurPool2d(c_in, stride=2),
                        nn.Conv2d(c_in, stem_hidden, 3, padding=1),
                        nn.SiLU(),
                    ]
                    c_in = stem_hidden
                    ds_left //= 2
                ops += [nn.Conv2d(stem_hidden, D, 1)]
                self.down = nn.Sequential(*ops)
                self.latent_proj = nn.Identity()
            else:
                raise ValueError(f"unknown downsample_mode={downsample_mode}")

        # ---- optional view embedding ----
        self.use_view_embed = view_embed
        self.max_views = max_views
        if view_embed:
            self.view_emb = nn.Embedding(max_views, D)

        # ---- pose encoding ----
        self.rot_scale = rot_scale
        self.trans_scale = trans_scale
        self.pose_fourier_bands = pose_fourier_bands
        self.pose_max_freq = pose_max_freq

        pose_in_dim = 6 * (2 * pose_fourier_bands + 1)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_in_dim, pose_pe_dim),
            nn.SiLU(),
            nn.Linear(pose_pe_dim, D),
        )

        self.gate = nn.Sequential(
            nn.Linear(6, D),
            nn.SiLU(),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )

        self.pe2d_proj = nn.Linear(pe_2d_dim, D) if pe_2d_dim > 0 else None

    def forward(self, hist_latent: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        assert hist_latent.dim() == 6
        B, C, V, T, H, W = hist_latent.shape
        assert C == 16
        if state is not None:
            assert state.shape == (B, T, 6)

        # (B,V,T,C,H,W) -> (BVT,C,H,W)
        x = hist_latent.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * V * T, C, H, W)

        # downsample + project
        x = self.down(x)
        x = self.latent_proj(x)  # if Identity, already D
        Hp, Wp = x.shape[-2], x.shape[-1]

        # tokens: (BVT, HpWp, D)
        x = x.flatten(2).transpose(1, 2).contiguous()

        # 2D pos enc (Hp,Wp)
        if self.pe2d_proj is not None:
            pe2d = _make_2d_sincos(Hp, Wp, self.pe_2d_dim, device=x.device)
            pe2d = self.pe2d_proj(pe2d.to(torch.bfloat16))  # (HpWp,D)
            x = x + pe2d.unsqueeze(0)

        # (B,V,T,HpWp,D)
        x = x.view(B, V, T, Hp * Wp, self.D)

        # pose enc per frame (broadcast)
        if state is not None:
            s = state.clone()
            s[..., :3] = s[..., :3] * self.trans_scale
            s[..., 3:] = s[..., 3:] * self.rot_scale
            pose_feat = _fourier_encode(s, self.pose_fourier_bands, self.pose_max_freq)  # (B,T,pose_in_dim)
            pose_emb = self.pose_mlp(pose_feat.to(torch.bfloat16))  # (B,T,D)
            g = self.gate(state)                 # (B,T,D)
            x = x + (g * pose_emb).unsqueeze(1).unsqueeze(3)  # (B,V,T,HpWp,D)

        # view emb
        if self.use_view_embed:
            if V > self.max_views:
                raise ValueError(f"V={V} exceeds max_views={self.max_views}")
            v_ids = torch.arange(V, device=x.device, dtype=torch.long)
            x = x + self.view_emb(v_ids).view(1, V, 1, 1, self.D)

        # flatten to (B, token_num, D)
        return x.reshape(B, V * T * Hp * Wp, self.D)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x
    
class MultiViewCrossAttention(Attention):
    def __init__(self, *args, state_t: int = None, **kwargs) -> None:  # noqa: RUF013
        super().__init__(*args, **kwargs)
        assert self.qkv_format == "bshd", "MultiViewCrossAttention only supports qkv_format='bshd'"
        self.state_t = state_t

    def forward(self, x, context=None, rope_emb=None):
        assert not self.is_selfattn, "MultiViewCrossAttention does not support self-attention"
        B, L, D = x.shape # L=3072=2*1536=2*6*256, i.e., 256 tokens per target latent view
        # n_cameras = context.shape[1] // 1280 # 512, 3072 / 2
        n_cameras = context.shape[1] // 1280
        # n_cameras = context.shape[1] // 512
        x_B_L_D = rearrange(x, "B (V L) D -> (V B) L D", V=n_cameras)
        context_B_M_D = rearrange(context, "B (V M) D -> (V B) M D", V=n_cameras) if context is not None else None
        x_B_L_D = super().forward(x_B_L_D, context_B_M_D, rope_emb=rope_emb)
        x_B_L_D = rearrange(x_B_L_D, "(V B) L D -> B (V L) D", V=n_cameras)
        return x_B_L_D
    
class CrossViewAttention(Attention):
    def __init__(self, *args, cross_view_attn_map: dict[int, list[int]], **kwargs):
        super().__init__(*args, **kwargs)
        del self.attn_op
        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type="padding",  # important
                attention_type="cross",  # important
            )
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")
        self.cross_view_attn_map = cross_view_attn_map
        self.max_neighbors = max(len(neighbors) for neighbors in cross_view_attn_map.values())
        self.neighbor_indices = None
        self.neighbor_mask = None

    def forward(self, x, view_indices_B_V, sv_video_size: VideoSize):
        """
        x: (B, V, L, D)
        view_indices_B_V: (B, V)
        sv_video_size: VideoSize (T, H, W), where T * H * W = L
        """
        assert not self.is_selfattn, "CrossViewAttention does not support self-attention"
        B, V, L, D = x.shape
        T, H, W = sv_video_size
        assert T * H * W == L, f"T * H * W != L: {T * H * W} != {L}"

        # move time dimension to batch dimension
        x = rearrange(x, "b v (t h w) d -> (b t) v (h w) d", t=T, h=H, w=W)
        B, V, L, D = x.shape

        view_indices_B_V = view_indices_B_V.repeat_interleave(T, dim=0).long()

        # Create neighbor indices and mask on the fly, only once.
        if self.neighbor_indices is None or self.neighbor_indices.device != x.device:
            num_total_views = len(self.cross_view_attn_map)
            neighbor_indices = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.long, device=x.device)
            neighbor_mask = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.bool, device=x.device)
            for i in range(num_total_views):
                neighbors = self.cross_view_attn_map[i]
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_indices[i, j] = neighbor_idx
                    neighbor_mask[i, j] = True
            self.neighbor_indices = neighbor_indices
            self.neighbor_mask = neighbor_mask

        num_total_views = len(self.cross_view_attn_map)
        view_indices_to_tensor_pos = torch.full(
            (B, num_total_views), -1, dtype=torch.long, device=x.device
        )  # include out of range view index
        b_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, V).long()
        view_indices_to_tensor_pos[b_indices, view_indices_B_V] = (
            torch.arange(V, device=x.device).unsqueeze(0).expand(B, -1)
        )

        neighbor_view_indices = self.neighbor_indices[view_indices_B_V]  # may include out of range view index
        gather_tensor_pos = view_indices_to_tensor_pos[
            b_indices.unsqueeze(2), neighbor_view_indices
        ]  # [B, V, max_neighbors], out of range view index will be -1

        # Sort to move all -1 to the end, which is convenient for creating attention mask.
        gather_tensor_pos, sorted_indices = torch.sort(gather_tensor_pos, dim=-1, descending=True)

        b_indices_for_gather = torch.arange(B, device=x.device)[:, None, None]
        # Clamp to avoid index error. Masked values will be ignored in attention.
        neighbor_features = x[
            b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
        ]  # [B, V, max_neighbors, L, C]

        # Prepare for attention
        query = self.q_proj(rearrange(x, "b v l c -> (b v) l c"))  # [B*V, L, C]
        context = rearrange(neighbor_features, "b v n l c -> (b v) (n l) c")  # [B*V, max_neighbors*L, C]
        key = self.k_proj(context)
        value = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (query, key, value),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Create attention mask
        is_neighbor_present = gather_tensor_pos != -1  # [B, V, max_neighbors]
        mask_for_input_views = self.neighbor_mask[view_indices_B_V]  # [B, V, n]

        # Reorder mask_for_input_views to match the sorted gather_tensor_pos
        mask_for_input_views = torch.gather(mask_for_input_views, -1, sorted_indices)
        final_mask = is_neighbor_present & mask_for_input_views

        mask_per_view = rearrange(final_mask, "b v n -> (b v) n")  # [BV, n]
        mask_kv = mask_per_view.repeat_interleave(L, dim=1)  # [BV, n*L]

        # Reshape mask to [batch_size, 1, 1, max_seqlen_kv] as per official documentation.
        mask = rearrange(mask_kv, "bv l_kv -> bv 1 1 l_kv")  # [BV, 1, 1, n*L]
        atten_mask_kv = ~mask  # 0 means keep, 1 means mask
        atten_mask_q = torch.zeros(query.shape[0], 1, 1, query.shape[1]).to(atten_mask_kv)

        attention_output = self.attn_op(q, k, v, attention_mask=(atten_mask_q, atten_mask_kv))
        attention_output = attention_output.flatten(2)  # [B*V, L, H*D]
        output = self.output_dropout(self.output_proj(attention_output))
        output = rearrange(output, "(b v) l d -> b v l d", v=V)
        # recover time dimension from batch to seq
        output = rearrange(output, "(b t) v (h w) d -> b v (t h w) d", t=T, h=H, w=W)
        return output

    def set_context_parallel_group(self, process_group, ranks, stream):
        raise NotImplementedError("Cross View Attention doesn't need communication")

    
class MultiViewCrossBlock(Block):
    """
    A transformer block that takes n_cameras as input. This block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        self_attention_backend: str = "transformer_engine",
        cross_attention_backend: str = "transformer_engine",
        natten_params: Mapping | None = None,
        state_t: int = None,  # noqa: RUF013
        enable_cross_view_attn: bool = True,
    ):
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            self_attention_backend,
            cross_attention_backend,
            natten_params,
        )
        self.state_t = state_t
        self.enable_cross_view_attn = enable_cross_view_attn
        del self.cross_attn
        # self.cross_attn = MultiViewCrossAttention(
        #     x_dim,
        #     context_dim,
        #     num_heads,
        #     x_dim // num_heads,
        #     qkv_format="bshd",
        #     state_t=state_t,
        #     backend=cross_attention_backend,
        # )
        self.cross_biased_attn = MultiViewCrossAttentionSDPA(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
        )
        
        # hard-coding for cross-view attn here
        cross_view_attn_map = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2],
        }
        if enable_cross_view_attn:
            self.cross_view_attn = CrossViewAttention(
                x_dim,
                x_dim,  # context_dim, can not set to None
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                cross_view_attn_map=cross_view_attn_map,
            )
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)
            
    def reset_parameters(self):
        super().reset_parameters()
        if self.enable_cross_view_attn:
            self.layer_norm_cross_view_attn.reset_parameters()
            
    # def init_weights(self):
    #     super().init_weights()
    #     if self.enable_cross_view_attn:
    #         self.cross_view_attn.init_weights()

    #         # Zero-initialize the output projection
    #         torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
    #         if self.cross_view_attn.output_proj.bias is not None:
    #             torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)
    
    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_biased_attn.init_weights()
        self.mlp.init_weights()
        if self.enable_cross_view_attn:
            self.cross_view_attn.init_weights()

            # Zero-initialize the output projection
            torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
            if self.cross_view_attn.output_proj.bias is not None:
                torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        view_indices_B_T: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        k_mem: torch.Tensor | None = None, # (B, t, D)
        v_mem: torch.Tensor | None = None, # (B, t, D)
        cross_attn_bias: torch.Tensor | None = None, # (B, M, N)
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)

        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        
        result_B_T_H_W_D = rearrange( # (B, tl, 15, 20, 2048)
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D, # (tl*15*20, 1, 1, 128)
                video_size=video_size, # (tl, 15, 20)
                k_mem=k_mem,
                v_mem=v_mem,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D
        
        # I think we should insert cross view attn here. x_B_T_H_W_D
        if self.enable_cross_view_attn:
            num_cameras = torch.unique(view_indices_B_T[0]).shape[0]
            x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=num_cameras)
            sv_video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])
            x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")
            view_indices_B_V = rearrange(view_indices_B_T, "b (v t) -> b v t", v=num_cameras)[..., 0]
            result_cross_view_attn_B_T_H_W_D = rearrange(
                self.cross_view_attn(self.layer_norm_cross_view_attn(x_B_V_L_D), view_indices_B_V, sv_video_size),
                "b v (t h w) d -> b (v t) h w d",
                v=num_cameras,
                t=sv_video_size.T,
                h=sv_video_size.H,
                w=sv_video_size.W,
            )
            x_B_T_H_W_D = x_B_T_H_W_D + result_cross_view_attn_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                # self.cross_attn(
                #     rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                #     crossattn_emb,
                #     rope_emb=rope_emb_L_1_1_D,
                # ),
                self.cross_biased_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    attn_bias=cross_attn_bias,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D

class ViewPairBias(nn.Module):
    def __init__(self, feat_dim=22, hidden=128, tokens_per_view=256, clamp=10.0, bias_scale=1.0):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.clamp = clamp
        self.bias_scale = bias_scale

    def forward(self, attn_bias_feat):  # (B,Vq,Vk,22)
        bias_view = self.mlp(attn_bias_feat).squeeze(-1)  # (B,Vq,Vk)
        bias_view = bias_view * self.bias_scale
        if self.clamp is not None:
            bias_view = bias_view.clamp(-self.clamp, self.clamp)

        tv = self.tokens_per_view
        return bias_view.repeat_interleave(tv, 1).repeat_interleave(tv, 2)  # (B,Lq,Lk)
    
class ActionConditionedMultiViewStatePredDiT(MultiViewDiT):
    def __init__(
        self, 
        *args, 
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        enable_cross_view_attn: bool = True,
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        layer_mask: list[bool] | None = None,
        sac_config: SACConfig = SACConfig(),  # noqa: B008
        natten_parameters: list[Mapping | None] | None = None,
        **kwargs
    ):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs.pop("action_dim")
        use_black = bool(kwargs.pop("use_black", False))
        use_history = bool(kwargs.pop("use_history", False))
        super().__init__(
            *args, 
            state_t=state_t,
            n_cameras_emb=n_cameras_emb,
            view_condition_dim=view_condition_dim,
            concat_view_embedding=concat_view_embedding,
            layer_mask=layer_mask,
            sac_config=sac_config,
            natten_parameters=natten_parameters,
            **kwargs,
        )
        self.dexterous_hands = False
        if not self.dexterous_hands:
            self.action_embedder_B_D_agi = Mlp(
                in_features=action_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.action_embedder_B_3D_agi = Mlp(
                in_features=action_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
        else:
            self.action_embedder_B_D_hand = Mlp(
                in_features=44 * 20,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.action_embedder_B_3D_hand = Mlp(
                in_features=44 * 20,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
        self.action_to_crossattn_emb = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels // 2,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        self.use_history = use_history
        if self.use_history:
            self.history_tokenizer = PoseAwareHistoryTokenizer(in_ch=16, D=self.model_channels//2)
            self.mem_to_k = nn.Linear(self.model_channels//2, self.model_channels, bias=False)
            self.mem_to_v = nn.Linear(self.model_channels//2, self.model_channels, bias=False)
            if not self.dexterous_hands:
                self.vpb = ViewPairBias() # libero, bridge
                # self.vpb = ViewPairBias(feat_dim=43) # flexiv
            else:
                self.vpb = ViewPairBias(feat_dim=133)
            

        # dit blocks with multiview-cross attn and cross-view attn
        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewCrossBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    self_attention_backend=self.atten_backend
                    if natten_parameters is None or natten_parameters[i] is None
                    else "natten",
                    cross_attention_backend=self.atten_backend,
                    natten_params=None if natten_parameters is None else natten_parameters[i],
                    state_t=self.state_t,
                    enable_cross_view_attn=enable_cross_view_attn,
                )
                for i in range(self.num_blocks)
            ]
        )
        
        # initilization
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)
            
    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder_multiview = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        
    def init_weights(self):
        self.x_embedder_multiview.init_weights()
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.init_weights()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()
        
    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        view_indices_B_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        try:
            process_group = parallel_state.get_context_parallel_group()
            cp_size = len(get_process_group_ranks(process_group))
        except:  # noqa: E722
            cp_size = 1
        n_cameras = (x_B_C_T_H_W.shape[2] * cp_size) // self.state_t
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
        if self.concat_view_embedding:
            if view_indices_B_T is None:
                view_indices = torch.arange(n_cameras).clamp(
                    max=self.n_cameras_emb - 1
                )  # View indices [0, 1, ..., V-1]
                view_indices = view_indices.to(x_B_C_T_H_W.device)
                view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
                view_embedding = rearrange(view_embedding, "V D -> D V")
                view_embedding = (
                    view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
                )  # Shape: [1, D, V, 1, 1, 1]
            else:
                view_indices_B_T = view_indices_B_T.clamp(max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
                view_embedding = self.view_embeddings(view_indices_B_T)  # B, (V T), D
                view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, D, V, T, 1, 1]
            x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
            view_embedding = view_embedding.expand(
                x_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
            x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=n_cameras)

        x_B_T_H_W_D = self.x_embedder_multiview(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[str(n_cameras)]
            extra_pos_emb = extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            raise NotImplementedError("FPS-aware positional embedding is not supported for multi-view DIT")

        x_B_T_H_W_D = x_B_T_H_W_D + pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,  # (B, 16, tl, hl, wl)；tl = view_num * tl (single)
        timesteps_B_T: torch.Tensor, # (B, tl)
        crossattn_emb: torch.Tensor, # (B, 1536, 1024)
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None, # (B, 1, tl, hl, wl)
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        action: torch.Tensor | None = None,  # (B, T_total, action_dim=14)
        blacks: torch.Tensor | None = None,  # (B, 16, tl, hl, wl)
        history: torch.Tensor | None = None, # (B, 16, th, hl, wh)
        state: torch.Tensor | None = None,  # (B, T_total, T_history, 22)
        view_indices_B_T: torch.Tensor | None = None,  # (B, T_total)
        **kwargs,
    ):
        del kwargs
        # 0. concat input masks
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],dim=1)
        else:
            B, C, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1)
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        
        # 1. project action to action embedding
        assert action is not None, "action must be provided"
        action = rearrange(action, "b t d -> b 1 (t d)")
        if not self.dexterous_hands:
            action_emb_B_D = self.action_embedder_B_D_agi(action) # (B, 1, 2048)
            action_emb_B_3D = self.action_embedder_B_3D_agi(action) # (B, 1, 2048 * 3)
        else:
            action_emb_B_D = self.action_embedder_B_D_hand(action) # (B, 1, 2048)
            action_emb_B_3D = self.action_embedder_B_3D_hand(action) # (B, 1, 2048 * 3)
        
        # 2. history-aware mechanism
        k_mem, v_mem, cross_attn_bias = None, None, None
        if self.use_history and history is not None:
            n_views = crossattn_emb.shape[1] // 512
            available_latent_per_history_view = history.shape[2] // n_views
            available_latent_per_target_view = x_B_C_T_H_W.shape[2] // n_views
            history = history.view(history.shape[0], history.shape[1], n_views, available_latent_per_history_view, history.shape[-2], history.shape[-1]) # (B, C, V, Tv, 32, 32)
            del crossattn_emb
            crossattn_emb = self.history_tokenizer(history.to(torch.bfloat16), None) 
            # (B, 2560=2*1280, 1024), 1280 / 5 = 256, i.e., 256 (16 * 16) tokens per latent history frame, 1800 tokens for flexiv
            B, Nt, D = crossattn_emb.shape
            k_mem = self.mem_to_k(crossattn_emb.view(B, n_views, Nt // n_views, D))
            v_mem = self.mem_to_v(crossattn_emb.view(B, n_views, Nt // n_views, D))
            # downsample the input action path to the latent size
            state_in = state.permute(0, 3, 2, 1).contiguous()
            state_in = F.interpolate(state_in, size=(available_latent_per_target_view, available_latent_per_history_view), mode='bilinear', align_corners=False)
            state_in = state_in.permute(0, 2, 3, 1).contiguous() # (B, tl=6, th=5, 22)
            cross_attn_bias = self.vpb(state_in)

        # 3. prepare embedding sequence
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            view_indices_B_T=view_indices_B_T,
        )
        
        # 4. timestamp embedding
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        
        # 5. black condition, not used in multi-view mode
        T_target = t_embedding_B_T_D.shape[1]
        
        # 6. add action-condition to timestamp embedding
        t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D  # (B,T,D) + (B,1,D=2048)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D # (B, 1, 6144)
        else:
            adaln_lora_B_T_3D = action_emb_B_3D
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
        
        # 7. for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb
        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb.shape}"
            )
            
        # 8. pass latent through timestamp embedding
        if use_cuda_graphs:
            with temporarily_remove_hooks(self):
                shapes_key = create_cuda_graph(
                    self.cuda_graphs,
                    self.blocks,
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_pos_emb,
                    k_mem=k_mem,
                    v_mem=v_mem,
                )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks
        
        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
            "k_mem": k_mem,
            "v_mem": v_mem,
            "cross_attn_bias": cross_attn_bias,
        }
        
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                view_indices_B_T,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        # 9. unpachify to original latent size
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, #(B, tl, hl//2, wl//2, 2048)
            t_embedding_B_T_D, #(B, tl, 2048)
            adaln_lora_B_T_3D=adaln_lora_B_T_3D, #(B, tl, 2048 * 3)
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O) #(B, 16, tl, hl, wl) <- (B, tl, hl//2, wl//2, 16 * 4)
        return x_B_C_Tt_Hp_Wp
