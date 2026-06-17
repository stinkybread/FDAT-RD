#!/usr/bin/env python3
"""
FDAT-RD Model Converter
========================
Standalone onnx converter for FDAT-RD (fdat_rd): FDAT + rectangular alternating
windows + token-dictionary cross-attention, dense spatial bias. The embedded
arch uses manual softmax attention (training arch uses SDPA) - identical
weights, cleanest export graph.

Converts: PyTorch (.pth/.pt), SafeTensors, ONNX (static / dynamic).

Usage:
    # 2x unshuffle, 720x540 input -> 1080p, static ONNX (H W = 540 720), verified:
    python fdat_rd_onnx_converter.py model.safetensors -f onnx-static --input-size 540 720 --split-size 10 30

    # tiny aligned verify first, then big export on GPU, no verify:
    python fdat_rd_onnx_converter.py model.safetensors -f onnx-static --input-size 60 60
    python fdat_rd_onnx_converter.py model.safetensors -f onnx-static --input-size 540 720 --device cuda --no-verify

    python fdat_rd_onnx_converter.py model.pth --info
    python fdat_rd_onnx_converter.py model.pth -f all

Notes:
    * --split-size carries the rectangular factorization (default 10 30); the
      spatial bias only stores the window product (s0*s1), validated.
    * scale is inferred for unshuffle models (in_ch 12 -> 2x, 48 -> 1x).
"""

import argparse
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# Standalone upsampler stack (verbatim; keys must match training)
# ============================================================================

SampleMods3 = Literal[
    "conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv",
    "dysample", "transpose+conv", "lda", "pa_up",
]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rt = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * rt.floor_()


class LayerNorm4D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DySample(nn.Module):
    def __init__(self, in_channels=64, out_ch=3, scale=2, groups=4,
                 end_convolution=True, end_kernel=1):
        super().__init__()
        if in_channels <= groups or in_channels % groups != 0:
            raise ValueError("Incorrect in_channels and groups values.")
        out_channels = 2 * groups * scale**2
        self.scale, self.groups = scale, groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, end_kernel, 1, end_kernel // 2)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (torch.stack(torch.meshgrid([h, h], indexing="ij"))
                .transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = (torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
                  .transpose(1, 2).unsqueeze(1).unsqueeze(0).to(x.device))
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
                  .view(B, 2, -1, self.scale * H, self.scale * W)
                  .permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1))
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords,
            mode="bilinear", align_corners=False, padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)
        if self.end_convolution:
            output = self.end_conv(output)
        return output


class LDA_AQU(nn.Module):
    def __init__(self, in_channels=48, reduction_factor=4, nh=1, scale_factor=2.0,
                 k_e=3, k_u=3, n_groups=2, range_factor=11, rpb=True):
        super().__init__()
        import numpy as np
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor
        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)
        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNorm4D(self.group_channel), nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2))
        self.layer_norm = LayerNorm4D(in_channels)
        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y, base_x = np.repeat(base, self.k_u), np.tile(base, self.k_u)
        base_offset = torch.tensor(np.stack([base_y, base_x], axis=1).flatten()).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head))

    def get_offset(self, offset, Hout, Wout):
        B = offset.shape[0]
        device = offset.device
        ri = torch.arange(Hout, device=device)
        ci = torch.arange(Wout, device=device)
        ri, ci = torch.meshgrid(ri, ci, indexing="ij")
        index_tensor = torch.stack((ri, ci), dim=-1).view(1, Hout, Wout, 2)
        B_off, C_off, H_off, W_off = offset.shape
        offset = offset.view(B_off, self.k_u, self.k_u, 2, H_off, W_off)
        offset = offset.permute(0, 1, 4, 2, 5, 3).contiguous()
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)
        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        return offset.flip(-1)

    def extract_feats(self, x, offset, ks=3):
        out = F.grid_sample(x, offset, mode="bilinear", padding_mode="zeros", align_corners=True)
        B, C, KH, KW = out.shape
        h, w = KH // ks, KW // ks
        out = out.view(B, C, ks, h, ks, w).permute(0, 2, 4, 1, 3, 5).contiguous()
        return out.view(B, ks * ks, C, h, w)

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q, k = self.proj_q(x), self.proj_k(x)
        q = F.interpolate(q, (out_H, out_W), mode="bilinear", align_corners=True)
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(x.dtype)
        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k, v = self.extract_feats(k, offset), self.extract_feats(v, offset)
        q = q.view(B, self.num_head, -1, out_H, out_W)
        q = q.view(B, self.num_head, q.shape[2], out_H * out_W).permute(0, 1, 3, 2).unsqueeze(3)
        _, n, c_k, h_k, w_k = k.shape
        k = k.view(B, self.n_groups, n, c_k, h_k, w_k).permute(0, 4, 5, 2, 1, 3).contiguous()
        k = k.view(B, h_k * w_k, n, self.n_groups * c_k)
        _, n, c_v, h_v, w_v = v.shape
        v = v.view(B, self.n_groups, n, c_v, h_v, w_v).permute(0, 4, 5, 2, 1, 3).contiguous()
        v = v.view(B, h_v * w_v, n, self.n_groups * c_v)
        k = k.view(B, k.shape[1], n, self.num_head, -1).permute(0, 3, 1, 2, 4)
        v = v.view(B, v.shape[1], n, self.num_head, -1).permute(0, 3, 1, 2, 4)
        if self.rpb:
            k = k + self.relative_position_bias_table
        attn = (q * self.scale @ k.transpose(-1, -2)).softmax(dim=-1)
        out = (attn @ v).squeeze(3).view(B, self.num_head, out_H, out_W, -1)
        return out.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, out_H, out_W)


class PA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(self, upsample="pa_up", scale=2, in_dim=48, out_dim=3,
                 mid_dim=48, group=4, dysample_end_kernel=1):
        m = []
        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend([nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)])
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(in_dim, in_dim, 3, 1, 1), nn.Upsample(scale_factor=2),
                              nn.LeakyReLU(negative_slope=0.2, inplace=True)])
                m.extend([nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True)])
            elif scale == 3:
                m.extend([nn.Conv2d(in_dim, in_dim, 3, 1, 1), nn.Upsample(scale_factor=scale),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
                          nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True)])
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            m.append(DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel))
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend([nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1), nn.GELU(),
                          nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1)])
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Upsample(scale_factor=2), nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                              PA(mid_dim), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                              nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                              nn.LeakyReLU(negative_slope=0.2, inplace=True)])
                    in_dim = mid_dim
            elif scale == 3:
                m.extend([nn.Upsample(scale_factor=3), nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                          PA(mid_dim), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                          nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True)])
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        super().__init__(*m)
        self.register_buffer("MetaUpsample", torch.tensor([
            3, ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv",
                "dysample", "transpose+conv", "lda", "pa_up"].index(upsample),
            scale, in_dim, out_dim, mid_dim, group], dtype=torch.uint8))


# ============================================================================
# FDAT-RD architecture (MANUAL attention for clean export; keys match training)
# ============================================================================

class TokenDictionaryCrossAttention(nn.Module):
    def __init__(self, dim, num_tokens=128, num_heads=4, qkv_bias=False):
        super().__init__()
        self.nh = num_heads
        self.hd = dim // num_heads
        self.scale = self.hd**-0.5
        self.dictionary = nn.Parameter(torch.zeros(num_tokens, dim))
        self.r_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.r_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.r_norm = nn.LayerNorm(dim)
        self.e_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.e_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _mha(self, q, k, v):
        B, Lq, C = q.shape
        Lk = k.shape[1]
        q = q.reshape(B, Lq, self.nh, self.hd).permute(0, 2, 1, 3)
        k = k.reshape(B, Lk, self.nh, self.hd).permute(0, 2, 1, 3)
        v = v.reshape(B, Lk, self.nh, self.hd).permute(0, 2, 1, 3)
        attn = F.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        return (attn @ v).permute(0, 2, 1, 3).reshape(B, Lq, C)

    def forward(self, x, H, W):
        B, N, C = x.shape
        d = self.dictionary.unsqueeze(0).expand(B, -1, -1)
        rk, rv = self.r_kv(self.r_norm(x)).chunk(2, dim=-1)
        d = d + self._mha(self.r_q(d), rk, rv)
        ek, ev = self.e_kv(d).chunk(2, dim=-1)
        return self.proj(self._mha(self.e_q(x), ek, ev))


class FastSpatialWindowAttention(nn.Module):
    def __init__(self, dim, window_size=(16, 16), num_heads=4, qkv_bias=False):
        super().__init__()
        self.wh, self.ww = window_size
        self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        n = self.wh * self.ww
        self.bias = nn.Parameter(torch.zeros(num_heads, n, n))

    def forward(self, x, H, W):
        B, L, C = x.shape
        wh, ww = self.wh, self.ww
        pad_b = (wh - H % wh) % wh
        pad_r = (ww - W % ww) % ww
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x.reshape(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b)).reshape(B, -1, C)
        H_pad, W_pad = H + pad_b, W + pad_r
        x = (x.reshape(B, H_pad // wh, wh, W_pad // ww, ww, C)
             .permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, wh * ww, C))
        N = wh * ww
        qkv = self.qkv(x).reshape(-1, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1)) + self.bias
        x = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(-1, N, C)
        x = (self.proj(x).reshape(B, H_pad // wh, W_pad // ww, wh, ww, C)
             .permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H_pad, W_pad, C))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.reshape(B, L, C)


class FastChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = F.normalize(q.transpose(-2, -1), dim=-1)
        k = F.normalize(k.transpose(-2, -1), dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        return self.proj((attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C))


class SimplifiedAIM(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(), nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, attn_feat, conv_feat, interaction_type, H, W):
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = self.sg(attn_feat.transpose(1, 2).reshape(B, C, H, W)).reshape(B, 1, L).transpose(1, 2)
            return attn_feat + (conv_feat * sm)
        cm = self.cg(conv_feat.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, 1).transpose(1, 2)
        return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0):
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hd, False), nn.GELU(), nn.Linear(hd, dim, False)
        self.drop = nn.Dropout(drop)
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        x_s = self.smix(x.transpose(1, 2).reshape(B, x.shape[-1], H, W)).reshape(B, x.shape[-1], L).transpose(1, 2)
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    def __init__(self, dim, nh, window_size, ffn_exp, aim_re, btype, dp, num_dict_tokens=128, qkv_b=False):
        super().__init__()
        self.btype = btype
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if btype == "spatial":
            self.attn = FastSpatialWindowAttention(dim, window_size, nh, qkv_b)
        elif btype == "channel":
            self.attn = FastChannelAttention(dim, nh, qkv_b)
        else:
            self.attn = TokenDictionaryCrossAttention(dim, num_dict_tokens, nh, qkv_b)
        if btype != "dictionary":
            self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU())
            self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x, H, W):
        B, L, C = x.shape
        return self.conv(x.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, L).transpose(1, 2)

    def forward(self, x, H, W):
        if self.btype == "dictionary":
            x = x + self.dp(self.attn(self.n1(x), H, W))
            return x + self.dp(self.ffn(self.n2(x), H, W))
        n1 = self.n1(x)
        itype = "channel_modulates_spatial" if self.btype == "spatial" else "spatial_modulates_channel"
        x = x + self.dp(self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W))
        return x + self.dp(self.ffn(self.n2(x), H, W))


class SimplifiedResidualGroup(nn.Module):
    def __init__(self, dim, depth, nh, split_size, ffn_exp, aim_re, pattern, dp_rates,
                 num_dict_tokens=128, start_parity=0):
        super().__init__()
        s0, s1 = split_size
        blocks, sp = [], start_parity
        for i in range(depth):
            btype = pattern[i % len(pattern)]
            if btype == "spatial":
                ws = (s0, s1) if sp % 2 == 0 else (s1, s0)
                sp += 1
            else:
                ws = (s0, s1)
            blocks.append(SimplifiedDATBlock(dim, nh, ws, ffn_exp, aim_re, btype, dp_rates[i], num_dict_tokens))
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).reshape(B, C, H, W)) + x


class FDATRD(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, embed_dim=128, num_groups=4,
                 depth_per_group=2, num_heads=4, split_size=(10, 30), num_dict_tokens=128,
                 ffn_expansion_ratio=2.0, aim_reduction_ratio=8, group_block_pattern=None,
                 drop_path_rate=0.1, mid_dim=64, upsampler_type="transpose+conv",
                 img_range=1.0, unshuffle_mod=False, **kw):
        super().__init__()
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel", "dictionary"]
        self.upscale = scale
        align = math.lcm(int(split_size[0]), int(split_size[1]))
        self.pad = align
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            scale = 4
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(num_in_ch * unshuffle**2, embed_dim, 3, 1, 1, bias=True))
            self.pad = unshuffle * align
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=True)
        ad = depth_per_group * len(group_block_pattern)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups * ad)]
        spatials_per_group = sum(
            1 for j in range(ad) if group_block_pattern[j % len(group_block_pattern)] == "spatial")
        self.groups = nn.Sequential(*[
            SimplifiedResidualGroup(
                embed_dim, ad, num_heads, split_size, ffn_expansion_ratio, aim_reduction_ratio,
                group_block_pattern, dpr[i * ad:(i + 1) * ad], num_dict_tokens,
                start_parity=(i * spatials_per_group) % 2)
            for i in range(num_groups)])
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(upsampler_type, scale, embed_dim, num_out_ch, mid_dim, 4)

    def check_img_size(self, x, h, w):
        if self.pad == 0:
            return x
        mph = (self.pad - h % self.pad) % self.pad
        mpw = (self.pad - w % self.pad) % self.pad
        mode = "replicate" if (mph >= h or mpw >= w) else "reflect"
        return F.pad(x, (0, mpw, 0, mph), mode)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.check_img_size(x, h, w)
        x_shallow = self.conv_first(x)
        x_deep = self.conv_after(self.groups(x_shallow))
        return self.upsampler(x_deep + x_shallow)[:, :, :h * self.upscale, :w * self.upscale]


# ============================================================================
# Detection / loading
# ============================================================================

def _block_type(sd, gi, bi):
    base = f"groups.{gi}.blocks.{bi}.attn."
    if base + "dictionary" in sd:
        return "dictionary"
    if base + "temp" in sd:
        return "channel"
    if base + "bias" in sd:
        return "spatial"
    return None


def detect_model_params(sd, split_size):
    p = dict(num_in_ch=3, num_out_ch=3, scale=4, embed_dim=128, num_groups=4, depth_per_group=2,
             num_heads=4, split_size=tuple(split_size), num_dict_tokens=128,
             ffn_expansion_ratio=2.0, aim_reduction_ratio=8,
             group_block_pattern=["spatial", "channel", "dictionary"], drop_path_rate=0.1,
             mid_dim=64, upsampler_type="transpose+conv", unshuffle_mod=False)

    if "conv_first.1.weight" in sd:
        p["embed_dim"] = sd["conv_first.1.weight"].shape[0]
        p["unshuffle_mod"] = True
        in_ch = sd["conv_first.1.weight"].shape[1]
        if in_ch == 12:
            p["scale"] = 2
        elif in_ch == 48:
            p["scale"] = 1
    elif "conv_first.weight" in sd:
        p["embed_dim"] = sd["conv_first.weight"].shape[0]
        p["num_in_ch"] = sd["conv_first.weight"].shape[1]

    gc = 0
    while f"groups.{gc}.conv.weight" in sd:
        gc += 1
    if gc:
        p["num_groups"] = gc

    types, bi = [], 0
    while f"groups.0.blocks.{bi}.n1.weight" in sd:
        types.append(_block_type(sd, 0, bi))
        bi += 1
    if types:
        n = len(types)
        period = n
        for c in range(1, n + 1):
            if n % c == 0 and all(types[i] == types[i % c] for i in range(n)):
                period = c
                break
        p["group_block_pattern"] = types[:period]
        p["depth_per_group"] = n // period

    s0, s1 = split_size
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn."
        if b + "temp" in sd:
            p["num_heads"] = sd[b + "temp"].shape[0]
            break
        if b + "bias" in sd:
            p["num_heads"] = sd[b + "bias"].shape[0]
            break
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn.bias"
        if b in sd:
            prod = sd[b].shape[1]
            if prod != s0 * s1:
                print(f"  WARNING: --split-size {split_size} product {s0*s1} "
                      f"!= detected window product {prod}. Fix --split-size.")
            break
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn.dictionary"
        if b in sd:
            p["num_dict_tokens"] = sd[b].shape[0]
            break

    if "groups.0.blocks.0.ffn.fc1.weight" in sd:
        w = sd["groups.0.blocks.0.ffn.fc1.weight"]
        r = w.shape[0] / w.shape[1]
        for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
            if abs(r - c) < 1e-3:
                r = c
                break
        p["ffn_expansion_ratio"] = r
    for bi in range(len(types)):
        k = f"groups.0.blocks.{bi}.inter.cg.1.weight"
        if k in sd:
            cg = sd[k]
            p["aim_reduction_ratio"] = max(1, cg.shape[1] // cg.shape[0])
            break

    if "upsampler.MetaUpsample" in sd:
        meta = sd["upsampler.MetaUpsample"]
        up = ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv",
              "dysample", "transpose+conv", "lda", "pa_up"]
        p["upsampler_type"] = up[int(meta[1])]
        p["mid_dim"] = int(meta[5])
        if not p["unshuffle_mod"]:
            p["scale"] = int(meta[2])

    last = None
    for k in sd:
        if k.startswith("upsampler.") and k.endswith(".weight") and "MetaUpsample" not in k:
            last = k
    if last:
        p["num_out_ch"] = sd[last].shape[0]
    return p


def clean_state_dict(sd):
    out = OrderedDict()
    for k, v in sd.items():
        for pre in ["module.", "model.", "net.", "network.", "state_dict."]:
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def load_model(path, device, split_size, override=None):
    print(f"Loading checkpoint: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ["params_ema", "params", "state_dict", "model", "net"]:
                if key in ckpt:
                    sd = ckpt[key]
                    break
            else:
                sd = ckpt
        else:
            sd = ckpt
    sd = clean_state_dict(sd)
    params = detect_model_params(sd, split_size)
    if override:
        params.update(override)
    print(f"  Detected: scale={params['scale']} embed_dim={params['embed_dim']} "
          f"groups={params['num_groups']} depth={params['depth_per_group']} "
          f"heads={params['num_heads']} split={params['split_size']} "
          f"dict_tokens={params['num_dict_tokens']} pattern={params['group_block_pattern']} "
          f"ffn={params['ffn_expansion_ratio']} up={params['upsampler_type']} "
          f"unshuffle={params['unshuffle_mod']}")
    model = FDATRD(**params)
    if hasattr(model.upsampler, "MetaUpsample"):
        sd["upsampler.MetaUpsample"] = model.upsampler.MetaUpsample
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, params


# ============================================================================
# Export
# ============================================================================

def export_pytorch(model, path, fp16=False):
    sd = model.state_dict()
    if fp16:
        sd = {k: v.half() for k, v in sd.items()}
    torch.save(sd, path)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.2f} MB)")


def export_safetensors(model, path, fp16=False):
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  safetensors not installed; skipping.")
        return
    sd = model.state_dict()
    if fp16:
        sd = {k: v.half() for k, v in sd.items()}
    save_file(sd, path)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.2f} MB)")


def export_onnx(model, path, dynamic=False, input_size=(1, 3, 540, 720), opset=17, fp16=False):
    model.eval()
    device = next(model.parameters()).device
    dtype = torch.float16 if fp16 else torch.float32
    if fp16:
        model = model.half()
    dummy = torch.randn(*input_size, device=device, dtype=dtype)
    dyn = ({"input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}} if dynamic else None)
    kwargs = dict(input_names=["input"], output_names=["output"], dynamic_axes=dyn,
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, path, external_data=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kwargs)
    mode = "dynamic" if dynamic else f"static {tuple(input_size[2:])}"
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.2f} MB) [{mode}]")


def verify_onnx(model, path, input_size):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed; skipping verification.")
        return
    device = next(model.parameters()).device
    x = torch.randn(*input_size, dtype=torch.float32)
    with torch.no_grad():
        t_out = model(x.to(device)).float().cpu().numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o_out = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(t_out - o_out).max())
    print(f"  ORT parity: max|torch-onnx| = {diff:.3e}  [{'OK' if diff < 1e-3 else 'WARN'}]")


def print_model_info(model, params):
    total = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 60 + "\nMODEL INFORMATION  (FDAT-RD)\n" + "=" * 60)
    for k in ("scale", "num_in_ch", "num_out_ch", "embed_dim", "num_groups", "depth_per_group",
              "num_heads", "split_size", "num_dict_tokens", "group_block_pattern",
              "ffn_expansion_ratio", "aim_reduction_ratio", "upsampler_type", "unshuffle_mod"):
        print(f"  {k}: {params[k]}")
    print(f"  Total parameters: {total:,} ({total/1e6:.2f}M)\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description="FDAT-RD Model Converter",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("-f", "--formats", nargs="+", default=["fp32"],
                    choices=["fp32", "fp16", "safetensors", "onnx-static", "onnx-dynamic", "all"])
    ap.add_argument("--scale", type=int)
    ap.add_argument("--split-size", type=int, nargs=2, default=[10, 30])
    ap.add_argument("--num-dict-tokens", type=int)
    ap.add_argument("--upsampler")
    ap.add_argument("--unshuffle-mod", action="store_true")
    ap.add_argument("--input-size", type=int, nargs=2, default=[540, 720],
                    help="Static ONNX input H W; default 540 720 (720x540 input -> 1080p)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--name")
    args = ap.parse_args()

    override = {}
    if args.scale:
        override["scale"] = args.scale
    if args.split_size:
        override["split_size"] = tuple(args.split_size)
    if args.num_dict_tokens:
        override["num_dict_tokens"] = args.num_dict_tokens
    if args.upsampler:
        override["upsampler_type"] = args.upsampler
    if args.unshuffle_mod:
        override["unshuffle_mod"] = True

    model, params = load_model(args.input, args.device, args.split_size, override or None)
    model = model.to(args.device)
    print_model_info(model, params)
    if args.info:
        return

    if args.output is None:
        out_dir = Path(args.input).parent
    elif os.path.isdir(args.output) or args.output.endswith("/"):
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or Path(args.input).stem
    if "all" in args.formats:
        args.formats = ["fp32", "fp16", "safetensors", "onnx-static", "onnx-dynamic"]

    print("\nExporting...")
    for fmt in args.formats:
        if fmt == "fp32":
            export_pytorch(model, str(out_dir / f"{base}_fp32.pth"))
        elif fmt == "fp16":
            export_pytorch(model, str(out_dir / f"{base}_fp16.pth"), fp16=True)
        elif fmt == "safetensors":
            export_safetensors(model, str(out_dir / f"{base}.safetensors"))
        elif fmt in ("onnx-static", "onnx-dynamic"):
            h, w = args.input_size
            isize = (1, params["num_in_ch"], h, w)
            dyn = fmt == "onnx-dynamic"
            path = str(out_dir / (f"{base}_dynamic.onnx" if dyn else f"{base}_static_{h}x{w}.onnx"))
            export_onnx(model, path, dynamic=dyn, input_size=isize, opset=args.opset)
            if not args.no_verify:
                verify_onnx(model, path, isize)
    print("\nDone!")


if __name__ == "__main__":
    main()
