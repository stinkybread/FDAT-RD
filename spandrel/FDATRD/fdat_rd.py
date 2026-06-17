# https://github.com/stinkybread/fdat/blob/main/fdat.py
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import _IncompatibleKeys  # type: ignore

from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util import store_hyperparameters
from spandrel.util.timm import DropPath

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
]

SampleMods3 = Literal[SampleMods, "transpose+conv", "lda", "pa_up"]


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LDA_AQU(nn.Module):
    def __init__(
        self,
        in_channels=48,
        reduction_factor=4,
        nh=1,
        scale_factor=2.0,
        k_e=3,
        k_u=3,
        n_groups=2,
        range_factor=11,
        rpb=True,
    ) -> None:
        super().__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )
        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset, Hout, Wout):
        B, _, _, _ = offset.shape
        device = offset.device
        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(
            1, Hout, Wout, 2
        )
        offset = rearrange(
            offset, "b (kh kw d) h w -> b kh h kw w d", kh=self.k_u, kw=self.k_u
        )
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)

        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x, offset, ks=3):
        out = nn.functional.grid_sample(
            x,
            offset,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(
            q, (out_H, out_W), mode="bilinear", align_corners=True
        )
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(
            x.dtype
        )

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out


class PA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods3 = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,  # Only DySample
        dysample_end_kernel=1,  # needed only for compatibility with version 2
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )
            # m.append(nn.Conv2d(mid_dim, out_dim, dysample_end_kernel, 1, dysample_end_kernel//2)) # kernel 1 causes chromatic artifacts
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2, 3, 4"
                )
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods3.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )



# --- fdat_rd components (manual attention for clean inference/export) ---
class TokenDictionaryCrossAttention(nn.Module):
    def __init__(self, dim, num_tokens=128, num_heads=4, qkv_bias=False) -> None:
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
    def __init__(self, dim, window_size=(16, 16), num_heads=4, qkv_bias=False) -> None:
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
    def __init__(self, dim, num_heads=4, qkv_bias=False) -> None:
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
    def __init__(self, dim, reduction_ratio=8) -> None:
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
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0) -> None:
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
    def __init__(self, dim, nh, window_size, ffn_exp, aim_re, btype, dp, num_dict_tokens=128, qkv_b=False) -> None:
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
                 num_dict_tokens=128, start_parity=0) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_seq = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).reshape(B, C, H, W)) + x


@store_hyperparameters()
class FDATRD(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        embed_dim: int = 128,
        num_groups: int = 4,
        depth_per_group: int = 2,
        num_heads: int = 4,
        split_size: tuple[int, int] = (10, 30),
        num_dict_tokens: int = 128,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: list[str] | None = None,
        drop_path_rate: float = 0.1,
        mid_dim: int = 64,
        upsampler_type: SampleMods3 = "transpose+conv",
        img_range: float = 1.0,
        unshuffle_mod: bool = False,
    ) -> None:
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel", "dictionary"]
        super().__init__()
        self.img_range, self.upscale = img_range, scale
        align = math.lcm(int(split_size[0]), int(split_size[1]))
        self.pad = align
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            scale = 4
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(num_in_ch * unshuffle**2, embed_dim, 3, 1, 1, bias=True),
            )
            self.pad = unshuffle * align
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=True)

        ad = depth_per_group * len(group_block_pattern)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups * ad)]
        spatials_per_group = sum(
            1 for j in range(ad) if group_block_pattern[j % len(group_block_pattern)] == "spatial"
        )
        self.groups = nn.Sequential(
            *[
                SimplifiedResidualGroup(
                    embed_dim, ad, num_heads, split_size, ffn_expansion_ratio,
                    aim_reduction_ratio, group_block_pattern, dpr[i * ad : (i + 1) * ad],
                    num_dict_tokens, start_parity=(i * spatials_per_group) % 2,
                )
                for i in range(num_groups)
            ]
        )
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(upsampler_type, scale, embed_dim, num_out_ch, mid_dim, 4)

    def load_state_dict(self, state_dict: StateDict, *args, **kwargs) -> _IncompatibleKeys:  # noqa: ANN002
        state_dict["upsampler.MetaUpsample"] = self.upsampler.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        if self.pad == 0:
            return x
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        mode = "replicate" if (mod_pad_h >= h or mod_pad_w >= w) else "reflect"
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode)

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, h, w)
        x_shallow = self.conv_first(x)
        x_deep = self.conv_after(self.groups(x_shallow))
        x_out = self.upsampler(x_deep + x_shallow)
        return x_out[:, :, : h * self.upscale, : w * self.upscale]
