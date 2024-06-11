# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer, Block, Attention

from utils import load_timm_weights
from models.encoder.utils import CrossAttention, load_milvt_ckpt

__all__ = ["MIL_VT_Concept", "ViTConcept", "vit_small_concept","vit_base_concept","deit3_small_concept","deit3_base_concept"]


def MIL_VT_Concept(pretrained: bool, **kwargs) -> "ViTConcept":
    """Load pretrain weight from distillation model, to train a plain model"""

    model = ViTConcept(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        model = load_milvt_ckpt(
            model, "/data0/wc_data/LesionDetect/weights/MIL_VIT_pretrain.pth.tar"
        )

    return model


def vit_small_concept(pretrained: bool, **kwargs) -> "ViTConcept":
    model = ViTConcept(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    if pretrained:
        model = load_timm_weights("vit_small_patch16_384", model)
    return model

def vit_base_concept(pretrained: bool, **kwargs) -> "ViTConcept":
    model = ViTConcept(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if pretrained:
        model = load_timm_weights("vit_base_patch16_384", model)
    return model

def deit3_small_concept(pretrained: bool, **kwargs) -> "ViTConcept":
    model = ViTConcept(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    if pretrained:
        model = load_timm_weights("deit3_small_patch16_384", model)
    return model

def deit3_base_concept(pretrained: bool, **kwargs) -> "ViTConcept":
    model = ViTConcept(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if pretrained:
        model = load_timm_weights("deit3_base_patch16_384", model)
    return model


class Attention(Attention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Block(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0,
        attn_drop: float = 0,
        init_values: float = None,
        drop_path: float = 0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = ...,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(o))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, weights


class ViTConcept(VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        num_lesions,
        img_size=224,
        *args,
        **kwargs,
    ):
        super().__init__(img_size=img_size, block_fn=Block, **kwargs)
        self.img_size = img_size
        self.num_lesions = num_lesions
        self.head = nn.Conv2d(self.embed_dim, self.num_lesions, kernel_size=[1, 1])
        self.head.apply(self._init_weights)
        self.num_patches = num_patches = self.patch_embed.num_patches

        self.lesion_tokens = nn.Parameter(
            torch.zeros(1, self.num_lesions, self.embed_dim)
        )
        self.pos_embed_lesion = nn.Parameter(
            torch.zeros(1, self.num_lesions, self.embed_dim)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.disease_tokens = nn.Parameter(
            torch.zeros(1, self.num_classes, self.embed_dim)
        )

        self.cross_attention = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=8,
        )

        trunc_normal_(self.pos_embed_lesion, std=0.02)
        trunc_normal_(self.lesion_tokens, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.disease_tokens, std=0.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_lesions
        N = self.num_patches
        if int(npatch) == N and int(w) == int(h):
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(int(w0) / math.sqrt(N), int(h0) / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x):
        B, nc, w, h = x.size()  # B * 3 * 224 * 224
        x = self.patch_embed(x)  # B * 196 * 384
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        lesion_tokens = self.lesion_tokens.expand(B, -1, -1)
        lesion_tokens = lesion_tokens + self.pos_embed_lesion

        x = torch.cat((lesion_tokens, x), dim=1)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        return x[:, 0 : self.num_lesions], x[:, self.num_lesions :], attn_weights

    def forward(
        self,
        x,
        n_layers=24,
        return_attn=False,
        attention_type="fused",
        lesion_lbls: torch.Tensor = None,
        intervene_sample_idx: int = None,
        intervene_cpt_idx: List[int] = None,
        int_prob: float = None,
    ):
        w, h = x.shape[2:]
        lesion_tokens, patch_tokens, attn_weights = self.forward_features(x)
        n, p, c = patch_tokens.shape

        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            patch_tokens = torch.reshape(patch_tokens, [n, w0, h0, c])
        else:
            patch_tokens = torch.reshape(patch_tokens, [n, int(p**0.5), int(p**0.5), c])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2])
        patch_tokens = patch_tokens.contiguous()
        local_lesion_tokens = self.head(patch_tokens)
        local_lesion_token_pooled = F.adaptive_max_pool2d(local_lesion_tokens, (1, 1))
        local_lesion_logits = torch.flatten(local_lesion_token_pooled, 1)

        global_lesion_logits = lesion_tokens.mean(-1)
        lesion_logits = (local_lesion_logits + global_lesion_logits) / 2.0

        if int_prob is not None:
            bs_size = lesion_logits.size(0)
            intervene_sample_idx = (
                torch.bernoulli(torch.tensor([int_prob] * bs_size)).nonzero().squeeze()
            )

        if intervene_sample_idx is not None:
            lesion_probs = torch.sigmoid(lesion_logits)
            int_precision = torch.clamp(lesion_lbls, 0.01, 0.99)
            target_scale = (
                torch.log(int_precision / (1 - int_precision)) / global_lesion_logits
            )
            lesion_int_scale = torch.where(
                lesion_lbls == lesion_probs.round(), 1.0, target_scale
            )
            mask = torch.zeros_like(lesion_probs)
            if intervene_cpt_idx is not None:
                mask[intervene_sample_idx, intervene_cpt_idx] = 1
            else:
                mask[intervene_sample_idx] = 1
            lesion_tokens_i = lesion_tokens * (
                1 - mask.unsqueeze(-1)
            ) + lesion_tokens * mask.unsqueeze(-1) * lesion_int_scale.unsqueeze(-1)
        else:
            lesion_tokens_i = lesion_tokens

        out_value, cross_attn_maps = self.cross_attention(
            self.disease_tokens.repeat(n, 1, 1), lesion_tokens_i
        )
        disease_logits = out_value.mean(dim=-1)

        if not return_attn:
            return disease_logits, lesion_logits, lesion_tokens

        feature_map = local_lesion_tokens.detach().clone()
        feature_map = F.relu(feature_map)
        n, c, h, w = feature_map.shape
        attn_weights = torch.stack(attn_weights)
        mtatt = (
            attn_weights[-n_layers:]
            .mean(2)
            .mean(0)[:, 0 : self.num_lesions, self.num_lesions :]
            .reshape([n, c, h, w])
        )
        patch_attn = attn_weights[:, :, self.num_lesions :, self.num_lesions :]
        if attention_type == "fused":
            cams = mtatt * feature_map  # B * num_lesions * 14 * 14
            cams = torch.sqrt(cams)
        elif attention_type == "patchcam":
            cams = feature_map
        elif attention_type == "mct":
            cams = mtatt
        else:
            raise f"Error! {attention_type} is not defined!"

        return disease_logits, lesion_logits, cams, patch_attn, cross_attn_maps

    def tune_setting(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        for param in self.cross_attention.parameters():
            param.requires_grad = True
        self.lesion_tokens.requires_grad = True
        self.pos_embed_lesion.requires_grad = True
        self.disease_tokens.requires_grad = True
