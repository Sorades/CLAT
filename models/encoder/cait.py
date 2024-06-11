# copy from cait_models.py in https://github.com/facebookresearch/deit/blob/main/cait_models.py

import math
from typing import List
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath

from utils import load_timm_weights
from .utils import CrossAttention

__all__ = [
    "cait_xs24_384_concept",
    "cait_xxs24_384_concept",
    "cait_xxs24_224_concept",
    "cait_s24_224_concept",
    "cait_xs24_384_eyepacs",
    "CaiTConcept",
]


def load_cait_concept(arch_name: str, **kwargs) -> "CaiTConcept":
    func = globals().get(arch_name)
    return func(**kwargs)


def cait_xs24_384_eyepacs(pretrained=False, **kwargs):
    model = CaiTConcept(
        embed_dim=288,
        depth=24,
        num_heads=6,
        qkv_bias=True,
        init_scale=1e-5,
        **kwargs,
    )
    eyepacs_ckpt_path = "/data1/wc_log/LesionDetect/EyePACS/pretrain/cait_xs24_384_/lr1e-4_bs64/checkpoints/epoch=10-step=6039.ckpt"
    ckpt = torch.load(eyepacs_ckpt_path, map_location="cpu")

    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    state_dict.pop("head.weight")
    state_dict.pop("head.bias")
    msg = model.load_state_dict(state_dict, strict=False)
    [print(f'missing: {key}') for key in msg.missing_keys]
    [print(f'unexpected: {key}') for key in msg.unexpected_keys]

    return model



def cait_xs24_384_concept(pretrained=False, **kwargs):
    model = CaiTConcept(
        embed_dim=288,
        depth=24,
        num_heads=6,
        qkv_bias=True,
        init_scale=1e-5,
        **kwargs,
    )
    if pretrained:
        model = load_timm_weights("cait_xs24_384.fb_dist_in1k", model)
    return model


def cait_xxs24_384_concept(pretrained=False, **kwargs):
    model = CaiTConcept(
        embed_dim=192,
        depth=24,
        num_heads=4,
        qkv_bias=True,
        init_scale=1e-5,
        **kwargs,
    )
    if pretrained:
        model = load_timm_weights("cait_xxs24_384.fb_dist_in1k", model)
        print("Loaded pretrained weights for cait_xxs24_224.fb_dist_in1k")
    return model


def cait_xxs24_224_concept(pretrained=False, **kwargs):
    model = CaiTConcept(
        embed_dim=192,
        depth=24,
        num_heads=4,
        qkv_bias=True,
        init_scale=1e-5,
        **kwargs,
    )
    if pretrained:
        model = load_timm_weights("cait_xxs24_224.fb_dist_in1k", model)
        print("Loaded pretrained weights for cait_xxs24_224.fb_dist_in1k")
    return model


def cait_s24_224_concept(pretrained=False, **kwargs):
    model = CaiTConcept(
        embed_dim=384,
        depth=24,
        num_heads=8,
        qkv_bias=True,
        init_scale=1e-5,
        **kwargs,
    )
    if pretrained:
        model = load_timm_weights("cait_s24_224.fb_dist_in1k", model)
        print("Loaded pretrained weights for cait_s24_224.fb_dist_in1k")
    return model


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_cls=1):
        B, N, C = x.shape
        q = (
            self.q(
                x[
                    :,
                    :num_cls,
                ]
            )
            .unsqueeze(1)
            .reshape(B, num_cls, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, num_cls, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, weights


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Class_Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)

        o, weights = self.attn(self.norm1(u), x_cls.shape[1])
        x_cls = x_cls + self.drop_path(self.gamma_1 * o)

        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls, weights


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention_talking_head,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma_1 * o)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, weights


class cait_models(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=LayerScale_Block,
        block_layers_token=LayerScale_Block_CA,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention_talking_head,
        Mlp_block=Mlp,
        init_scale=1e-4,
        Attention_block_token_only=Class_Attention,
        Mlp_block_token_only=Mlp,
        depth_token_only=2,
        mlp_ratio_clstk=4.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = img_size

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.blocks_token_only = nn.ModuleList(
            [
                block_layers_token(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio_clstk,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block_token_only,
                    Mlp_block=Mlp_block_token_only,
                    init_values=init_scale,
                )
                for i in range(depth_token_only)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens, _ = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        x = self.head(x)

        return x


class CaiTConcept(cait_models):
    def __init__(
        self,
        num_lesions,
        decay_parameter=0.996,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_concepts = num_lesions
        self.decay_parameter = decay_parameter
        self.head = nn.Conv2d(self.embed_dim, self.num_concepts, kernel_size=[1, 1])
        self.head.apply(self._init_weights)

        img_size = to_2tuple(self.img_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.concept_cls_token = nn.Parameter(
            torch.zeros(1, self.num_concepts, self.embed_dim)
        )
        self.pos_embed_concept_cls = nn.Parameter(
            torch.zeros(1, self.num_concepts, self.embed_dim)
        )
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.class_tokens = nn.Parameter(
            torch.zeros(1, self.num_classes, self.embed_dim)
        )

        self.cross_attention = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=8,
        )

        trunc_normal_(self.pos_embed_concept_cls, std=0.02)
        trunc_normal_(self.concept_cls_token, std=0.02)
        trunc_normal_(self.pos_embed_pat, std=0.02)
        trunc_normal_(self.class_tokens, std=0.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_concepts
        N = self.num_patches
        if int(npatch) == N and int(w) == int(h):
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
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
            x = x + self.pos_embed_pat
        x = self.pos_drop(x)

        cls_tokens = self.concept_cls_token.expand(B, -1, -1)

        attn_weights_concepts = []
        attn_weights_patches = []
        concept_embeddings = []

        for _, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights_patches.append(weights_i)

        for _, blk in enumerate(self.blocks_token_only):
            cls_tokens, weights_i = blk(x, cls_tokens)
            attn_weights_concepts.append(weights_i)
            concept_embeddings.append(cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)

        return (
            x[:, 0 : self.num_concepts],
            x[:, self.num_concepts :],
            attn_weights_concepts,
            attn_weights_patches,
            concept_embeddings,
        )

    def forward(
        self,
        x,
        n_layers=2,
        return_attn=False,
        attention_type="fused",
        lesion_lbls: torch.Tensor = None,
        intervene_sample_idx: int = None,
        intervene_cpt_idx: List[int] = None,
        int_prob: float = None,
    ):
        w, h = x.shape[2:]
        (
            concept_tokens,
            patch_tokens,
            attn_weights_concepts,
            attn_weights_patches,
            all_layers_concept_token,
        ) = self.forward_features(x)

        n, p, c = patch_tokens.shape  # B * 196 * 384
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            patch_tokens = torch.reshape(patch_tokens, [n, w0, h0, c])
        else:
            patch_tokens = torch.reshape(
                patch_tokens, [n, int(p**0.5), int(p**0.5), c]
            )  # B * 14 * 14 * 384
        patch_tokens = patch_tokens.permute([0, 3, 1, 2])  # B * 384 * 14 * 14
        patch_tokens = patch_tokens.contiguous()
        concept_patch = self.head(patch_tokens)  # B * num_lesions * 14 * 14

        # concept_patch maxpooling
        concept_patch_pooled = F.adaptive_max_pool2d(
            concept_patch, (1, 1)
        )  # B * num_lesions * 1 * 1
        concept_patch_logits = torch.flatten(concept_patch_pooled, 1)  # B * num_lesions

        concept_logits = concept_tokens.mean(-1)
        # B * num_lesions, mean of class tokens to get class logits
        concept_logits = (concept_logits + concept_patch_logits) / 2

        if int_prob is not None:
            bs_size = concept_logits.size(0)
            intervene_sample_idx = (
                torch.bernoulli(torch.tensor([int_prob] * bs_size))
                .nonzero()
                .squeeze()
            )

        if intervene_sample_idx is not None:
            lesion_probs = torch.sigmoid(concept_logits)
            int_precision = torch.clamp(lesion_lbls,0.01,0.99)
            target_scale = torch.log(int_precision / (1 - int_precision))/concept_patch_logits
            lesion_int_scale = torch.where(lesion_lbls == lesion_probs.round(), 1., target_scale)
            mask = torch.zeros_like(lesion_probs)
            if intervene_cpt_idx is not None:
                mask[intervene_sample_idx, intervene_cpt_idx] = 1
            else:
                mask[intervene_sample_idx] = 1
            concept_logits_i = concept_tokens * (1-mask.unsqueeze(-1)) + concept_tokens * mask.unsqueeze(-1) * lesion_int_scale.unsqueeze(-1)
        else:
            concept_logits_i = concept_tokens

        out_value, cross_attn_maps = self.cross_attention(
            self.class_tokens.repeat(n, 1, 1), concept_logits_i
        )
        class_logits = out_value.mean(dim=-1)

        if not return_attn:
            return class_logits, concept_logits, concept_tokens

        if return_attn:
            feature_map = concept_patch.detach().clone()  # B * num_lesions * 14 * 14
            feature_map = F.relu(feature_map)
            n, c, h, w = feature_map.shape

            attn_weights_concepts = torch.stack(
                attn_weights_concepts
            )  # 12 * B * num_heads * num_lesions * num_lesions+196

            mtatt = (
                attn_weights_concepts[-n_layers:]
                .mean(2)
                .mean(0)[:, :, self.num_concepts :]
                .reshape([n, c, h, w])
            )  # B * num_lesions * 196 => B * num_lesions * 14 * 14, attn of class token and patch token
            patch_attn = torch.stack(attn_weights_patches).mean(
                2
            )  # 12 * B * 196 * 196, attn of patch token and patch token
            if attention_type == "fused":
                cams = mtatt * feature_map  # B * num_lesions * 14 * 14
                cams = torch.sqrt(cams)
            elif attention_type == "patchcam":
                cams = feature_map
            elif attention_type == "mct":
                cams = mtatt
            else:
                raise f"Error! {attention_type} is not defined!"

            return class_logits, concept_logits, cams, patch_attn, cross_attn_maps

        return (
            class_logits,
            concept_logits,
            concept_tokens,
        )

    def learn_concept_embed(self):
        """freeze all layers except concept token and head"""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True
        self.concept_cls_token.requires_grad = True

    def learn_all(self):
        """unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True

    def learn_class_embed(self):
        """freeze all layers except class token"""
        for param in self.parameters():
            param.requires_grad = False

        self.class_tokens.requires_grad = True

    def freeze_lesion_token(self):
        """freeze lesion tokens only"""
        self.concept_cls_token.requires_grad = False
