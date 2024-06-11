from collections import OrderedDict
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_outputs=None,
        num_heads=8,
        attention_dropout=0.1,
        projection_dropout=0.0,
    ) -> None:
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y, weights=None):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = (
            self.q(x)
            .reshape(B, Nx, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if weights is not None:
            attn = attn * weights.repeat(1, self.num_heads, 1, 1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def interpolate_pos_embed(model, checkpoint_model) -> None:
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def load_vit_ckpt(model:nn.Module, ckpt_path:str, pos_embed_modify:bool=True) -> nn.Module:
    state_dict = model.state_dict()
    ckpt_model = torch.load(ckpt_path,map_location='cpu')['model']
    for k in ['head.weight', 'head.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    if pos_embed_modify:
        ckpt_model['pos_embed'] = ckpt_model['pos_embed'][:,1:]
    interpolate_pos_embed(model, ckpt_model)
    msg = model.load_state_dict(ckpt_model, strict=False)
    # [print(f'missing: {key}') for key in msg.missing_keys if 'head' not in key]
    # [print(f'unexpected: {key}') for key in msg.unexpected_keys if 'decoder' not in key and key != 'mask_token']
    return model

def rename_pretrain_weight(checkpoint):
    state_dict_old = checkpoint['state_dict']
    state_dict_new = OrderedDict()
    for key, value in state_dict_old.items():
        state_dict_new[key[len('module.'):]] = value
    return state_dict_new

def load_milvt_ckpt(model:nn.Module, ckpt_path:str) -> nn.Module:
    state_dict = model.state_dict()
    checkpoint0 = torch.load(ckpt_path,map_location='cpu')

    ckpt_model = rename_pretrain_weight(checkpoint0)
    state_dict = model.state_dict()
    checkpoint_keys = list(ckpt_model.keys())
    for tempKey in list(state_dict.keys()):
        if tempKey not in checkpoint_keys:
            print("Missing Key not in pretrain model: ", tempKey)

    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if (
            k in ckpt_model
        ):  # and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    if ckpt_model['pos_embed'].shape[1] != state_dict['pos_embed'].shape[1]:
        ckpt_model['pos_embed'] = ckpt_model['pos_embed'][:,2:]
    interpolate_pos_embed(model, ckpt_model)
    msg = model.load_state_dict(ckpt_model, strict=False)
    # [print(f'missing: {key}') for key in msg.missing_keys if 'head' not in key]
    # [print(f'unexpected: {key}') for key in msg.unexpected_keys if 'decoder' not in key and key != 'mask_token']
    return model