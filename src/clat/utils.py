import logging
import os
from collections import OrderedDict, namedtuple
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
import prettytable as pt
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    Metric,
    MetricCollection,
    Recall,
    Specificity,
)

CLATOutput = namedtuple(
    "CLATOutput",
    [
        "disease_logits",
        "lesion_logits",
        "lesion_tokens",
        "cams",
        "patch_attns",
        "cross_attn_maps",
    ],
)


# region logging
def logging_config(log_dir: Optional[str] = None, rank: Optional[int] = None):
    """Configure logging

    Args:
        log_dir (str, optional): The directory to save the log file. Defaults to None.
    """
    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = f"records_r{rank}" if rank is not None else "records"
        handler = logging.FileHandler(f"{log_dir}/{log_name}.log")
        formatter = logging.Formatter(
            "%(asctime)s%(message)s", datefmt="[%y/%m%d %H:%M:%S]"
        )
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().handlers = [handler]


# endregion


# region data
def kfold_split(
    kfold: int, fold_num: int, disease_labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    train_idx, val_idx = list(
        skf.split(disease_labels, y=disease_labels.iloc[:, 1:].values.argmax(axis=1))
    )[fold_num]
    train_disease_labels = disease_labels.iloc[train_idx, :]
    val_disease_labels = disease_labels.iloc[val_idx, :]
    test_disease_labels = val_disease_labels
    return train_disease_labels, val_disease_labels, test_disease_labels  # type: ignore


def handout_split(
    val_size: Optional[float], test_size: float, disease_labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size == 1:
        return None, None, disease_labels  # type: ignore

    train_disease_labels, test_disease_labels = train_test_split(
        disease_labels,
        test_size=test_size,
        stratify=disease_labels.iloc[:, 1:],
        random_state=42,
    )

    if val_size:
        train_disease_labels, val_disease_labels = train_test_split(
            train_disease_labels,
            test_size=val_size,
            stratify=train_disease_labels.iloc[:, 1:],
            random_state=42,
        )
    else:
        val_disease_labels = test_disease_labels

    return train_disease_labels, val_disease_labels, test_disease_labels


# endregion


# region metrics
def configure_metrics(
    num_disease: int,
    num_lesion: int,
    disease_avg: Literal["micro", "macro", "weighted"] = "micro",
    lesion_avg: Literal["micro", "macro", "weighted"] = "micro",
    device: str = "cpu",
) -> tuple[MetricCollection, MetricCollection]:
    """Configure metrics for classification and concept detection tasks

    Args:
        n_cls (int): Integer specifying the number of classes. Defaults to None.
        n_cpt (int): Integer specifying the number of concepts. Defaults to None.
        average (Literal["micro", "macro"], optional): Defines the reduction that is applied over labels. Defaults to "micro".

    Returns:
        Tuple[MetricCollection, MetricCollection]: 1st element is classification metrics, 2rd element is concept detection metrics
    """
    if num_disease is None:
        raise ValueError("`num_disease` must be specified")
    if num_lesion is None:
        raise ValueError("`num_lesion` must be specified")
    task = "multiclass" if num_disease > 1 else "binary"
    cls_metrics = MetricCollection(
        {
            "kappa": CohenKappa(
                task=task, num_classes=num_disease, weights="quadratic"
            ),
            "sensitivity": (
                Recall(task=task, num_classes=num_disease, average=disease_avg)
                if task == "multiclass"
                else Recall(task=task, num_labels=num_disease, average=disease_avg)
            ),
            "specificity": (
                Specificity(task=task, num_classes=num_disease, average=disease_avg)
                if task == "multiclass"
                else Specificity(task=task, num_labels=num_disease, average=disease_avg)
            ),
            "auc": (
                AUROC(task=task, num_classes=num_disease, average="macro")
                if task == "multiclass"
                else AUROC(task=task, num_labels=num_disease, average="macro")
            ),
        }
    )
    if disease_avg != "micro":
        cls_metrics.add_metrics(
            {
                "acc": (
                    Accuracy(task=task, num_classes=num_disease, average=disease_avg)
                    if task == "multiclass"
                    else Accuracy(
                        task=task, num_labels=num_disease, average=disease_avg
                    )
                ),
                "f1": (
                    F1Score(task=task, num_classes=num_disease, average=disease_avg)
                    if task == "multiclass"
                    else F1Score(task=task, num_labels=num_disease, average=disease_avg)
                ),
            }
        )

    cpt_metrics = MetricCollection(
        {
            "f1": F1Score(task="multilabel", num_labels=num_lesion, average=lesion_avg),
            "acc": Accuracy(
                task="multilabel", num_labels=num_lesion, average=lesion_avg
            ),
            "auc": AUROC(task="multilabel", num_labels=num_lesion, average="macro"),
        },
        prefix="cpt_",
    )

    return cls_metrics.to(device), cpt_metrics.to(device)


class CLSandCPTMetrics(Metric):
    def __init__(
        self,
        cls_names: list[str],
        cpt_names: list[str],
        cls_avg: Literal["micro", "macro", "weighted"] = "micro",
        cpt_avg: Literal["micro", "macro", "weighted"] = "micro",
        **kwargs,
    ):
        """
        Metrics for classification and concept detection tasks,
        inherited from torchmetrics.Metric

        Args:
            cls_names (List[str]): The list of class names
            cpt_names (List[str]): The list of concept names
            cls_avg (Literal["micro", "macro"], optional): The average method
                for classification metrics. Defaults to "micro".
            cpt_avg (Literal["micro", "macro"], optional): The average method
                for concept detection metrics. Defaults to "micro".
        """
        super().__init__(**kwargs)
        self.n_cls = len(cls_names)
        self.n_cpt = len(cpt_names)
        self.cls_names = cls_names
        self.cpt_names = cpt_names
        self.cls_metrics, self.cpt_metrics = configure_metrics(
            self.n_cls, self.n_cpt, cls_avg, cpt_avg
        )

    def reset(self) -> None:
        self.cls_metrics.reset()
        self.cpt_metrics.reset()

    def update(self, cls_logits, cls_lbls, cpt_logits, cpt_lbls) -> None:
        self.cls_metrics.update(
            (
                cls_logits.detach().softmax(1)
                if self.n_cls > 1
                else cls_logits.detach().sigmoid()
            ),
            cls_lbls,
        )
        self.cpt_metrics.update(cpt_logits.detach().sigmoid(), cpt_lbls)

    def compute(self) -> dict:
        cls_metrics = self.cls_metrics.compute()
        cpt_metrics = self.cpt_metrics.compute()
        self.result = {**cls_metrics, **cpt_metrics}
        return self.result


# endregion


# region visualization
def fit_rs2table(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    best_metrics: Optional[dict] = None,
    best_epoch: Optional[int] = None,
) -> pt.PrettyTable:
    table = pt.PrettyTable()
    table.field_names = [f"Epoch {epoch}", *train_metrics.keys()]
    table.add_row(["Train"] + [f"{v:.2%}" for v in train_metrics.values()])
    table.add_row(["Val"] + [f"{v:.2%}" for v in val_metrics.values()])
    if best_metrics and best_epoch:
        table.add_row(
            [f"Best ep{best_epoch}"] + [f"{v:.2%}" for v in best_metrics.values()]
        )
    return table


def test_rs2table(rs: dict) -> pt.PrettyTable:
    table = pt.PrettyTable()
    table.field_names = [""] + list(rs.keys())
    row_data = [f"{v * 100:.2f}" for v in rs.values()]
    table.add_row(["Test"] + row_data)
    return table


def get_heatmap(cam_: np.ndarray, img_path: str, img_size: int) -> np.ndarray:
    raw_img = Image.open(img_path).convert("RGB")
    raw_img = np.array(raw_img.resize((img_size, img_size)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_), cv2.COLORMAP_JET)  # type: ignore
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = 0.5 * raw_img / 255.0 + 0.5 * heatmap / 255.0
    cam = cam.transpose((2, 0, 1))
    return cam


# endregion

# region load ckpts


def interpolate_pos_embed(model, checkpoint_model) -> None:
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}"
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def load_timm_weights(model_name: str, model: torch.nn.Module):
    checkpoint = timm.create_model(model_name, pretrained=True).state_dict()
    model_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
            # print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k not in ["cls_token", "pos_embed"]
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded pretrained weights from {model_name}")
    return model


def load_vit_ckpt(
    model: nn.Module, ckpt_path: str, pos_embed_modify: bool = True
) -> nn.Module:
    state_dict = model.state_dict()
    ckpt_model = torch.load(ckpt_path, map_location="cpu")["model"]
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    if pos_embed_modify:
        ckpt_model["pos_embed"] = ckpt_model["pos_embed"][:, 1:]
    interpolate_pos_embed(model, ckpt_model)
    _ = model.load_state_dict(ckpt_model, strict=False)
    return model


def rename_pretrain_weight(checkpoint):
    state_dict_old = checkpoint["state_dict"]
    state_dict_new = OrderedDict()
    for key, value in state_dict_old.items():
        state_dict_new[key[len("module.") :]] = value
    return state_dict_new


def load_milvt_ckpt(model: nn.Module, ckpt_path: str):
    state_dict = model.state_dict()
    checkpoint0 = torch.load(ckpt_path, map_location="cpu")

    ckpt_model = rename_pretrain_weight(checkpoint0)
    state_dict = model.state_dict()
    checkpoint_keys = list(ckpt_model.keys())
    for tempKey in list(state_dict.keys()):
        if tempKey not in checkpoint_keys:
            print("Missing Key not in pretrain model: ", tempKey)

    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if k in ckpt_model:  # and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]
    if ckpt_model["pos_embed"].shape[1] != state_dict["pos_embed"].shape[1]:
        ckpt_model["pos_embed"] = ckpt_model["pos_embed"][:, 2:]
    interpolate_pos_embed(model, ckpt_model)
    _ = model.load_state_dict(ckpt_model, strict=False)
    return model


# endregion


# region module
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


# endregion
