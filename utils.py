import os
import cv2
import numpy as np
import timm
from PIL import Image
import torch
import logging
import pandas as pd
import prettytable as pt
from typing import List, Literal, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import (
    Metric,
    Accuracy,
    Recall,
    F1Score,
    AUROC,
    Specificity,
    MetricCollection,
    CohenKappa,
)

def logging_config(log_dir: str = None, rank: int = None):
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


def kfold_split(
    kfold: int, fold_num: int, disease_labels: pd.DataFrame
) -> List[pd.DataFrame]:
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    train_idx, val_idx = list(
        skf.split(disease_labels, y=disease_labels.iloc[:, 1:].values.argmax(axis=1))
    )[fold_num]
    train_disease_labels = disease_labels.iloc[train_idx, :]
    val_disease_labels = disease_labels.iloc[val_idx, :]
    test_disease_labels = val_disease_labels
    return train_disease_labels, val_disease_labels, test_disease_labels


def handout_split(
    val_size: float, test_size: float, disease_labels: pd.DataFrame
) -> List[pd.DataFrame]:
    if test_size == 1:
        return None, None, disease_labels

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


def configure_metrics(
    num_disease: int = None,
    num_lesion: int = None,
    disease_avg: Literal["micro", "macro", "weighted"] = "micro",
    lesion_avg: Literal["micro", "macro", "weighted"] = "micro",
    device: str = "cpu",
) -> Tuple[MetricCollection, MetricCollection]:
    """Configure metrics for classification and concept detection tasks

    Args:
        n_cls (int, optional): Integer specifying the number of classes. Defaults to None.
        n_cpt (int, optional): Integer specifying the number of concepts. Defaults to None.
        average (Literal["micro", "macro"], optional): Defines the reduction that is applied over labels. Defaults to "micro".

    Returns:
        Tuple[MetricCollection, MetricCollection]: 1st element is classification metrics, 2rd element is concept detection metrics
    """
    if num_disease:
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
                    else Specificity(
                        task=task, num_labels=num_disease, average=disease_avg
                    )
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
                        Accuracy(
                            task=task, num_classes=num_disease, average=disease_avg
                        )
                        if task == "multiclass"
                        else Accuracy(
                            task=task, num_labels=num_disease, average=disease_avg
                        )
                    ),
                    "f1": (
                        F1Score(task=task, num_classes=num_disease, average=disease_avg)
                        if task == "multiclass"
                        else F1Score(
                            task=task, num_labels=num_disease, average=disease_avg
                        )
                    ),
                }
            )
    if num_lesion:
        cpt_metrics = MetricCollection(
            {
                "f1": F1Score(
                    task="multilabel", num_labels=num_lesion, average=lesion_avg
                ),
                "acc": Accuracy(
                    task="multilabel", num_labels=num_lesion, average=lesion_avg
                ),
                "auc": AUROC(task="multilabel", num_labels=num_lesion, average="macro"),
            },
            prefix="cpt_",
        )

    return (
        cls_metrics.to(device) if num_disease else None,
        cpt_metrics.to(device) if num_lesion else None,
    )


class CLSandCPTMetrics(Metric):
    def __init__(
        self,
        cls_names: List[str],
        cpt_names: List[str],
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


def fit_rs2table(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    best_metrics: dict = None,
    best_epoch: int = None,
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
    row_data = [f"{v*100:.2f}" for v in rs.values()]
    table.add_row(["Test"] + row_data)
    return table


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


def get_heatmap(cam_, img_path, img_size):
    raw_img = Image.open(img_path).convert("RGB")
    raw_img = np.array(raw_img.resize((img_size, img_size)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = 0.5 * raw_img / 255.0 + 0.5 * heatmap / 255.0
    cam = cam.transpose((2, 0, 1))
    return cam