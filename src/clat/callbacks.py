import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer

from clat.data import DataItem
from clat.model import CLAT
from clat.utils import CLSandCPTMetrics, fit_rs2table, test_rs2table


class MetricsCaculator(Callback):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: CLAT) -> None:
        self.best_metrics = None
        self.best_epoch = None
        assert isinstance(pl_module, CLAT), (
            f"`pl_module` must be CLAT, got {type(pl_module)}"
        )
        self.train_metrics = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)
        self.val_metrics = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)

    def on_test_start(self, trainer: Trainer, pl_module: CLAT) -> None:
        assert isinstance(pl_module, CLAT), (
            f"`pl_module` must be CLAT, got {type(pl_module)}"
        )
        self.test_metrics = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
    ) -> None:
        self.train_metrics.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            outputs["lesion_logits"],
            batch.lesion_lbls,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.val_metrics.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            outputs["lesion_logits"],
            batch.lesion_lbls,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_metrics.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            outputs["lesion_logits"],
            batch.lesion_lbls,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_rs = self.train_metrics.compute()
        val_rs = self.val_metrics.compute()
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.current_val_rs = val_rs

        table = fit_rs2table(
            trainer.current_epoch, train_rs, val_rs, self.best_metrics, self.best_epoch
        )
        logging.info(f"\n{table}") if self.verbose else None

        for name, value in train_rs.items():
            pl_module.log(
                f"train/{name}",
                value,
                prog_bar=True if name in ["kappa"] else False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        for name, value in val_rs.items():
            pl_module.log(
                f"val/{name}",
                value,
                prog_bar=True if name in ["kappa"] else False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        table = test_rs2table(self.test_metrics.compute())
        logging.info(f"\n{table}") if self.verbose else None

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint,
    ) -> None:
        self.best_metrics = self.current_val_rs
        self.best_epoch = trainer.current_epoch


class GenHeatmap(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_test_start(self, trainer: Trainer, pl_module: CLAT) -> None:
        assert isinstance(pl_module, CLAT), (
            f"`pl_module` must be CLAT, got {type(pl_module)}"
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAT,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        img_size = pl_module.img_size
        logger = pl_module.logger.experiment.add_image  # type: ignore
        cams = outputs["cams"]
        disease_logits = outputs["disease_logits"]
        disease_lbls = batch.disease_lbls
        img_ids = batch.id
        img_paths = batch.img_path

        cams = F.interpolate(
            cams,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        cam_list = []
        concept_confidence = torch.sigmoid(outputs["lesion_logits"])
        cam_list.append(cams)
        cls_attentions = torch.sum(torch.stack(cam_list), dim=0)

        for b in range(batch.image.shape[0]):
            for lesion_ind in range(pl_module.num_lesions):
                lesion_cls_score = format(
                    concept_confidence[b, lesion_ind].cpu().numpy(), ".4f"
                )

                lesion_cls_attention = cls_attentions[b, lesion_ind]

                lesion_cls_attention = (
                    lesion_cls_attention - lesion_cls_attention.min()
                ) / (lesion_cls_attention.max() - lesion_cls_attention.min() + 1e-8)

                lesion_cls_attention = lesion_cls_attention.detach().cpu().numpy()

                img_path = img_paths[b]
                raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_img = np.array(cv2.resize(raw_img, (img_size, img_size)))
                heatmap = cv2.applyColorMap(
                    (255 * lesion_cls_attention).astype(np.uint8), cv2.COLORMAP_JET
                )
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                cam = 0.5 * raw_img.astype(np.float32) / 255.0 + 0.5 * heatmap / 255.0
                cam = cam.transpose((2, 0, 1))

                line = np.ones((3, img_size, 2))
                vis = np.concatenate([cam, line], axis=2)
                ds = disease_logits[b].argmax().item()
                logger(
                    f"{pl_module.lesion_names[lesion_ind]}/ls_{lesion_cls_score}/ds_{ds}/lbl_{disease_lbls[b]}/id_{img_ids[b]}",
                    vis,
                    global_step=trainer.global_step,
                )
