import logging
from typing import Any, Union
from lightning import LightningModule, Trainer
import torch
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from models import AbstractCPTModule
from utils import (
    fit_rs2table,
    test_rs2table,
    CLSandCPTMetrics,
)


class MetricsCaculator(Callback):
    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.cls_avg = "micro"
        self.cpt_avg = "micro"

    def _unpack_outputs(self, pl_module, outputs: STEP_OUTPUT):
        if isinstance(pl_module, AbstractCPTModule):
            ret_keys = [
                "disease_logits",
                "disease_lbls",
                "lesion_logits",
                "lesion_lbls",
            ]
        else:
            raise ValueError(
                f"pl_module must be AbstractCPTModule, got {type(pl_module)}"
            )

        return (outputs[key] for key in ret_keys)

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.best_metrics = None
        self.best_epoch = None
        if isinstance(pl_module, AbstractCPTModule):
            self.train_metrics = CLSandCPTMetrics(
                pl_module.disease_names,
                pl_module.lesion_names,
                self.cls_avg,
                self.cpt_avg,
            ).to(pl_module.device)
            self.val_metrics = CLSandCPTMetrics(
                pl_module.disease_names,
                pl_module.lesion_names,
                self.cls_avg,
                self.cpt_avg,
            ).to(pl_module.device)
        else:
            raise ValueError(
                f"pl_module must be AbstractCPTModule, got {type(pl_module)}"
            )

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if isinstance(pl_module, AbstractCPTModule):
            self.test_metrics = CLSandCPTMetrics(
                pl_module.disease_names,
                pl_module.lesion_names,
                self.cls_avg,
                self.cpt_avg,
            ).to(pl_module.device)
        else:
            raise ValueError(
                f"pl_module must be AbstractCPTModule, got {type(pl_module)}"
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.train_metrics.update(*self._unpack_outputs(pl_module, outputs))

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.val_metrics.update(*self._unpack_outputs(pl_module, outputs))

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_metrics.update(*self._unpack_outputs(pl_module, outputs))

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
        checkpoint: torch.Dict[str, Any],
    ) -> None:
        self.best_metrics = self.current_val_rs
        self.best_epoch = trainer.current_epoch
