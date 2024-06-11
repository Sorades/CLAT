from typing import Dict, Set
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning import Trainer

from models.clat import CLAT
from utils import logging_config


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_and_test(
        self,
        model: "LightningModule",
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule: "LightningDataModule" = None,
        ckpt_path: str = None,
    ) -> None:
        """fit and test the model"""
        self.fit(model, datamodule, ckpt_path)
        self.test(ckpt_path="best", datamodule=datamodule)

    def exp_int(
        self,
        model: CLAT,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule: "LightningDataModule" = None,
        ckpt_path: str = None,
    ) -> None:
        """fit, test, and intervene the model"""

        self.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        model.eval_int = False
        self.test(ckpt_path="best", datamodule=datamodule)
        model.eval_int = True
        self.test(ckpt_path="best", datamodule=datamodule)


class MyCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(run=True, trainer_class=MyTrainer, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.init_args.disease_names", "model.init_args.disease_names"
        )
        parser.link_arguments(
            "data.init_args.lesion_names", "model.init_args.lesion_names"
        )
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.add_argument(
            "--config_overwrite",
            default=False,
            action="store_true",
            help="whether to overwrite the config file",
        )
        return super().add_arguments_to_parser(parser)

    def instantiate_classes(self) -> None:
        try:
            config_overwrite = self.config["config_overwrite"]
        except KeyError:
            config_overwrite = self.config[self.subcommand]["config_overwrite"]
        if config_overwrite:
            print("Overwriting config file")
            self.save_config_kwargs = {"overwrite": True}
        super().instantiate_classes()
        logging_config(self.trainer.log_dir, self.trainer.local_rank)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands.update(
            {
                "fit_and_test": {
                    "model",
                    "train_dataloaders",
                    "val_dataloaders",
                    "datamodule",
                },
                "exp_int": {
                    "model",
                    "train_dataloaders",
                    "val_dataloaders",
                    "datamodule",
                },
            }
        )
        return subcommands


def cli_main():
    cli = MyCLI()

if __name__ == "__main__":
    cli_main()
