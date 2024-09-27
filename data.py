import cv2
# avoid overload of CPU with multiple GPU envs
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import re
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pandas import DataFrame
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utils import kfold_split, handout_split


class FundusDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        disease_names: List[str],
        lesion_names: List[str],
        val_size: float = None,
        test_size: float = 0.2,
        kfold: int = 0,
        fold_num: int = -1,
        batch_size: int = 16,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.val_size = val_size
        self.test_size = test_size
        self.kfold = kfold
        self.fold_num = fold_num
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_transforms = A.Compose(
            [
                A.Resize(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.GaussianBlur(p=0.5),
                A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )
        self.eval_transforms = A.Compose(
            [
                A.Resize(width=img_size, height=img_size),
                A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )

        self.num_workers = 4

    def setup(self, stage: str) -> None:
        method_name = f"setup_{self.dataset_name}_dataset"
        setup_method = getattr(self, method_name, None)
        assert (
            setup_method is not None
        ), f"Dataset {self.dataset_name} not supported, please choose from {self.dataset_support_list}."

        trainset, valset, testset = setup_method()
        if stage == "fit" or stage is None:
            self.trainset = trainset
            self.valset = valset
        elif stage == "test" or stage == "predict":
            self.testset = testset

        return super().setup(stage)

    def setup_DDR_dataset(self) -> None:
        disease_annotation_file = "data/annotation_DDR_disease.csv"
        lesion_annotation_file = "data/annotation_DDR_lesion.csv"
        root_dir = "/data0/wc_data/LesionDetect/DDR/fundus_384" # Modify this to your own path

        disease_df = pd.read_csv(disease_annotation_file)
        lesion_df = pd.read_csv(lesion_annotation_file)
        # split data
        train_disease_annotation, val_disease_annotation, test_disease_annotation = (
            kfold_split(self.kfold, self.fold_num, disease_df)
            if self.kfold > 1
            else handout_split(self.val_size, self.test_size, disease_df)
        )
        trainset = FundusDatasetWithLesion(
            root_dir,
            ids=train_disease_annotation["ID"].values,
            disease_lbls=train_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.train_transforms,
            file_ext="jpg",
        )
        valset = FundusDatasetWithLesion(
            root_dir,
            ids=val_disease_annotation["ID"].values,
            disease_lbls=val_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="jpg",
        )
        testset = FundusDatasetWithLesion(
            root_dir,
            ids=test_disease_annotation["ID"].values,
            disease_lbls=test_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="jpg",
        )
        return trainset, valset, testset

    def setup_RAO_dataset(self) -> None:
        disease_annotation_file = "/data0/wc_data/LesionDetect/RAO/annotations/new_label_stage.csv"
        lesion_annotation_file = "/data0/wc_data/LesionDetect/RAO/annotations/new_label_lesion.csv" 

        root_dir = "/data0/wc_data/LesionDetect/RAO/fundus_512"

        disease_df = pd.read_csv(disease_annotation_file)
        lesion_df = pd.read_csv(lesion_annotation_file).loc[:, ["ID"] + self.lesion_names]
        # split data
        train_disease_annotation, val_disease_annotation, test_disease_annotation = (
            kfold_split(self.kfold, self.fold_num, disease_df)
            if self.kfold > 1
            else handout_split(self.val_size, self.test_size, disease_df)
        )
        trainset = FundusDatasetWithLesion(
            root_dir,
            ids=train_disease_annotation["ID"].values,
            disease_lbls=train_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.train_transforms,
            file_ext="jpg",
        )
        valset = FundusDatasetWithLesion(
            root_dir,
            ids=val_disease_annotation["ID"].values,
            disease_lbls=val_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="jpg",
        )
        testset = FundusDatasetWithLesion(
            root_dir,
            ids=test_disease_annotation["ID"].values,
            disease_lbls=test_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="jpg",
        )

        return trainset, valset, testset

    def setup_FGADR_dataset(self) -> None:
        disease_annotation_file = "data/annotation_FGADR_disease.csv"
        lesion_annotation_file = "data/annotation_FGADR_lesion.csv"

        root_dir = "/data0/wc_data/LesionDetect/FGADR/fundus_384" # Modify this to your own path

        disease_df = pd.read_csv(disease_annotation_file)
        lesion_df = pd.read_csv(lesion_annotation_file)
        # split data
        train_disease_annotation, val_disease_annotation, test_disease_annotation = (
            kfold_split(self.kfold, self.fold_num, disease_df)
            if self.kfold > 1
            else handout_split(self.val_size, self.test_size, disease_df)
        )
        trainset = FundusDatasetWithLesion(
            root_dir,
            ids=train_disease_annotation["ID"].values,
            disease_lbls=train_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.train_transforms,
            file_ext="png",
        )
        valset = FundusDatasetWithLesion(
            root_dir,
            ids=val_disease_annotation["ID"].values,
            disease_lbls=val_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="png",
        )
        testset = FundusDatasetWithLesion(
            root_dir,
            ids=test_disease_annotation["ID"].values,
            disease_lbls=test_disease_annotation.iloc[:, 1:].values.argmax(axis=1),
            lesion_annotations=lesion_df,
            transforms=self.eval_transforms,
            file_ext="png",
        )

        return trainset, valset, testset

    def setup_FGADDR_dataset(self) -> None:
        from torch.utils.data import ConcatDataset

        DDR_train, DDR_val, DDR_test = self.setup_DDR_dataset()
        FGADR_train, FGADR_val, FGADR_test = self.setup_FGADR_dataset()
        trainset = ConcatDataset([DDR_train, FGADR_train])
        valset = ConcatDataset([DDR_val, FGADR_val])
        testset = ConcatDataset([DDR_test, FGADR_test])
        return trainset, valset, testset


    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @classmethod
    def dataset_support_list(cls):
        return [
            re.match("setup_(.*)_dataset", name).group(1)
            for name in dir(cls)
            if re.match("setup_(.*)_dataset", name)
        ]


class FundusDatasetWithLesion(Dataset):
    def __init__(
        self,
        root_dir: str,
        ids: np.ndarray,
        disease_lbls: np.ndarray,
        lesion_annotations: DataFrame,
        transforms: A.Compose,
        file_ext: str = "jpg",
    ) -> None:
        """Dataset class for fundus image classification with retinal lesion labels.

        Args:
            root_dir (str): The root directory of the dataset.
            ids (np.ndarray): The id of the images.
            disease_lbls (np.ndarray): The disease labels of the images. Doesn't matter if it's one-hot or not.
            lesion_annotations (DataFrame): The lesion annotations of the images.
            transforms (T.Compose): The transforms to apply to the images.
        """
        super().__init__()
        self.root_dir = root_dir
        self.file_ext = file_ext
        self.ids = ids
        self.disease_lbls = disease_lbls
        self.lesion_annotations = lesion_annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        disease_lbls = self.disease_lbls[index]
        img_path = f"{self.root_dir}/{id}.{self.file_ext}"

        assert (
            id in self.lesion_annotations["ID"].values
        ), f"{id} does not exist in lesion annotations file."
        assert isinstance(self.transforms, A.Compose), "Invalid transforms."

        # Get concept labels
        lesion_lbls = self.lesion_annotations[self.lesion_annotations["ID"] == id].iloc[
            :, 1:
        ]

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        image = np.array(pil_img)

        image = self.transforms(image=image)["image"]

        return {
            "image": image,
            "disease_lbls": disease_lbls,
            "lesion_lbls": lesion_lbls.values[0],
            "id": id,
            "img_path": img_path,
        }

