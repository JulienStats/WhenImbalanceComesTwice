import os
import pandas as pd
import numpy as np
import cv2
from typing import List
from torch.utils.data import Dataset
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def get_biased_indices(labels: List[bool], bias: float, dataset_size: int) -> List:
    """Return a list of indices

    Args:
        labels (List[bool]): list of Os and 1s
        bias (float): the proportion of 1 we want in the final dataset

    Returns:
        List: the indices with the required proportion of ones and zeros
    """
    required_number_of_defects = int(bias * dataset_size)
    remaining_number_of_healthy = dataset_size - required_number_of_defects

    if labels is np.ndarray:
        labels = torch.from_numpy(labels)

    defect_indices = np.random.choice(
        torch.where(labels == 1)[0], required_number_of_defects, replace=False
    )
    healthy_indices = np.random.choice(
        torch.where(labels == 0)[0], remaining_number_of_healthy, replace=False
    )
    to_ret = np.concatenate([defect_indices, healthy_indices])
    assert len(to_ret) == dataset_size, f"{len(to_ret)=} != {dataset_size=}"
    return to_ret


def get_biased_indices_augment(labels: List[bool], bias: float) -> List:
    """Return a list of indices

    Args:
        labels (List[bool]): list of Os and 1s
        bias (float): the proportion of defect required

    Returns:
        List: the indices with the required proportion of ones and zeros
    """

    labels: np.ndarray
    labels = torch.from_numpy(labels)

    defect_indices = torch.where(labels == 1)[0]
    number_of_defects = len(defect_indices)

    number_of_healthy = int((number_of_defects * (1 - bias)) / bias)
    healthy_indices = np.random.choice(
        torch.where(labels == 0)[0], number_of_healthy, replace=False
    )
    to_ret = np.concatenate([defect_indices, healthy_indices])
    return to_ret


class imagenetpreprocess(A.ImageOnlyTransform):
    def __init__(self, encoder_name="efficientnet-b3", always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(
            encoder_name, pretrained="imagenet"
        )

    def apply(self, image, **params):
        # Apply the preprocessing function
        image = self.preprocess_fn(image)
        return image


potato_tf_train = A.Compose(
    [
        imagenetpreprocess(),
        A.RandomCrop(256, 256),
        ToTensorV2(),
    ]
)

potato_tf_test = A.Compose([imagenetpreprocess(), A.Resize(256, 256), ToTensorV2()])


tf_train = A.Compose(
    [
        imagenetpreprocess(),
        A.Resize(64, 64),
        ToTensorV2(),
    ]
)

tf_test = A.Compose([imagenetpreprocess(), A.Resize(64, 64), ToTensorV2()])


class PotatoDataset(Dataset):
    def __init__(self, ml_set: str, folder_path: str, bias: int = 0, transform=None):
        self.data_path = os.path.join(folder_path, f"{ml_set}")
        self.data = os.listdir(self.data_path)
        self.ml_set = ml_set
        self._transform = transform
        self.masks = pd.Series(self.data)[
            pd.Series(self.data).str.startswith("mask")
        ].sort_values()
        self.images = pd.Series(self.data)[
            pd.Series(self.data).str.startswith("scan")
        ].sort_values()
        self.names = np.arange(len(self.images))
        self.bias = bias

        assert len(self.masks) == len(self.images)

        self.clsNames = {
            # occ one class defect
            0: "background",
            1: "defect",
        }

    def randomize(self):
        self.masks = self.masks.sample(frac=1)
        self.images = self.images.sample(frac=1)

    def __getitem__(self, index):
        original_image = cv2.imread(
            os.path.join(self.data_path, self.images.iloc[index])
        )
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.data_path, self.masks.iloc[index]))

        assert (
            self.images.iloc[index].split("_")[1]
            == self.masks.iloc[index].split("_")[1]
        )

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).reshape(*mask.shape[:2], 1)

        if self._transform:
            image = np.array(original_image)
            mask = np.array(mask)
            seg_item = self._transform(image=image, mask=mask)

            return {
                "image": seg_item["image"].float(),
                "mask": seg_item["mask"].permute(2, 0, 1),
                "name": self.images.iloc[index].split("_")[1],
                "original_image": original_image,
            }

        return {
            "original_image": original_image,
            "mask": mask,
            "name": self.images.iloc[index].split("_")[1],
        }

    def __len__(self):
        return len(self.images)



if __name__ == "__main__":
    pass