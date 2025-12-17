from pathlib import Path
import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import torch
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2

severstal_transform = A.Compose(
    [
        A.PadIfNeeded(
            min_height=256,
            min_width=416,
            # Dimension constraints for FPN models
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        A.GaussianBlur(blur_limit=0, sigma_limit=(0.5, 2), p=0.5),
        A.HorizontalFlip(),
        A.Normalize(),
        ToTensorV2(),
    ]
)

patates_transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.GaussianBlur(blur_limit=0, sigma_limit=(0.5, 2), p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)

transforms = {
    "severstal_patches": severstal_transform,
    "potato_disease_256": patates_transform,
}


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

    if isinstance(labels, np.ndarray):
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


class PatchDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        transform=None,
    ):
        self.images_path = Path(rootdir)
        self.image_base_path = self.images_path / "data"
        self.masks_base_path = self.images_path / "labels"

        self.image_names = np.array(list((self.image_base_path).glob("*")))
        self.mask_names = np.array(list((self.masks_base_path).glob("*")))

        print(f"Found {len(self.image_names)} images, {len(self.mask_names)} masks")

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.images_path.stem == "severstal_patches":
            image = cv2.imread(self.image_names[idx], cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(
                self.image_names[idx],
            )

        mask_path = str(self.image_names[idx]).replace("/data/", "/labels/")
        if Path(mask_path) in self.mask_names:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)

        transformed.update(
            {
                "name": self.image_names[idx].stem,
                "original_image": image,
            }
        )

        # we binarize output (severstal has 4 classes)
        transformed["mask"] = (transformed["mask"] > 0).long()
        return transformed

    def do_bias(self, bias: float = 0.5, dataset_size: int = 1000) -> Dataset:
        """Create a biased dataset with a proportion pi0 of defective samples

        Args:
            pi0 (float, optional): Proportion of defective samples. Defaults to 0.5.

        Returns:
            Dataset: _description_
        """
        images_stems = pd.Series(self.image_names).apply(lambda s: s.stem)
        masks_stems = pd.Series(self.mask_names).apply(lambda s: s.stem)

        labels = images_stems.isin(masks_stems).to_numpy()
        print(f"{np.bincount(labels)=}, {labels.mean()=}")
        biased_indices = get_biased_indices(labels, bias, dataset_size)
        biased_ds = deepcopy(self)
        biased_ds.image_names = self.image_names[biased_indices]

        return biased_ds

    @property
    def bias(self):
        images_stems = pd.Series(self.image_names).apply(lambda s: s.stem)
        masks_stems = pd.Series(self.mask_names).apply(lambda s: s.stem)

        return images_stems.isin(masks_stems).mean()


if __name__ == "__main__":
    pass
