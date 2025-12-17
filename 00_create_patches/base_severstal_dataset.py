from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def _pixels2mask(mask: np.ndarray, pixels: List[int], class_id: int):
    fill_val = class_id
    for i in range(0, len(pixels), 2):
        mask[pixels[i] - 1 : pixels[i] - 1 + pixels[i + 1]] = fill_val


def pixels2mask(pixels: List[int], mask: np.ndarray, class_id: int) -> np.ndarray:
    flattened_mask = mask.reshape(-1, order="F")
    _pixels2mask(flattened_mask, pixels, class_id)
    return flattened_mask.reshape(mask.shape, order="F")


def _mask2pixels(flattened_mask: np.ndarray, class_id: int) -> List[int]:
    fill_val = class_id
    pixels = List()
    idx = 0
    start_idx = 0
    count = 0
    for i in flattened_mask:
        if i == fill_val:
            if count == 0:
                start_idx = idx
            count += 1
        else:
            if count > 0:
                pixels.append(start_idx + 1)
                pixels.append(count)
                count = 0
        idx += 1
    if count > 0:
        pixels.append(start_idx + 1)
        pixels.append(count)
    return pixels


def mask2pixels(mask: np.ndarray, class_id: int) -> List[int]:
    return _mask2pixels(mask.flatten("F"), class_id)


class SeverstalDataset(Dataset):
    def __init__(
        self,
        rootdir: str,
        binary_output: bool = False,
    ):
        self.images_path = Path(rootdir)
        self.csv_path = Path(rootdir) / "train.csv"

        self.binary_output = binary_output
        try:
            self.image_names = np.load("../severstal_imnames.npy", allow_pickle=True)
        except Exception:
            self.image_names = np.array(list(self.images_path.glob("*.jpg")))

        self.df = pd.read_csv(self.csv_path)
        self.defect_images_names = set(self.df["ImageId"].unique())

        self.transform = A.Compose([A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name: Path = self.image_names[idx]
        image = cv2.imread(self.images_path / image_name, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        mask: np.ndarray = np.zeros((height, width), dtype=image.dtype)

        image_id: str = image_name.stem
        image_filename = image_id + ".jpg"

        if image_filename not in self.defect_images_names:
            # if the image is healthy the mask is full of 0
            seg_item = {
                "image": np.expand_dims(image, axis=0),
                "mask": mask,
                "name": image_filename,
            }

        else:
            # if the mask contains some defects
            for i, row in self.df[self.df["ImageId"] == image_filename].iterrows():
                mask = pixels2mask(
                    list(map(int, row.EncodedPixels.split())), mask, row.ClassId
                )
            seg_item = {
                "image": np.expand_dims(image, axis=0),
                "mask": mask,
            }
        transformed = self.transform(
            image=seg_item["image"].squeeze(), mask=seg_item["mask"]
        )

        transformed.update(
            {
                "name": image_filename,
                "original_image": seg_item["image"],
            }
        )
        transformed["mask"] = transformed["mask"].long()
        if self.binary_output:
            transformed["mask"] = (transformed["mask"] > 0).long()

        return transformed

    def stats_summary(self):
        num_images = len(self)
        num_faulty = len(self.defect_images_names)
        num_healthy = num_images - num_faulty
        prop_healthy = num_healthy / num_images
        prop_faulty = 1 - prop_healthy

        return f""" 
            {num_images=}
            {num_faulty=}
            {num_healthy=}
            {prop_healthy=}
            {prop_faulty=}      
        """


if __name__ == "__main__":
    pass
