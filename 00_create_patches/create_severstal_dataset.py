import os

import lightning as L
from base_severstal_dataset import SeverstalDataset
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from typer import Typer
import cv2
from tqdm.auto import tqdm
import pandas as pd

logger.add("logs/al_logger.txt")


app = Typer(pretty_exceptions_enable=False)



@app.command()
def main(
    datadir: str,
    datasetdir: str = "severstal_steel_detection",
):
    print(f"""
    ================================
    Arguments : 
    {datadir=}
    {datasetdir=}

    ================================
    """)
    L.pytorch.seed_everything(15, workers=True)
    Path(datadir)
    config = OmegaConf.load("config.yaml")


    new_data_dir = os.path.join(datadir, "severstal_patches")
    image_dir = os.path.join(new_data_dir, "data")
    masks_dir = os.path.join(new_data_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    datadir = os.path.join(datadir, datasetdir)

    ds = SeverstalDataset(rootdir=datadir, binary_output=False)

    logger.info(f"Dataset length: {len(ds)}")

    defects = np.zeros(len(ds))
    rate_of_black = np.zeros(len(ds))
    defects = []
    names = []

    for image_index in tqdm(range(len(ds)), desc = "Creating dataset splits ..."):
        seg_item = ds[image_index]
        image = seg_item["original_image"]
        mask = seg_item["mask"]
        name = seg_item["name"]
        rate_of_black = (image[0,:,:].mean(0)<=5).mean()

        if rate_of_black >= 0.02:
            continue

        else:
            image = image.squeeze()
            h,w = image.shape
        
        assert h == 256, f"image {name} has a height problem {h=}"
        assert w == 1600, f"image {name} has a width problem {w=}"

        num_splits = 4
        stripe_w = w // num_splits  # 400
        for i in range(num_splits):
            x0 = i * stripe_w
            x1 = (i + 1) * stripe_w
            crop = image[:, x0:x1]
            mask_crop = mask[:, x0:x1].numpy()
            crop.squeeze().shape == mask_crop.shape, f"split {i} of image {name} has a shape problem {crop.squeeze().shape=} {mask_crop.shape=}"
            cv2.imwrite(os.path.join(image_dir,f"{name.split('.')[0]}_{i}.jpg"), crop)

            if mask_crop.sum() > 0:
                cv2.imwrite(os.path.join(masks_dir,f"{name.split('.')[0]}_{i}.jpg"), mask_crop)

            names.append(f"{name.split('.')[0]}_{i}")
            defects.append((mask_crop>0).sum())

    defects = np.array(defects)
    names = np.array(names)
    
    pd.DataFrame({"name":names, "defects":defects}).to_csv(os.path.join(new_data_dir, "labels.csv"))
    
if __name__ == "__main__":
    app()
