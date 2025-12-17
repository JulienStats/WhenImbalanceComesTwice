from typer import Typer
from base_potato_dataset import PotatoDataset
import os
from pathlib import Path
import cv2
from einops import rearrange
from tqdm.auto import tqdm

app = Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    dataset_rootdir:str, # directory path where the base dataset are stored
    psize:int = 256
):
    rootdir = Path(dataset_rootdir)
    train_ds  = PotatoDataset(folder_path=rootdir / "potato_disease", ml_set =  "train_")
    valid_ds  = PotatoDataset(folder_path=rootdir / "potato_disease", ml_set = "valid_")
    test_ds  = PotatoDataset(folder_path=rootdir / "potato_disease", ml_set =  "test_")


    # Save of patched dataset
    output_base_path = rootdir / f"potato_disease_{psize}"
    image_save_path = output_base_path / "data"
    masks_save_path = output_base_path / "labels"
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(masks_save_path, exist_ok=True)
    
    global_counter = 0

    for i in tqdm(range(len(train_ds)), desc="Creating patches for training set"):
        image = train_ds[i]["original_image"]
        mask = train_ds[i]["mask"]
        name = train_ds[i]["name"]
        img = rearrange(image, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        msk = rearrange(mask, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        for j in range(img.shape[0]):
            cv2.imwrite(str(image_save_path / f"{global_counter}_{j}.png"), img[j])
            if (msk[j]>0).astype(int).sum()>0:
                # save mask only if exists
                cv2.imwrite(str(masks_save_path /f"{global_counter}_{j}.png"), (msk[j]>0).astype(int))

        global_counter +=1

    for i in tqdm(range(len(valid_ds)), desc="Creating patches for validation set"):
        image = valid_ds[i]["original_image"]
        mask = valid_ds[i]["mask"]
        name = valid_ds[i]["name"]
        img = rearrange(image, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        msk = rearrange(mask, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        for j in range(img.shape[0]):
            cv2.imwrite(str(image_save_path / f"{global_counter}_{j}.png"), img[j])
            if (msk[j]>0).astype(int).sum()>0:
                # save mask only if exists
                cv2.imwrite(str(masks_save_path /f"{global_counter}_{j}.png"), (msk[j]>0).astype(int))

        global_counter +=1

    for i in tqdm(range(len(test_ds)), desc="Creating patches for test set"):
        image = test_ds[i]["original_image"]
        mask = test_ds[i]["mask"]

        img = rearrange(image, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        msk = rearrange(mask, "(nh ph) (nw pw) c-> (nh nw) ph pw c", ph=psize, pw=psize)
        for j in range(img.shape[0]):
            cv2.imwrite(str(image_save_path / f"{global_counter}_{j}.png"), img[j])
            if (msk[j]>0).astype(int).sum()>0:
                # save mask only if exists
                cv2.imwrite(str(masks_save_path /f"{global_counter}_{j}.png"), (msk[j]>0).astype(int))

        global_counter +=1

    return 0


if __name__ == "__main__":
    app()