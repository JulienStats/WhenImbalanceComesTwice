import os

import lightning as L
import numpy as np
from active_learning import al_loop
from semantic_segmentation import (
    AVAILABLE_SCORING_FUNCTIONS,
)
from pathlib import Path
from copy import deepcopy
from patch_dataset import PatchDataset, transforms
from sklearn.model_selection import train_test_split
from loguru import logger
from omegaconf import OmegaConf
from lightning_module import SsegmentationMulticlass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typer import Typer
from typing import List
import pandas as pd


logger.add("logs/al_logger.txt")

app = Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    datadir: str,
    datasetdir: str = "potato_disease_256",  # potato_disease_256  severstal_patches
    bias: float = 1.0,
    logdir: str = "./logs/",
    strat: str = "deepcoresetSampling",
    start_random: bool = True,
    pi1: float = 0.05,
    run_id: int = 0,
    constant_steps: int = -1,
    seed: int = 30,
):
    print(f"""
    ================================
    Arguments : 
    {datadir=}
    {datasetdir=}
    {bias=}
    {logdir=}
    {strat=}
    {start_random=} 
    {pi1=}
    {run_id=}
    {seed=}
    ================================
    """)

    L.pytorch.seed_everything(seed, workers=True)
    assert datasetdir in ["severstal_patches", "potato_disease_256"]
    config = OmegaConf.load("config.yaml")
    config.strat = strat
    config.seed = seed

    assert config.strat in AVAILABLE_SCORING_FUNCTIONS, (
        f"Strat {config.strat} not available. Available: {AVAILABLE_SCORING_FUNCTIONS.keys()}"
    )

    config.bias = bias
    test = 1

    if test:
        config.batch_size = 2  # max found by my function
        config.val_batch_size = 4 * config.batch_size  # max found by my function
    else:
        config.batch_size = 50  # max found by my function
        config.val_batch_size = 4 * config.batch_size  # max found by my function

    log_dir = (
        Path(logdir)
        / f"epochs_{config.epochs}_cycles_{config.cycles}_reps_{config.repetitions}_budget_{config.budget}_strat_{strat}_bias_{bias}_cstSteps_{constant_steps}_dataset_{datasetdir}_runid_{run_id}_seed_{seed}_test_{test}"
    )

    config.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    datadir = os.path.join(datadir, datasetdir)

    config.workers = int(os.cpu_count()) // 2

    ds = PatchDataset(rootdir=datadir, transform=transforms[datasetdir])
    logger.info(f"Dataset length: {len(ds)}")

    labels = (
        pd.Series(ds.image_names)
        .apply(lambda s: s.stem)
        .isin(pd.Series(ds.mask_names).apply(lambda s: s.stem))
    )
    print(f"{np.bincount(labels)=}, {labels.mean()=}")

    train_indices, test_indices = train_test_split(
        np.arange(len(ds)), test_size=0.2, random_state=15, stratify=labels
    )
    train_ds = deepcopy(ds)
    train_ds.image_names = ds.image_names[train_indices]

    val_ds = deepcopy(ds)
    val_ds.image_names = ds.image_names[test_indices[::2]]

    test_ds = deepcopy(ds)
    test_ds.image_names = ds.image_names[test_indices[1::2]]

    assert np.intersect1d(train_ds.image_names, test_ds.image_names).size == 0
    assert np.intersect1d(val_ds.image_names, test_ds.image_names).size == 0

    # We bias the train dataset
    logger.info(f"Train ds : {np.bincount(labels[train_indices])=}")
    logger.info(f"val ds : {np.bincount(labels[test_indices[::2]])=}")
    logger.info(f"Test ds : {np.bincount(labels[test_indices[1::2]])=}")
    logger.info(f"{bias=}")

    biased_train: PatchDataset = train_ds.do_bias(bias, dataset_size=4300)
    biased_train.bias

    pi0_test_ds = test_ds.do_bias(bias, 500)
    pi0_test_ds.bias

    pi1_test_ds = test_ds.do_bias(pi1, 500)
    pi1_test_ds.bias


    # make sure no data leak
    assert (
        np.intersect1d(
            biased_train.image_names,
            test_ds.image_names,
        ).size
        == 0
    )
    assert (
        np.intersect1d(
            biased_train.image_names,
            pi1_test_ds.image_names,
        ).size
        == 0
    )
    writer = SummaryWriter(f"{config.log_dir}/global_writer/{config.strat}")

    logger.info(f"Running the experiment with bias of {bias:.2%}")
    logger.info(f"Running the experiment with pi1 of {pi1:.2%}")
    # RuntimeError: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.

    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        pi0_test_ds,
        batch_size=config.val_batch_size,
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
    )

    pi1_test_loader = DataLoader(
        pi1_test_ds,
        batch_size=config.val_batch_size,
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
    )

    logger.info(f"Length of validation loader {len(val_loader)}")
    logger.info(f"Length of test loader {len(test_loader)}")
    logger.info(f"Length of test pi1 loader {len(pi1_test_loader)}")

    logger.info(f"bias of train dataset {biased_train.bias}")
    logger.info(f"bias of test ds {pi0_test_ds.bias}")
    logger.info(f"bias of test pi1 ds {pi1_test_ds.bias}")

    def create_model(config):
        categories: List[str] = ["BG", "defect"]
        return SsegmentationMulticlass(
            config,
            categories=categories,
            num_channels=3 if datasetdir == "potato_disease_256" else 1,
        )

    def get_trainer(config, test: int = test):
        return L.Trainer(
            max_epochs=config.epochs,
            max_steps=2 if test else -1,
            log_every_n_steps=1,
            profiler="simple",
            deterministic="warn",
            precision="16-mixed",
            enable_progress_bar=True,
            check_val_every_n_epoch=config.epochs,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            default_root_dir=os.path.join(log_dir, f"{config.strat}"),
            callbacks=[],
        )

    al_loop(
        config,
        biased_train,
        biased_train.image_names,
        val_loader,
        [test_loader, pi1_test_loader],
        AVAILABLE_SCORING_FUNCTIONS[config.strat],
        get_trainer,
        create_model,
        logger,
        writer,
        deterministic="warn",
        start_random=start_random,
        constant_steps=constant_steps,
    )


if __name__ == "__main__":
    app()
