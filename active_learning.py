from torch.utils.data import DataLoader, Dataset

from typing import Callable, List, Dict, Optional, Union

import numpy as np
import torch
import gc
import lightning as L
import pandas as pd
import os
from time import time
import math
from al_dataset import ActiveLearningDataset

def make_budget(config, al_dataset, logger):
    """Create a budjets list given a budget argument of a config

    Args:
        config (_type_): _description_
        al_dataset (_type_): _description_
        logger (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    We can create 3 types of budget;
         int given an int and a number of cycles -> The budget will be the int repeated cycle times
         float -> like int but the float represents a proportion of the total dataset to be labeled at each cycle
         List[float]: Will budget percentages of the total dataset
         List[int]: the number of images to add at each cycle. 

    """    
    if type(config.budget) is int:
        # if you provide budget as an integer, the AL procudeure will consider <cycles> labeling cycles adding
        budgets = [config.budget for _ in range(config.cycles)]
    
    elif type(config.budget) is float:
        # if you provide budget as a float, the AL procudeure will consider <cycles> labeling cycles adding
        int_budget = int(config.budget * len(al_dataset))
        budgets = [int_budget for _ in range(config.cycles)]

    elif isiterable(config.budget) and type(config.budget[0]) is float:
        # here you provide proportion of full dataset to be labeled at each cycle
        cycles = len(config.budget)
        if logger is not None:
            logger.info(
                f"Ignoring cycles provided in config file ({config.cycles}) and running {cycles} cycles according to len of provided budget"
            )
        budgets_cum = np.array([int(prop * len(al_dataset)) for prop in config.budget])
        budgets = budgets_cum.copy()
        for i in range(1, cycles):
            budgets[i] = budgets_cum[i] - budgets_cum[i - 1]
        for i in range(cycles):
            # we check wether computed budgets correspond to the provided proportions
            assert math.isclose(
                np.cumsum(budgets)[i] / len(al_dataset), config.budget[i], abs_tol=0.1
            ), (
                f"budgets {np.cumsum(budgets)[i] / len(al_dataset)} is not equal to {config.budget[i]}"
            )
        if logger is not None:
            logger.info(f"Budget for this experiment is going to be : {budgets}")
    elif isiterable(config.budget) and type(config.budget[0]) is int:
        cycles = len(config.budget)
        if logger is not None:
            logger.info(
                f"Ignoring cycles provided in config file ({config.cycles}) and running {cycles} cycles according to len of provided budget"
            )
        budgets = config.budget
    else:
        raise ValueError(
            f"budget must be an int, a list of int or a list of float {print(type(config.budget))}"
        )
    
    assert sum(budgets) <= len(al_dataset), "Total number of selected items is bigger than the size of the dataset"
    config.cycles = len(budgets)
    return budgets


def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def write_dict_to_file(
    log_dir: str, dico: Dict[str, float], cycle: int, rep: int, logger, strat: str
) -> None:
    logger.info(f"Writing metrics to {log_dir}/dict_{strat}.csv")

    log_dfpath = f"{log_dir}/dict_{strat}.csv"
    index = f"{cycle}_{rep}"
    if os.path.isfile(log_dfpath):
        saved_df = pd.read_csv(log_dfpath, sep=";", index_col=0)
        current_df = pd.DataFrame(dico, index=[index])

        updated_df = pd.concat([saved_df, current_df])
        updated_df.to_csv(log_dfpath, sep=";")
    else:
        df = pd.DataFrame(dico, index=[index])
        df.to_csv(log_dfpath, sep=";")


def al_loop(
    config,
    dataset: Dataset,
    dataset_index: List[str],
    val_loader: DataLoader,
    test_loaders: List[DataLoader],
    strategy,
    get_trainer: Callable,
    create_module: Callable,
    logger,
    writer,
    precomputed_features: Optional[Union[str, np.ndarray]] = None,
    start_random=True,
    deterministic: Union[bool, str] = False,  # True False "warn"
    constant_steps:int = -1,
):
    """_summary_

    Args:
        config (_type_): Dictconfig with all hyperparameter
        dataset (Dataset): Dataset used as starting Du
        val_loader (DataLoader): Validation dataloader used to validate along each AL cycle
        strategy (baseStrat): Al strategy
        get_trainer (Callable): function that take no input and return a new trainer
        create_module (Callable): function that take config as input and return a lightning module
        writer (_type_): tensorboard SummaryWriter that logs all metrics logged by the lightning module
        hook_process_func (_type_, optional): function that parse embeddings hooked at config.hook_layer Defaults to lambdax:x.squeeze().
        precomputed_features (str, optional): path to file containing pre-computed feature to run core-set selection algorithm. Defaults to None.
        constant_steps (int, optional): if different from -1, set a specific number of optim steps, change the number of epochs accordingly. Defaults to -1.
    """
    repetitions: int = config.repetitions
    seed: int = config.seed
    subset: int = config.subset
    
    np.random.seed(seed)

    assert len(dataset) == len(dataset_index), (
        f"dataset and dataset_index must have the same length, here {len(dataset)=}, {len(dataset_index)=}"
    )

    al_dataset = ActiveLearningDataset(dataset, config, dataset_index)

    # setup budgets_list
    # at each cycle, budget should be the number of data ADDED to the current labeled set

    budgets = make_budget(config, al_dataset, logger)
    cycles = len(budgets)
    logger.info(f"Budget for this experiment is going to be : {budgets}")

    if deterministic or deterministic == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)

    # We have a tensorboard writer for logging at the base level (among all reps and all cycles, i will log at te globla_step steps.)
    global_step = 0

    for repetition in range(repetitions):
        model = create_module(config)
        al_dataset.start_rep()
        t0_rep = time()

        for cycle in range(cycles):
            t0_cycle = time()
            config.current_cycle = cycle
            budget = budgets[cycle]

            t1_score = time()

            # Random selection if first cycle.
            if cycle == 0 and start_random:
                # first selection
                selected = {
                    "ids": np.random.choice(len(dataset), budget, replace=False),
                    "scores": np.zeros(budget, dtype=np.float32) - 1,
                }
                logger.info("FIRST CYCLE : RANDOM SELECTION")

            else:
                # normal scoring on a subset
                strat = strategy(
                    base_dataset=al_dataset,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    logger=logger,
                    budget=budget,
                    save_freq=-1,
                    precomputed_features=precomputed_features,
                )
                selected: dict = strat.select(al_dataset.unlabeled_indices)

            al_dataset.label(selected, budget, cycle, repetition)

            # =========================================================
            #  Training
            # =========================================================

            labeled_loader = al_dataset.labeled_dataloader()

            t1_start_train = time()
            
            if constant_steps != -1:
                config.epochs = int(constant_steps / len(labeled_loader)) + 1
                logger.info(f"Setting epochs to {config.epochs} to have {constant_steps} steps")

            trainer: L.Trainer = get_trainer(config)
            model = create_module(config)

            
            trainer.fit(
                model, train_dataloaders=labeled_loader, val_dataloaders=val_loader
            )

            if isinstance(test_loaders, DataLoader):
                # convert to list in case only one dataloader has been passed
                test_loaders = [test_loaders]
            metrics_dicts = []
            for dataloader in test_loaders:
                trainer.test(model, dataloaders=dataloader)
                metrics_dicts.append(trainer.logged_metrics)

            # =========================================================
            #  LOGGING
            # =========================================================
            for loader_idx, metrics in enumerate(metrics_dicts):
                metrics = {k: v.item() for k, v in metrics.items()}
                al_dict = {
                    "cycle": cycle,
                    "rep": repetition,
                    "budget": budget,
                    "labeled_items": len(al_dataset.labeled_indices),
                    "dataloader_idx" : loader_idx
                }
                metrics.update(al_dict)

                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step=global_step)    

                RANK = int(os.environ.get("RANK", 0))
                if RANK == 0:
                    write_dict_to_file(
                        config.log_dir,
                        metrics,
                        cycle,
                        repetition,
                        logger,
                        strat=config.strat,
                    )

            t2_end_train = time()
            global_step += 1

            logger.info(f"""
            Cycle {cycle + 1}/{cycles} ended. rep {repetition + 1} / {repetitions}
            {(t2_end_train - t0_cycle) / 60:.4f} minutes for the cycle
            {(t1_start_train - t1_score) / 60:4f} for scoring
            {(t2_end_train - t1_start_train) / 60:.4f} for training + val 
            {(t1_start_train - t1_score) / 60:4f} for scoring 
            {(t2_end_train - t1_start_train) / 60:.4f} for training + val
            {len(al_dataset.labeled_indices)=} {budget=} {round(len(al_dataset.labeled_indices)/len(al_dataset), 3)=}
            {len(al_dataset.unlabeled_indices)=}, {len(al_dataset)=}
            {subset=}
            """)

        # clear memory because of a cuda OOM error
        del model
        a = gc.collect()
        print(a)
        torch.cuda.empty_cache()
        logger.info(f"""
        Repetition : {repetition + 1} ended
        {(time() - t0_rep) / 60:.4f} minutes taken
        estimated time for full AL procedure {repetitions * (time() - t0_rep) / 60:.4f} minutes
        estimated time for full AL procedure {repetitions * (time() - t0_rep) / 60 / 60:.4f} hours
        """)
