from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
)
import torch
from typing import List, Dict, Optional
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os


class SubsetLinearSampler(torch.utils.data.Sampler):
    """Sample a dataset from given indices not randomly

    Args:
        Sampler (_type_): _description_
    """    
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



def test_strat(
    labeled: np.ndarray,
    unlabeled: np.ndarray,
    selected: np.ndarray,
    budget: int,
    full_names: np.ndarray,
) -> None:
    """check 3 conditions on selected items
        - budget items have been selected
        - selected indices or not in the labeled_indices
        - selected items are all coming from unlabeled indices

    Args:
        labeled (np.ndarray[int]): _description_
        unlabeled (np.ndarray[int]): _description_
        selected (np.ndarray[int]): _description_
        full_names (np.ndarray[str]): list of the names identifying each row of the complete dataset
        budget (int): _description_
    """
    # Check if we selected the good number of items
    assert len(np.unique(selected)) == len(selected), (
        f"strat should select only unique items but {len(np.unique(selected))} unique items have been selected"
    )
    # Check if we selected the good number of items
    assert len(np.unique(selected)) == budget, (
        f"selected {len(np.unique(selected))} unique items instead of {budget}"
    )
    # check if we selected items from the unlabeled dataset
    assert np.isin(selected, labeled).sum() == 0, (
        f"{np.isin(selected, labeled).sum()} labeled items have been selected"
    )
    # check if all the selected items are in the unlabeled dataset
    assert np.isin(selected, unlabeled).sum() == budget, (
        f"{budget - np.isin(selected, unlabeled).sum()} selected items"
    )

    labeled_names = pd.Series(full_names[labeled])
    selected_names = pd.Series(full_names[selected])
    unlabeled_names = pd.Series(full_names[unlabeled])

    assert labeled_names.is_unique, "labeled names should be unique"
    assert unlabeled_names.is_unique, "unlabeled names should be unique"
    assert selected_names.is_unique, "selected names should be unique"

    # check if all the selected items are in the unlabeled dataset
    assert selected_names.isin(unlabeled_names).sum() == budget, (
        "You didn't select all your items from the unlabeled dataset"
    )
    # check if we selected items from the labeled dataset
    assert selected_names.isin(labeled_names).sum() == 0, (
        "You selected names in the labeled names"
    )



class ActiveLearningDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        config: DictConfig,
        names: List[str],
    ):
        """
        Args:
            dataset (Dataset): The original dataset.
            config (DictConfig): Configuration object.
            names (List[str]): List of unique names/identifiers for each data point.
        """
        self.dataset: Dataset = dataset
        self.config = config

        self.names: np.ndarray = names

        self.pb = self.config.problem


        self.validate()
        self.start_rep()

    def start_rep(self):
        """Initializes the labeled and unlabeled indices."""
        self.labeled_indices = np.array([]).astype(int)
        self.unlabeled_indices = torch.randperm(len(self.dataset)).numpy().astype(int)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def unlabeled_subset(self) -> np.ndarray:
        if (self.config.subset == -1) or (
            self.config.subset >= len(self.unlabeled_indices)
        ):
            subset = self.unlabeled_indices
        else:
            subset = np.random.choice(
                self.unlabeled_indices, self.config.subset, replace=False
            )
        # return self.unlabeled_indices
        return subset

    def labeled_dataloader(self) -> DataLoader:
        """Return a dataloader that sample randomly from labeled indices

        Returns:
            _type_: Dataloader
        """
        # the drop_last can remove the first cycle if the labeled items are less thant the batch_size
        if len(self.labeled_indices) < self.config.batch_size:
            bs = len(self.labeled_indices)
        else:
            bs = self.config.batch_size

        print(f"{self.config.batch_size=}, {bs=}, {len(self.labeled_indices)=}")
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.labeled_indices),
            drop_last=True,
            batch_size=bs,
            num_workers=self.config.workers,
        )

    def full_dataloader(self) -> DataLoader:
        """Return a dataloader that samples sequentially from unlabeled indices

        Returns:
            _type_: DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=2 * self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
        )

    def unlabeled_dataloader(self) -> DataLoader:
        """Return a dataloader that samples sequentially from unlabeled indices

        Returns:
            _type_: DataLoader
        """
        return DataLoader(
            self.dataset,
            sampler=SubsetLinearSampler(self.unlabeled_subset),
            batch_size=2 * self.config.batch_size,
            num_workers=self.config.workers,
        )

    def save(self):
        pd.DataFrame({}, columns=[]).to_csv(self.config.log_dir + "")

    def label(
        self,
        selected_items: Dict[str, np.ndarray],
        budget: int,
        cycle: int = -1,
        rep: int = -1,
    ) -> None:
        """_summary_

        Args:
            selected_items (Dict[str,np.ndarray]): _description_
                - "ids": np.ndarray of selected in the global dataset
                - "scores": np.ndarray of scores for the selected indices
            cycle (int): _description_
            rep (int): _description_
            budget (int): _description_
        """
        # indices returned by strats are indexed in the unlabeled subset
        global_index_to_label = selected_items["ids"]
        test_strat(
            selected=global_index_to_label,
            labeled=self.labeled_indices,
            unlabeled=self.unlabeled_indices,
            budget=budget,
            full_names=self.names,
        )

        RANK = int(os.environ.get("RANK", 0))
        print(f"{RANK=}")
        if RANK == 0:
            self.log_selected(selected_items, cycle, rep)

        self.labeled_indices = np.concatenate(
            (self.labeled_indices, global_index_to_label)
        )
        self.unlabeled_indices = np.setdiff1d(
            self.unlabeled_indices, global_index_to_label
        )

    def log_selected(self, selected_items: Dict, cycle: int, rep: int) -> None:
        config = self.config

        selection_path = config.log_dir
        os.makedirs(selection_path, exist_ok=True)

        pd.DataFrame(
            {
                "repetition": rep,
                "cycle": cycle,
                **selected_items,
                "aldataset_name": self.names[selected_items["ids"]],
                "strat": config.strat,
            },
            index=np.arange(len(selected_items["ids"])),
        ).to_csv(
            os.path.join(
                selection_path,
                f"selected_items_{config.strat}_{cycle}_{rep}.csv",
            ),
            sep=";",
        )
        print(
            f"logged selected items to {os.path.join(selection_path, 'selected_items_')}{config.strat}_{cycle}_{rep}.csv"
        )

    def __repr__(self):
        return f"ActiveLearningDataset with {len(self.labeled_indices)} labeled items and {len(self.unlabeled_indices)} unlabeled items."

    def validate(self):
        # need this for the maskrcnn collate_func when constructing the dataloaders in instance seg
        assert "problem" in self.config, (
            f"problem key should appear in the config file, problem should be one of {['classif', 'odetection', 'iseg', 'sseg']}"
        )
        assert self.pb in ["classif", "odetection", "iseg", "sseg"], (
            f"{self.pb} is not a valid problem type, pb should be in {['classif', 'odetection', 'iseg', 'sseg']}"
        )

        assert len(self.names) == len(self.dataset)
        assert len(np.unique(self.names)) == len(self.names), (
            f"names should be unique: {len(self.names) - len(np.unique(self.names))}"
        )
        assert hasattr(self.config, "log_dir"), "config should have a log_dir attribute"
