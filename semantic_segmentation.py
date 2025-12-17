import torch
from tqdm.auto import tqdm
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from torch import Tensor
from dataclasses import dataclass
from abc import ABC, abstractmethod
from logging import Logger
import math
import os


def safe_put_on_device(batch, device: Union[torch.device, str]):
    """Put a batch of device on target device

    Args:
        batch (_type_): _description_
        device (torch.device): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [safe_put_on_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(safe_put_on_device(v, device) for v in batch)
    else:
        return batch


def add_complement_to_one(tensor, dim) -> torch.Tensor:
    """Complement to one in case of binary tasks

    Args:
        tensor (_type_): _description_
        dim (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert tensor.shape[dim] == 1, (
        f"we can only complement a binary mask, got {tensor.shape=}, complement on {dim=}, it should be one"
    )
    assert tensor.max() <= 1, "shoudl be a probability mask"
    complement = 1 - tensor
    res = torch.cat((tensor, complement), dim=dim)
    return res


def torch_entropy(pk: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute the entropy of tensor alog specific dimension

    Args:
        pk (torch.Tensor): tensor ex for image [B, K, H, W] where K is the number of classes.
        dim (int, optional): _description_. Defaults to 1.

    Returns:
        torch.Tensor: _description_ ex [B, 1, H, W] where the entropy is computed along the class dimension.

    Compute the entropy along a specific dimension.

    If a binary mask with probability in [0,1] is given, a complement to 1 matrix is concatenated to create a probabilistic tensor.
    """
    if pk.nelement() == 0:
        # Tensor subseting with rounding of floats values can be empty
        return torch.zeros_like(pk).sum(dim=dim, keepdim=True).to(pk.device).float()

    if pk.ndim == 1:
        # unidimensional case, add a dimension to store the complement
        pk = pk.unsqueeze(0)
        dim = 0

    if pk.shape[dim] == 1:
        # if the tensor is a binary mask we complement to one to get a probability vector
        pk = add_complement_to_one(pk, dim=dim)

    assert math.isclose(pk.sum(dim=dim).mean(), 1, rel_tol=1 / 1000), (
        f"Should provide probabilities for each class on dimension 1 (channel dimension), {pk.sum(dim=dim).mean()}"
    )
    return -(pk * pk.log()).nansum(
        dim, keepdim=True
    )  # nansum because for determinist cases (null entropy) it returns none


@torch.no_grad()
def get_representation(
    model,  # Cant type hint because of circular import
    D: torch.utils.data.Dataset,  # al dataset
    config: Dict,
    logger,
) -> np.ndarray:
    """_summary_

    Args:
        model (_type_): _description_
        D (torch.utils.data.Dataset): _description_
        config (_type_): _description_
        logger (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loader = D.full_dataloader()  # ty:ignore

    features: List[float] = []
    print(f"len(dataLoader): {len(loader)}")
    for i, batch in enumerate(tqdm(loader, desc="Extracting Representations...")):
        x, _, _ = model.prepare_batch(batch)
        with torch.no_grad():
            x = safe_put_on_device(x, device)
            x = x
            temp_z = model.forward_embeddings(x)
            features.append(temp_z.cpu().numpy())

    features: np.ndarray = np.concatenate(features, axis=0)

    logger.info(f"Embeddings matrix shape {features.shape=}")
    assert features.shape[0] == len(D), (
        f"Features shape {features.shape}, but dataset length is {len(D)}"
    )

    return torch.nn.functional.normalize(torch.from_numpy(features), p=2, dim=1).numpy()


@dataclass
class baseStrat(ABC):
    """Base class for all AL strategies

    Args:
        ABC (_type_): abstract class that can't be used as is.

    Returns:
        _type_: _description_
    """

    base_dataset: torch.utils.data.Dataset
    val_loader: torch.utils.data.DataLoader
    model: torch.nn.Module
    config: Dict
    logger: Logger
    budget: int
    save_freq: int = -1
    precomputed_features: Optional[str] = (
        None  # path to a numpy file with embeddings for the full dataset
    )

    def __post_init__(self):
        self.repetitions = self.config.repetitions
        self.labeled_indices = self.base_dataset.labeled_indices
        self.unlabeled_indices = self.base_dataset.unlabeled_indices
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    @torch.no_grad()
    def select(self, unlabeled_subset: np.ndarray) -> Dict:
        selected: Dict = self._select(unlabeled_subset)

        # ensure all values have the same size
        sizes = [len(v) for v in selected.values()]
        assert all([s == sizes[0] for s in sizes]), (
            f"Selected dict values have different sizes: {sizes}"
        )

        return selected

    @abstractmethod
    @torch.no_grad()
    def _select(self, unlabeled_subset: np.ndarray) -> Dict:
        """Score the nulabeled subset of an Array.

        Args:
            unlabeled_subset (np.ndarray):

        Returns:
            Dict: with the following keys:
                - "ids": np.ndarray of selected indices [0, |Du|-1]
                - "scores": corresponding scores

        """
        pass

    @torch.no_grad()
    def compute_embeddings_on_subset(self) -> np.ndarray:
        dataset = self.base_dataset

        # return compute_embeddings(
        return get_representation(
            model=self.model,
            D=dataset,
            config=self.config,
            logger=self.logger,
        )

    @property
    def feature_matrix(self) -> np.ndarray:
        """Load or compute the feature matrix for the full dataset.

        uses the self.precomputed_features to create the matrix

        Returns:
            np.ndarray: _description_
        """
        if (self.precomputed_features is not None) and (
            isinstance(self.precomputed_features, str)
        ):
            self.logger.info(
                f"Loading precomputed features from path: {self.precomputed_features}"
            )
            assert os.path.exists(self.precomputed_features), (
                f"Precomputed features path {self.precomputed_features} does not exist"
            )
            matrix = np.load(self.precomputed_features)

        elif isinstance(self.precomputed_features, np.ndarray):
            self.logger.info("Using precomputed features from numpy array")
            matrix = self.precomputed_features
        else:
            matrix = get_representation(
                model=self.model,
                D=self.base_dataset,
                config=self.config,
                logger=self.logger,
            )

        assert matrix.shape[0] == len(self.base_dataset), (
            f"Precomputed features shape {matrix.shape[0]} does not match dataset length {len(self.base_dataset)}"
        )
        return matrix

    def convert_instance(self, child_class):
        """
        Create a new instance of the given child class with the same attributes as the current instance.

        Args:
            child_class (type): The child class to create an instance of.

        Returns:
            baseStrat: A new instance of the child class.
        """
        new_instance = child_class(
            base_dataset=self.base_dataset,
            val_loader=self.val_loader,
            model=self.model,
            config=self.config,
            logger=self.logger,
            budget=self.budget,
            precomputed_features=self.precomputed_features,
        )
        return new_instance


@dataclass
class randomSampling(baseStrat):
    name: str = "random"

    def _select(self, unlabeled_subset):
        selected = np.random.choice(unlabeled_subset, self.budget, replace=False)
        return {
            "ids": selected.astype(int),
            "scores": np.random.rand(self.budget).astype(np.float32),
        }


@dataclass
class EntropySampling(baseStrat):
    """Construct strats that iterate over a dataset and score each outputs.
    Children classes needs to re implement the score output function

    Args:
        baseStrat (_type_): _description_
    """

    name: str = "entropy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _select(self, unlabeled_subset) -> Dict:
        """Select unlabaled items based on any score provided by the score_outputs method.

        Args:
            unlabeled_subset (List): _description_

        Returns:
            Dict: dict with following keys
                - ids: indices of selected items in the full dataset
                - scores: scores for those items
                - detected_classes: np.ndarray of detected classes for each selected ids
                - names: np.ndarray of names for each selected ids (not used)
        """

        unlab_loader = self.base_dataset.unlabeled_dataloader()

        scores = []
        detected_classes = []
        names = []

        d_model = self.model.to(self.device)
        d_model.eval()

        i = 0
        pbar = tqdm(
            unlab_loader, desc=f"Scoring unlabeled dataloader using {self.name}"
        )
        for ii, batchs in enumerate(pbar):
            imgs, target, name = d_model.prepare_batch(batchs)
            bs = len(imgs)
            imgs = self.put_data_on_device(imgs)
            outputs = d_model.forward_proba(imgs)

            scores.extend(self.score_outputs(outputs).float())
            detected_classes.extend(self.get_detected_classes(outputs))
            names.extend(name)
            pbar.set_postfix(
                {"Batch mean score": torch.tensor(scores[i : i + bs]).mean().item()}
            )
            i += bs

        scores = torch.tensor(scores)

        self.logger.info(
            f"{(scores.isnan()).sum()=} {scores.mean()=} {scores.std()=} {scores.min()=} {scores.max()=} {(scores==0).sum()=} {(scores==-1).sum()=}"
        )

        topscores = torch.topk(scores, self.budget)
        best_scores_indices = topscores.indices
        best_scores = topscores.values

        return {
            "du_ids": best_scores_indices.cpu().numpy(),
            "ids": self.base_dataset.unlabeled_subset[
                best_scores_indices.cpu().numpy()
            ],
            "scores": best_scores.cpu().numpy(),
            "detected_classes": np.array(detected_classes)[
                best_scores_indices.cpu().numpy()
            ],
            "names": np.array(names)[best_scores_indices],
        }

    def score_outputs(self, outputs) -> torch.Tensor:
        """_summary_

        Args:
            outputs (_type_): _description_

        Returns:
            torch.Tensor: Should return a tensor containing one element per image.
        """
        return torch_entropy(outputs, dim=1).nanmean(dim=[1, 2, 3])

    def put_data_on_device(self, images) -> List:
        return images.to(self.device)

    def get_detected_classes(self, outputs) -> List[str]:
        return [
            "|".join(output.unique().cpu().numpy().astype(str).tolist())
            for output in outputs.argmax(1)
        ]


def furthest_first(
    unlabeled_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    device,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray]:
    unlabeled_embeddings: Tensor = torch.from_numpy(unlabeled_embeddings).to(device)
    labeled_embeddings: Tensor = torch.from_numpy(labeled_embeddings).to(device)

    m: int = unlabeled_embeddings.shape[0]

    if labeled_embeddings.shape[0] == 0:
        min_dist = torch.tile(torch.tensor(float("inf")), (m,)).to(device)
    else:
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0]

    idxs: np.ndarray = np.zeros(budget).astype(int)
    distances: np.ndarray = np.zeros(budget)

    i: int = 0
    for i in tqdm(range(budget), desc="Finding a core-set of items."):
        idx = torch.argmax(min_dist)
        idxs[i] = idx.item()
        dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx], :])
        min_dist = torch.minimum(min_dist, dist_new_ctr[:, 0])
        distances[i] = min_dist[idx].item()

    return idxs, distances


class deepcoresetSampling(baseStrat):
    @torch.no_grad()
    def _select(self, unlabeled_subset):
        matrix = self.feature_matrix

        selected, distances = furthest_first(
            labeled_embeddings=matrix[self.labeled_indices, :],
            unlabeled_embeddings=matrix[unlabeled_subset, :],
            budget=self.budget,
            device=self.device,
        )
        # selected indexed on Du
        selected = unlabeled_subset[selected]

        self.logger.info(
            f"{distances.mean()=}  {distances.std()=} {distances.min()=} {distances.max()=} {(distances==0).sum()=}{(np.isnan(distances)).sum()=}"
        )

        return {"ids": selected, "scores": distances}


AVAILABLE_SCORING_FUNCTIONS = {
    "randomSampling": randomSampling,
    "entropySampling": EntropySampling,
    "deepcoresetSampling": deepcoresetSampling,
}
