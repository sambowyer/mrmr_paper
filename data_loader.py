import os
import joblib as jbl
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

from zarth_utils.config import Config
from data_utils import (
    HelmLite, OpenLLM, load_glue_predictions, load_continuous_cat_scores,
    parse_pass_at_k_dataset, dir_pass_at_k_v3_full, dir_pass_at_k_v3_open,
)


class DataLoader(ABC):
    """Abstract base class for data loading"""

    @abstractmethod
    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load dataset and return (scores, model_names, true_acc)

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Tuple containing:
                - scores: 2D array of model predictions (num_models, num_items)
                - model_names: List of model names
                - true_acc: 1D array of true accuracies (num_models,)
        """
        pass


class OpenLLMLoader(DataLoader):
    """Loader for OpenLLM leaderboard datasets"""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load OpenLLM dataset"""
        # Load scores from existing cache
        scores_path = os.path.join("data", "scores", f"{dataset_name}.jbl")
        scores = jbl.load(scores_path)

        # Get model names and true accuracies
        model_names_path = os.path.join("data", "scores", f"{dataset_name}_models.jbl")
        model_info = jbl.load(model_names_path)

        # model_info is a list of model names
        model_names = model_info

        # Calculate true accuracies - for OpenLLM, we need to compute from scores
        # scores shape: (num_models, num_items)
        true_acc = scores.mean(axis=1)

        return scores, model_names, true_acc


class HelmsLoader(DataLoader):
    """Loader for HELMS benchmark datasets"""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load HELMS dataset"""
        # Use existing HelmLite class
        helm = HelmLite(tasks=[dataset_name])
        helm.download_and_check()
        dataset = helm.get_datasets()

        # scores shape: (num_models, num_items)
        scores = np.array(dataset["acc"]).T
        model_names = helm.models
        true_acc = scores.mean(axis=1)

        return scores, model_names, true_acc


class GlueLoader(DataLoader):
    """Loader for GLUE benchmark datasets"""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load GLUE dataset"""
        # Load predictions and gold labels
        all_preds, gold_labels = load_glue_predictions(dataset_name)

        # Calculate accuracy for each model
        scores = np.zeros((all_preds.shape[0], all_preds.shape[1]))
        for m in range(all_preds.shape[0]):
            scores[m] = all_preds[m].argmax(-1) == gold_labels

        # For GLUE, we don't have model names in the same format
        model_names = [f"model_{i}" for i in range(scores.shape[0])]
        true_acc = scores.mean(axis=1)

        return scores, model_names, true_acc


class HelmsLoader(DataLoader):
    """Loader for HELMS benchmark datasets"""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load HELMS dataset"""
        # Use existing HelmLite class
        helm = HelmLite(tasks=[dataset_name])
        helm.download_and_check()
        dataset = helm.get_datasets()

        # scores shape: (num_models, num_items)
        scores = np.array(dataset["acc"]).T
        model_names = helm.models
        true_acc = scores.mean(axis=1)

        return scores, model_names, true_acc


class GlueLoader(DataLoader):
    """Loader for GLUE benchmark datasets"""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load GLUE dataset"""
        # Load predictions and gold labels
        all_preds, gold_labels = load_glue_predictions(dataset_name)

        # Calculate accuracy for each model
        scores = np.zeros((all_preds.shape[0], all_preds.shape[1]))
        for m in range(all_preds.shape[0]):
            scores[m] = all_preds[m].argmax(-1) == gold_labels

        # For GLUE, we don't have model names in the same format
        model_names = [f"model_{i}" for i in range(scores.shape[0])]
        true_acc = scores.mean(axis=1)

        return scores, model_names, true_acc


class ContinuousCATLoader(DataLoader):
    """Loader for continuous-cat evaluation datasets (continuous scores)."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        scores, model_names = load_continuous_cat_scores(dataset_name)
        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class PassAtKCodeLoader(DataLoader):
    """Loader for pass@k code evaluation datasets (continuous scores)."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        scores_path = os.path.join("data", "scores", f"{dataset_name}.jbl")
        scores = jbl.load(scores_path)
        model_names = jbl.load(
            os.path.join("data", "scores", f"{dataset_name}_models.jbl")
        )
        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class PassAtKCodeV2Loader(DataLoader):
    """Loader for v2 pass@k code datasets (all items kept, incomplete models dropped)."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        scores_path = os.path.join("data", "scores", f"{dataset_name}.jbl")
        scores = jbl.load(scores_path)
        model_names = jbl.load(
            os.path.join("data", "scores", f"{dataset_name}_models.jbl")
        )
        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class _PassAtKV3Loader(DataLoader):
    """Base loader for v3 pass@k code datasets (full or open)."""

    _scores_dir: str = ""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        file_name = dataset_name
        if file_name.endswith("_open"):
            file_name = file_name[: -len("_open")]

        scores = jbl.load(os.path.join(self._scores_dir, f"{file_name}.jbl"))

        parsed = parse_pass_at_k_dataset(dataset_name)
        if parsed is None:
            raise ValueError(f"Cannot parse benchmark from dataset name: {dataset_name}")
        benchmark = parsed[0]

        info = jbl.load(os.path.join(self._scores_dir, f"{benchmark}_info.jbl"))
        model_names = list(info["models"])

        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class PassAtKFullLoader(_PassAtKV3Loader):
    """Loader for v3 pass@k code datasets -- full (all models)."""
    _scores_dir = dir_pass_at_k_v3_full


class PassAtKOpenLoader(_PassAtKV3Loader):
    """Loader for v3 pass@k code datasets -- open-source models only."""
    _scores_dir = dir_pass_at_k_v3_open


class LoaderRegistry:
    """Registry for data loaders"""

    _loaders: Dict[str, DataLoader] = {
        "openllm": OpenLLMLoader(),
        "helm": HelmsLoader(),
        "glue": GlueLoader(),
        "continuous_cat": ContinuousCATLoader(),
        "pass_at_k_code": PassAtKCodeLoader(),
        "pass_at_k_code_v2": PassAtKCodeV2Loader(),
        "pass_at_k_full": PassAtKFullLoader(),
        "pass_at_k_open": PassAtKOpenLoader(),
    }

    @classmethod
    def get_loader(cls, source: str) -> DataLoader:
        """Get loader for specified data source"""
        loader = cls._loaders.get(source.lower())
        if loader is None:
            raise ValueError(
                f"Unknown data source: {source}. Available: {list(cls._loaders.keys())}"
            )
        return loader

    @classmethod
    def register_loader(cls, source: str, loader: DataLoader):
        """Register a new data loader"""
        cls._loaders[source.lower()] = loader
