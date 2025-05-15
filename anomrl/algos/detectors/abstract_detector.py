import logging
from abc import ABC, abstractmethod
from os import PathLike

import lightning as L
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig

from anomrl.common.trainers import LitModelTrainer
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import AbstractDataModule

logger = logging.getLogger(__name__)


class AbstractDetector(ABC):
    """
    Base class for all detectors. A detector is a model that takes in a batch of episode data and predicts
    anomaly scores for each step in the episode
    """

    @abstractmethod
    def fit(self, data_module: L.LightningDataModule, trainer_cfg: DictConfig) -> None:
        """fit the detector on the given data_module. The datamodule should have a train and val dataloader
        each yielding batches of EpisodeData
        """
        ...

    @abstractmethod
    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]: ...

    def predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        """predict anomaly scores for batched episode data

        Returns:
            np.ndarray: anomaly scores for each step for each episode in the batch
        """
        scores = self._predict_scores(batch)
        assert len(scores) == len(batch)
        assert [(len(s), len(b)) for s, b in zip(scores, batch.actions, strict=True)]
        return scores

    @abstractmethod
    def save(self, model_path: str) -> None: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    def post_train_callback(self, data_module: L.LightningDataModule):
        """Hook to run after training is complete"""
        return


class LitModelDetector(AbstractDetector):
    """
    Base Class for all detectors that use a LightningModule as the model
    """

    model_class: type

    def __init__(
        self,
        model: L.LightningModule | None = None,
        model_path: None | PathLike = None,
        model_kwargs: None | dict = None,
        observation_size=None | tuple[int] | torch.Size,
        action_size=None | tuple[int] | torch.Size,
    ):
        if model is not None:
            assert isinstance(model, self.model_class), (
                f"model must be an instance of {self.model_class}, got {type(model)}"
            )
            self.model = model

        elif model_path is not None:
            self.model = self.model_class.load_from_checkpoint(model_path)

        else:
            assert observation_size is not None and action_size is not None
            self.model = self._init_model(observation_size, action_size, **model_kwargs or {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _init_model(self, observation_size, action_size, model_kwargs=None) -> L.LightningModule: ...

    def save(self, model_path: str):
        raise NotImplementedError

    def fit(self, data_module: AbstractDataModule, trainer_cfg: dict):
        self.trainer = LitModelTrainer(**trainer_cfg)._trainer
        self.trainer.fit(model=self.model, datamodule=data_module)

        if trainer_cfg.get("save_best_checkpoint"):
            checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
            if checkpoint_callback and hasattr(checkpoint_callback, "best_model_path"):
                best_model_path = checkpoint_callback.best_model_path
                logger.info(f"Loading best model from checkpoint: {best_model_path}")
                self.model = self.model_class.load_from_checkpoint(best_model_path)
                mlflow.log_metric("best_val_loss", checkpoint_callback.best_model_score)
            else:
                logger.warning("No valid checkpoint callback or best_model_path found.")

        else:
            logger.info("No checkpoint saved, using the current model")

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        model_repr = self.model.__repr__().replace("\n", "\n  ")
        return f"{class_name}(\n (model): {model_repr} \n)"
