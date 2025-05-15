import numpy as np
import torch

from anomrl.algos.detectors.abstract_detector import LitModelDetector
from anomrl.algos.models.latent_dynamics import LatentDynamicsModel
from anomrl.algos.models.lit_autoencoders import LitAutoEncoder, LitPredictiveAutoEncoder
from anomrl.data.data_utils import BatchEpisodeData


class AEDetector(LitModelDetector):
    model_class = LitAutoEncoder

    def _init_model(self, observation_size: torch.Size, action_size: torch.Size, **model_kwargs):
        return self.model_class(img_size=observation_size, **(model_kwargs or {}))

    def _predict_scores(self, batch: BatchEpisodeData):
        self.model.eval()
        self.model.to(self.device)
        batch = batch.to(self.device)
        pred = self.model.predict_step(batch)
        targets = batch.observations[:, :-1]
        scores = self._score_fn(pred, targets)
        return scores

    def _score_fn(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        assert predictions.shape == targets.shape
        return torch.mean((predictions - targets) ** 2, dim=(2, 3, 4)).cpu().numpy()


class PredAEDetector(AEDetector):
    model_class = LitPredictiveAutoEncoder

    def _predict_scores(self, batch: BatchEpisodeData):
        self.model.eval()
        self.model.to(self.device)
        batch = batch.to(self.device)
        pred = self.model.predict_step(batch)
        targets = batch.observations[:, 1:]
        scores = self._score_fn(pred, targets)
        return scores


class LatentDynamicsDetector(PredAEDetector):
    model_class = LatentDynamicsModel

    def _init_model(self, observation_size: torch.Size, action_size: torch.Size, **model_kwargs):
        return self.model_class(observation_size=observation_size, action_size=action_size, **(model_kwargs or {}))
