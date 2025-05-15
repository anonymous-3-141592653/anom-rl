import numpy as np
import torch

from anomrl.algos.detectors.abstract_detector import LitModelDetector
from anomrl.algos.models.prednet import LitPredNet
from anomrl.data.data_utils import BatchEpisodeData


class PredNetDetector(LitModelDetector):
    model_class = LitPredNet

    def _init_model(self, observation_size: torch.Size, action_size: torch.Size, **model_kwargs):
        return self.model_class()

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
