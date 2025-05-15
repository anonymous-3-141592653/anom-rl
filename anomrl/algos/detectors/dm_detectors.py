import torch

from anomrl.algos.detectors.abstract_detector import LitModelDetector
from anomrl.algos.models.lit_models import (
    LitDynamicsModel,
    LitEnsembleDynamicsModel,
    LitProbDynamicsModel,
    LitProbEnsembleDynamicsModel,
    LSTMDynamicsModel,
)
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import EnsembleDataModule


class SimpleDMDetector(LitModelDetector):
    model_class = LitDynamicsModel

    def _init_model(self, observation_size, action_size, **model_kwargs):
        return self.model_class(observation_size=observation_size, action_size=action_size, **model_kwargs)

    def _predict_scores(self, batch: BatchEpisodeData):
        self.model.eval()
        self.model.to(self.device)
        batch = batch.to(self.device)
        targets = batch.observations[:, 1:, :]
        predictions = self.model.predict_step(batch)
        scores = self._score_fn(predictions, targets)
        return scores

    def _score_fn(self, predictions, targets):
        assert predictions.shape == targets.shape
        return torch.mean((predictions - targets) ** 2, dim=-1).cpu().numpy()


class LSTMDMDetector(SimpleDMDetector):
    model_class = LSTMDynamicsModel


class ProbDMDetector(SimpleDMDetector):
    model_class = LitProbDynamicsModel

    def _score_fn(self, predictions, targets):
        assert predictions.shape == (self.model.n_samples, *targets.shape)
        err = torch.mean((predictions - targets) ** 2, dim=-1).cpu().numpy()  # (n_samples, batch_size, seq_len)
        aggregate = err.min(0)
        return aggregate  # (batch_size, seq_len)


class EnsDMDetector(SimpleDMDetector):
    model_class = LitEnsembleDynamicsModel

    def fit(self, data_module, **kwargs):
        data_module = EnsembleDataModule(data_module, self.model.ens_size)
        super().fit(data_module, **kwargs)

    def _score_fn(self, predictions, targets):
        assert predictions.shape == (self.model.ens_size, *targets.shape)
        err = torch.mean((predictions - targets) ** 2, dim=-1).cpu().numpy()
        aggregate = err.min(0)
        return aggregate


class ProbEnsDMDetector(SimpleDMDetector):
    model_class = LitProbEnsembleDynamicsModel

    def fit(self, data_module, **kwargs):
        data_module = EnsembleDataModule(data_module, self.model.ens_size)
        super().fit(data_module, **kwargs)

    def _score_fn(self, predictions, targets):
        assert predictions.shape == (self.model.n_samples, self.model.ens_size, *targets.shape)
        err = torch.mean((predictions.flatten(0, 1) - targets) ** 2, dim=-1).cpu().numpy()
        aggregate = err.min(0)

        return aggregate
