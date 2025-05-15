import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from anomrl.algos.detectors.abstract_detector import AbstractDetector
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import AbstractDataModule


class IsoforestDetector(AbstractDetector):
    def __init__(self, n_estimators=100, **kwargs):
        self.n_estimators = n_estimators
        self.model = IsolationForest(n_jobs=4, bootstrap=True, n_estimators=self.n_estimators)

    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        scores = []
        for i in range(len(batch)):
            x = batch.observations[i, 1:, :]
            d = self.model.decision_function(x)
            scores.append(-d)
        return scores

    def fit(self, data_module: AbstractDataModule, **kwargs):
        X = [x.observations.flatten(0, 1) for x in data_module.train_dataloader()]
        X = torch.cat(X, dim=0).cpu().numpy()
        self.model.fit(X)

    def save(self, model_path: str):
        raise NotImplementedError

    def __repr__(self):
        return f"IsoforestDetector(n_estimators={self.n_estimators})"
