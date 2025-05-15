import numpy as np
import torch
from sklearn.svm import OneClassSVM

from anomrl.algos.detectors.abstract_detector import AbstractDetector
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import AbstractDataModule


class OCSVMDetector(AbstractDetector):
    def __init__(self, model_kwargs, **kwargs):
        self.model = OneClassSVM(**model_kwargs)

    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        scores = []
        for i in range(len(batch)):
            x = batch.observations[i, 1:, :]
            score = -self.model.score_samples(x)  # type: ignore
            scores.append(score)
        return scores

    def fit(self, data_module: AbstractDataModule, **kwargs):
        X = [x.observations.flatten(0, 1) for x in data_module.train_dataloader()]
        X = torch.cat(X, dim=0).cpu().numpy()
        self.model.fit(X)

    def save(self, model_path: str):
        raise NotImplementedError

    def __repr__(self):
        return f"OCSVMdetector(kernel={self.model.kernel})"  # type: ignore
