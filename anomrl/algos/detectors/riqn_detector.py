import numpy as np
import torch

from anomrl.algos.detectors.abstract_detector import AbstractDetector
from anomrl.algos.models.riqn import RIQN_Predictor
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import AbstractDataModule


class RIQN_Detector(AbstractDetector):
    def __init__(self, model_kwargs=None, *args, **kwargs):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = RIQN_Predictor(input_features=kwargs["observation_size"][0], device=device, **model_kwargs or {})

    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        scores = []
        for i in range(len(batch)):
            x = batch.observations[i]
            s = self.model.predict_episode(x)
            scores.append(s)

        return scores

    def fit(self, data_module: AbstractDataModule, trainer_cfg: dict):
        X = torch.nn.utils.rnn.pad_sequence(
            [b.observations.flatten(0, 1) for b in data_module.train_dataloader()], batch_first=True
        )
        self.model.fit(train_ep_obs=X.numpy(), **trainer_cfg)

    def save(self, model_path: str):
        pass

    def __repr__(self):
        pass
