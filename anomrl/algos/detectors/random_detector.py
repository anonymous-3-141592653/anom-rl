import numpy as np

from anomrl.data.data_utils import BatchEpisodeData


class RandomDetector:
    def __init__(self, **kwargs):
        pass

    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        all_scores = []
        batch_size, seq_len, _ = batch.actions.shape
        scores = np.random.uniform(size=(batch_size, seq_len))
        all_scores.append(scores)
        return all_scores

    def fit(self, data_module, **kwargs):
        pass
