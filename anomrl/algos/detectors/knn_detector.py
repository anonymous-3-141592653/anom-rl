import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models.resnet import ResNet, resnet18

from anomrl.algos.detectors.abstract_detector import AbstractDetector, LitModelDetector
from anomrl.algos.models.lit_autoencoders import LitAutoEncoder
from anomrl.data.data_utils import BatchEpisodeData
from anomrl.data.datamodules import AbstractDataModule


class KnnDetector(AbstractDetector):
    def __init__(self, model_kwargs, aggregation="none", **kwargs):
        self.model = NearestNeighbors(**model_kwargs)
        self.k = self.model.n_neighbors  # type: ignore
        self.aggregation = aggregation

    def _predict_scores(self, batch: BatchEpisodeData) -> list[np.ndarray]:
        scores = []
        for i in range(len(batch)):
            x = batch.observations[i, 1:, :]
            dist, _ = self.model.kneighbors(x, return_distance=True)  # type: ignore
            aggr = self.aggregate(dist)
            scores.append(aggr)
        return scores

    def aggregate(self, dists: np.ndarray):
        if self.aggregation == "none":
            return dists[:, self.k - 1 : self.k]
        elif self.aggregation == "mean":
            return dists.mean(-1, keepdims=True)
        elif self.aggregation == "max":
            return dists.max(-1, keepdims=True)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def fit(self, data_module: AbstractDataModule, **kwargs):
        X = [x.observations.flatten(0, 1) for x in data_module.train_dataloader()]
        X = torch.cat(X, dim=0).cpu().numpy()
        self.model.fit(X)

    def save(self, model_path: str):
        raise NotImplementedError

    def __repr__(self):
        return f"KnnDetector(n_neighbors={self.k})"


class KnnAEDetector(LitModelDetector):
    model_class = LitAutoEncoder

    def __init__(self, n_neighbors=1, **model_kwargs):
        super().__init__(**model_kwargs)
        self.k = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def _init_model(self, observation_size: torch.Size, action_size: torch.Size, **model_kwargs):
        return self.model_class(img_size=observation_size, **(model_kwargs or {}))

    def _predict_scores(self, batch: BatchEpisodeData):
        self.model.eval()
        self.model.to(self.device)
        batch = batch.to(self.device)
        z = self.model.encode(batch)
        z = z[:, :-1, :]  # remove last step
        dists, _ = self.knn.kneighbors(z.flatten(0, 1).cpu().numpy(), return_distance=True)
        dists = dists.reshape(len(batch), -1, self.k)
        scores = dists.mean(-1, keepdims=True)
        return scores

    def post_train_callback(self, data_module):
        encodings = []
        for batch in data_module.train_dataloader():
            batch = batch.to(self.model.device)
            z = self.model.encode(batch)
            z = z[:, :-1, :]  # remove last step
            encodings.append(z.flatten(0, 1).cpu().numpy())

        self.knn.fit(np.concatenate(encodings, axis=0))


class KnnResNetDetector(AbstractDetector):
    model_class = ResNet

    def __init__(self, n_neighbors, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.model = resnet18(weights="IMAGENET1K_V1")
        self.model.eval()
        self.model.to(self.device)
        self._prepocess = transforms.Compose(
            [transforms.Resize(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.activations: torch.Tensor

        def hook_fn(module, input, output):
            self.activations = output

        self.model.avgpool.register_forward_hook(hook_fn)

    def _predict_scores(self, batch: BatchEpisodeData):
        z = self._extract_features(batch)
        dists, _ = self.knn.kneighbors(z.flatten(0, 1).cpu().numpy(), return_distance=True)
        dists = dists.reshape(len(batch), -1, self.k)
        scores = dists.mean(-1, keepdims=True)
        return scores

    @torch.no_grad()
    def _extract_features(self, batch: BatchEpisodeData) -> torch.Tensor:
        x = batch.observations[:, 1:]
        x = x.flatten(0, 1)
        x = self._prepocess(x)
        x = x.to(self.device)
        self.model(x)
        activation = self.activations
        return activation.view(len(batch), -1, 512)

    def fit(self, data_module, **kwargs):
        pass

    def post_train_callback(self, data_module):
        latent_features = []
        for batch in data_module.train_dataloader():
            batch = batch.to(self.device)
            z = self._extract_features(batch)
            latent_features.append(z.flatten(0, 1).cpu().numpy())

        self.knn.fit(np.concatenate(latent_features, axis=0))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        model_repr = self.model.__repr__().replace("\n", "\n  ")
        return f"{class_name}(\n (model): {model_repr} \n)"

    def save(self, model_path: str):
        raise NotImplementedError
