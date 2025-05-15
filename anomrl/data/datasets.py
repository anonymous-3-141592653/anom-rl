import os
import warnings
from abc import abstractmethod
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from gymnasium import spaces
from minari import MinariDataset
from minari.dataset.minari_dataset import MinariDatasetSpec
from torch.utils.data import Dataset
from torchvision import transforms

from anomrl.common.helpers import flatten_dict, sorted_alphanum
from anomrl.data.data_utils import BatchEpisodeData, EpisodeData, Normal, NpToTensorImg

warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")


class AbstractDataset(Dataset):
    observation_size: torch.Size
    action_size: torch.Size
    metadata: dict
    spec: MinariDatasetSpec

    def __init__(
        self,
        data_dir: str | os.PathLike,
        normalize=True,
        obs_normal: Normal | None = None,
        act_normal: Normal | None = None,
    ):
        self.normalize = normalize
        self.obs_normal = obs_normal
        self.act_normal = act_normal
        minari_dataset = MinariDataset(Path(data_dir) / "data")
        self.n_episodes = len(minari_dataset)
        self.spec = minari_dataset.spec
        self.metadata = minari_dataset.storage.metadata
        self.seed = self.metadata.get("seed", None)
        self._init_data(minari_dataset)

        self.observation_size = self.__getitem__(0).observations.shape[1:]
        self.action_size = self.__getitem__(0).actions.shape[1:]
        assert isinstance(minari_dataset.observation_space, spaces.Box | spaces.Dict), (
            f"only Box or Dict observation space is supported, got {type(minari_dataset.observation_space)}"
        )

        assert isinstance(minari_dataset.action_space, spaces.Box), (
            f"only Box action space is supported, got {type(minari_dataset.action_space)}"
        )

    @property
    def env_metadata(self):
        return {
            "env_id": self.spec.env_spec.id,
            "anomaly_type": self.metadata["anomaly_type"],
            "anomaly_strength": self.metadata["anomaly_strength"],
            "reward_avg": self.metadata["statistics"]["reward_avg"],
            "normalized_score": self.metadata["statistics"]["normalized_score"],
        }

    def __len__(self):
        return self.n_episodes

    def _compute_normal(self, data: list[np.ndarray]):
        assert len(data) > 0, "Data list is empty"
        assert isinstance(data[0], np.ndarray), "Data list must contain np.ndarrays"
        cat = np.concatenate([d for d in data], axis=0)
        mean = cat.mean(0)
        std = cat.std(0)
        return Normal(mean, std)

    @abstractmethod
    def _init_data(self, minari_dataset: MinariDataset) -> None:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> EpisodeData:
        """Return sample from the dataset at the given idx)
        Args:
            idx: index of the sample

        Returns: EpisodeData
        """


class VectorObsDataset(AbstractDataset):
    def _init_data(self, minari_dataset: MinariDataset):
        self._observations, self._actions, self._rewards, self._infos = [], [], [], []
        for ep in minari_dataset:
            if isinstance(ep.observations, dict):
                obs = flatten_dict(ep.observations)
            else:
                obs = ep.observations
            acs = np.atleast_2d(ep.actions)
            assert len(obs) - 1 == len(acs) == len(ep.rewards)

            self._observations.append(obs)
            self._actions.append(acs)
            self._rewards.append(ep.rewards)
            self._infos.append(ep.infos)

        if self.normalize:
            if self.obs_normal is None:
                self.obs_normal = self._compute_normal(self._observations)
            if self.act_normal is None:
                self.act_normal = self._compute_normal(self._actions)

    def __getitem__(self, idx: int) -> EpisodeData:
        observations = self._observations[idx]
        actions = self._actions[idx]
        if self.normalize:
            observations = (observations - self.obs_normal.mean) / (self.obs_normal.std + 1e-6)
            actions = (actions - self.act_normal.mean) / (self.act_normal.std + 1e-6)
        return EpisodeData(observations, actions, rewards=self._rewards[idx], infos=self._infos[idx])


class EnsembleDataset(Dataset):
    def __init__(self, dataset, ens_size):
        self._dataset = dataset
        self.ens_size = ens_size
        self.ens_ids = np.random.randint(0, len(self._dataset), (5, len(self._dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self._dataset, name, None)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            items = [self._dataset[self.ens_ids[e][idx]] for e in range(self.ens_size)]
            return BatchEpisodeData(items)
        else:
            raise NotImplementedError


class ImgObsDataset(AbstractDataset):
    def __init__(self, data_dir, normalize=True, **kwargs):
        if normalize:
            self.observation_transforms = transforms.Compose([NpToTensorImg(), transforms.Normalize((0.5), (0.5))])

        video_dir = Path(data_dir) / "videos"
        self.video_paths = sorted_alphanum([str(f) for f in video_dir.iterdir() if f.is_file()])
        assert all([f"_{i}_" in os.path.basename(p) for i, p in enumerate(self.video_paths)])
        super().__init__(data_dir, normalize, **kwargs)
        assert len(self.video_paths) == len(self)

    def _init_data(self, minari_dataset: MinariDataset):
        self._actions, self._rewards, self._infos = [], [], []
        for ep in minari_dataset:
            self._actions.append(np.atleast_2d(ep.actions))
            self._rewards.append(ep.rewards)
            self._infos.append(ep.infos)

        if self.normalize and self.act_normal is None:
            self.obs_normal = None
            self.act_normal = self._compute_normal(self._actions)

    def _load_video(self, path):
        frames = iio.imread(path, plugin="pyav")  # Reads the entire video
        return torch.tensor(frames)  # (num_frames, height, width, channels)

    def __getitem__(self, idx: int) -> EpisodeData:
        observations = self._load_video(self.video_paths[idx])
        actions = self._actions[idx]
        rewards = self._rewards[idx]
        assert len(observations) == len(actions) + 1
        if self.normalize:
            observations = self.observation_transforms(observations)
            actions = (actions - self.act_normal.mean) / (self.act_normal.std + 1e-6)

        return EpisodeData(observations, actions, rewards, self._infos[idx])
