import logging
from abc import abstractmethod
from collections.abc import Generator
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from anomrl.data.data_utils import pad_collate
from anomrl.data.datasets import AbstractDataset, EnsembleDataset, ImgObsDataset, VectorObsDataset

logger = logging.getLogger(__name__)


class AbstractDataModule(L.LightningDataModule):
    def __init__(
        self,
        env_id: str,
        batch_size=1,
        data_dir: str | Path | None = None,
        data_loader_workers=1,
        normalize=True,
    ):
        super().__init__()
        self.env_id = env_id
        self.batch_size = batch_size
        data_dir = data_dir
        assert data_dir is not None, "dataset_root must be provided or set as an environment variable: data_dir"
        self.data_dir = Path(data_dir) / env_id
        self._splits = self._find_all_splits()
        self.data_loader_workers = data_loader_workers
        self._normalize = normalize
        self._obs_normal = None
        self._act_normal = None
        if self._normalize:
            self._obs_normal = self.train_set.obs_normal
            self._act_normal = self.train_set.act_normal

    def _find_all_splits(self) -> dict[str, Path]:
        splits = {}
        for p in self.data_dir.iterdir():
            if p.name in splits:
                raise ValueError(f"Duplicate split found: {p.name}")

            if p.is_dir():
                splits[p.name] = p
        return splits

    def get_split(self, split_name: str) -> AbstractDataset:
        if split_name not in self._splits:
            raise ValueError(f"Split {split_name} not found in {self.data_dir}")
        return self._init_dataset(self._splits[split_name])

    @abstractmethod
    def _init_dataset(self, _id) -> AbstractDataset: ...

    @property
    def observation_size(self):
        return self.train_set.observation_size

    @property
    def action_size(self):
        return self.train_set.action_size

    @property
    def train_set(self) -> AbstractDataset:
        if not hasattr(self, "_train_set"):
            self.setup("fit")
        return self._train_set

    @property
    def val_set(self) -> AbstractDataset:
        if not hasattr(self, "_val_set"):
            self.setup("fit")
        return self._val_set

    @property
    def test_set(self) -> AbstractDataset:
        if not hasattr(self, "_test_set"):
            self.setup("test")
        return self._test_set

    def collate_fn(self, batch):
        return pad_collate(batch)

    def iterate_anomaly_sets(
        self, anomaly_strengths: list[str] | None = None, anomaly_types: list[str] | None = None
    ) -> Generator[AbstractDataset]:
        anomaly_splits = {k for k in self._splits if k not in ["train", "val", "test"]}
        if anomaly_strengths is not None:
            anomaly_splits = {k for k in anomaly_splits if any(s in k for s in anomaly_strengths)}
        if anomaly_types is not None:
            anomaly_splits = {k for k in anomaly_splits if any(t in k for t in anomaly_types)}

        for split in anomaly_splits:
            dataset = self.get_split(split)
            if dataset is not None:
                yield dataset

    def setup(self, stage="fit") -> None:
        if stage == "fit":
            if not hasattr(self, "_train_set"):
                self._train_set = self.get_split("train")
            if not hasattr(self, "_val_set"):
                self._val_set = self.get_split("val")

        elif stage == "test":
            if not hasattr(self, "_test_set"):
                self._test_set = self.get_split("test")

        elif stage == "predict":
            pass

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _dataloader(self, dataset, batch_size=None, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
        )

    def train_dataloader(self, batch_size=None) -> DataLoader[AbstractDataset]:
        return self._dataloader(self.train_set, shuffle=True, batch_size=batch_size or self.batch_size)

    def val_dataloader(self, batch_size=None) -> DataLoader[AbstractDataset]:
        return self._dataloader(self.val_set, batch_size=batch_size or self.batch_size)

    def test_dataloader(self, batch_size=None) -> DataLoader[AbstractDataset]:
        return self._dataloader(self.test_set, batch_size=batch_size or self.batch_size)

    def anomaly_dataloaders(
        self, batch_size=None, anomaly_strengths: list[str] | None = None, anomaly_types: list[str] | None = None
    ) -> Generator[DataLoader[AbstractDataset]]:
        for dataset in self.iterate_anomaly_sets(anomaly_strengths=anomaly_strengths, anomaly_types=anomaly_types):
            yield self._dataloader(dataset, batch_size=batch_size or self.batch_size)


class VectorObsDataModule(AbstractDataModule):
    def _init_dataset(self, data_path):
        dataset = VectorObsDataset(
            data_path, normalize=self._normalize, obs_normal=self._obs_normal, act_normal=self._act_normal
        )
        return dataset


class ImgObsDataModule(AbstractDataModule):
    def _init_dataset(self, data_path):
        dataset = ImgObsDataset(data_path, normalize=self._normalize, act_normal=self._act_normal)
        return dataset


class EnsembleDataModule(VectorObsDataModule):
    def __init__(self, data_module: VectorObsDataModule, ens_size=5):
        self._data_module = data_module
        self._data_module.setup("fit")
        self.ens_size = ens_size
        self._data_module._train_set = EnsembleDataset(data_module.train_set, self.ens_size)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self._data_module, name, None)
