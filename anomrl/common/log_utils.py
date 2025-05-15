import logging
import os
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.logger import KVWriter


class VideoBuffer:
    """
    Video data class storing the video frames and the frame per seconds

    :param frames: frames to create the video from
    :param fps: frames per second
    """

    def __init__(self, fps: float = 30, file_prefix: str = "video"):
        self._frames = []
        self.fps = fps
        self.file_prefix = file_prefix

    def add_frame(self, frame):
        self._frames.append(frame)

    def as_array(self):
        return np.array(self._frames)

    def as_tensor(self):
        return torch.ByteTensor(self.as_array())

    def write_video(self, filename: str, verbose: bool = False):
        self.filename = filename

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if verbose:
            print(f"writing video to: {filename}")
        torchvision.io.write_video(filename=filename, video_array=self.as_tensor(), fps=self.fps)

        self._empty_buffer()

    def _empty_buffer(self):
        self._frames = []

    def __len__(self):
        return len(self._frames)


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, str | tuple[str, ...]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)  # type: ignore

            if isinstance(value, VideoBuffer):
                with TemporaryDirectory() as temp_dir:
                    video_name = f"{value.file_prefix}_step_{step:04}.mp4" if step else f"{value.file_prefix}.mp4"
                    with open(os.path.join(temp_dir, video_name), "wb") as f:
                        value.write_video(f.name, verbose=False)
                        mlflow.log_artifact(f.name, "videos")

            if isinstance(value, plt.Figure):
                with TemporaryDirectory() as temp_dir:
                    name, i = key.split("_")
                    fig_name = f"{name}_{step:04}_{i}.png" if step else "figure.png"
                    with open(os.path.join(temp_dir, fig_name), "wb") as f:
                        value.savefig(f.name)
                        mlflow.log_artifact(f.name, "figures")

            if isinstance(value, DictConfig):
                # log params recursively, log dicts as parent_key.child_key...: value
                for k, v in value.items():
                    if isinstance(v, DictConfig):
                        mlflow.log_param(f"{k}", OmegaConf.to_yaml(v))
                    else:
                        mlflow.log_param(f"{k}", v)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s - %(funcName)s] --- %(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )
