import os
from collections import defaultdict
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing_extensions import override

from anomrl.common.plot_utils import plot_episode
from anomrl.common.wrappers import RecordVideo
from anomrl.data.data_utils import collect_rollouts


class RecordEvalCallback(BaseCallback):
    """
    based on: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-videos
    """

    def __init__(self, eval_env, freq: int, n_episodes: int = 1, record_video: bool = True, record_plots: bool = True):
        """
        Records videos and plots of an agent's trajectory
        """
        super().__init__()
        self._env = eval_env
        self._freq = freq
        self._n_episodes = n_episodes
        self._record_plots = record_plots
        self._record_video = record_video
        if record_video:
            self._env = RecordVideo(eval_env)

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.num_timesteps % self._freq == 0:
            episodes = collect_rollouts(self._env, self.model, self._n_episodes, render=False)

            if self._record_video:
                self.logger.record(
                    "trajectory/video",
                    self._env.video_buffer,
                    exclude=("stdout", "log", "json", "csv"),
                )
            if self._record_plots:
                for i, episode in enumerate(episodes):
                    if episode is not None:
                        obs_plot = plot_episode(episode.observations)
                        self.logger.record(f"plt-observations_{i}", obs_plot, exclude=("stdout", "log", "json", "csv"))
                        act_plot = plot_episode(episode.actions)
                        self.logger.record(f"plt-actions_{i}", act_plot, exclude=("stdout", "log", "json", "csv"))
                        rew_plot = plot_episode(episode.rewards)
                        self.logger.record(f"plt-rewards_{i}", rew_plot, exclude=("stdout", "log", "json", "csv"))

            self.logger.dump(self.num_timesteps)
        return True


class MLflowEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        if self.n_calls == 1:
            self.eval_freq = self.eval_freq / self.training_env.num_envs

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            with TemporaryDirectory() as temp_dir:
                self.best_model_save_path = temp_dir
                model_path = os.path.join(temp_dir, "best_model.zip")
                ret = super()._on_step()
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path, "models")
            return ret
        else:
            return True


class RecordInfoCallback(BaseCallback):
    """Callback that records the info dicts of a certain key."""

    def __init__(self, log_keys: dict[str], *args, **kwargs):
        """Initialize the callback class."""
        super().__init__(*args, **kwargs)
        self.log_keys = log_keys
        self.info_buffer = defaultdict(list)

    @override
    def _on_step(self):
        """Save in the buffer the info of the episode and record it if done."""
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=False):
            for key in self.log_keys:
                self.info_buffer[key].append(info[key])
            if done:
                self.logger.record(key, info[key])
                self.info_buffer[key].clear()
        return True
