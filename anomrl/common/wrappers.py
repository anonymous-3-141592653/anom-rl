import os
from collections import Counter
from datetime import datetime
from typing import Any

import gymnasium
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv

from anomrl.common.log_utils import VideoBuffer


class RecordVideo(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        video_dir: str = "",
        video_format: str = ".mp4",
        name_prefix=None,
        write_on_ep_end: bool = False,
    ):
        super().__init__(env)
        self.video_dir = video_dir
        self.video_format = video_format
        self.write_on_ep_end = write_on_ep_end
        self.ep_ctr = 0
        if name_prefix is None:
            self.name_prefix = "episode"
        elif name_prefix == "auto":
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            maybe_anomaly = getattr(self, "anomaly_type", "default")
            self.name_prefix = f"{self.env.spec.id}_{maybe_anomaly}_{now}"
        else:
            self.name_prefix = name_prefix
        self._init_buffer()

    def _init_buffer(self):
        if isinstance(self.env, VecEnv):
            _env = self.env.envs[0]
        else:
            _env = self.env
        self.video_buffer = VideoBuffer(fps=_env.metadata["render_fps"], file_prefix=self.name_prefix)

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)
        frame = self.render()
        self.video_buffer.add_frame(frame)

        if self.write_on_ep_end and (term or trunc):
            self.write_video()
            self._init_buffer()
            self.ep_ctr += 1

        return observation, reward, term, trunc, info

    def render(self):
        return self.env.render()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = self.render()
        self.video_buffer.add_frame(frame)
        return observation, info

    def write_video(self, filename: str = ""):
        if filename == "":
            filename = os.path.join(self.video_dir, f"{self.name_prefix}_{self.ep_ctr}_{self.video_format}")

        assert len(self.video_buffer) > 0, "no frames in buffer"

        self.video_buffer.write_video(filename)

    def close(self):
        if len(self.video_buffer) > 0:
            self.write_video()
        return super().close()


class RecordStatistics(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._init_buffer()

    def _init_buffer(self):
        self._data_buffer = {
            "termination_causes": [],
            "terminated": [],
            "truncated": [],
            "is_success": [],
            "is_critical": [],
            "cum_reward": [],
            "cum_cost": [],
            "episode_length": [],
        }

    def _flush_episode_data(self, info, term, trunc):
        self._data_buffer["termination_causes"].append(info.get("term_cause", "None"))
        self._data_buffer["terminated"].append(term)
        self._data_buffer["truncated"].append(trunc)
        self._data_buffer["is_success"].append(info.get("is_success", "None"))
        self._data_buffer["is_critical"].append(info.get("is_critical", "None"))
        self._data_buffer["cum_reward"].append(self.cum_reward)
        self._data_buffer["cum_cost"].append(self.cum_cost)
        self._data_buffer["episode_length"].append(self.ep_length)

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)

        self.cum_reward += float(reward)
        self.cum_cost += float(info.get("cost", 0))
        self.ep_length += 1

        if term or trunc:
            self._flush_episode_data(info, term, trunc)

        return observation, reward, term, trunc, info

    def reset(self, seed=None, **kwargs):
        observation, info = self.env.reset(seed=seed, **kwargs)
        self.cum_reward = 0.0
        self.cum_cost = 0.0
        self.ep_length = 0
        return observation, info

    def get_statistics(self, as_df=False):
        statistics = {
            "termination_causes": dict(Counter(self._data_buffer["termination_causes"])),
            "n_episodes": len(self._data_buffer["terminated"]),
            "n_steps": sum(self._data_buffer["episode_length"]),
            "n_terminated": sum(self._data_buffer["terminated"]),
            "n_truncated": sum(self._data_buffer["truncated"]),
            "reward_avg": np.mean(self._data_buffer["cum_reward"]),
            "reward_std": np.std(self._data_buffer["cum_reward"]),
            "cost_avg": np.mean(self._data_buffer["cum_cost"]),
            "cost_std": np.std(self._data_buffer["cum_cost"]),
            "episode_length_avg": np.mean(self._data_buffer["episode_length"]),
            "episode_length_std": np.std(self._data_buffer["episode_length"]),
            "success_rate": self._data_buffer["is_success"].count(True) / len(self._data_buffer["is_success"]),
            "critical_rate": self._data_buffer["is_critical"].count(True) / len(self._data_buffer["is_success"]),
        }

        for k in statistics:
            if isinstance(statistics[k], float):
                statistics[k] = round(statistics[k], 3)

        if as_df:
            return pd.DataFrame(statistics, index=[self.spec.id])

        else:
            return statistics


class RecordImageObs(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        assert self.render_mode == "rgb_array", "Render mode must be 'rgb_array'"

    def _get_img_obs(self):
        return self.render()

    def step(self, action: Any) -> tuple:
        obs, rew, term, trunc, info = self.env.step(action)
        info["img_obs"] = self._get_img_obs()
        return obs, rew, term, trunc, info

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        info["img_obs"] = self._get_img_obs()
        return obs, info


class PersistentInfos(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

    def _get_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "is_success": bool(info.get("is_success", False)),
            "is_critical": bool(info.get("is_critical", False)),
            "cost": float(info.get("cost", 0.0)),
            "is_anomaly": bool(info.get("is_anomaly", False)),
        }

    def step(self, action: Any) -> tuple:
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, rew, term, trunc, self._get_info(info)

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        return obs, self._get_info(info)
