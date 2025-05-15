import os
import warnings

import minari
import numpy as np
import torch
import tqdm
from anomaly_gym.common.metrics import get_normalized_score
from minari import MinariDataset
from minari.utils import _generate_dataset_metadata

from anomrl.common.helpers import atleast_2d, can_cast_to_float_array, flatten_dict, group_dict
from anomrl.common.wrappers import PersistentInfos, RecordStatistics, RecordVideo

warnings.filterwarnings("ignore", ".*is set to None. *")


class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class EpisodeData:
    """Contains a sequence/episode of n transitions, consisting of n+1 observations and n actions"""

    observations: torch.Tensor  # (ep_len+1, obs_dims...)
    actions: torch.Tensor  # (ep_len, act_dims...)
    rewards: torch.Tensor  # (ep_len,)
    infos: dict[str, torch.Tensor] | dict[str, list[str]]

    def __init__(
        self,
        observations: np.ndarray | torch.Tensor | list,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        infos: dict,
    ):
        if isinstance(observations[0], dict):
            observations = flatten_dict(group_dict(observations))

        self.observations = atleast_2d(torch.as_tensor(np.asarray(observations), dtype=torch.float32))
        self.actions = atleast_2d(torch.as_tensor(np.asarray(actions), dtype=torch.float32))
        self.rewards = atleast_2d(torch.as_tensor(np.asarray(rewards), dtype=torch.float32))
        infos = group_dict(infos) if isinstance(infos, list) else infos
        for k, v in infos.items():
            if can_cast_to_float_array(v):
                infos[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32)

        self.infos = infos

    def __len__(self):
        return self.actions.shape[0] if len(self.actions.shape) == 3 else self.actions.shape[1]

    def __repr__(self) -> str:
        return (
            f"EpisodeData(\n"
            f"  observations:{self.observations.shape}, \n"
            f"  actions:{self.actions.shape}, \n"
            f"  reward:{self.rewards.sum():.2f}, \n"
            f"  infos:{self.infos.keys()} \n"
            "\n)"
        )

    def __iter__(self):
        yield from (
            self.observations,
            self.actions,
            self.rewards,
            self.infos,
        )


class BatchEpisodeData:
    """Contains a batch of episodes"""

    def __init__(self, episodes: list[EpisodeData]):
        self.observations = torch.nn.utils.rnn.pad_sequence([ep.observations for ep in episodes], batch_first=True)
        self.actions = torch.nn.utils.rnn.pad_sequence([ep.actions for ep in episodes], batch_first=True)
        self.rewards = [ep.rewards for ep in episodes]
        self.infos = group_dict([ep.infos for ep in episodes])

    def __len__(self):
        return self.actions.shape[0]

    def __repr__(self) -> str:
        return (
            f"BatchEpisodeData("
            f"observations:{self.observations.shape},"
            f"actions:{self.actions.shape},"
            f"rewards:{self.rewards},"
            ")"
        )

    def to(self, device):
        """Move all tensors to the specified device"""
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)

        return self

    def __iter__(self):
        yield from (
            self.observations,
            self.actions,
            self.rewards,
            self.infos,
        )


class NpToTensorImg(torch.nn.Module):
    """Converts array or tensor from numpy image format to tensor image format"""

    def forward(self, x: torch.Tensor | np.ndarray):
        assert x.ndim == 4
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        x = x.permute((0, 3, 1, 2))
        x = x.contiguous()
        x = x.div(255)
        return x


def collect_rollouts(
    env, agent, n_episodes, verbose=False, render=False, seed=None, deterministic=True
) -> list[EpisodeData]:
    episodes = []
    trange = tqdm.trange(n_episodes, desc=env.spec.id, miniters=10) if verbose else range(n_episodes)

    for i in trange:
        obs, _ = env.reset(seed=seed + i if seed is not None else None)

        ep = {"observations": [], "actions": [], "rewards": [], "infos": []}
        ep["observations"].append(obs)

        while True:
            action, _ = agent.predict(obs.copy(), deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(action)
            ep["observations"].append(obs)
            ep["actions"].append(action)
            ep["rewards"].append(rew)
            ep["infos"].append(info)

            if render:
                env.render()

            if term or trunc:
                episodes.append(EpisodeData(**ep))
                break

    return episodes


def run_rollouts(env, agent, n_episodes, verbose=True, render=False, seed=1001, deterministic=True):
    trange = tqdm.trange(n_episodes, desc=env.spec.id, miniters=10) if verbose else range(n_episodes)
    for i in trange:
        obs, _ = env.reset(seed=seed + i)
        while True:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                break
            if render:
                env.render()


def collect_dataset(
    env,
    agent,
    n_episodes,
    save_dir,
    tag="",
    record_img_obs=False,
    record_statistics=True,
    verbose=False,
    render=False,
    seed=1001,
    deterministic=True,
) -> MinariDataset:
    dataset_id = make_dataset_id(env, tag)
    dataset_path = os.path.join(save_dir, tag, "data")
    assert not os.path.exists(dataset_path), f"Dataset {dataset_id} already exists at {dataset_path}"

    env = PersistentInfos(env)

    if record_img_obs:
        video_dir = os.path.join(save_dir, tag, "videos")
        env = RecordVideo(env, video_dir, write_on_ep_end=True)

    if record_statistics:
        env = RecordStatistics(env)

    env = minari.DataCollector(
        env, observation_space=env.observation_space, action_space=env.action_space, record_infos=True
    )

    run_rollouts(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        verbose=verbose,
        render=render,
        seed=seed,
        deterministic=deterministic,
    )
    metadata = _generate_dataset_metadata(dataset_id, env_spec=env.spec, **default_keys)  # type: ignore
    dataset = save_dataset(env, dataset_path, metadata)

    if record_statistics:
        statistics = env.get_wrapper_attr("get_statistics")()
        base_env_id = getattr(env.unwrapped, "base_env_id", env.spec.id)
        if verbose:
            print(base_env_id, statistics)
        normalized_score = get_normalized_score(base_env_id, statistics["reward_avg"])
        statistics["normalized_score"] = normalized_score
        dataset.storage.update_metadata({"statistics": statistics})

    dataset.storage.update_metadata(
        {
            "algorithm_name": agent.__class__.__name__,
            "anomaly_type": getattr(env.unwrapped, "anomaly_type", "None"),
            "anomaly_strength": getattr(env.unwrapped, "anomaly_strength", "None"),
            "anomaly_onset": getattr(env.unwrapped, "anomaly_onset", "None"),
            "anomaly_param": getattr(env.unwrapped, "anomaly_param", "None"),
            "seed": seed,
        }
    )
    return dataset


default_keys = dict(
    eval_env=None,
    algorithm_name=None,
    author=None,
    author_email=None,
    code_permalink=None,
    ref_min_score=None,
    ref_max_score=None,
    expert_policy=None,
    num_episodes_average_score=100,
    description=None,
    requirements=None,
)


def save_dataset(env, dataset_path, metadata) -> MinariDataset:
    assert isinstance(env, minari.DataCollector)
    os.makedirs(dataset_path)
    env._save_to_disk(dataset_path, metadata)
    dataset = MinariDataset(dataset_path)
    dataset.storage.update_metadata({"dataset_size": dataset.storage.get_size()})
    return dataset


def make_dataset_id(env, tag="", version="v0"):
    dataset_id = f"{env.spec.id}-{tag}-{version}"
    return dataset_id


def pad_collate(batch):
    return BatchEpisodeData(batch)
