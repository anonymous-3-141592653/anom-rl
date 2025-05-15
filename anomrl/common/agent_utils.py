import importlib
from typing import Callable, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import HerReplayBuffer

NET_ARCHS_DICT = {
    "ac_small": dict(pi=[64, 64], vf=[64, 64]),
    "ac_medium": dict(pi=[256, 256], vf=[256, 256]),
    "ac_large": dict(pi=[512, 512], vf=[512, 512]),
    "pg_small": [64, 64],
    "pg_medium": [256, 256],
    "pg_big": [400, 300],
    "pg_large": [256, 256, 256],
    "pg_verybig": [512, 512, 512],
}

ACTIVATION_FN_DICT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}


# from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_schedule(initial_value: float, schedule_type: str) -> Callable[[float], float]:
    if schedule_type == "linear":
        return linear_schedule(initial_value)
    else:
        raise ValueError("learning rate schedule not known")


def create_agent(agent_name, agent_module, agent_kwargs=None):
    """function to create agent from class name

    -> does not use gymnasium.make which does not work with ray multiprocessing

    """
    if agent_kwargs is None:
        agent_kwargs = {}
    m = importlib.import_module(agent_module)
    agent_cls = getattr(m, agent_name)
    agent = agent_cls(**agent_kwargs)
    return agent


def build_agent(
    agent_name,
    agent_module,
    agent_kwargs: dict | DictConfig | None = None,
    env=None,
    device="cuda",
    tb_log=None,
    seed=None,
    verbose=True,
):
    if isinstance(agent_kwargs, DictConfig):
        agent_kwargs = OmegaConf.to_container(agent_kwargs)

    if agent_kwargs is None:
        agent_kwargs = {}

    if "replay_buffer_class" in agent_kwargs:
        agent_kwargs["replay_buffer_class"] = eval(agent_kwargs["replay_buffer_class"])

    if "learning_rate_schedule" in agent_kwargs:
        agent_kwargs["learning_rate"] = get_schedule(
            agent_kwargs["learning_rate"], agent_kwargs.pop("learning_rate_schedule")
        )

    policy_kwargs = agent_kwargs.get("policy_kwargs", {})
    if isinstance(policy_kwargs.get("net_arch"), str):
        policy_kwargs["net_arch"] = NET_ARCHS_DICT[policy_kwargs.pop("net_arch")]

    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = ACTIVATION_FN_DICT[policy_kwargs.pop("activation_fn")]

    agent_kwargs.update(
        dict(env=env, device=device, verbose=verbose, seed=seed, tensorboard_log=tb_log, policy_kwargs=policy_kwargs)
    )

    agent = create_agent(agent_name, agent_module, agent_kwargs)
    return agent


def load_agent(agent_name, agent_module, path, env=None, device="cpu"):
    m = importlib.import_module(agent_module)
    agent_cls = getattr(m, agent_name)
    agent = agent_cls.load(path, env=env, device=device)
    return agent
