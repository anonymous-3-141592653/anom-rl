import anomaly_gym
import gymnasium
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation, TransformObservation
from stable_baselines3.common.env_util import is_wrapped, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def make_env(env_id, env_kwargs, n_envs=0, frame_stack=1, monitor_wrapper=True):
    def _make():
        env = gymnasium.make(env_id, **env_kwargs)

        if frame_stack > 1:
            h, w, c = env.observation_space.shape
            env = FrameStackObservation(env, frame_stack)
            env = TransformObservation(env, lambda x: np.array(x).reshape(h, w, c * frame_stack))
            env.observation_space = Box(low=0, high=255, shape=(h, w, c * frame_stack), dtype=np.uint8)

        if not is_wrapped(env, Monitor) and monitor_wrapper:
            print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        return env

    if n_envs > 1:
        env = DummyVecEnv([_make for _ in range(n_envs)])
    else:
        env = _make()

    return env
