# %%
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import anomaly_gym
import gymnasium
import numpy as np
from anomaly_gym.common.metrics import get_normalized_score
from dotenv import load_dotenv
from IPython.display import Video, display
from sb3_contrib import TQC

from anomrl.common.wrappers import RecordVideo
from anomrl.data.data_utils import collect_rollouts

load_dotenv()
TRAINED_AGENTS_DIR = os.environ.get("TRAINED_AGENTS_DIR")
if TRAINED_AGENTS_DIR is None:
    TRAINED_AGENTS_DIR = str(Path(__file__).parents[2] / "data/trained_agents")
RENDER_ENV = False
RECORD_VIDEO = True
N_EPISODES = 1
# %%
# choose from
# ENV_ID = "Carla-LaneKeep"
# ENV_ID = "Sape-Goal0"
# ENV_ID = "Sape-Goal1"
# ENV_ID = "Sape-Goal2"
# ENV_ID = "Mujoco-CartpoleSwingup"
# ENV_ID = "Mujoco-HalfCheetah"
# ENV_ID = "Mujoco-Reacher3D"
# ENV_ID = "URMujoco-Reach"
ENV_ID = "URMujoco-PnP"
env = gymnasium.make(ENV_ID, render_mode="rgb_array")


# %%
# use this to test anomalous envs
# env = gymnasium.make(
#     "Anom_" + ENV_ID,
#     anomaly_type="obs_factor",
#     # anomaly_strength="extreme",
#     anomaly_onset="start",
# )

# %%
agent_path = os.path.join(TRAINED_AGENTS_DIR, ENV_ID, "TQC/best_model.zip")
agent = TQC.load(path=agent_path, device="cpu")


# %%
if RECORD_VIDEO:
    env = RecordVideo(env)

episodes = collect_rollouts(env=env, agent=agent, n_episodes=N_EPISODES, verbose=1, render=RENDER_ENV, seed=1)

if RECORD_VIDEO:
    with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        env.write_video(tmp.name)
        video = Video(tmp.name, embed=True, width=480, height=480)
        display(video)
        print("saved video to:", tmp.name)


# %%
cum_rewards = [e.rewards.sum() for e in episodes]
avg_reward = np.mean(cum_rewards)
std_reward = np.std(cum_rewards)
normalized_score = get_normalized_score(ENV_ID, avg_reward)
print("policy: ", agent.__class__.__name__)
print("cum_rewards: ", cum_rewards)
print("avg_reward: ", avg_reward)
print("std_reward: ", std_reward)
print("normalized_score: ", normalized_score)
