# %%
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anomaly_gym
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sb3_contrib import TQC
from tqdm import tqdm

from anomrl.common.log_utils import setup_logger
from anomrl.data.data_utils import collect_rollouts, get_normalized_score

load_dotenv()
setup_logger()

# %%

SEARCH_SPACES = {
    "Mujoco-CartpoleSwingup": {
        "robot_mass": np.linspace(1, 3.5, 500),
        "robot_force": np.linspace(0, 25, 500),
        "robot_friction": np.linspace(0, 10.0, 500),
        "action_offset": np.linspace(0, 1, 500),
        "action_factor": np.linspace(1, 5, 500),
        "action_noise": np.linspace(0, 3, 500),
        "obs_offset": np.linspace(0, 1, 500),
        "obs_factor": np.linspace(1, 1.1, 500),
        "obs_noise": np.linspace(0, 1, 500),
    },
}


def get_anomaly_statistics(env_id, a_type, a_param, n_episodes):
    env = gymnasium.make("Anom_" + env_id, anomaly_type=a_type, anomaly_param=a_param, anomaly_onset="start")
    episodes = collect_rollouts(env=env, agent=agent, n_episodes=n_episodes, seed=20000)
    cum_rewards = np.array([e.rewards.sum() for e in episodes])
    normalized_scores = np.array([get_normalized_score(env_id, r) for r in cum_rewards])

    result = {
        "env_id": env_id,
        "anomaly_type": a_type,
        "anomaly_param": a_param,
        "normalized_score": normalized_scores.mean(),
    }

    return result


def get_all_statistics(env_id, n_episodes, parallel=True):
    all_statistics = []
    search_spaces = SEARCH_SPACES[env_id]

    n_cpus = len(os.sched_getaffinity(0))

    logging.info("Number of CPUs:", n_cpus)

    if parallel:
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            futures = []
            for anomaly_type, search_params in search_spaces.items():
                for param in search_params:
                    f = executor.submit(get_anomaly_statistics, env_id, anomaly_type, param, n_episodes)
                    futures.append(f)

            for f in tqdm(futures):
                all_statistics.append(f.result())
    else:
        with tqdm(total=sum(len(v) for v in search_spaces.values())) as pbar:
            for anomaly_type, search_params in search_spaces.items():
                for param in search_params:
                    statistics = get_anomaly_statistics(env_id, anomaly_type, param, n_episodes)
                    all_statistics.append(statistics)
                    pbar.update(1)

    df = pd.DataFrame(all_statistics)
    return df


def get_anomaly_strength_dict(df):
    anomaly_strength_dict = {}

    # get different values for column "anomaly type" if not nan or None
    anomaly_types = df["anomaly_type"].dropna().unique()

    strength_levels = {"tiny": 0.99, "light": 0.95, "medium": 0.9, "strong": 0.75, "extreme": 0.5}

    for anomaly_type in anomaly_types:
        anomaly_strength_dict[anomaly_type] = {}
        sdf = df[df["anomaly_type"] == anomaly_type]
        # for each strength level, get the index of the closest value to the strength level in the column "normalized_score"
        for strength, level in strength_levels.items():
            idx = sdf["normalized_score"].sub(level).abs().idxmin()
            anomaly_strength_dict[anomaly_type][strength] = {
                "param": sdf.loc[idx, "anomaly_param"].item(),
                "normalized_score": sdf.loc[idx, "normalized_score"].item(),
            }

    return anomaly_strength_dict


def plot_anomaly_parameters(df):
    anomaly_types = df["anomaly_type"].dropna().unique()
    fig, ax = plt.subplots(len(anomaly_types) + 1, 1, tight_layout=True, figsize=(5, 3 * len(anomaly_types)))
    plt.style.use("ggplot")
    for i, anomaly_type in enumerate(anomaly_types):
        ax[i].set_title(anomaly_type)
        df[df["anomaly_type"] == anomaly_type].plot(x="anomaly_param", y="normalized_score", ax=ax[i], kind="scatter")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("-n", "--n_episodes", type=int, default=100)
    parser.add_argument("--agent", type=str, default="TQC")
    parser.add_argument("--agent_path", type=str, default=None)
    args = parser.parse_args()

    if args.agent_path is None:
        TRAINED_AGENTS_DIR = os.environ.get("TRAINED_AGENTS_DIR")
        if TRAINED_AGENTS_DIR is None:
            TRAINED_AGENTS_DIR = str(Path(__file__).parents[3] / "data/trained_agents")
            logging.warning(f"env variable TRAINED_AGENTS_DIR not set, using default: {TRAINED_AGENTS_DIR}")

        args.agent_path = os.path.join(TRAINED_AGENTS_DIR, f"{args.env}/{args.agent}/best_model.zip")
        assert os.path.exists(args.agent_path), f"Agent path {args.agent_path} does not exist"

    if args.agent == "TQC":
        agent = TQC.load(args.agent_path, device="cpu")
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    df = get_all_statistics(args.env_id, parallel=args.parallel, n_episodes=args.n_episodes)

    file_name = f"data/results/anomaly_parameters_norm_score/{args.env_id}_anomaly_parameters_{args.n_episodes}"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    logging.info(f"Saving results to {file_name}")
    df.to_csv(file_name + ".csv", index=False)
    anomaly_strength_dict = get_anomaly_strength_dict(df)

    # save anomaly dict as yaml
    with open(file_name + ".yaml", "w") as file:
        logging.info(yaml.dump(anomaly_strength_dict))
        yaml.dump(anomaly_strength_dict, file)

    plot_anomaly_parameters(df)
    plt.savefig(file_name + ".png")
