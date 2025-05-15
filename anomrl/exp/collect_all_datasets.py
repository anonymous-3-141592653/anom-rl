# %%
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anomaly_gym
import gymnasium
from dotenv import load_dotenv
from sb3_contrib import TQC

from anomrl.common.log_utils import setup_logger
from anomrl.data.data_utils import collect_dataset

load_dotenv(override=True)
setup_logger()


def _run_task(task, agent=None):
    if agent is None:
        agent = globals().get("agent", None)

    if agent is None:
        agent = TQC.load(task["agent_path"], device="cpu")

    assert agent is not None, "Agent not found"

    if task["record_img_obs"]:
        task["env_kwargs"]["render_mode"] = "rgb_array"

    logging.info(f"Collecting dataset for: {task}")

    env = gymnasium.make(task["env_id"], **task["env_kwargs"])
    collect_dataset(
        env=env,
        agent=agent,
        record_statistics=True,
        verbose=True,
        n_episodes=task["n_episodes"],
        tag=task["tag"],
        seed=task["seed"],
        record_img_obs=task["record_img_obs"],
        save_dir=task["save_dir"],
    )


def _create_tasks(env_id, save_dir, n_episodes, anomaly_onset, record_img_obs, seed):
    tasks = [
        {
            "env_id": env_id,
            "env_kwargs": {},
            "tag": "train",
            "seed": seed + 1000,
            "n_episodes": n_episodes,
            "record_img_obs": record_img_obs,
            "save_dir": save_dir,
        },
        {
            "env_id": env_id,
            "env_kwargs": {},
            "tag": "val",
            "seed": seed + 2000,
            "n_episodes": math.ceil(n_episodes / 10),
            "record_img_obs": record_img_obs,
            "save_dir": save_dir,
        },
        {
            "env_id": env_id,
            "env_kwargs": {},
            "tag": "test",
            "seed": seed + 3000,
            "n_episodes": n_episodes,
            "record_img_obs": record_img_obs,
            "save_dir": save_dir,
        },
    ]

    strengths = ["tiny", "medium", "strong", "extreme"]
    anomaly_types = gymnasium.make(
        "Anom_" + env_id, anomaly_strength="tiny", anomaly_type="obs_noise"
    ).unwrapped.anomaly_types

    for strength in strengths:
        for a_type in anomaly_types:
            env_kwargs = dict(anomaly_type=a_type, anomaly_strength=strength, anomaly_onset=anomaly_onset)
            tasks.append(
                {
                    "env_id": "Anom_" + env_id,
                    "env_kwargs": env_kwargs,
                    "tag": f"{a_type}_{strength}",
                    "seed": seed + 4000,
                    "n_episodes": n_episodes,
                    "record_img_obs": record_img_obs,
                    "save_dir": save_dir,
                }
            )

    return tasks


def collect_all_datasets(env_id, save_dir, n_episodes, anomaly_onset, record_img_obs, n_jobs, seed):
    tasks = _create_tasks(env_id, save_dir, n_episodes, anomaly_onset, record_img_obs, seed)

    if n_jobs == 1:
        for task in tasks:
            _run_task(task)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for task in tasks:
                result = executor.submit(_run_task, task)
                futures.append(result)

            for f in futures:
                f.result()


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="exp")
    parser.add_argument("--agent", type=str, default="TQC")
    parser.add_argument("--agent_path", type=str, default=None)
    parser.add_argument("--anomaly_onset", type=str, default="start")
    parser.add_argument("--record_img_obs", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    if args.save_dir is None:
        DATASETS_DIR = os.environ.get("DATASETS_DIR")
        if DATASETS_DIR is None:
            DATASETS_DIR = str(Path(__file__).parents[2] / "data/datasets")
            logging.warning(f"env variable DATASETS_DIR not set, using default: {DATASETS_DIR}")

        args.save_dir = os.path.join(
            DATASETS_DIR, f"{args.agent}-{args.data_tag}-onset_{args.anomaly_onset}-seed_{args.seed}/{args.env}"
        )

    if args.agent_path is None:
        TRAINED_AGENTS_DIR = os.environ.get("TRAINED_AGENTS_DIR")
        if TRAINED_AGENTS_DIR is None:
            TRAINED_AGENTS_DIR = str(Path(__file__).parents[2] / "data/trained_agents")
            logging.warning(f"env variable TRAINED_AGENTS_DIR not set, using default: {TRAINED_AGENTS_DIR}")

        args.agent_path = os.path.join(TRAINED_AGENTS_DIR, f"{args.env}/{args.agent}/best_model.zip")
        assert os.path.exists(args.agent_path), f"Agent path {args.agent_path} does not exist"

    if args.agent == "TQC":
        agent = TQC.load(args.agent_path, device="cpu")
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    n_cpus = len(os.sched_getaffinity(0)) if args.parallel else 1

    collect_all_datasets(
        env_id=args.env,
        save_dir=args.save_dir,
        n_episodes=args.n_episodes,
        anomaly_onset=args.anomaly_onset,
        record_img_obs=args.record_img_obs,
        n_jobs=n_cpus,
        seed=args.seed,
    )
