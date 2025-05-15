import os
import sys
from multiprocessing import Pool
from tempfile import TemporaryDirectory

import dotenv
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat, Logger

from anomrl.common.agent_utils import build_agent
from anomrl.common.env_utils import make_env
from anomrl.common.log_utils import MLflowOutputFormat
from anomrl.common.sb3_callbacks import MLflowEvalCallback, RecordEvalCallback, RecordInfoCallback

dotenv.load_dotenv(override=True)
OmegaConf.register_new_resolver("eval", lambda s: eval(s))


def train_run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    mlflow.log_params(OmegaConf.to_object(cfg))

    loggers = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()])

    train_env = make_env(**cfg.env, n_envs=cfg.train_envs)
    if cfg.eval_envs > 0:
        eval_env = make_env(**cfg.env, n_envs=cfg.eval_envs)
    else:
        eval_env = train_env

    agent = build_agent(
        agent_name=cfg.agent.name,
        agent_module=cfg.agent.module,
        agent_kwargs=cfg.agent.kwargs,
        env=train_env,
        device=cfg.train.device,
        seed=cfg.seed,
    )

    agent.set_logger(loggers)

    cb_list = []
    if cfg.get("record_eval", None) is not None:
        record_cb = RecordEvalCallback(
            eval_env,
            freq=cfg.record_eval.freq,
            n_episodes=cfg.record_eval.n_episodes,
            record_video=cfg.record_eval.video,
            record_plots=cfg.record_eval.plots,
        )
        cb_list.append(record_cb)

    if cfg.get("record_info", None) is not None:
        record_info_cb = RecordInfoCallback(log_keys=cfg.record_info.log_keys)
        cb_list.append(record_info_cb)

    if cfg.get("eval", None) is not None:
        eval_callback = MLflowEvalCallback(
            eval_env=eval_env,
            eval_freq=cfg.eval.freq,
            n_eval_episodes=cfg.eval.n_episodes,
            deterministic=True,
            render=False,
        )
        cb_list.append(eval_callback)

    agent.learn(
        total_timesteps=cfg.train.n_steps,
        progress_bar=True,
        callback=CallbackList(cb_list),
        log_interval=25,
    )

    if cfg.ckpts.save_final:
        with TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "final_model.zip"), "wb") as f:
                agent.save(f.name)
                mlflow.log_artifact(f.name, "models")


def train_run_with_seed(cfg: DictConfig, seed=42, parent_run_id=None) -> None:
    with mlflow.start_run(nested=True, run_name=f"seed_{seed}", parent_run_id=parent_run_id) as child_run:
        parent_run = mlflow.get_parent_run(child_run.info.run_id)
        mlflow.set_tag("parent_run", parent_run.info.run_name)
        mlflow.set_tags(cfg.mlflow.tags)
        with open_dict(cfg):
            cfg.seed = seed
        train_run(cfg)


@hydra.main(version_base=None, config_path="./cfgs", config_name="train_agent_cfg")
def train(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment)

    if not OmegaConf.is_list(cfg.seeds):
        raise ValueError(f"seeds must be a list, got: {type(cfg.seeds)}")

    with mlflow.start_run() as parent_run:
        mlflow.set_tags(cfg.mlflow.tags)

        if len(cfg.seeds) == 1:
            with open_dict(cfg):
                cfg.seed = cfg.seeds[0]
            train_run(cfg)

        else:
            with Pool(len(cfg.seeds)) as pool:
                pool.starmap(train_run_with_seed, [(cfg, s, parent_run.info.run_id) for s in cfg.seeds])


if __name__ == "__main__":
    """call this script with
        - command line arguments only

            python exp/train/train_agent.py +agent.name=PPO +env.id=Sape-Goal1

        - cfg files (specified by the arguments)

            python exp/train/train_agent.py +agen=ppo +env=sape_goal1

    """

    train()
