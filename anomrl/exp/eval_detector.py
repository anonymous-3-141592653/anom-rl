import logging
import warnings
from pathlib import Path

import dotenv
import hydra
import lightning as L
import mlflow
import mlflow.entities
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from anomrl.algos.detectors import AbstractDetector, LitModelDetector
from anomrl.common.eval_utils import (
    compute_detection_times,
    compute_global_metrics,
    compute_local_metrics,
    compute_thresholds,
)
from anomrl.common.helpers import flatten_dict_to_str, save_yaml
from anomrl.common.mlflow_utils import setup_experiment
from anomrl.data.datamodules import AbstractDataModule

torch.set_float32_matmul_precision("high")
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*You are using a CUDA device *")
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*The `srun` command is available on your system but is not used*")
warnings.filterwarnings("ignore", module="mlflow.utils.environment")

dotenv.load_dotenv(override=True)

# logger for this file
logger = logging.getLogger(__name__)


def load_model_registry(model_registry_dir: str):
    model_registry_dir = Path(model_registry_dir)
    model_registry_dir.mkdir(parents=True, exist_ok=True)
    model_registry = {}
    for path in model_registry_dir.rglob("*.pt"):
        model_id = path.stem
        model_registry[model_id] = path
    return model_registry


def make_summary(df, metric="GLOB_AUROC"):
    return {
        "avg. AUROC": df[metric].mean(),
        "tiny avg. AUROC": df[df["anomaly_strength"] == "tiny"][metric].mean(),
        "medium avg. AUROC": df[df["anomaly_strength"] == "medium"][metric].mean(),
        "strong avg. AUROC": df[df["anomaly_strength"] == "strong"][metric].mean(),
        "extreme avg. AUROC": df[df["anomaly_strength"] == "extreme"][metric].mean(),
    }


def create_tags(cfg: DictConfig):
    detector_cls = hydra.utils.instantiate(cfg.detector, _partial_=True).func
    model_cls = getattr(detector_cls, "model_class", None)
    model_cls_name = model_cls.__name__ if model_cls is not None else "None"
    model_kwargs = flatten_dict_to_str(cfg.detector.get("model_kwargs", {}))
    data_name = Path(cfg.data_module.data_dir).name
    model_id = f"{model_cls_name}_{cfg.tags.model_tag}_{model_kwargs}_{data_name}_{cfg.data_module.env_id}"
    tags = {
        "data_module": cfg.data_module._target_.split(".")[-1],
        "data_name": data_name,
        "detector_cls": detector_cls.__name__,
        "model_cls": model_cls_name,
        "model_kwargs": model_kwargs,
        "model_id": model_id,
    }
    return tags


def update_tags(cfg: DictConfig):
    new_tags = create_tags(cfg)
    with open_dict(cfg):
        cfg.tags.update(new_tags)


def load_detector(cfg: DictConfig, model_path: str):
    model = torch.load(model_path)
    detector = hydra.utils.instantiate(cfg.detector, model=model)
    return detector


def train_detector(cfg: DictConfig, data_module: AbstractDataModule):
    detector: LitModelDetector = hydra.utils.instantiate(
        cfg.detector, observation_size=data_module.observation_size, action_size=data_module.action_size
    )
    detector.fit(data_module, trainer_cfg=cfg.trainer)
    logger.info(f"Finished training {detector.__class__.__name__}")
    if cfg.register_model and isinstance(detector.model, L.LightningModule):
        logger.info(f"registering model {cfg.tags.model_id}")
        torch.save(detector.model, Path(cfg.model_registry_dir, f"{cfg.tags.model_id}.pt"))
    return detector


def _eval_detector(detector: AbstractDetector, dataloader):
    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader):
        assert len(batch) == 1, "Batch size must be 1 for evaluation"
        batch_scores = detector.predict_scores(batch)
        scores = np.concatenate(batch_scores)
        labels = np.concatenate([a[1:] for a in batch.infos["is_anomaly"]])
        assert len(scores) == len(labels)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_scores, all_labels


def evaluate_detector(detector: AbstractDetector, data_module: AbstractDataModule, anomaly_onset_mode: str):
    logger.info(f"Evaluating {detector.__class__.__name__} on {data_module.env_id}")
    logger.info(f"anomaly onset mode: {anomaly_onset_mode}")

    if anomaly_onset_mode == "start":
        normal_scores, normal_labels = _eval_detector(detector, data_module.test_dataloader(batch_size=1))

    else:
        val_scores, _ = _eval_detector(detector, data_module.val_dataloader(batch_size=1))
        thresholds = compute_thresholds(val_scores)

    all_results = []
    for anomaly_loader in data_module.anomaly_dataloaders(batch_size=1):
        anomaly_scores, anomaly_labels = _eval_detector(detector, anomaly_loader)

        if anomaly_onset_mode == "start":
            global_metrics = compute_global_metrics(
                y_true_list=anomaly_labels + normal_labels,
                y_score_list=anomaly_scores + normal_scores,
            )
            det_stats = {**global_metrics}
        else:
            global_metrics = compute_global_metrics(y_true_list=anomaly_labels, y_score_list=anomaly_scores)
            local_metrics = compute_local_metrics(y_true_list=anomaly_labels, y_score_list=anomaly_scores, vus=True)
            det_times = compute_detection_times(
                y_true_list=anomaly_labels, y_score_list=anomaly_scores, thresholds=thresholds
            )

            det_stats = {**global_metrics, **local_metrics, **det_times}

        env_meta = anomaly_loader.dataset.env_metadata
        result = {**env_meta, **det_stats, "dataset_seed": anomaly_loader.dataset.seed}
        logger.info(env_meta)
        logger.info(det_stats)
        all_results.append(result)

    result_df = pd.DataFrame(all_results)
    return result_df


def save_results(result_df, metadata, save_dir, log_mlflow=True):
    save_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_dir / "results.csv", index=False)
    save_yaml(metadata, save_dir / "metadata.yaml")
    summary = make_summary(result_df, metric="GLOB_AUROC")
    logger.info(summary)
    logger.info("Results saved to: " + str(save_dir))
    if log_mlflow:
        mlflow.log_metrics(summary)
        mlflow.log_dict(summary, "summary.txt")
        mlflow.log_artifact(str(save_dir / "results.csv"))
        mlflow.log_artifact(str(save_dir / "metadata.yaml"))


@hydra.main(config_path="cfgs", config_name="eval_detector", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
    setup_experiment(cfg.mlflow.experiment_name)

    # try:
    parent_run = mlflow.start_run(log_system_metrics=True)
    update_tags(cfg)
    mlflow.set_tags(cfg.tags)
    logger.info(f"Started new run: {cfg.mlflow.experiment_name}/{parent_run.info.run_name}({parent_run.info.run_id})")
    logger.info(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.run.seed)

    data_module = hydra.utils.instantiate(cfg.data_module)

    model_registry = load_model_registry(cfg.model_registry_dir)

    with mlflow.start_run(nested=True, run_name="train"):
        if cfg.tags.model_id in model_registry:
            logger.info(f"Model found, loaded model: {cfg.tags.model_id}")
            detector = load_detector(cfg, model_registry[cfg.tags.model_id])

        else:
            logger.info(f"No model found, training new detector with model: {cfg.tags.model_id}")
            detector = train_detector(cfg, data_module)

    with mlflow.start_run(nested=True, run_name="post_train"):
        detector.post_train_callback(data_module)

    with mlflow.start_run(nested=True, run_name="eval"):
        anomaly_onset_mode = "start" if "onset_start" in cfg.tags.data_name else "random"
        result_df = evaluate_detector(detector=detector, data_module=data_module, anomaly_onset_mode=anomaly_onset_mode)
    save_dir = Path(cfg.results_dir, f"{cfg.run.timestamp}_{parent_run.info.run_id}")
    save_results(result_df=result_df, metadata=cfg, save_dir=save_dir)
    status = "FINISHED"

    # except Exception as e:
    #     status = "FAILED"
    #     logger.error(e)

    # finally:
    #     # log logfile as artifact to mlflow
    #     mlflow.log_artifact(cfg.log_dir, "logs")
    #     mlflow.end_run(status)


if __name__ == "__main__":
    main()
