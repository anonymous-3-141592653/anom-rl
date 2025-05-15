import logging
import os

import hydra
import lightning as L
import mlflow
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import ListConfig

logger = logging.getLogger(__name__)


class LitModelTrainer:
    def __init__(
        self,
        max_epochs,
        check_val_every_n_epoch=1,
        early_stopping=False,
        progress_bar=True,
        save_best_checkpoint=True,
        log_mlflow=True,
        accelerator="cuda",
        extra_callbacks: None | ListConfig = None,
        save_last_checkpoint=True,
    ):
        if save_best_checkpoint:
            assert max_epochs > check_val_every_n_epoch, "Check val every epoch should be less than max epochs"

        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=25))

        if progress_bar:
            from anomrl.common.lit_utils import TrainingProgressBar

            callbacks.append(TrainingProgressBar())

        # always cache checkpoints
        import tempfile

        from lightning.pytorch.callbacks import ModelCheckpoint

        ckpt_dir = tempfile.TemporaryDirectory().name
        logger.info(f"caching checkpoints to {ckpt_dir}")
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_last=save_last_checkpoint,
                save_top_k=int(save_best_checkpoint),
                filename="{epoch:02d}-{val_loss:.2f}_best",
                dirpath=ckpt_dir,
            )
        )

        if extra_callbacks is not None:
            for cb in extra_callbacks:
                callbacks.append(hydra.utils.instantiate(cb))

        if log_mlflow:
            active_run = mlflow.active_run()
            assert active_run is not None, "No active mlflow run found, need to start mlflow run first"
            experiment = mlflow.get_experiment(active_run.info.experiment_id)
            tracking_uri = mlflow.get_tracking_uri()

            mlflow_logger = MLFlowLogger(
                experiment_name=experiment.name,
                tracking_uri=tracking_uri,
                run_id=active_run.info.run_id,
                log_model=True,
            )

        self._trainer = L.Trainer(
            max_epochs=max_epochs,
            enable_model_summary=True,
            enable_checkpointing=True,
            accelerator=accelerator,
            logger=mlflow_logger,
            callbacks=callbacks,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
