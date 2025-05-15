from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from tqdm import tqdm as _tqdm


class TrainingProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diable_validation_progress_bar = True

    def on_train_batch_end(self, *args, **kwargs) -> None:
        pass

    def on_train_epoch_start(self, trainer: L.Trainer, *_: Any) -> None:
        # init progress bar
        if self.train_progress_bar.total is None:
            self.train_progress_bar.reset(total=trainer.max_epochs)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # update the progress bar
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch + 1}")
        self.train_progress_bar.n = trainer.current_epoch + 1
        self.train_progress_bar.refresh()
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def init_validation_tqdm(self):
        if self.diable_validation_progress_bar:
            return _tqdm(disable=True)
        else:
            return super().init_validation_tqdm()


class PlotPredictionCallback(Callback):
    def __init__(self, *args, plot_every_n_epoch=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_every_n_epoch = plot_every_n_epoch

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        if batch_idx == 0:
            if trainer.current_epoch == 0 or (trainer.current_epoch + 1) % self.plot_every_n_epoch == 0:
                fig = pl_module.plot_predictions(batch)
                trainer.logger.experiment.log_figure(
                    run_id=trainer.logger.run_id,
                    figure=fig,
                    artifact_file=f"reconstruction_epoch_{trainer.current_epoch}.png",
                )
