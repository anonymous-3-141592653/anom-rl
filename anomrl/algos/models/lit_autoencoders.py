import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torchvision import transforms

from anomrl.common.plot_utils import plot_img_targets_predictions


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: torch.Size | None = None,
        n_channels: int = 32,
        n_features: int = 128,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        if input_size is None:
            input_size = torch.Size((3, 64, 64))

        assert input_size[0] == 3 or input_size[0] == 1, "Only RGB or grayscale images supported; Channel must be first"

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 1 * n_channels, kernel_size=4, stride=2),
            act_fn(),
            nn.Conv2d(1 * n_channels, 2 * n_channels, kernel_size=4, stride=2),
            act_fn(),
            nn.Conv2d(2 * n_channels, 4 * n_channels, kernel_size=4),
            act_fn(),
            nn.Conv2d(4 * n_channels, 8 * n_channels, kernel_size=4, stride=2),
            act_fn(),
        )
        with torch.no_grad():
            x = torch.zeros(1, *input_size)
            self.conv_size = self.conv(x).size()[1:]

        self.linear = nn.Linear(self.conv_size.numel(), n_features)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_size: torch.Size,
        conv_size: torch.Size,
        n_channels: int = 32,
        n_features: int = 128,
        act_fn=nn.ReLU,
        stride=2,
    ):
        super().__init__()
        # self.conv_size = conv_size
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape(output_size[1:], padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        self.conv_shape = (32 * n_channels, *conv4_shape)
        self.linear = nn.Linear(n_features, 32 * n_channels * np.prod(conv4_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 * n_channels, 4 * n_channels, conv4_kernel_size, stride, output_padding=0),
            act_fn(),
            nn.ConvTranspose2d(4 * n_channels, 2 * n_channels, conv3_kernel_size, stride, output_padding=0),
            act_fn(),
            nn.ConvTranspose2d(2 * n_channels, 1 * n_channels, conv2_kernel_size, stride, output_padding=0),
            act_fn(),
            nn.ConvTranspose2d(1 * n_channels, output_size[0], conv1_kernel_size, stride, output_padding=0),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], *self.conv_shape)
        out = self.decoder(x)
        return out


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: torch.Size,
        n_channels: int = 64,
        n_features: int = 128,
        encoder_class=Encoder,
        decoder_class=Decoder,
    ):
        super().__init__()
        self.encoder = encoder_class(input_size, n_channels, n_features)
        self.decoder = decoder_class(input_size, self.encoder.conv_size, n_channels, n_features)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class LitAutoEncoder(L.LightningModule):
    base_model_cls = Autoencoder

    def __init__(
        self,
        img_size: torch.Size | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 5,
        T_0: int = 100,
        T_mult: int = 2,
        eta_min: int = 0,
        base_model_kwargs: dict | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self._init_base_model(**base_model_kwargs or {})
        self.save_hyperparameters()

    def _init_base_model(self, **base_model_kwargs):
        assert self.img_size is not None
        self.base_model = self.base_model_cls(input_size=self.img_size, **base_model_kwargs)

    def _forward(self, batch):
        x = self._process_input(batch)
        batch_size, seq_len, n_channels, height, width = x.shape
        x = x.view(seq_len * batch_size, n_channels, height, width)
        out, z = self.base_model(x)
        z = z.view(batch_size, seq_len, -1)
        out = out.view(batch_size, seq_len, n_channels, height, width)
        return out, z

    def _compute_loss(self, pred, target):
        assert pred.shape == target.shape
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean()
        return loss

    def _process_input(self, batch):
        assert batch.observations.ndim == 5  # (batch_size, seq_len+1,  height, width, n_channels)
        return batch.observations

    def _process_predictions(self, batch, prediction):
        assert prediction.shape == batch.observations.shape  # (batch_size, seq_len+1,  height, width, n_channels)
        return prediction[:, :-1]  # (batch_size, seq_len,  height, width, n_channels)

    def _compute_target(self, batch):
        return batch.observations

    def _step(self, batch, batch_idx=None):
        pred, z = self._forward(batch)
        target = self._compute_target(batch)
        loss = self._compute_loss(pred, target)
        return pred, target, loss

    def training_step(self, batch, batch_idx=None):
        pred, target, loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"pred": pred, "target": target, "loss": loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx=None):
        pred, target, loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"pred": pred, "target": target, "loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=None):
        pred, target, loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=False, on_epoch=True)
        return {"pred": pred, "target": target, "loss": loss}

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        pred, z = self._forward(batch)
        output = self._process_predictions(batch, pred)
        return output

    @torch.no_grad()
    def encode(self, batch, batch_idx=None):
        _, z = self._forward(batch)
        return z

    def _warmup_lr_lambda(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            return (current_epoch + 1) / self.warmup_epochs
        return 1.0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup_sched = LambdaLR(optimizer, self._warmup_lr_lambda)
        cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[self.warmup_epochs]
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _make_plot(self, targets, outputs, n_steps=10):
        def _compute_delta_img(img_1, img_2, threshold=0.1):
            difference = torch.abs(img_1 - img_2)
            mask = difference.sum(0) > threshold
            diff_img = torch.zeros_like(img_1)
            diff_img[0] += mask  # add to red channel
            return diff_img

        fig, axs = plt.subplots(n_steps, 3, figsize=(3, n_steps), tight_layout=True)
        for i in range(n_steps):
            axs[i, 0].set_title(f"step {i}", fontsize=5)
            axs[i, 0].imshow(targets[i].permute(1, 2, 0).float().clamp(0, 1))
            axs[i, 0].axis("off")
            axs[i, 1].imshow(outputs[i].permute(1, 2, 0).float().clamp(0, 1))
            axs[i, 1].axis("off")
            axs[i, 2].imshow(_compute_delta_img(targets[i], outputs[i]).permute(1, 2, 0).float().clamp(0, 1))
            axs[i, 2].axis("off")

        return fig

    def plot_predictions(self, batch):
        predictions = self.predict_step(batch)
        denormalize = transforms.Normalize((-1), (1 / 0.5))
        targets = denormalize(batch.observations[0, :-1]).cpu()  # take first episdoe in batch
        predictions = denormalize(predictions[0]).cpu()  # take first episdoe in batch
        fig = plot_img_targets_predictions(targets, predictions)
        return fig


class LitPredictiveAutoEncoder(LitAutoEncoder):
    base_model_cls = Autoencoder

    def _process_input(self, batch):
        assert batch.observations.ndim == 5  # (batch_size, seq_len+1,  height, width, n_channels)
        return batch.observations[:, :-1]

    def _process_predictions(self, batch, prediction):
        assert prediction.shape == batch.observations[:, 1:].shape
        return batch.observations[:, :-1] + prediction  # (batch_size, seq_len,  height, width, n_channels)

    def _compute_target(self, batch):
        return batch.observations[:, 1:] - batch.observations[:, :-1]

    def plot_predictions(self, batch):
        predictions = self.predict_step(batch)
        denormalize = transforms.Normalize((-1), (1 / 0.5))
        targets = denormalize(batch.observations[0, 1:]).cpu()  # take first episdoe in batch
        predictions = denormalize(predictions[0]).cpu()  # take first episdoe in batch
        fig = plot_img_targets_predictions(targets, predictions)
        return fig
