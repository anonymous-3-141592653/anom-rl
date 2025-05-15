import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anomrl.algos.models.core_models import LSTM, MLP, ProbMLP


class LitDynamicsModel(L.LightningModule):
    base_model_cls = MLP

    def __init__(
        self,
        observation_size,
        action_size,
        lr=0.001,
        lr_schedule=False,
        weight_decay=0.0001,
        base_model_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observation_n = observation_size[0]
        self.action_n = action_size[0]
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self._init_base_model(base_model_kwargs)
        self.save_hyperparameters()

    def _init_base_model(self, base_model_kwargs=None):
        input_dim = self.observation_n + self.action_n
        output_dim = self.observation_n
        self.base_model = self.base_model_cls(input_dim, output_dim, **(base_model_kwargs or {}))

    def _forward(self, batch):
        assert batch.observations.ndim == 3  # (batch_size, seq_len, obs_dim)
        assert batch.actions.ndim == 3  # (batch_size, seq_len, act_dim)
        assert batch.observations.shape[-1] == self.observation_n
        assert batch.actions.shape[-1] == self.action_n
        return self.base_model(self._process_input(batch))

    def _compute_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def _process_input(self, batch):
        return torch.cat((batch.observations[..., :-1, :], batch.actions), dim=-1)

    def _process_predictions(self, batch, pred):
        next_obs = batch.observations[..., :-1, :]
        return next_obs + pred

    def _compute_target(self, batch):
        return batch.observations[..., 1:, :] - batch.observations[..., :-1, :]

    def training_step(self, batch, batch_idx=None):
        pred = self._forward(batch)
        target = self._compute_target(batch)
        loss = self._compute_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        pred = self._forward(batch)
        output = self._process_predictions(batch, pred)
        return output

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_schedule:
            return {
                "optimizer": optim,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim),
                "monitor": "val_loss",
            }
        else:
            return optim


class LSTMDynamicsModel(LitDynamicsModel):
    base_model_cls = LSTM


class GaussianNLLLoss(nn.Module):
    def __init__(self):
        # Compute the Negative Log-Likelihood (NLL) loss for a Gaussian distribution.
        # NLL(x, μ, var) = 0.5 * [ log(var) + ((x - μ)^2 / var) + log(2π) ]

        # where:
        # - x: target
        # - μ: Predicted mean
        # - logvar: Predicted log variance (logvar = log(σ^2))
        super().__init__()

    def forward(self, mean, log_var, target):
        inv_var = torch.exp(-log_var)
        nll_loss = 0.5 * log_var + 0.5 * ((target - mean) ** 2) * inv_var
        return nll_loss


def variance_penalty(log_var):
    # Penalize high variance (or log variance) predictions
    return log_var**2


def kl_divergence(mean, log_var):
    # Calculate KL divergence between predicted distribution and standard normal prior
    kl_loss = -0.5 * (1 + log_var - mean.pow(2) - torch.exp(log_var))
    return kl_loss


class LitProbDynamicsModel(LitDynamicsModel):
    base_model_cls = ProbMLP

    def __init__(self, n_samples=250, lambda_var=0.01, lambda_kl=1e-6, lambda_regu=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.lambda_var = lambda_var
        self.lambda_kl = lambda_kl
        self.lambda_regu = lambda_regu
        self.loss_fn = GaussianNLLLoss()

    def _compute_loss(self, prediction, targets):
        pred_mean, pred_logvar = prediction

        nll_loss = self.loss_fn(pred_mean, pred_logvar, targets).mean()
        var_loss = variance_penalty(pred_logvar).mean() * self.lambda_var
        kl_loss = kl_divergence(pred_mean, pred_logvar).sum() * self.lambda_kl
        loss = nll_loss + var_loss + kl_loss
        self.log("nll_loss", nll_loss, prog_bar=True)
        self.log("var_loss", var_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)

        regu_loss = self.lambda_regu * (self.base_model.max_logvar.sum() - self.base_model.min_logvar.sum())
        loss += regu_loss
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        pred_mean, pred_logvar = self._forward(batch)
        samples = torch.stack([torch.normal(pred_mean, pred_logvar.exp().sqrt()) for s in range(self.n_samples)])
        output = self._process_predictions(batch, samples)  # (n_samples, batch_size, seq_len, obs_dim)
        return output


class LitEnsembleDynamicsModel(LitDynamicsModel):
    def __init__(self, ens_size=5, **kwargs):
        self.ens_size = ens_size
        super().__init__(**kwargs)
        self.loss_fn = nn.MSELoss()

    def _init_base_model(self, base_model_kwargs=None):
        input_dim = self.observation_n + self.action_n
        output_dim = self.observation_n
        self.base_models = nn.ModuleList(
            [self.base_model_cls(input_dim, output_dim, **base_model_kwargs or {}) for _ in range(self.ens_size)]
        )

    def _forward(self, x):
        assert x.shape[1] == self.ens_size
        assert x.shape[-1] == self.observation_n + self.action_n
        return torch.stack([self.base_models[i](x[:, i, ...]) for i in range(self.ens_size)])

    def _compute_loss(self, preds, targets):
        assert preds.shape == targets.shape
        loss = self.loss_fn(preds, targets)
        return loss

    def training_step(self, batch, batch_idx=None):
        # different inputs and targets for each ensemble member
        assert batch.observations.ndim == 4  # (batch_size, ens_size, seq_len, obs_dim)
        assert batch.actions.ndim == 4  # (batch_size, ens_size, seq_len, act_dim)
        assert batch.observations.shape[1] == len(self.base_models)
        assert batch.actions.shape[1] == len(self.base_models)
        assert batch.observations.shape[-1] == self.observation_n
        assert batch.actions.shape[-1] == self.action_n

        preds = self._forward(self._process_input(batch))
        targets = self._compute_target(batch).swapaxes(0, 1)  # -> (ens_size, batch_size, seq_len, obs_dim)
        loss = self._compute_loss(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        # same inputs and targets for all ens members
        assert batch.observations.ndim == 3  # (batch_size, seq_len, obs_dim)
        assert batch.actions.ndim == 3  # (batch_size, seq_len, act_dim)
        assert batch.observations.shape[-1] == self.observation_n
        assert batch.actions.shape[-1] == self.action_n
        # inputs = [batch] * (self.ens_size - 1)
        inputs = self._process_input(batch).unsqueeze(1).expand(-1, self.ens_size, -1, -1)
        preds = self._forward(inputs)
        targets = self._compute_target(batch).unsqueeze(0).expand(self.ens_size, -1, -1, -1)
        loss = self._compute_loss(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        # same inputs for all ens members
        assert batch.observations.ndim == 3  # (batch_size, seq_len, obs_dim)
        assert batch.actions.ndim == 3  # (batch_size, seq_len, act_dim)
        assert batch.observations.shape[-1] == self.observation_n
        assert batch.actions.shape[-1] == self.action_n

        inputs = self._process_input(batch).unsqueeze(1).expand(-1, self.ens_size, -1, -1)  # -> (bs, es, sl, od)
        preds = self._forward(inputs)
        output = self._process_predictions(batch, preds)  # -> (ens_size, batch_size, seq_len, obs_dim)
        return output


class LitProbEnsembleDynamicsModel(LitEnsembleDynamicsModel):
    base_model_cls = ProbMLP

    def __init__(self, ens_size=5, n_samples=1_000, lambda_var=1e-4, lambda_kl=0.0, lambda_regu=0.001, **kwargs):
        self.n_samples = n_samples
        self.lambda_var = lambda_var
        self.lambda_kl = lambda_kl
        self.lambda_regu = lambda_regu
        super().__init__(ens_size=ens_size, **kwargs)
        self.loss_fn = GaussianNLLLoss()

    def _forward(self, x):
        assert x.shape[1] == self.ens_size
        assert x.shape[-1] == self.observation_n + self.action_n
        preds = [self.base_models[i](x[:, i, ...]) for i in range(self.ens_size)]

        pred_mean = torch.stack([p for p, _ in preds])
        pred_logvar = torch.stack([l for _, l in preds])
        return pred_mean, pred_logvar

    def _compute_loss(self, prediction, target):
        pred_mean, pred_logvar = prediction

        nll_loss = self.loss_fn(pred_mean, pred_logvar, target).mean((1, 2, 3))
        var_loss = variance_penalty(pred_logvar).mean((1, 2, 3)) * self.lambda_var
        kl_loss = kl_divergence(pred_mean, pred_logvar).sum((1, 2, 3)) * self.lambda_kl
        loss = nll_loss + var_loss + kl_loss

        regu_loss = torch.stack(
            [self.lambda_regu * (m.max_logvar.sum() - m.min_logvar.sum()) for m in self.base_models]
        )
        loss += regu_loss

        loss = loss.sum()
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        inputs = self._process_input(batch).unsqueeze(1).expand(-1, self.ens_size, -1, -1)  # -> (bs, es, sl, od)
        pred_mean, pred_logvar = self._forward(inputs)
        samples = torch.stack([torch.normal(pred_mean, pred_logvar.exp().sqrt()) for s in range(self.n_samples)])
        output = self._process_predictions(batch, samples)  # -> (n_samples, ens_size, batch_size, seq_len, obs_dim)
        return output
