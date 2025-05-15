from typing import Any

import torch
from torch import nn

from .lit_autoencoders import Decoder, Encoder, LitPredictiveAutoEncoder


class LatentDynamicsModel(LitPredictiveAutoEncoder):
    def __init__(
        self,
        observation_size: torch.Size,
        action_size: torch.Size,
        prediction_mode: str = "delta",
        **kwargs,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        super().__init__(**kwargs)

    def _init_base_model(
        self,
        n_channels: int = 64,
        n_features: int = 128,
        encoder_class=Encoder,
        decoder_class=Decoder,
    ):
        self.encoder = encoder_class(self.observation_size, n_channels, n_features)
        self.decoder = decoder_class(self.observation_size, self.encoder.conv_size, n_channels, n_features)
        n_actions = self.action_size[0]
        self.latent_model = self.layers = nn.Sequential(
            nn.Linear(n_features + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )

    def _forward(self, batch):
        obs = batch.observations[:, :-1]
        batch_size, seq_len, n_channels, height, width = obs.shape
        obs = obs.view(seq_len * batch_size, n_channels, height, width)
        acs = batch.actions.view(seq_len * batch_size, -1)
        z = self.encoder(obs)
        z_prime = self.latent_model(torch.cat([z, acs], dim=1))
        obs_prime = self.decoder(z_prime)
        obs_prime = obs_prime.view(batch_size, seq_len, n_channels, height, width)
        return obs_prime, z

    def _compute_target(self, batch):
        return batch.observations[:, 1:]

    def _process_predictions(self, batch, prediction):
        assert prediction.shape == batch.observations[:, :-1].shape  # (batch_size, seq_len,  height, width, n_channels)
        return prediction
