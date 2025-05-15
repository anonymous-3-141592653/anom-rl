import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NET_ARCHS = {
    "tiny": [64, 64],
    "small": [256, 256],
    "medium": [512, 256, 128],
    "large": [1024, 1024, 1024],
    "deep": [200, 200, 200, 200],
}


ACTIVATION_FN_DICT = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "tanh": nn.Tanh,
}


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=1.0 / (2.0 * np.sqrt(m.in_features)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ProbMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        net_arch="small",
        p_dropout=0.0,
        activation_fn="LeakyReLU",
        trunc_norm_init=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        hidden_dims = NET_ARCHS[net_arch]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(ACTIVATION_FN_DICT[activation_fn]())
            layers.append(nn.Dropout(p=p_dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, output_dim)
        self.logvar_layer = nn.Linear(prev_dim, output_dim)

        self.max_logvar = nn.parameter.Parameter(torch.ones(1, output_dim, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.parameter.Parameter(-torch.ones(1, output_dim, dtype=torch.float32) * 10.0)

        if trunc_norm_init:
            self.apply(init_weights)

    def forward(self, x):
        x = self.hidden_layers(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=1, fc_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, output_dim))

    def forward(self, x):
        batch_size, _, _ = x.shape
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(lstm_out)
        return out
