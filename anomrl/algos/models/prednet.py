import math
from collections import deque

import lightning as L
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class LitPredNet(L.LightningModule):
    def __init__(
        self,
        lr=0.001,
        A_channels=(3, 48, 96, 192),
        R_channels=(3, 48, 96, 192),
        output_mode="error",
        nt=10,
    ):
        super(LitPredNet, self).__init__()
        self.lr = lr
        self.nt = nt
        self.model = PredNet(R_channels, A_channels, output_mode)
        self.layer_loss_weights = torch.FloatTensor([[1.0], [0.0], [0.0], [0.0]])
        self.time_loss_weights = 1.0 / (nt - 1) * torch.ones(nt, 1)
        self.time_loss_weights[0] = 0

    def forward(self, x):
        return self.model(x)

    def _training_step(self, batch, batch_idx):
        assert len(batch) == 1
        self.model.output_mode = "error"
        chunks = torch.split(batch.observations, self.nt, dim=1)
        if chunks[-1].size(1) != self.nt:
            chunks = chunks[:-1]
        if len(chunks) == 0:
            return torch.tensor(0.0, requires_grad=True)
        all_errors = []
        for x in chunks:
            errors = self.model(x)
            loc_batch = errors.size(0)
            errors = torch.mm(errors.view(-1, self.nt), self.time_loss_weights.to(self.device))  # batch*n_layers x 1
            errors = torch.mm(errors.view(loc_batch, -1), self.layer_loss_weights.to(self.device))  # batch x n_layers
            errors = torch.mean(errors)
            all_errors.append(errors)
        return torch.mean(torch.stack(all_errors))

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self._training_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = self._training_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx=None):
        self.model.eval()
        self.model.to(self.device)
        self.model.output_mode = "prediction"
        assert len(batch) == 1
        preds = []

        for t in range(1, len(batch.observations[0])):
            p = self.model(batch.observations[:, max(0, t - self.nt) : t])
            preds.append(p)
        preds = torch.cat(preds).unsqueeze(0)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


# https://github.com/leido/pytorch-prednet/tree/master


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode="error"):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0,)  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.output_mode = output_mode

        default_output_modes = ["prediction", "error"]
        assert output_mode in default_output_modes, "Invalid output_mode: " + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1], self.r_channels[i], (3, 3))
            setattr(self, "cell{}".format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module("satlu", SatLU())
            setattr(self, "conv{}".format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(
                nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], (3, 3), padding=1), self.maxpool
            )
            setattr(self, "update_A{}".format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, "cell{}".format(l))
            cell.reset_parameters()

    def forward(self, input):
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2 * self.a_channels[l], w, h)).cuda()
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()
            w = w // 2
            h = h // 2
        time_steps = input.size(1)
        total_error = []

        for t in range(time_steps):
            A = input[:, t]
            A = A.type(torch.cuda.FloatTensor)

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, "cell{}".format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l + 1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx

            for l in range(self.n_layers):
                conv = getattr(self, "conv{}".format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, "update_A{}".format(l))
                    A = update_A(E)
            if self.output_mode == "error":
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)

        if self.output_mode == "error":
            return torch.stack(total_error, 2)  # batch x n_layers x nt
        elif self.output_mode == "prediction":
            return frame_prediction


class SatLU(nn.Module):
    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            self.__class__.__name__
            + " ("
            + "min_val="
            + str(self.lower)
            + ", max_val="
            + str(self.upper)
            + inplace_str
            + ")"
        )


# https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
            self.register_parameter("bias_ch", None)
        self.register_buffer("wc_blank", torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        wx = F.conv2d(input, self.weight_ih, self.bias_ih, self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride, self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride, self.padding_h, self.dilation, self.groups)

        wxhc = (
            wx
            + wh
            + torch.cat(
                (
                    wc[:, : 2 * self.out_channels],
                    Variable(self.wc_blank).expand(wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)),
                    wc[:, 2 * self.out_channels :],
                ),
                1,
            )
        )

        i = F.sigmoid(wxhc[:, : self.out_channels])
        f = F.sigmoid(wxhc[:, self.out_channels : 2 * self.out_channels])
        g = F.tanh(wxhc[:, 2 * self.out_channels : 3 * self.out_channels])
        o = F.sigmoid(wxhc[:, 3 * self.out_channels :])

        c_1 = f * c_0 + i * g
        h_1 = o * F.tanh(c_1)
        return h_1, (h_1, c_1)
