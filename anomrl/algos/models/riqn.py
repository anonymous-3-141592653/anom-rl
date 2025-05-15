# code modified from: https://github.com/modanesh/recurrent_implicit_quantile_networks

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class AutoregressiveRecurrentIQN_v2(nn.Module):
    def __init__(self, feature_len, gru_size, quantile_embedding_dim, num_quantile_sample, device, fc1_units=64):
        super(AutoregressiveRecurrentIQN_v2, self).__init__()
        self.gru_size = gru_size
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device
        self.feature_len = feature_len
        self.fc_1 = nn.Linear(feature_len, fc1_units)
        self.gru = nn.GRUCell(fc1_units, gru_size)
        self.fc_2 = nn.Linear(gru_size, gru_size)
        self.fc_3 = nn.Linear(gru_size, feature_len)

        self.phi = nn.Linear(self.quantile_embedding_dim, gru_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, hx, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(
            input_size * num_quantiles, self.quantile_embedding_dim
        )
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = functional.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.feature_len)
        state_tile = state_tile.flatten().view(-1, self.feature_len).to(self.device)

        x = functional.relu(self.fc_1(state_tile))
        ghx = self.gru(x, hx)
        x = ghx + functional.relu(self.fc_2(ghx))
        x = self.fc_3(x * phi)
        z = x.view(-1, num_quantiles, self.feature_len)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z, ghx


class RIQN_Predictor(AutoregressiveRecurrentIQN_v2):
    def __init__(
        self,
        input_features,
        gru_units=64,
        quantile_embedding_dim=128,
        num_quantile_sample=64,
        device=torch.device("cuda"),
        lr=0.001,
        num_tau_sample=1,
    ):
        super().__init__(input_features, gru_units, quantile_embedding_dim, num_quantile_sample, device)
        self.num_tau_sample = num_tau_sample
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)
        self.constructor_kwargs = {
            "input_features": input_features,
            "gru_units": gru_units,
            "quantile_embedding_dim": quantile_embedding_dim,
            "num_quantile_sample": num_quantile_sample,
            "lr": lr,
            "num_tau_sample": num_tau_sample,
        }

    def fit(self, train_ep_obs, *args, **kwargs):
        checkpoint_path = os.path.join("./tmp", "riqn_chk.pt")
        X_train = self.prepare_data(observations=train_ep_obs)
        self._fit(X_train, checkpoint_path=checkpoint_path, *args, **kwargs)

    def _fit(self, X_train, checkpoint_path, epochs=1_000, batch_size=128, clip_value=10, test_interval=10):
        self.train()
        states_min, states_max = states_min_max_finder(X_train)
        train_rb, test_rb, max_len = data_splitting(X_train, batch_size, states_min, states_max, self.device)
        epsilon = 1
        all_train_losses, all_test_losses = [], []
        best_loss = float("inf")
        for i in range(epochs):
            train_loss = self.train_epoch(train_rb, max_len, epsilon, clip_value=clip_value)
            if i % test_interval == 0:
                all_train_losses.append(train_loss)
                eval_loss, best_loss = self.eval_poch(
                    test_rb, max_len, best_loss=best_loss, epsilon=epsilon, path=checkpoint_path
                )
                all_test_losses.append(eval_loss)
                print(
                    f"ep: {i}/{epochs},  train_loss: {train_loss}, eval_loss: {eval_loss}, best_loss: {best_loss}",
                    end="\r",
                    flush=True,
                )
                # plot_losses(all_train_losses, all_test_losses, env_dir, memory=True, scheduled_sampling=True)
            epsilon = epsilon_decay(epsilon, epochs, i)
        self.eval()

    def train_epoch(self, train_rb, max_len, epsilon, clip_value):
        total_loss = ss_learn_model(
            model=self,
            optimizer=self.optimizer,
            memory=train_rb,
            max_len=max_len,
            epsilon=epsilon,
            clip_value=clip_value,
            num_tau_sample=self.num_tau_sample,
            gru_size=self.gru_size,
            feature_len=self.feature_len,
            device=self.device,
            has_memory=True,
        )
        return total_loss

    def eval_poch(self, test_rb, max_len, epsilon, best_loss, path):
        eval_loss, best_loss = ss_evaluate_model(
            model=self,
            memory=test_rb,
            max_len=max_len,
            best_total_loss=best_loss,
            path=path,
            epsilon=epsilon,
            num_tau_sample=self.num_tau_sample,
            gru_size=self.gru_size,
            feature_len=self.feature_len,
            device=self.device,
            has_memory=True,
        )
        return eval_loss, best_loss

    def feed_forward(self, hx, states, batch_size, sampling_size, tree_root=False):
        states = states.reshape(states.shape[0], 1, -1)
        if tree_root:
            tau = torch.Tensor(np.random.rand(batch_size * sampling_size, 1))
            if hx is not None:
                z, hx = self.forward(states, hx, tau, sampling_size)
            else:
                z = self.forward(states, tau, sampling_size)
        else:
            tau = torch.Tensor(np.random.rand(batch_size * self.num_tau_sample, 1))
            if hx is not None:
                z, hx = self.forward(states, hx, tau, self.num_tau_sample)
            else:
                z = self.forward(states, tau, self.num_tau_sample)
        return z, hx

    def predict_episode(self, episode_obs, sampling_size=8, horizon=1, normalzie=True):
        if episode_obs.ndim != 3:
            episode_obs = np.expand_dims(episode_obs, 1)

        self.eval()
        estimated_dists = []
        anomaly_scores = []
        h_memory = torch.zeros(len(episode_obs[0]) * sampling_size, self.gru_size)
        for i in range(len(episode_obs) - horizon):
            state = episode_obs[i][:, : self.feature_len]
            state = torch.Tensor(state)
            value_return, h_memory = self.feed_forward(
                h_memory.detach().to(self.device), state, len(state), sampling_size, tree_root=True
            )
            unaffected_h_memory = h_memory
            for j in range(1, horizon):
                tmp_h_memory = []
                tmp_value_return = []
                value_return_t = value_return
                h_memory_t = h_memory
                for sample in range(sampling_size):
                    value_return, h_memory = self.feed_forward(
                        h_memory_t[sample, :].detach().reshape(1, -1),
                        value_return_t[:, :, sample],
                        len(value_return_t),
                        sampling_size,
                        tree_root=False,
                    )
                    tmp_h_memory.append(h_memory)
                    tmp_value_return.append(value_return)
                h_memory = torch.stack(tmp_h_memory).squeeze(1)
                value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, self.feature_len, -1)
            h_memory = unaffected_h_memory
            estimated_dists.append(value_return.squeeze(0).detach().cpu().numpy())
            anomaly_score = measure_as(
                value_return.squeeze(0).detach().cpu().numpy(),
                episode_obs[i + horizon][:, : self.feature_len].squeeze(0),
                self.feature_len,
            )
            anomaly_scores.append(anomaly_score)

        mean_anomaly_scores = np.array(anomaly_scores).mean(axis=1)
        return mean_anomaly_scores

    def prepare_data(self, observations, has_time_feature=False):
        tensor_observations = construct_batch_data(
            observations.shape[-1], observations, device=self.device, has_time_feature=has_time_feature
        )
        states_min, states_max = states_min_max_finder(tensor_observations)
        ep_lengths = [len(ep) for ep in observations]
        self.states_min = states_min
        self.states_max = states_max
        self.ep_length_min = min(ep_lengths)
        self.ep_length_max = max(ep_lengths)
        return tensor_observations

    def save(self, path):
        print("Saving model at: ", path)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "constructor_kwargs": self.constructor_kwargs,
                "attributes": {},
            },
            f=path,
        )

    @classmethod
    def load(cls, path, device, **model_kwargs):
        saved_variables = torch.load(path)
        model = cls(**saved_variables["constructor_kwargs"], device=device)
        model.load_state_dict(saved_variables["state_dict"])
        for k, v in saved_variables["attributes"].items():
            model.__setattr__(k, v)
        model.to(device)
        model.eval()
        return model


def ss_evaluate_model(
    model, memory, max_len, gru_size, num_tau_sample, device, best_total_loss, path, epsilon, feature_len, has_memory
):
    total_loss = 0
    count = 0
    model.eval()
    s_hat = None
    for s_batch, mc_returns in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = test_model(
                        model,
                        h_memory.detach().to(device),
                        s,
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        feature_len,
                    )
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[: len(s)]
                    s_hat, loss, h_memory = test_model(
                        model,
                        h_memory.detach().to(device),
                        s_hat.detach(),
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        feature_len,
                    )
                s_hat = s_hat.squeeze(2)
                total_loss += loss
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, _ = test_model(
                        model, None, s, mc_return, len(s_batch), num_tau_sample, device, feature_len
                    )
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[: len(s)]
                    s_hat, loss, _ = test_model(
                        model, None, s_hat.detach(), mc_return, len(s_batch), num_tau_sample, device, feature_len
                    )
                s_hat = s_hat.squeeze(2)
                total_loss += loss
                count += 1
    # print("test loss :", total_loss.item() / count)
    if total_loss.item() / count <= best_total_loss:
        # print("Saving the best model!")
        best_total_loss = total_loss.item() / count
        # torch.save(model.state_dict(), path)
    return total_loss.item() / count, best_total_loss


def data_splitting(tensor_dataset, batch_size, features_min, features_max, device):
    # prevent division by zero in normalization
    no_need_normalization = np.where((features_min == features_max))[0]
    normalized_states = (tensor_dataset[0][0].cpu().numpy() - features_min) / (features_max - features_min)
    normalized_n_states = (tensor_dataset[0][1].cpu().numpy() - features_min) / (features_max - features_min)
    for index in no_need_normalization:
        normalized_states[:, index] = features_min[index]
        normalized_n_states[:, index] = features_min[index]
    normalized_tensor_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(normalized_states).to(device), torch.Tensor(normalized_n_states).to(device)
    )
    all_indices = np.arange(len(normalized_tensor_dataset))
    max_len = len(normalized_tensor_dataset[0][0])
    np.random.shuffle(all_indices)
    train_indices = all_indices[: int(len(all_indices) * 90 / 100)]
    test_indices = all_indices[int(len(all_indices) * 90 / 100) :]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dl = DataLoader(normalized_tensor_dataset, batch_size, sampler=train_sampler)
    test_dl = DataLoader(normalized_tensor_dataset, batch_size, sampler=test_sampler)
    return train_dl, test_dl, max_len


def train_model(model, optimizer, hx, states, target, batch_size, num_tau_sample, device, clip_value, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if hx is not None:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.reshape(target.shape[0], 1, -1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

    error_loss = T_z - z
    huber_loss = functional.smooth_l1_loss(z, T_z.detach(), reduction="none")
    if num_tau_sample == 1:
        tau = torch.arange(0, 1, 1 / 100).view(1, 100)
    else:
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

    loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    if clip_value is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    return z.squeeze(2), loss, hx


def test_model(model, hx, states, target, batch_size, num_tau_sample, device, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if hx is not None:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.reshape(target.shape[0], 1, -1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

    error_loss = T_z - z
    huber_loss = functional.smooth_l1_loss(z, T_z.detach(), reduction="none")
    if num_tau_sample == 1:
        tau = torch.arange(0, 1, 1 / 100).view(1, 100)
    else:
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

    loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
    loss = loss.mean()
    return z, loss, hx


def epsilon_decay(epsilon, num_iterations, iteration, decay_type="linear", k=0.997):
    if decay_type == "linear":
        step = 1 / (num_iterations * 2)
        return round(epsilon - step, 6)
    elif decay_type == "exponential":
        return max(k**iteration, 0.5)


def states_min_max_finder(train_dataset):
    features_min = train_dataset[0][0].min(axis=0).values.cpu().numpy()
    features_max = train_dataset[0][0].max(axis=0).values.cpu().numpy()
    return features_min, features_max


def ss_learn_model(
    model, optimizer, memory, max_len, gru_size, num_tau_sample, device, epsilon, clip_value, feature_len, has_memory
):
    total_loss = 0
    count = 0
    model.train()
    s_hat = None
    for s_batch, mc_returns in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = train_model(
                        model,
                        optimizer,
                        h_memory.detach().to(device),
                        s,
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        clip_value,
                        feature_len,
                    )
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[: len(s)]
                    s_hat, loss, h_memory = train_model(
                        model,
                        optimizer,
                        h_memory.detach().to(device),
                        s_hat.detach(),
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        clip_value,
                        feature_len,
                    )
                total_loss += loss.item()
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, _ = train_model(
                        model,
                        optimizer,
                        None,
                        s,
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        clip_value,
                        feature_len,
                    )
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[: len(s)]
                    s_hat, loss, _ = train_model(
                        model,
                        optimizer,
                        None,
                        s_hat.detach(),
                        mc_return,
                        len(s_batch),
                        num_tau_sample,
                        device,
                        clip_value,
                        feature_len,
                    )
                total_loss += loss.item()
                count += 1
    return total_loss / count


def construct_batch_data(feature_len, dataset, device, has_time_feature=False):
    episodes_states = []
    episodes_next_states = []
    episodes_len = []
    for i, episode in enumerate(dataset):
        episodes_len.append(len(episode))
    max_len = max(episodes_len) - 1
    for i, episode in enumerate(dataset):
        try:
            episode = np.array(episode).squeeze(1)
        except:
            pass
        if has_time_feature:
            # print('get rid of features added by TimeFeatureWrapper ')
            episode = episode[:, :-1]
        episodes_states.append(
            torch.Tensor(
                np.concatenate((episode[:-1, :], np.zeros((max_len - len(episode[:-1, :]), feature_len))), axis=0)
            )
        )
        episodes_next_states.append(
            torch.Tensor(
                np.concatenate((episode[1:, :], np.zeros((max_len - len(episode[1:, :]), feature_len))), axis=0)
            )
        )

    episodes_states = torch.stack(episodes_states).to(device)
    episodes_next_states = torch.stack(episodes_next_states).to(device)

    tensor_dataset = torch.utils.data.TensorDataset(episodes_states, episodes_next_states)
    return tensor_dataset


def measure_as(distribution, actual_return, input_len):
    anomaly_scores = []
    for i in range(input_len):
        anomaly_scores.append(k_nearest_neighbors(distribution[i, :], actual_return[i]))
    return np.array(anomaly_scores)


def k_nearest_neighbors(distribution, actual_return):
    neigh = NearestNeighbors(n_neighbors=distribution.shape[0])
    neigh.fit(distribution.reshape(-1, 1))
    distances, indices = neigh.kneighbors(np.array(actual_return).reshape(-1, 1))
    return distances.mean()
