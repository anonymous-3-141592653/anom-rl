import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.transforms import transforms


def plot_episode(ep_values: np.ndarray, title=""):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        with plt.rc_context({"font.size": 5}):
            assert ep_values.ndim == 2
            n_axes = ep_values.shape[1]
            fig, axes = plt.subplots(n_axes, 1, figsize=(3, n_axes))
            fig.suptitle(title, fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            if n_axes == 1:
                axes = [axes]
            for i in range(n_axes):
                axes[i].plot(ep_values[:, i])
                axes[i].set_title(f"Axis {i}")
            return fig


def plot_img_targets_predictions(targets, predictions, n_steps=10):
    def _compute_delta_img(img_1, img_2, threshold=0.1):
        difference = torch.abs(img_1 - img_2)
        mask = difference.sum(0) > threshold
        diff_img = torch.zeros_like(img_1)
        diff_img[0] += mask  # add to red channel
        return diff_img

    assert len(targets) == len(predictions)
    assert n_steps <= len(targets)

    fig, axs = plt.subplots(n_steps, 3, figsize=(3, n_steps), tight_layout=True)
    for i in range(n_steps):
        axs[i, 0].set_title(f"step {i}", fontsize=5)
        axs[i, 0].imshow(targets[i].permute(1, 2, 0).float().clamp(0, 1))
        axs[i, 0].axis("off")
        axs[i, 1].imshow(predictions[i].permute(1, 2, 0).float().clamp(0, 1))
        axs[i, 1].axis("off")
        axs[i, 2].imshow(_compute_delta_img(targets[i], predictions[i]).permute(1, 2, 0).float().clamp(0, 1))
        axs[i, 2].axis("off")

    return fig
