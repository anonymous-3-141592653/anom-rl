import re

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def save_yaml(dict, path):
    with open(path, "w") as f:
        OmegaConf.save(config=dict, f=f)


def load_yaml(path):
    return OmegaConf.load(path)


def flatten_dict_to_str(dic) -> str:
    if dic is None:
        return ""
    flat = []
    for k, v in dic.items():
        if isinstance(v, dict) or isinstance(v, DictConfig):
            s = k + "_<" + flatten_dict_to_str(v) + ">"
        else:
            s = f"{k}_{v}"
        flat.append(s)
    return "_".join(flat)


def atleast_2d(x):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return x


def can_cast_to_float_array(value) -> bool:
    try:
        np.array(value, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def flatten_dict(x: dict[str, list | np.ndarray | torch.Tensor]) -> torch.Tensor | np.ndarray:
    dtype = type(next(iter(x.values())))
    if dtype is np.ndarray or dtype is list:
        return np.hstack([np.asarray(v, dtype=np.float32) for v in x.values()])
    elif dtype is torch.Tensor:
        return torch.hstack([torch.as_tensor(v, dtype=torch.float32) for v in x.values()])
    else:
        raise TypeError(f"Unsupported data type: {type(dtype)}. Expected np.ndarray, list, or torch.Tensor.")


def group_dict(x) -> dict:
    """Converts a list of dictionaries to a dictionary of lists"""
    return {k: [v[k] for v in x] for k in x[0].keys()}


def ungroup_dict(x):
    """Converts a dictionary of lists to a list of dictionaries"""
    return [dict(zip(x.keys(), v)) for v in zip(*x.values())]


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """

    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def sorted_alphanum(x: list):
    return sorted(x, key=alphanum_key)
