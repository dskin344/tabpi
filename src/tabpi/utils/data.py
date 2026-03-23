"""Data loading and preprocessing (HDF5, demonstrations)."""

from __future__ import annotations

import h5py
import numpy as np
from rich import print


def h5_to_tree(path: str) -> dict[str, h5py.Group]:
    def read_node(node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        if isinstance(node, h5py.Group):
            return {k: read_node(node[k]) for k in node}
        raise TypeError(type(node))

    with h5py.File(path, "r") as f:
        return read_node(f)


def extract(tree: dict, selection: str, demo, action_key: str = "actions") -> tuple[np.ndarray, np.ndarray]:
    demos = tree["data"]

    def _get(d, key):
        for part in key.split("/"):
            d = d[part]
        return d

    if demo is None:
        sa_by_demo = {k: (d["states"], _get(d, action_key)) for k, d in demos.items()}

    elif isinstance(demo, int):
        key = f"demo_{demo}"
        sa_by_demo = {key: (demos[key]["states"], _get(demos[key], action_key))}

    keys = sa_by_demo.keys()

    if selection == "steps":
        states = np.concatenate([sa_by_demo[k][0] for k in keys], axis=0)
        actions = np.concatenate([sa_by_demo[k][1] for k in keys], axis=0)

    elif selection == "demos":  # Can't use np.stack since different dimensions
        states = np.array([sa_by_demo[k][0] for k in keys], dtype=object)
        actions = np.array([sa_by_demo[k][1] for k in keys], dtype=object)

    return states, actions


def shuffle(
    training: float, features, actions, shuffle: str, test: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if shuffle is not None:
        x = "Step Shuffle" if shuffle == "step" else "Demo Shuffle"
        print(x)

        rng = np.random.default_rng(seed=42)
        indices = np.arange(features.shape[0])

        rng.shuffle(indices)
        features = features[indices]
        actions = actions[indices]
    else:
        print("No Shuffle")

    x_fit, x_test, y_fit, y_test = split(training, test, features, actions)
    return x_fit, x_test, y_fit, y_test


# Assume test param is for the percentage of tail end of data you want to test on
def split(training: float, test: float, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(x.shape[0] * training)
    n_test = int(x.shape[0] * test)

    # ndarray with state values
    if x.dtype == float and y.dtype == float:
        x_train, x_test = x[:n_train], x[-n_test:]
        y_train, y_test = y[:n_train], y[-n_test:]

    # nparray with ndarrays
    elif x.dtype == np.ndarray and y.dtype == np.ndarray:
        indices = np.arange(x.shape[0])
        train = indices[:n_train]
        test = indices[-n_test:]

        x_train = np.concatenate([x[i] for i in train])
        y_train = np.concatenate([y[i] for i in train])
        x_test = np.concatenate([x[i] for i in test])
        y_test = np.concatenate([y[i] for i in test])

    return x_train, x_test, y_train, y_test
