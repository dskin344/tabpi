from __future__ import annotations

import os
from pathlib import Path

import h5py
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from tabpfn import TabPFNRegressor


def check_download(suites, task_suite_name):
    suite_dir = suites.joinpath(task_suite_name)

    if os.path.exists(suite_dir):
        print("Datasets found:")
        t_names = [f.stem for f in suite_dir.glob("*.hdf5")]
        for index, name in enumerate(t_names):
            print(index, ": ", name)
    else:
        print("Task suite datasets not found. Downloading now")
        libero_dataset_download(datasets=task_suite_name, use_huggingface=True)


def h5_to_tree(path: str):
    def read_node(node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        if isinstance(node, h5py.Group):
            return {k: read_node(node[k]) for k in node}
        raise TypeError(type(node))

    with h5py.File(path, "r") as f:
        return read_node(f)


def extract(suites, task_suite, task_id):
    suite: str = task_suite.get_task_demonstration(task_id)
    full_path: Path = suites / suite

    tree = h5_to_tree(full_path)
    demos = tree["data"]

    sa_by_demo = {k: (d["states"], d["actions"]) for k, d in demos.items()}
    keys = sa_by_demo.keys()

    states = np.concatenate([sa_by_demo[k][0] for k in keys], axis=0)
    actions = np.concatenate([sa_by_demo[k][1] for k in keys], axis=0)

    return states, actions


class MyMultiTPFN:
    def __init__(self, dim: int):
        self.dim = dim
        self.models = [TabPFNRegressor() for _ in range(dim)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        print("Fitting...")
        for i in range(self.dim):
            self.models[i].fit(x, y[:, i])  # Train on dimension i
        print("Done fitting")

    def predict(self, x: np.ndarray) -> np.ndarray:
        print("Predicting...")
        predictions = []
        for i in range(self.dim):
            pred = self.models[i].predict(x)
            predictions.append(pred)
        print("Done Predicting")
        return np.column_stack(predictions)


class MyLibero:
    def __init__(self, task_suite_name: str, task_id: int):
        self.task_suite = benchmark.get_benchmark(task_suite_name)()
        self.task_id = task_id
        self.current_task = None
        self.env = None

    def get_task_suite(self):
        return self.task_suite

    def get_env(self):
        return self.env

    def build_env(self):
        self.current_task = self.task_suite.get_task(self.task_id)
        bddl_file_path = self.task_suite.get_task_bddl_file_path(self.task_id)

        print(f"Using task: {self.current_task.name}")

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
        }

        self.env = OffScreenRenderEnv(**env_args)
        self.env.reset()
