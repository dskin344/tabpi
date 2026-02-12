from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, TypeAlias

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


def extract(tree: dict) -> tuple[np.ndarray, np.ndarray]:
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


Env: TypeAlias = Any


@dataclass
class EnvFactory:
    pass


@dataclass
class LiberoFactory(EnvFactory):
    suite: str = "libero_object"  # used to select group of envs
    id: int = 0
    max_steps: int | None = None  # TODO fix

    task: str = field(init=False)  # used to search for dataset name

    def __post_init__(self):
        bench = self.get_benchmark(self.suite)
        self.task = bench.get_task(self.id)

    def get_benchmark(self, suite: str) -> benchmark.Benchmark:
        return benchmark.get_benchmark(self.suite)()

    def build(self) -> Env:
        bench = self.get_benchmark(self.suite)

        bddl_file_path: Path = bench.get_task_bddl_file_path(self.id)

        print(f"Using task: {self.task.name}")

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
            # TODO max steps ...
        }

        env = OffScreenRenderEnv(**env_args)
        return env

    def get_data_path(suites_root: Path):
        bench = self.get_benchmark(self.suite)
        demo_path: str = bench.get_task_demonstration(self.id)
        full_path: Path = suites_root / demo_path
        return full_path

    def load_data(suites_root: Path):
        data_path = self.get_data_path(suites_root)
        tree = h5_to_tree(data_path)
        return tree
