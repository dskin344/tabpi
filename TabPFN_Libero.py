from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import h5py
import imageio
import jax
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.utils.download_utils
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import tyro

suites = Path(libero.__file__).parents[0] / "datasets"


def h5_to_tree(path: str):
    def read_node(node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        if isinstance(node, h5py.Group):
            return {k: read_node(node[k]) for k in node}
        raise TypeError(type(node))

    with h5py.File(path, "r") as f:
        return read_node(f)


def spec(x: dict):
    return jax.tree.map(lambda y: y.shape, x)


def extract(task_suite, task_id):
    suite: str = task_suite.get_task_demonstration(task_id)
    full_path: Path = suites / suite

    tree = h5_to_tree(full_path)
    demos = tree["data"]

    sa_by_demo = {k: (d["states"], d["actions"]) for k, d in demos.items()}
    keys = sa_by_demo.keys()

    states = np.concatenate([sa_by_demo[k][0] for k in keys], axis=0)
    actions = np.concatenate([sa_by_demo[k][1] for k in keys], axis=0)

    return states, actions


def check_download(task_suite_name):
    suite_dir = suites.joinpath(task_suite_name)

    if os.path.exists(suite_dir):
        print("Datasets found:")
        t_names = [f.stem for f in suite_dir.glob("*.hdf5")]
        for index, name in enumerate(t_names):
            print(index, ": ", name)
    else:
        print("Task suite datasets not found. Downloading now")
        libero_dataset_download(datasets=task_suite_name, use_huggingface=True)


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


@dataclass
class Config:
    suite: str = "libero_spatial"
    task_id: int = 0
    training: float = 0.05
    steps: int = 500


def main(cfg: Config):
    task_suite_name = cfg.suite
    task_id = cfg.task_id

    task_suite = benchmark.get_benchmark(task_suite_name)()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()

    check_download(task_suite_name)

    features, actions = extract(task_suite, task_id)
    print(features.shape)
    print(actions.shape)

    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    actions = actions[indices]

    n_fit = int(features.shape[0] * cfg.training)
    x_fit, x_test = features[:n_fit], features[n_fit:]
    y_fit, y_test = actions[:n_fit], actions[n_fit:]

    policy = MyMultiTPFN(dim=7)
    policy.fit(x_fit, y_fit)
    yh = policy.predict(x_test)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    task = task_suite.get_task(task_id)
    bddl_file_path = task_suite.get_task_bddl_file_path(task_id)
    print(f"Using task: {task_names[task_id]}")

    # Create environment arguments dictionary
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": 720,  # HD resolution
        "camera_widths": 1280,
        "camera_names": "galleryview",
    }

    # Create environment
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    frames = []
    total_time = 0
    done, step, max_steps = False, 0, cfg.steps

    dir_path = Path(f"ObsVids/{cfg.training}{task_names[task_id]}")
    dir_path.mkdir(exist_ok=False)

    while not done and step < max_steps:
        env_state = env.get_sim_state()

        # print(f"Env State Shape: {env_state.shape}")
        start = time.time()
        action = policy.predict(env_state.reshape(1, -1))
        end = time.time()
        action = np.concatenate(action, axis=0)
        # print(action.shape)

        iteration_time = start - end
        total_time += iteration_time
        obs, _reward, done, _info = env.step(action)

        frames.append(obs["galleryview_image"][::-1])

        print(f"step={step}")
        print(f"Inference time={iteration_time}")

        if done:
            print("Resetting the env")
            env.reset()

        step += 1

        if step % 10 == 0:
            imageio.mimsave(f"ObsVids/{cfg.training}{task_names[task_id]}/Step{step}.mp4", frames, fps=5)
            avg_time = iteration_time / (step + 1)
            print(f"Average Inference Time={avg_time}")
            print(f"{cfg.training}{task_names[task_id]} Step{step}.mp4 saved")

        step += 1

    env.close()

    # Save video
    imageio.mimsave(f"ObsVids/{cfg.training}{task_names[task_id]}/All.mp4", frames, fps=5)
    print(f"ObsVids/{cfg.training}{task_names[task_id]}/All.mp4 saved")


if __name__ == "__main__":
    main(tyro.cli(Config))
