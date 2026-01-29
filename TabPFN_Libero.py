from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio
import jax
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.download_utils import check_libero_dataset
import numpy as np
from rich import print
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
    # optional: post-process (example: cast all float64 -> float32)
    # tree = jtu.tree_map(lambda x: x.astype(np.float32) if x.dtype == np.float64 else x, tree)
    demos = tree["data"]
    d0 = tree["data"]["demo_0"]
    # print(spec(demos))

    sa_by_demo = {k: (d["states"], d["actions"]) for k, d in demos.items()}
    keys = sorted(sa_by_demo.keys())

    states = np.concatenate([sa_by_demo[k][0] for k in keys], axis=0)
    actions = np.concatenate([sa_by_demo[k][1] for k in keys], axis=0)

    # states, actions = d0['states'], d0['actions']
    return states, actions


@dataclass
class Config:
    task: str | None = None


def main(cfg: Config):
    task_suite_name = "libero_spatial"
    task_id = 0

    task_suite = benchmark.get_benchmark(task_suite_name)()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()

    if not check_libero_dataset():
        print("Datasets not found")
        # libero_dataset_download(datasets="all", use_huggingface=True)
    else:
        print("Datasets found, no need to download")

    features, actions = extract(task_suite, task_id)
    print(features.shape)
    print(actions.shape)
    quit()

    # Fit the model here
    n_train = int(features.shape[0] * 0.80)
    max_steps = features.shape[0] - n_train
    x_train, x_test = features[:n_train], features[n_train:]
    y_train, y_test = actions[:n_train], actions[n_train:]

    """regressor = TabPFNRegressor()
    regressor.fit(x_train, y_train)
    prediction_policy = regressor.predict(x_test)"""

    prediction_policy = []
    for i in range(7):
        regressor = TabPFNRegressor()
        regressor.fit(x_train, y_train[:, i])  # Train on dimension i
        pred = regressor.predict(x_test)
        prediction_policy.append(pred)

    # Combine into (69, 7) shape
    action_policy = np.column_stack(prediction_policy)

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

    frames = []
    done, step = False, 0
    while not done and step < max_steps:
        action = action_policy[step]
        obs, reward, done, _info = env.step(action)

        frames.append(obs["galleryview_image"][::-1])

        print(f"step={step}, reward={reward}")

        if done:
            print("Resetting the env")
            obs = env.reset()

        step += 1

    env.close()

    # Save video
    imageio.mimsave("Libero.mp4", frames, fps=5)
    print("Libero.mp4 saved")


if __name__ == "__main__":
    main(tyro.cli(Config))
