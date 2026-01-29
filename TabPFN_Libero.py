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


class MyMultiTPFN:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.models = [TabPFNRegressor(**kwargs) for _ in range(dim)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        for i in range(self.dim):
            self.models[i].fit(x, y[:, i])  # Train on dimension i

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = []
        for i in range(self.dim):
            pred = self.models[i].predict(x)
            predictions.append(pred)
        return np.column_stack(predictions)


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

    # Fit the model here
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    actions = actions[indices]

    n_fit = int(features.shape[0] * 0.80)
    n_test = features.shape[0] - n_fit
    x_fit, x_test = features[:n_fit], features[n_fit:]
    y_fit, y_test = actions[:n_fit], actions[n_fit:]

    """regressor = TabPFNRegressor()
    regressor.fit(x_train, y_train)
    prediction_policy = regressor.predict(x_test)"""

    policy = MyMultiTPFN(dim=7)
    # fit the policy
    policy.fit(x_fit, y_fit)
    # predict the policy
    yh = policy.predict(x_test)

    # import mse metric from tabpfn.metrics
    # Evaluate the model
    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    """
    prediction_policy = []
    for i in range(7):
        regressor = TabPFNRegressor()
        regressor.fit(x_fit, y_fit[:, i])  # Train on dimension i
        pred = regressor.predict(x_test)
        prediction_policy.append(pred)
    """

    # Combine into (69, 7) shape
    # action_policy = np.column_stack(prediction_policy)

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
    thing = env.reset()
    print(spec(thing))

    frames = []
    done, step, max_steps = False, 0, 500
    while not done and step < max_steps:
        # action = action_policy[step]
        print(spec(obs))
        action = policy.predict(obs.reshape(1, -1))
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
