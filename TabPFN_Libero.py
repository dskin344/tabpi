from __future__ import annotations

import os

import h5py
import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.download_utils import check_libero_dataset
import numpy as np
from tabpfn import TabPFNRegressor


def extract(task_suite, task_id):
    demo_path = task_suite.get_task_demonstration(task_id)
    full_path = os.path.join("/home/dskinner2/repo/tabpi/.venv/lib/python3.13/site-packages/libero/datasets", demo_path)

    with h5py.File(full_path, "r") as f:
        # Only using demo 0 for task_id
        demo = f["data"]["demo_0"]
        print(task_suite.get_task(task_id).name)

        # Extract all the state-based features (not im
        features = np.array(demo["states"])  # (329, 47)

        # Get targets (example: predict next action)
        actions = np.array(demo["actions"])  # (329, 7)

        return features, actions


def main():
    task_suite_name = "libero_spatial"
    task_id = 0

    task_suite = benchmark.get_benchmark(task_suite_name)()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()

    if not check_libero_dataset():
        print("Datasets not found")
        # libero_dataset_download(datasets="libero_100", use_huggingface=True)
    else:
        print("Datasets found, no need to download")

    features, actions = extract(task_suite, task_id)
    print(features.shape)
    print(actions.shape)

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
    main()
