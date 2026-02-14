from __future__ import annotations
import torch

from dataclasses import dataclass, field
from pathlib import Path
import time

import imageio
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import tyro
import wandb

import libero
from tabpi.utils.deco import avgtime
from tabpi.utils.util import check_download, EnvFactory, extract, LiberoFactory
from tabpi.wab import Wandb

data_dir = Path(libero.__file__).parents[0] / "datasets"


@dataclass
class Config:
    task_suite: str = "libero_object"
    training: float = 0.10

    task_id: int = 0
    steps: int = 10
    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)


def main(cfg: Config):
    print(data_dir)
    check_download(data_dir, cfg.task_suite)

    raw: dict[str, Any] = cfg.env.load_data(data_dir)
    features, actions = extract(raw)
    venv = cfg.env.build()
    _ = venv.reset()

    print(features.shape)
    print(actions.shape)

    # Fitting Global Step Shuffle
    rng = np.random.default_rng(seed=42)

    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    features = features[indices]
    actions = actions[indices]

    n_fit = int(features.shape[0] * cfg.training)
    n_test = int(features.shape[0] * 0.10)
    x_fit, x_test = features[:n_fit], features[n_test:]
    y_fit, y_test = actions[:n_fit], actions[n_test:]
    print("Globally Shuffled")

    act_dim = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        n_estimators=act_dim,
        device=device,
        # device='cuda:0',
        fit_mode="fit_preprocessors", # cannot use batched yet
        # n_preprocessing_jobs=act_dim*2,
        # memory_saving_mode=False,
        inference_precision="autocast",
    )

    """
# If multiple GPUs are detected, wrap the model in DataParallel
        if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model)
            model.executor_.model = torch.nn.DataParallel(model.executor_.model)
            # model.executor_.model.to(device) # from https://github.com/PriorLabs/TabPFN/issues/215

        model.executor_.model.to(device)
    """


    print(f"Fitting on {cfg.training * 100}%")
    start = time.time()
    model.fit(x_fit, y_fit)
    end = time.time()
    fit_time = end - start
    print(f"Done fitting in {fit_time} seconds")

    print("Predicting on last 10%")

    """
    predict = avgtime(10)(model.predict)
    yh = predict(x_test)

    print("Initializing Wandb")
    # run = cfg.wandb.initialize(cfg)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    quit()
    """

    frames = []
    total_time = 0
    done, steps, max_steps, reward = False, 0, cfg.steps, 0

    vid_path = "ObsVids/"
    dir_path = Path(vid_path)
    dir_path.mkdir(exist_ok=True)

    while not done and steps < max_steps:
        steps += 1
        states = np.array(venv.get_sim_state())
        print(f"Type: {type(states)}, Value: {states.shape}")

        # Track inference time
        start = time.time()
        # for states in states:
        actions = model.predict(states)
        end = time.time()
        print(f'time for {act_dim} actions: {end - start} seconds')

        iteration_time = end - start
        total_time += iteration_time

        print(actions.shape)
        obs, env_reward, done, _info = venv.step(actions)

        # @dskinner TODO
        # frames.append(obs["galleryview_image"][::-1])

        print(f"steps={steps}")
        print(f"Inference time={iteration_time}")

        reward += env_reward

        # step() sets done=self._check_success()
        if done:
            print("Task completed successfully")
            break

        # sucesses = rewards.sum(axis=-1) # did any step have success?
        # sr = successes.mean()

    avg_time = total_time / steps
    venv.close()

    # Save video
    imageio.mimsave(f"{vid_path}{int(cfg.training * 100)}%{task_names[cfg.task_id]}All.mp4", frames, fps=30)
    print(f"{vid_path}{int(cfg.training * 100)}%{task_names[cfg.task_id]}All.mp4 saved")

    cfg.wandb.log(
        {
            "Fit Time": fit_time,
            "MSE": mse,
            "R^2": r2,
            "Average Inference Time": avg_time,
            "Reward": reward,
            "Steps until done": steps,
            f"sim/video_{cfg.training * 100}%": wandb.Video(
                f"{vid_path}{int(cfg.training * 100)}%{task_names[cfg.task_id]}All.mp4", format="mp4"
            ),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
