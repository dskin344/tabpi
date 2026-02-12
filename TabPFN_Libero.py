from __future__ import annotations

from dataclasses import dataclass
import time

import imageio
import libero
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
import tyro

from utils.util import *
import wandb

suites = Path(libero.__file__).parents[0] / "datasets"


@dataclass
class Config:
    task_suite: str
    training: float

    task_id: int = 0
    steps: int = 1000


def main(cfg: Config):
    libero = MyLibero(cfg.task_suite, cfg.task_id)
    task_names = libero.get_task_suite().get_task_names()

    check_download(suites, cfg.task_suite)

    features, actions = extract(suites, libero.get_task_suite(), cfg.task_id)
    print(features.shape)
    print(actions.shape)

    # Fitting Global Step Shuffle
    rng = np.random.default_rng(seed=42)

    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    features = features[indices]
    actions = actions[indices]

    libero.build_env()

    wandb.init(
        project=f"{cfg.task_suite}; {task_names[cfg.task_id]} ",
        name=f"{cfg.training * 100}%",
        config={"training": cfg.training},
    )

    n_fit = int(features.shape[0] * cfg.training)
    n_test = int(features.shape[0] * 0.10)
    x_fit, x_test = features[:n_fit], features[n_test:]
    y_fit, y_test = actions[:n_fit], actions[n_test:]

    policy = MyMultiTPFN(dim=7)

    start = time.time()
    policy.fit(x_fit, y_fit)
    end = time.time()

    fit_time = end - start
    yh = policy.predict(x_test)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    frames = []
    total_time = 0
    done, steps, max_steps, reward = False, 0, cfg.steps, 0

    vid_path = "ObsVids/"
    dir_path = Path(vid_path)
    dir_path.mkdir(exist_ok=True)

    while not done and steps < max_steps:
        steps += 1
        env_state = libero.get_env().get_sim_state()

        # Track inference time
        start = time.time()
        action = policy.predict(env_state.reshape(1, -1))
        end = time.time()

        iteration_time = end - start
        total_time += iteration_time

        action = np.concatenate(action, axis=0)
        obs, env_reward, done, _info = libero.get_env().step(action)

        frames.append(obs["galleryview_image"][::-1])

        print(f"steps={steps}")
        print(f"Inference time={iteration_time}")

        reward += env_reward

        # step() sets done=self._check_success()
        if done:
            print("Task completed successfully")
            break

    avg_time = total_time / steps
    libero.get_env().close()

    # Save video
    imageio.mimsave(f"{vid_path}{int(cfg.training * 100)}%{task_names[cfg.task_id]}All.mp4", frames, fps=30)
    print(f"{vid_path}{int(cfg.training * 100)}%{task_names[cfg.task_id]}All.mp4 saved")

    wandb.log(
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
