"""Reusable evaluation helpers for model validation and environment rollout."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from tabpi.envs.env import EnvFactory
import wandb


def val_metrics(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    print("Predicting on last 10%")
    yh = model.predict(x_test)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    return {"mse": mse, "r2": r2}


def rollout(
    env: EnvFactory,
    max_steps: int,
    policy: Any,
    venv: Any,
    timer: Any,
    cfg: Any,
    overfit: bool = False,
    demo: bool = False,
    init_state: Any = None,
) -> dict[str, Any]:
    env_name = "demo" if demo else "sim"
    if overfit:
        print("Overfit")
        env.set_init_state(init_state)

    frames = []
    success = 0

    bar = tqdm(range(max_steps), desc="Rollout")

    for i in bar:
        states = np.array(env.get_state())

        with timer("fwd"):
            actions = policy(states) if not isinstance(policy, np.ndarray) else np.stack([policy[i]])  # * env.n_envs)
        with timer(env_name):
            obs, reward, done, _info = env.step(actions)

        # frames.append(obs[0]["galleryview_image"][::-1])
        frames.append(obs["agentview_image"][::-1])

        dones = done  # np.array(dones)
        success: bool = int(env.env._check_success())

        payload = {f"{env_name}/reward": reward}

        joint_pos = env.env.robots[0]._joint_positions
        eef_xpos = obs["robot0_eef_pos"]

        payload = payload | {
            **{f"{env_name}/actions{i}": wandb.Histogram(actions[..., i]) for i in range(actions.shape[-1])},
            **{f"{env_name}/joint{i}": wandb.Histogram(actions[..., i]) for i in range(joint_pos.shape[-1])},
            **{f"{env_name}/eef_{i}": wandb.Histogram(actions[..., i]) for i in range(eef_xpos.shape[-1])},
        }

        cfg.log(payload)

        desc = f"Step: {len(frames)}/{max_steps} SR: {success}"
        bar.set_description(desc)

        if success == 1:  # dones.all():
            bar.write("Task Completed!")
            break

    env.reset()

    # wandb.Video expects (T, C, H, W) for raw numpy input.
    video = np.array(frames)
    video = np.transpose(video, (0, 3, 1, 2))

    return {f"{env_name}": {"video": wandb.Video(video, fps=30, format="mp4"), "len": len(frames), "sr": success}}
