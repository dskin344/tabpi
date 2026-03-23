from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
from rich import print
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
from tqdm import tqdm
import tyro

from tabpi.envs.env import EnvFactory
from tabpi.envs.robosuite import RoboSuiteFactory
from tabpi.models.tabpfn_policy import ModelPolicy
from tabpi.utils.data import extract, shuffle
from tabpi.utils.eval import rollout
from tabpi.utils.timer import Timer
from tabpi.utils.wab import Wandb
import wandb


@dataclass
class Config:
    fit: float = 0.10
    selection: str = "steps"
    n_runs: int = 5  # for non VecEnv. sequential rollouts

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=RoboSuiteFactory)
    debug: bool = False

    action: Literal["absolute", "relative", "next_obs/robot0_joint_pos"] = "relative"
    n_estimators: int = 4

    def __post_init__(self):
        if self.debug:
            self.wandb.use = False
        if self.action in ("relative", "absolute"):
            self.env.controller = "OSC_POSE"
        elif self.action == "next_obs/robot0_joint_pos":
            self.env.controller = "JOINT_POSITION"
            self.env.file_name = "low_dim.hdf5"


def main(cfg: Config):
    venv = cfg.env.build(cfg.action)

    cfg.env.check_download()

    raw_data: dict[str, Any] = cfg.env.load_data()
    features, actions = extract(raw_data, cfg.selection, cfg.env.demo, cfg.action)
    if cfg.action == "next_obs/robot0_joint_pos":
        _, gripper_actions = extract(raw_data, cfg.selection, cfg.env.demo)
        actions = np.concatenate([actions, gripper_actions[:, -1:]], axis=1)

    print(features.shape, actions.shape)

    x_fit, _x_test, y_fit, _y_test = shuffle(cfg.fit, features, actions, None)

    print(x_fit.shape, y_fit.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        n_estimators=cfg.n_estimators,
        device=device,
        # device='cuda:0',:
        # n_preprocessing_jobs=act_dim*2,
        # memory_saving_mode=False,
        inference_precision="autocast",
    )

    t = Timer()

    print(f"Fitting on {cfg.fit * 100}%")
    with t("fit"):
        model.fit(x_fit, y_fit)

    """with t("val"):
        val = val_metrics(model, x_test, y_test)"""

    print("Initializing Wandb")
    cfg.wandb.initialize(cfg)

    if cfg.env.overfit:
        print(f"Running demo_{cfg.env.demo}")
        demo_result = rollout(cfg.env, cfg.env.horizon, actions, venv, t, cfg.wandb, cfg.env.overfit, True, features[0])
        cfg.wandb.log(demo_result)

    pi = ModelPolicy(model)

    roll = partial(rollout, cfg.env, cfg.env.horizon, pi, venv, t, cfg.wandb, cfg.env.overfit, False, features[0])
    for i in tqdm(range(cfg.n_runs), desc="Rollouts", leave=False):
        result = roll()
        cfg.wandb.log(result)

    times = t.get_average_times()

    # wandb.define_metric("val", step_metric="step")
    wandb.define_metric("times", step_metric="step")

    metrics = {  # "val": val,
        "times": times,
    }

    cfg.wandb.log(metrics)

    venv.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
