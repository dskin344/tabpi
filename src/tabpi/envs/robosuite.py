"""Robotsuite environment integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import robomimic
import robosuite as suite
from robosuite.controllers import load_controller_config

from tabpi.utils.data import h5_to_tree

from .env import EnvFactory


@dataclass
class RoboSuiteFactory(EnvFactory):
    task: str = "Lift"
    # n_envs: int = 1 # for vec env ... not supported yet
    vectorized: bool = False
    horizon: int = 150
    controller: str = "OSC_POSE"
    file_name: str = "demo.hdf5"

    overfit: bool = False
    demo: int | None = None

    def __post_init__(self):
        self.suites_path = (Path(robomimic.__file__).parents[0] / "../datasets" / self.task.lower()).resolve()

        if self.overfit and self.demo is None:
            self.demo = 0

    def build(self, action):
        controller_config = load_controller_config(default_controller=self.controller)
        if action == "absolute":
            print("Using absolute OSC space")
            controller_config["control_delta"] = False
        elif action == "next_obs/robot0_joint_pos":
            print("Using absolute joint space")
            controller_config["input_type"] = "absolute"

        env_kwargs = {
            "env_name": self.task,
            "robots": "Panda",
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": True,
            "camera_names": "agentview",
            "camera_heights": 720,
            "camera_widths": 1280,
            "control_freq": 20,
            "horizon": self.horizon,
            "reward_shaping": True,
        }

        self.env = suite.make(**env_kwargs)
        return self.env

    def load_data(self):
        data_path = self.suites_path / "ph" / f"{self.file_name}"

        tree = h5_to_tree(data_path)
        return tree

    def check_download(self):
        if os.path.exists(self.suites_path):
            print("Datasets found:")
            t_names = [f.relative_to(self.suites_path.parent) for f in self.suites_path.glob("**/*.hdf5")]
            for index, name in enumerate(t_names):
                print(index, ": ", name)
        else:
            print(f"{self.task} dataset not found. Downloading now")
            script_path = Path(robomimic.__file__).parent / "scripts/download_datasets.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--tasks",
                self.task.lower(),
                "--dataset_types",
                "ph",
                "--hdf5_types",
                "low_dim",
            ]
            subprocess.run(cmd, check=True)

    def set_init_state(self, init_state):
        self.env.sim.set_state_from_flattened(init_state)
        self.env.sim.forward()

    def get_state(self):
        state_data = self.env.sim.get_state().flatten().reshape(1, -1)
        return state_data

    def step(self, action):
        action = action.flatten()
        return self.env.step(action)

    def reset(self):
        self.env.reset()
