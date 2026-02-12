"""Test script for LIBERO SubprocVectorEnv integration.

This demonstrates:
1. Creating vectorized environments with SubprocVectorEnv
2. Using the EnvFactory pattern for configuration
3. Calling LIBERO-specific methods (check_success, get_sim_state, etc.)

Reference: /data/projects/tabpi/src/tabpi/utils/util.py:113-116
The EnvFactory.build() method should use this pattern when n_envs > 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tyro import cli

from tabpi.utils.util import EnvFactory, LiberoFactory


@dataclass
class Config:
    env: EnvFactory = field(default_factory=LiberoFactory)


def main(cfg: Config):
    """Test vectorized environment setup.

    This mirrors the EnvFactory.build() pattern from util.py.
    When EnvFactory.build() is called with n_envs > 1, it should do:

        from libero.libero.envs.venv import SubprocVectorEnv
        env_fns = [lambda: OffScreenRenderEnv(**env_args) for _ in range(self.n_envs)]
        return SubprocVectorEnv(env_fns)
    """

    venv = cfg.env.build()

    # Get action space from first env
    action_spaces = venv.get_env_attr("action_space")
    print(f"  action_spaces: {action_spaces}")
    action_space = action_spaces[0] if action_spaces[0] is not None else None

    # Step loop
    print(f"\nStepping for {cfg.env.max_steps} iterations...")
    for step in range(cfg.env.max_steps):
        # Sample random actions (7-dim for panda arm)
        if action_space is not None:
            actions = np.array([action_space.sample() for _ in range(cfg.env.max_envs)])
        else:
            # Fallback: create random actions (panda = 7 DOF)
            actions = np.random.randn(cfg.env.n_envs, 7) * 0.1

        obs, rewards, dones, infos = venv.step(actions)

        print(f"  Step {step + 1}:")
        print(f"    obs shape: {obs.shape}")
        print(f"    rewards: {rewards}")
        print(f"    dones: {dones}")
        print(f"    env_ids: {[info.get('env_id') for info in infos]}")

    # Test LIBERO-specific methods
    print("\nTesting LIBERO-specific methods...")

    # check_success
    success = venv.check_success()
    print(f"✓ check_success(): {success}")

    # get_sim_state
    states = venv.get_sim_state()
    print(f"✓ get_sim_state(): {len(states)} states retrieved")
    if states:
        print(f"  state[0] shape: {states[0].shape}")

    # Cleanup
    print("\nClosing environments...")
    venv.close()
    print("✓ Done!")


if __name__ == "__main__":
    main(cli(Config))
