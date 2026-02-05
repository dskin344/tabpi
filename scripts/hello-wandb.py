from __future__ import annotations

from dataclasses import dataclass, field

import tyro

from tabpi.wab import Wandb


@dataclass
class Config:
    wandb: Wandb = field(default_factory=Wandb)


def main(cfg: Config):
    run = cfg.wandb.initialize(cfg)
    cfg.wandb.log({"test": {"mse": 0.5, "r2": 0.8}})


if __name__ == "__main__":
    main(tyro.cli(Config))
