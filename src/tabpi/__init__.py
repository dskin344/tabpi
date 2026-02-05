from __future__ import annotations

from pathlib import Path

# tmp = Path(__file__).parent.resolve()
ROOT = Path(__file__).parents[2].resolve()
WANDB = Path().home() / "wandb" / "tabpi"

print(WANDB)
