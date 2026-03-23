from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC

import tyro

from tabpi.utils.wab import Wandb
import wandb


@dataclass
class Config:
    wandb: Wandb = field(default_factory=Wandb)
    max_minutes: float = 5.0
    include_running: bool = False
    delete: bool = False
    yes: bool = False


def parse_wandb_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    # W&B timestamps are typically ISO-8601 with trailing Z.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def runtime_seconds(run: wandb.apis.public.Run, include_running: bool) -> float | None:
    runtime = run.summary.get("_runtime")
    if runtime is not None:
        try:
            return float(runtime)
        except (TypeError, ValueError):
            return None

    if run.state == "running" and include_running:
        created_at = parse_wandb_datetime(getattr(run, "created_at", None))
        if created_at is None:
            return None
        return (datetime.now(UTC) - created_at).total_seconds()

    return None


def main(cfg: Config) -> None:
    threshold_seconds = cfg.max_minutes * 60.0
    api = wandb.Api()
    project_path = f"{cfg.wandb.group}/{cfg.wandb.project}"

    runs = list(api.runs(project_path))
    matches: list[tuple[wandb.apis.public.Run, float]] = []

    for run in runs:
        seconds = runtime_seconds(run, include_running=cfg.include_running)
        if seconds is None:
            continue
        if seconds < threshold_seconds:
            matches.append((run, seconds))

    if not matches:
        print(f"No runs found in {project_path} with runtime < {cfg.max_minutes:g} minutes.")
        return

    print(f"Found {len(matches)} run(s) in {project_path} with runtime < {cfg.max_minutes:g} minutes:")
    for run, seconds in matches:
        print(f"- {run.id}\t{run.name}\tstate={run.state}\truntime={seconds:.1f}s")

    if not cfg.delete:
        print("\nDry run only. Re-run with --delete to delete these runs.")
        return

    if not cfg.yes:
        response = input("\nDelete these runs permanently? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return

    deleted = 0
    for run, _ in matches:
        run.delete()
        deleted += 1
        print(f"Deleted: {run.id} ({run.name})")

    print(f"\nDeleted {deleted} run(s).")


if __name__ == "__main__":
    main(tyro.cli(Config))
