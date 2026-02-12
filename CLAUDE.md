# CLAUDE.md

## Project Overview

**TabPi** is a reinforcement/imitation learning project that integrates TabPFN (a pre-trained foundation model for tabular data) with robotic manipulation environments. The goal is to train policies for complex robotic tasks (e.g., LIBERO benchmarks) by leveraging TabPFN's regression capabilities for fast policy learning with minimal training data.

**Key workflow**: Extract state-action pairs from demonstration HDF5 files → Train TabPFN regression models → Deploy learned policies on MuJoCo-based robot simulations.

## Package Layout

```
src/tabpi/
├── __init__.py              # Package initialization, path constants
├── config.py                # Configuration dataclasses (Config, WandbConfig, etc.)
├── data.py                  # Data loading and preprocessing (HDF5, demonstrations)
├── train.py                 # Training loops and procedures
├── eval.py                  # Evaluation and policy testing
│
├── models/                  # Core model implementations
│   ├── __init__.py
│   ├── tabpfn_policy.py    # TabPFN-based policy wrapper/multi-output handler
│   └── base.py             # Base policy interface
│
├── envs/                    # Environment wrappers and utilities
│   ├── __init__.py
│   ├── libero.py           # LIBERO environment integration
│   ├── cartpole.py         # Gym/Gymnasium wrappers
│   └── base.py             # Base environment interface
│
└── utils/                   # Utilities
    ├── __init__.py
    ├── logging.py          # W&B integration, logging helpers
    └── video.py            # Video recording and rendering

experiments/                # Standalone reproducible experiments
├── libero_policy.py        # LIBERO policy training & evaluation
├── cartpole_baseline.py    # CartPole baseline
└── __init__.py

tests/                      # Unit and integration tests
├── conftest.py
├── unit/
│   ├── test_data.py
│   └── test_models.py
└── integration/
    └── test_libero.py

docs/                       # Project documentation
├── DESIGN.md              # Design decisions and patterns
├── TYRO.md                # CLI argument handling conventions
└── ZEN.md                 # Code philosophy
```

**Key principles**:
- **Core logic** in `src/tabpi/` (models, data, training, eval)
- **Experiments** separate in `experiments/` (not in src/, easier to modify without breaking package)
- **Feature-based** organization (models/, envs/, utils/) not layer-based
- **Tests** co-located by purpose (unit/ for isolated tests, integration/ for end-to-end)

## Build & Run

- **Package manager**: `uv` (with `uv_build` backend)
- **Python**: >=3.11 (target 3.11)
- **Install deps**: `uv sync --group dev`
- **Run scripts**: `uv run script.py` (never `python` directly or activate venv)
- **JAX**: 0.5.3 with CUDA 12

## Linting & Formatting

- **Do not lint or use ruff** — it wastes tokens. Skip `ruff check .` and `ruff format .`
- Pre-commit hooks may run ruff, but don't manually invoke linting for code review or polish
- **Required import**: every `.py` file must have `from __future__ import annotations`
- **Type checking**: `uv run pyright` (basic mode)
- **isort**: via ruff, force-sort-within-sections, no order-by-type

## Testing

- **Run**: `uv run pytest` (defaults: `-q --maxfail=1 --disable-warnings`)
- **With coverage**: `uv run pytest --cov` (fail_under=80, branch coverage)
- **Markers**: `unit` (fast), `integration` (requires GPU), `multinode`
- **Frameworks**: pytest + hypothesis (property-based tests)
- Tests use real JAX/TF/NumPy -- no mocking of numerical backends
- Seed explicitly: `np.random.default_rng(seed)`, `jax.random.PRNGKey(n)`
- Shape assertions are the primary contract (`assert output.shape == (...)`)

## Code Style

### General
- Concise code and docstrings; purpose should be clear from reading the code
- Short, meaningful variable names (avoid long names that clutter)
- Keep functions small and single-purpose (cyclomatic complexity <= 8, prefer 4-5)
- DRY -- no duplicated code
- Flat over nested; avoid excessive nesting
- Feature-based file organization, not layer-based
- OOP patterns for components; decorators/wrappers when appropriate
- `config.create(*args, **kwargs)` pattern for component instantiation

### Python
- Type hints on public API; `X | None` union syntax (via `__future__` annotations)
- Google-style docstrings with inline field comments for dataclasses
- snake_case functions/variables, PascalCase classes, UPPER_CASE constants
- Private functions prefixed with `_`
- `functools.partial` for currying, `einops.rearrange` for tensor reshaping

### JAX/Flax
- `nn.Module` with `@nn.compact` for neural network components
- `@flax.struct.dataclass` for PyTree-compatible data containers
- `module.apply({"params": params}, ...)` for inference
- `jax.tree.map()` for tree operations; nested dict trees throughout
- `@partial(jax.jit, static_argnames=(...))` for JIT
- `jax.lax.scan` for iterative processes, `jax.vmap` for batching
- `ModuleSpec` system (`crossformer/utils/spec.py`) for config-driven instantiation

## Git Conventions

- Main branch: `main`
- Feature branches: `feat-*` or topic names
- Commit messages: short, lowercase, descriptive (conventional commits optional)
- CI: pre-commit hooks only (GitHub Actions on PRs and pushes to main)
- No force-push to main

## Things to Avoid

- Excessive nesting
- Long variable names
- Long functions with multiple responsibilities
- Duplicated code
- Mocking JAX/NumPy in tests (use real backends)
- Happy-path-only tests (write negative tests, edge cases)
- Adding features or refactoring beyond what was asked
- Shims unless explicitly needed
