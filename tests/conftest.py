from __future__ import annotations

import jax.random
import numpy as np
import pytest


@pytest.fixture
def rng_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def np_rng(rng_seed):
    """NumPy random number generator with fixed seed."""
    return np.random.default_rng(rng_seed)


@pytest.fixture
def jax_key(rng_seed):
    """JAX PRNG key with fixed seed."""
    return jax.random.PRNGKey(rng_seed)


@pytest.fixture
def sample_state():
    """Sample observation/state vector for testing."""
    return np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)


@pytest.fixture
def sample_action():
    """Sample action vector for testing."""
    return np.array([0.3], dtype=np.float32)


@pytest.fixture
def sample_batch_states(np_rng):
    """Batch of sample states for testing."""
    batch_size = 8
    state_dim = 4
    return np_rng.standard_normal(size=(batch_size, state_dim)).astype(np.float32)


@pytest.fixture
def sample_batch_actions(np_rng):
    """Batch of sample actions for testing."""
    batch_size = 8
    action_dim = 1
    return np_rng.standard_normal(size=(batch_size, action_dim)).astype(np.float32)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests (may require GPU)")
    config.addinivalue_line("markers", "multinode: multinode tests")
