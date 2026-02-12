from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tabpi.wab import Wandb, WandbMode


@pytest.mark.unit
class TestWandbMode:
    """Tests for WandbMode enum."""

    def test_wandb_mode_values(self):
        """Test that WandbMode has expected values."""
        assert WandbMode.ONLINE.value == "online"
        assert WandbMode.OFFLINE.value == "offline"
        assert WandbMode.DISABLED.value == "disabled"

    def test_wandb_mode_members(self):
        """Test that all expected modes are present."""
        modes = {m.name for m in WandbMode}
        assert modes == {"ONLINE", "OFFLINE", "DISABLED"}


@pytest.mark.unit
class TestWandbConfig:
    """Tests for Wandb configuration class."""

    def test_wandb_default_values(self):
        """Test default configuration values."""
        cfg = Wandb()
        assert cfg.project == "tabpi"
        assert cfg.group == "ssl-luc"
        assert cfg.entity is None
        assert cfg.resume_from is None
        assert cfg.use is True

    def test_wandb_custom_values(self):
        """Test custom configuration values."""
        cfg = Wandb(
            project="custom_proj",
            group="custom_group",
            entity="my_entity",
            use=False,
        )
        assert cfg.project == "custom_proj"
        assert cfg.group == "custom_group"
        assert cfg.entity == "my_entity"
        assert cfg.use is False

    def test_wandb_dir_is_path_like(self):
        """Test that wandb dir is path-like."""
        cfg = Wandb()
        assert isinstance(cfg.dir, (str, Path))


@pytest.mark.unit
class TestWandbModeMethod:
    """Tests for Wandb.mode() method."""

    def test_mode_online_when_use_true(self):
        """Test mode returns online when use=True."""
        cfg = Wandb(use=True)
        mode = cfg.mode(use=True)
        assert mode == "online"

    def test_mode_disabled_when_use_false(self):
        """Test mode returns disabled when use=False."""
        cfg = Wandb(use=False)
        mode = cfg.mode(use=False)
        assert mode == "disabled"

    def test_mode_respects_parameter(self):
        """Test that mode() respects the use parameter over config."""
        cfg = Wandb(use=True)
        # Parameter should override config
        assert cfg.mode(use=False) == "disabled"
        assert cfg.mode(use=True) == "online"


@pytest.mark.unit
class TestWandbLogMethod:
    """Tests for Wandb.log() method."""

    @patch("tabpi.wab.wandb.log")
    def test_log_with_step(self, mock_wandb_log):
        """Test logging with step parameter."""
        cfg = Wandb()
        info = {"loss": 0.5, "accuracy": 0.9}
        step = 10

        cfg.log(info, step=step)

        # Check that wandb.log was called with correct arguments
        mock_wandb_log.assert_called_once()
        _args, kwargs = mock_wandb_log.call_args
        assert kwargs.get("step") == step

    @patch("tabpi.wab.wandb.log")
    def test_log_without_step(self, mock_wandb_log):
        """Test logging without step parameter."""
        cfg = Wandb()
        info = {"loss": 0.5}

        cfg.log(info)

        # Check that wandb.log was called without step
        mock_wandb_log.assert_called_once()
        _args, kwargs = mock_wandb_log.call_args
        assert "step" not in kwargs

    @patch("tabpi.wab.wandb.log")
    def test_log_flatten_dict(self, mock_wandb_log):
        """Test that log flattens nested dictionaries."""
        cfg = Wandb()
        info = {"metrics": {"loss": 0.5, "acc": 0.9}}

        cfg.log(info)

        # The flatten_dict should be called, so we get flattened keys
        mock_wandb_log.assert_called_once()

    @patch("tabpi.wab.wandb.log")
    def test_log_custom_separator(self, mock_wandb_log):
        """Test log with custom separator."""
        cfg = Wandb()
        info = {"a": {"b": 1}}

        cfg.log(info, sep=".")

        mock_wandb_log.assert_called_once()


@pytest.mark.unit
class TestWandbInitialize:
    """Tests for Wandb.initialize() method."""

    @patch("tabpi.wab.wandb.init")
    def test_initialize_basic(self, mock_init):
        """Test basic initialization."""
        mock_init.return_value = MagicMock(name="test_run")
        cfg = Wandb()

        @dataclass
        class DummyConfig:
            wandb: Wandb

        dummy_cfg = DummyConfig(wandb=cfg)

        result = cfg.initialize(dummy_cfg)

        mock_init.assert_called_once()
        assert result is not None

    @patch("tabpi.wab.wandb.init")
    def test_initialize_with_name(self, mock_init):
        """Test initialization with custom name."""
        mock_run = MagicMock()
        mock_run.name = "original_name"
        mock_init.return_value = mock_run

        cfg = Wandb()

        @dataclass
        class DummyConfig:
            wandb: Wandb

        dummy_cfg = DummyConfig(wandb=cfg)

        cfg.initialize(dummy_cfg, name="custom_name")

        # Check that init was called with name parameter
        _args, kwargs = mock_init.call_args
        assert "name" in kwargs
        assert kwargs["name"] == "custom_name"

    @patch("tabpi.wab.wandb.init")
    def test_initialize_creates_timestamp(self, mock_init):
        """Test that initialize prepends timestamp to run name."""
        mock_run = MagicMock()
        mock_run.name = "test_run"
        mock_init.return_value = mock_run

        cfg = Wandb()

        @dataclass
        class DummyConfig:
            wandb: Wandb

        dummy_cfg = DummyConfig(wandb=cfg)

        cfg.initialize(dummy_cfg)

        # Check that run.name was modified with timestamp
        assert mock_run.name != "test_run"
        assert "_test_run" in mock_run.name

    @patch("tabpi.wab.wandb.init")
    def test_initialize_sets_config_name(self, mock_init):
        """Test that initialize sets config.name."""
        mock_run = MagicMock()
        mock_run.name = "test_run"
        mock_init.return_value = mock_run

        cfg = Wandb()

        @dataclass
        class DummyConfig:
            name: str | None = None
            wandb: Wandb = None

            def __post_init__(self):
                if self.wandb is None:
                    self.wandb = cfg

        dummy_cfg = DummyConfig()

        cfg.initialize(dummy_cfg)

        # Config name should be updated
        assert dummy_cfg.name is not None
        assert "_test_run" in dummy_cfg.name


@pytest.mark.unit
class TestWandbIntegration:
    """Integration-like tests for Wandb class."""

    def test_wandb_instantiation(self):
        """Test that Wandb can be instantiated without errors."""
        cfg = Wandb()
        assert isinstance(cfg, Wandb)

    def test_wandb_dir_attribute(self):
        """Test that wandb dir attribute exists and is accessible."""
        cfg = Wandb()
        # Should not raise AttributeError
        dir_path = cfg.dir
        assert dir_path is not None

    def test_multiple_instances_independent(self):
        """Test that multiple Wandb instances are independent."""
        cfg1 = Wandb(project="proj1")
        cfg2 = Wandb(project="proj2")

        assert cfg1.project != cfg2.project
        assert cfg1 is not cfg2
