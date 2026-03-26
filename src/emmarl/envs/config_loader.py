"""Configuration loader for FireSim environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emmarl.envs.agent import AgentType


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default_config.json"


class SimulationConfig:
    """Container for simulation configuration loaded from JSON."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Load configuration from JSON file.

        Args:
            config_path: Path to JSON config file. Uses default if None.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        self._config = self._load_config(config_path)

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load and parse JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def environment(self) -> dict[str, Any]:
        """Get environment configuration."""
        return self._config.get("environment", {})

    @property
    def agents(self) -> dict[str, Any]:
        """Get agent configuration."""
        return self._config.get("agents", {})

    @property
    def rewards(self) -> dict[str, float]:
        """Get reward weights configuration."""
        return self._config.get("rewards", {})

    @property
    def terrain(self) -> dict[str, Any]:
        """Get terrain configuration."""
        return self._config.get("terrain", {})

    @property
    def fire_dynamics(self) -> dict[str, Any]:
        """Get fire dynamics configuration."""
        return self._config.get("fire_dynamics", {})

    @property
    def map_config(self) -> dict[str, Any]:
        """Get map configuration."""
        return self._config.get("map", {})

    @property
    def rendering(self) -> dict[str, Any]:
        """Get rendering configuration."""
        return self._config.get("rendering", {})

    @property
    def graph_filter(self) -> dict[str, Any]:
        """Get graph filter configuration."""
        return self._config.get("graph_filter", {})

    @property
    def action_ranges(self) -> dict[str, float]:
        """Get action ranges configuration."""
        return self._config.get("action_ranges", {})

    @property
    def protection(self) -> dict[str, float]:
        """Get protection configuration."""
        return self._config.get("protection", {})

    @property
    def movement(self) -> dict[str, float]:
        """Get movement configuration."""
        return self._config.get("movement", {})

    def get_agent_type_config(self, agent_type: AgentType) -> dict[str, Any]:
        """Get configuration for a specific agent type.

        Args:
            agent_type: The agent type to get config for.

        Returns:
            Dictionary containing agent type configuration.
        """
        type_configs = self.agents.get("type_configs", {})
        return type_configs.get(agent_type.name, {})

    def get_terrain_properties(self, terrain_type: str) -> dict[str, float]:
        """Get properties for a specific terrain type.

        Args:
            terrain_type: The terrain type name.

        Returns:
            Dictionary containing terrain properties.
        """
        return self.terrain.get(terrain_type, {})


def load_config(config_path: str | Path | None = None) -> SimulationConfig:
    """Load simulation configuration from JSON file.

    Args:
        config_path: Path to JSON config file. Uses default if None.

    Returns:
        SimulationConfig instance with loaded configuration.
    """
    return SimulationConfig(config_path)
