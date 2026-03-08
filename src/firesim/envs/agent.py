"""Agent types for FireSim emergency response environment."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any
import numpy as np


class AgentType(Enum):
    """Types of agents in the emergency response simulation."""

    MEDIC = auto()
    FIRE_FORCE = auto()
    POLICE = auto()
    CIVILIAN = auto()


@dataclass
class MovementConfig:
    """Movement configuration for an agent type."""

    max_speed: float = 10.0
    max_acceleration: float = 5.0
    stamina_cost_per_step: float = 1.0
    can_run: bool = True
    run_multiplier: float = 2.0
    can_climb: bool = False
    can_swim: bool = False


class AgentTypeConfig:
    """Pre-defined movement configs for each agent type."""

    MEDIC = MovementConfig(
        max_speed=8.0,
        max_acceleration=4.0,
        stamina_cost_per_step=1.5,
        can_run=True,
        run_multiplier=1.5,
    )
    FIRE_FORCE = MovementConfig(
        max_speed=10.0,
        max_acceleration=6.0,
        stamina_cost_per_step=2.0,
        can_run=True,
        run_multiplier=1.8,
        can_climb=True,
    )
    POLICE = MovementConfig(
        max_speed=11.0,
        max_acceleration=7.0,
        stamina_cost_per_step=1.8,
        can_run=True,
        run_multiplier=2.0,
    )
    CIVILIAN = MovementConfig(
        max_speed=7.0,
        max_acceleration=3.0,
        stamina_cost_per_step=1.0,
        can_run=True,
        run_multiplier=1.2,
    )


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_type: AgentType
    agent_id: str
    health: float = 100.0
    max_health: float = 100.0
    stamina: float = 100.0
    max_stamina: float = 100.0
    position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    resources: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.resources is None:
            self.resources = self._default_resources()

    @property
    def movement_config(self) -> MovementConfig:
        """Get movement config for this agent type."""
        return getattr(AgentTypeConfig, self.agent_type.name)

    def _default_resources(self) -> dict[str, float]:
        """Get default resources based on agent type."""
        defaults = {
            AgentType.MEDIC: {"medkits": 5, "medication": 3},
            AgentType.FIRE_FORCE: {"water": 100.0, "foam": 50.0},
            AgentType.POLICE: {"barriers": 10, "flare": 5},
            AgentType.CIVILIAN: {},
        }
        return defaults.get(self.agent_type, {}).copy()

    def has_resource(self, resource: str, amount: float = 1.0) -> bool:
        """Check if agent has sufficient resource."""
        return self.resources.get(resource, 0.0) >= amount

    def consume_resource(self, resource: str, amount: float = 1.0) -> bool:
        """Consume a resource if available."""
        if self.has_resource(resource, amount):
            self.resources[resource] -= amount
            return True
        return False

    def add_resource(self, resource: str, amount: float) -> None:
        """Add resources to agent."""
        self.resources[resource] = self.resources.get(resource, 0.0) + amount


@dataclass
class AgentState:
    """Mutable state for an agent during simulation."""

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)
    health: float = 100.0
    stamina: float = 100.0
    status: str = "active"
    carrying: dict[str, float] | None = None
    target: tuple[float, float] | None = None
    action_taken: bool = False
    observation: Any = None
    is_running: bool = False
    is_moving: bool = False
    distance_traveled: float = 0.0

    def __post_init__(self) -> None:
        if self.carrying is None:
            self.carrying = {}

    def is_alive(self) -> bool:
        """Check if agent is alive."""
        return self.health > 0 and self.status != "dead"

    def is_exhausted(self) -> bool:
        """Check if agent is exhausted."""
        return self.stamina <= 0

    def take_damage(self, damage: float) -> None:
        """Take damage."""
        self.health = max(0.0, self.health - damage)
        if self.health <= 0:
            self.status = "dead"

    def use_stamina(self, amount: float) -> bool:
        """Use stamina, returns True if successful."""
        if self.stamina >= amount:
            self.stamina -= amount
            return True
        return False

    def restore_stamina(self, amount: float) -> None:
        """Restore stamina."""
        self.stamina = min(100.0, self.stamina + amount)

    def heal(self, amount: float) -> None:
        """Heal the agent."""
        self.health = min(100.0, self.health + amount)

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        return np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

    def get_heading(self) -> float:
        """Get heading angle in radians."""
        if self.get_speed() < 0.01:
            return 0.0
        return np.arctan2(self.velocity[1], self.velocity[0])
