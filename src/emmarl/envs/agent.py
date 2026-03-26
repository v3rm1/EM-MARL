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


class AgentStatus(Enum):
    """Health/operational status of agents."""

    HEALTHY = auto()
    INJURED = auto()
    AFFECTED = auto()  # Affected by smoke/heat
    CRITICAL = auto()
    DECEASED = auto()
    SURVIVED = auto()
    EVACUATED = auto()
    RESCUED = auto()


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

    @classmethod
    def from_dict(
        cls, agent_type: AgentType, config_dict: dict[str, Any]
    ) -> MovementConfig:
        """Create MovementConfig from dictionary.

        Args:
            agent_type: The agent type.
            config_dict: Dictionary with movement config values.

        Returns:
            MovementConfig instance.
        """
        return MovementConfig(
            max_speed=config_dict.get("max_speed", 10.0),
            max_acceleration=config_dict.get("max_acceleration", 5.0),
            stamina_cost_per_step=config_dict.get("stamina_cost_per_step", 1.0),
            can_run=config_dict.get("can_run", True),
            run_multiplier=config_dict.get("run_multiplier", 2.0),
            can_climb=config_dict.get("can_climb", False),
            can_swim=config_dict.get("can_swim", False),
        )

    @classmethod
    def get_default(cls, agent_type: AgentType) -> MovementConfig:
        """Get default config for an agent type.

        Args:
            agent_type: The agent type.

        Returns:
            MovementConfig instance.
        """
        return getattr(cls, agent_type.name)

    @classmethod
    def set_config(cls, agent_type: AgentType, config: MovementConfig) -> None:
        """Set custom config for an agent type.

        Args:
            agent_type: The agent type.
            config: MovementConfig to set.
        """
        setattr(cls, agent_type.name, config)


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

    @classmethod
    def from_config(
        cls,
        agent_type: AgentType,
        agent_id: str,
        position: tuple[float, float],
        resources: dict[str, float] | None = None,
    ) -> "AgentConfig":
        """Create AgentConfig from simulation config.

        Args:
            agent_type: The type of agent.
            agent_id: Unique identifier for the agent.
            position: Initial position of the agent.
            resources: Optional resources dict. Uses defaults if None.

        Returns:
            AgentConfig instance.
        """
        return cls(
            agent_type=agent_type,
            agent_id=agent_id,
            position=position,
            resources=resources,
        )

    def _default_resources(self) -> dict[str, float]:
        """Get default resources based on agent type."""
        defaults = {
            AgentType.MEDIC: {"medkits": 5, "medication": 3},
            AgentType.FIRE_FORCE: {"water": 100.0, "foam": 50.0},
            AgentType.POLICE: {"barriers": 10, "flare": 5},
            AgentType.CIVILIAN: {},
        }
        return defaults.get(self.agent_type, {}).copy()

    def set_resources(self, resources: dict[str, float]) -> None:
        """Set resources from config dictionary.

        Args:
            resources: Dictionary of resource_name -> amount.
        """
        self.resources = resources.copy()

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
    agent_status: AgentStatus = AgentStatus.HEALTHY
    carrying: dict[str, float] | None = None
    target: tuple[float, float] | None = None
    action_taken: bool = False
    observation: Any = None
    is_running: bool = False
    is_moving: bool = False
    distance_traveled: float = 0.0
    smoke_exposure: float = 0.0
    heat_exposure: float = 0.0
    time_in_fire: float = 0.0
    rescued: bool = False
    evacuated: bool = False

    def __post_init__(self) -> None:
        if self.carrying is None:
            self.carrying = {}

    def is_alive(self) -> bool:
        """Check if agent is alive."""
        return self.health > 0 and self.agent_status != AgentStatus.DECEASED

    def is_exhausted(self) -> bool:
        """Check if agent is exhausted."""
        return self.stamina <= 0

    def take_damage(self, damage: float) -> None:
        """Take damage."""
        self.health = max(0.0, self.health - damage)
        if self.health <= 0:
            self.agent_status = AgentStatus.DECEASED
        elif self.health < 30:
            self.agent_status = AgentStatus.CRITICAL
        elif self.health < 60:
            self.agent_status = AgentStatus.INJURED

    def take_fire_damage(self, fire_intensity: float, dt: float) -> None:
        """Take damage from fire exposure.

        Args:
            fire_intensity: Fire intensity (0-1)
            dt: Time step
        """
        damage_rate = fire_intensity * 50.0
        self.take_damage(damage_rate * dt)
        self.time_in_fire += dt

    def take_smoke_damage(self, smoke_density: float, dt: float) -> None:
        """Take damage from smoke inhalation.

        Args:
            smoke_density: Smoke density (0-1)
            dt: Time step
        """
        damage_rate = smoke_density * 20.0
        self.take_damage(damage_rate * dt)
        self.smoke_exposure += smoke_density * dt

    def take_heat_damage(self, heat_intensity: float, dt: float) -> None:
        """Take damage from heat exposure.

        Args:
            heat_intensity: Heat intensity (0-1)
            dt: Time step
        """
        damage_rate = heat_intensity * 30.0
        self.take_damage(damage_rate * dt)
        self.heat_exposure += heat_intensity * dt

    def update_status(self) -> None:
        """Update agent status based on health and conditions."""
        if self.agent_status == AgentStatus.DECEASED:
            return

        if self.health <= 0:
            self.agent_status = AgentStatus.DECEASED
        elif self.health < 20:
            self.agent_status = AgentStatus.CRITICAL
        elif self.health < 50:
            self.agent_status = AgentStatus.INJURED
        elif self.smoke_exposure > 10 or self.heat_exposure > 5:
            self.agent_status = AgentStatus.AFFECTED
        else:
            self.agent_status = AgentStatus.HEALTHY

    def mark_survived(self) -> None:
        """Mark agent as survived."""
        if self.is_alive():
            self.agent_status = AgentStatus.SURVIVED

    def mark_evacuated(self) -> None:
        """Mark agent as evacuated."""
        if self.is_alive():
            self.evacuated = True
            self.agent_status = AgentStatus.EVACUATED

    def mark_rescued(self) -> None:
        """Mark agent as rescued."""
        if self.is_alive():
            self.rescued = True
            self.agent_status = AgentStatus.RESCUED

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
        self.update_status()

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        return np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

    def get_heading(self) -> float:
        """Get heading angle in radians."""
        if self.get_speed() < 0.01:
            return 0.0
        return np.arctan2(self.velocity[1], self.velocity[0])
