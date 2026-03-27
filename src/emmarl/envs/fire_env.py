"""FireSim: PettingZoo environment for emergency response multi-agent simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from emmarl.envs.agent import (
    AgentConfig,
    AgentState,
    AgentType,
    AgentStatus,
    AgentTypeConfig,
    MovementConfig,
)
from emmarl.envs.config_loader import SimulationConfig, load_config
from emmarl.envs.map import ZoneType, create_default_map
from emmarl.envs.fire_dynamics import (
    FireModel,
    create_default_fire_model,
    FuelProperties,
    SuppressionPhysics,
    FoamPhysics,
    AerialSuppressionPhysics,
    SuppressionLinePhysics,
)

try:
    from emmarl.envs.fire_jax import JAXFireModel, create_jax_fire_model

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JAXFireModel = None
    create_jax_fire_model = None
from emmarl.envs.render import (
    FireSimRenderer,
    RenderConfig,
    RenderMode,
    GraphFilter,
)
from emmarl.envs.metrics import EpisodeMetrics


@dataclass
class FireEnvConfig:
    """Configuration for FireEnv."""

    num_medics: int = 2
    num_fire_force: int = 2
    num_police: int = 2
    num_civilians: int = 5
    map_width: float = 1000.0
    map_height: float = 1000.0
    max_steps: int = 1000
    agent_speed: float = 10.0
    agent_vision_radius: float = 100.0
    enable_fire_dynamics: bool = True
    wind_speed: float = 10.0
    wind_direction: float = 0.0
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "incident_resolved": 100.0,
            "casualty_prevented": 50.0,
            "damage_reduced": 25.0,
            "survived": 20.0,
            "died": -100.0,
            "stamina_penalty": -0.1,
            "time_penalty": -0.01,
        }
    )
    action_ranges: dict[str, float] = field(
        default_factory=lambda: {
            "heal_range": 30.0,
            "medication_range": 20.0,
            "firefighter_range": 50.0,
            "foam_range": 40.0,
            "police_range": 60.0,
            "civilian_seek_help_distance": 100.0,
        }
    )
    protection: dict[str, float] = field(
        default_factory=lambda: {
            "civilian_fire_range": 100.0,
            "civilian_protection": 0.3,
            "firefighter_protection": 0.1,
            "medic_protection": 0.2,
            "police_protection": 0.2,
        }
    )
    movement: dict[str, float] = field(
        default_factory=lambda: {
            "safe_position_margin": 50.0,
            "danger_threshold": 0.8,
            "stamina_restore_rate": 0.5,
            "stamina_run_cost_multiplier": 1.5,
            "turn_rate": 0.2,
        }
    )
    suppression: dict[str, float] = field(
        default_factory=lambda: {
            "water_effectiveness": 1.0,
            "foam_effectiveness": 1.5,
            "retardant_effectiveness": 2.0,
            "aerial_drop_accuracy": 0.7,
            "line_construction_rate": 1.0,
        }
    )
    water_temperature: float = 20.0

    @classmethod
    def from_config(cls, config: SimulationConfig) -> FireEnvConfig:
        """Create FireEnvConfig from SimulationConfig.

        Args:
            config: SimulationConfig instance with loaded JSON config.

        Returns:
            FireEnvConfig instance.
        """
        env = config.environment
        agents = config.agents
        rewards = config.rewards
        action_ranges = config.action_ranges
        protection = config.protection
        movement = config.movement
        suppression = config.suppression if hasattr(config, "suppression") else {}

        return cls(
            num_medics=agents.get("num_medics", 2),
            num_fire_force=agents.get("num_fire_force", 2),
            num_police=agents.get("num_police", 2),
            num_civilians=agents.get("num_civilians", 5),
            map_width=env.get("map_width", 1000.0),
            map_height=env.get("map_height", 1000.0),
            max_steps=env.get("max_steps", 1000),
            agent_speed=env.get("agent_speed", 10.0),
            agent_vision_radius=env.get("agent_vision_radius", 100.0),
            enable_fire_dynamics=env.get("enable_fire_dynamics", True),
            wind_speed=env.get("wind_speed", 10.0),
            wind_direction=env.get("wind_direction", 0.0),
            reward_weights=rewards.copy(),
            action_ranges=action_ranges.copy(),
            protection=protection.copy(),
            movement=movement.copy(),
            suppression=suppression.copy() if suppression else {},
            water_temperature=env.get("water_temperature", 20.0),
        )


class FireEnv(AECEnv):
    """Emergency response multi-agent environment using PettingZoo AEC API.

    This environment simulates an emergency response scenario with multiple
    agent types (medics, fire force, police, civilians) working together
    to handle various incidents.

    Agent Types:
        - MEDIC: Can heal civilians and other agents, use medkits
        - FIRE_FORCE: Can extinguish fires, use water/foam
        - POLICE: Can control crowds, set barriers
        - CIVILIAN: Can move, seek safety, be rescued
    """

    metadata: dict = {"render_modes": ["human"], "name": "emmarl_v0"}

    def __init__(
        self,
        config: FireEnvConfig | SimulationConfig | str | Path | None = None,
        use_jax_fire: bool = False,
    ) -> None:
        """Initialize the FireSim environment.

        Args:
            config: Environment configuration. Can be:
                - FireEnvConfig: Use directly
                - SimulationConfig: Convert to FireEnvConfig
                - str/Path: Load JSON file and convert
                - None: Use defaults
            use_jax_fire: Use JAX-accelerated fire physics (requires JAX)
        """
        super().__init__()

        self._simulation_config: SimulationConfig | None = None

        if config is None:
            self.config = FireEnvConfig()
        elif isinstance(config, FireEnvConfig):
            self.config = config
        elif isinstance(config, SimulationConfig):
            self._simulation_config = config
            self.config = FireEnvConfig.from_config(config)
        elif isinstance(config, (str, Path)):
            loaded_config = load_config(config)
            self._simulation_config = loaded_config
            self.config = FireEnvConfig.from_config(loaded_config)
        else:
            self.config = FireEnvConfig()

        self.possible_agents = self._create_agent_ids()
        self.agents = self.possible_agents.copy()

        self._agent_configs: dict[str, AgentConfig] = {}
        self._agent_states: dict[str, AgentState] = {}

        self._emergency_map = create_default_map(
            width=self.config.map_width,
            height=self.config.map_height,
        )

        self._current_step = 0
        self._agent_selector: AgentSelector | None = None
        self._cumulative_rewards: dict[str, float] = {}

        self._action_spaces = self._create_action_spaces()
        self._observation_spaces = self._create_observation_spaces()

        self._total_incidents = len(self._emergency_map.incidents)
        self._resolved_incidents = 0

        self._renderer: FireSimRenderer | None = None
        if self._simulation_config is not None:
            self._render_config = RenderConfig.from_config(
                self._simulation_config.rendering
            )
        else:
            self._render_config = RenderConfig()

        self._fire_model: FireModel | None = None
        self._jax_fire_model: JAXFireModel | None = None
        self._use_jax_fire = use_jax_fire and JAX_AVAILABLE

        if self._use_jax_fire:
            self._initialize_jax_fire_model()
        else:
            self._initialize_fire_model()

        self._suppression_physics = SuppressionPhysics()
        self._foam_physics = FoamPhysics()
        self._aerial_physics = AerialSuppressionPhysics()
        self._line_physics = SuppressionLinePhysics()

        self._suppression_applications: list[dict] = []

        self._episode_metrics = EpisodeMetrics()

        self._apply_agent_type_configs()

    def _apply_agent_type_configs(self) -> None:
        """Apply agent type configurations from simulation config."""
        if self._simulation_config is None:
            return

        config_dict = self._simulation_config.agents.get("type_configs", {})
        for agent_type_str, type_config in config_dict.items():
            try:
                agent_type = AgentType[agent_type_str]
                movement_config = MovementConfig(
                    max_speed=type_config.get("max_speed", 10.0),
                    max_acceleration=type_config.get("max_acceleration", 5.0),
                    stamina_cost_per_step=type_config.get("stamina_cost_per_step", 1.0),
                    can_run=type_config.get("can_run", True),
                    run_multiplier=type_config.get("run_multiplier", 2.0),
                    can_climb=type_config.get("can_climb", False),
                    can_swim=type_config.get("can_swim", False),
                )
                AgentTypeConfig.set_config(agent_type, movement_config)
            except KeyError:
                pass

    def _create_agent_ids(self) -> list[str]:
        """Create agent IDs for all agents."""
        agents = []
        for i in range(self.config.num_medics):
            agents.append(f"medic_{i}")
        for i in range(self.config.num_fire_force):
            agents.append(f"firefighter_{i}")
        for i in range(self.config.num_police):
            agents.append(f"police_{i}")
        for i in range(self.config.num_civilians):
            agents.append(f"civilian_{i}")
        return agents

    def _get_agent_type(self, agent: str) -> AgentType:
        """Get agent type from agent ID."""
        if agent.startswith("medic"):
            return AgentType.MEDIC
        elif agent.startswith("firefighter"):
            return AgentType.FIRE_FORCE
        elif agent.startswith("police"):
            return AgentType.POLICE
        else:
            return AgentType.CIVILIAN

    def _create_action_spaces(self) -> dict[str, spaces.Space]:
        """Create action spaces for all agent types."""
        action_spaces = {}
        for agent in self.possible_agents:
            agent_type = self._get_agent_type(agent)
            action_spaces[agent] = self._get_action_space_for_type(agent_type)
        return action_spaces

    def _get_action_space_for_type(self, agent_type: AgentType) -> spaces.Space:
        """Get action space for a specific agent type."""
        if agent_type == AgentType.MEDIC:
            return spaces.Dict(
                {
                    "move": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "run": spaces.Discrete(2),
                    "heal": spaces.Discrete(2),
                    "use_medication": spaces.Discrete(2),
                    "communicate": spaces.Discrete(2),
                }
            )
        elif agent_type == AgentType.FIRE_FORCE:
            return spaces.Dict(
                {
                    "move": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "run": spaces.Discrete(2),
                    "extinguish": spaces.Discrete(2),
                    "use_foam": spaces.Discrete(2),
                    "communicate": spaces.Discrete(2),
                }
            )
        elif agent_type == AgentType.POLICE:
            return spaces.Dict(
                {
                    "move": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "run": spaces.Discrete(2),
                    "control_crowd": spaces.Discrete(2),
                    "place_barrier": spaces.Discrete(2),
                    "communicate": spaces.Discrete(2),
                }
            )
        else:
            return spaces.Dict(
                {
                    "move": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "run": spaces.Discrete(2),
                    "seek_help": spaces.Discrete(2),
                }
            )

    def _create_observation_spaces(self) -> dict[str, spaces.Space]:
        """Create observation spaces for all agents."""
        obs_spaces = {}
        vision_size = int(self.config.agent_vision_radius * 2)
        for agent in self.possible_agents:
            obs_spaces[agent] = spaces.Box(
                low=0, high=1, shape=(vision_size, vision_size, 4), dtype=np.float32
            )
        return obs_spaces

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self._current_step = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self._episode_metrics = EpisodeMetrics()

        self._emergency_map = create_default_map(
            width=self.config.map_width,
            height=self.config.map_height,
        )
        self._resolved_incidents = 0
        self._total_incidents = self._get_total_incidents()

        self._initialize_agents()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _get_total_incidents(self) -> int:
        """Get total number of incidents based on map type."""
        if self._emergency_map:
            return len(self._emergency_map.incidents)
        return 0

    def _initialize_fire_model(self) -> None:
        """Initialize the fire dynamics model."""
        if not self.config.enable_fire_dynamics:
            return

        self._fire_model = create_default_fire_model(
            wind_speed=self.config.wind_speed,
            wind_direction=self.config.wind_direction,
        )

    def _initialize_jax_fire_model(self) -> None:
        """Initialize the JAX fire dynamics model."""
        if not self.config.enable_fire_dynamics:
            return

        if not JAX_AVAILABLE:
            import warnings

            warnings.warn("JAX not available, falling back to NumPy fire model")
            self._use_jax_fire = False
            self._initialize_fire_model()
            return

        grid_size = int(max(self.config.map_width, self.config.map_height) / 10)
        self._jax_fire_model = create_jax_fire_model(
            grid_height=grid_size,
            grid_width=grid_size,
            wind_speed=self.config.wind_speed,
            wind_direction=self.config.wind_direction,
            enable_diurnal=True,
        )

    def _update_fire_dynamics(self) -> None:
        """Update fire dynamics if enabled."""
        if not self.config.enable_fire_dynamics:
            return

        if self._use_jax_fire and self._jax_fire_model is not None:
            self._update_fire_dynamics_jax()
        elif self._fire_model is not None:
            self._update_fire_dynamics_numpy()

    def _update_fire_dynamics_jax(self) -> None:
        """Update fire dynamics using JAX."""
        import jax.numpy as jnp

        if self._jax_fire_model is None:
            return

        grid_size = int(max(self.config.map_width, self.config.map_height) / 10)
        fuel_map = jnp.full((grid_size, grid_size), 0.5, dtype=jnp.float32)
        slope_map = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)

        if self._emergency_map.terrain:
            grid = self._emergency_map.terrain
            for gy in range(min(grid.height, grid_size)):
                for gx in range(min(grid.width, grid_size)):
                    world_x, world_y = grid.to_world_coords(gx, gy)
                    fuel_load = grid.get_fuel_load(world_x, world_y)
                    if fuel_load > 0:
                        fg_x = int(gx * grid_size / grid.width)
                        fg_y = int(gy * grid_size / grid.height)
                        if 0 <= fg_x < grid_size and 0 <= fg_y < grid_size:
                            fuel_map = fuel_map.at[fg_y, fg_x].set(fuel_load)

        self._jax_fire_model.step(fuel_map, slope_map, dt=1.0)

    def _update_fire_dynamics_numpy(self) -> None:
        """Update fire dynamics using NumPy."""
        if self._fire_model is None:
            return

        fuel_map: dict[tuple[float, float], FuelProperties] = {}

        if self._emergency_map.terrain:
            grid = self._emergency_map.terrain
            for gy in range(grid.height):
                for gx in range(grid.width):
                    world_x, world_y = grid.to_world_coords(gx, gy)
                    fuel_load = grid.get_fuel_load(world_x, world_y)
                    if fuel_load > 0:
                        key = (int(world_x), int(world_y))
                        fuel_map[key] = FuelProperties.from_terrain(
                            "grass", fuel_load, 0.3
                        )

        self._fire_model.update(fuel_map)

    def _apply_fire_damage_to_agents(self) -> None:
        """Apply fire, smoke, and heat damage to agents.

        Civilians take damage only when within fire range.
        Responders take reduced damage based on their protective gear.
        """
        if not self.config.enable_fire_dynamics:
            return

        civilian_fire_range = self.config.protection.get("civilian_fire_range", 100.0)
        civilian_protection = self.config.protection.get("civilian_protection", 0.3)
        firefighter_protection = self.config.protection.get(
            "firefighter_protection", 0.1
        )
        medic_protection = self.config.protection.get("medic_protection", 0.2)
        police_protection = self.config.protection.get("police_protection", 0.2)

        for agent in self.agents:
            state = self._agent_states[agent]
            if not state.is_alive():
                continue

            agent_type = self._get_agent_type(agent)
            position = state.position

            if self._use_jax_fire and self._jax_fire_model is not None:
                fire_intensity = self._jax_fire_model.get_intensity_at(
                    position[0], position[1]
                )
                flame_length = fire_intensity * 2.0
            elif self._fire_model is not None:
                fire_intensity = self._fire_model.get_intensity_at(position)
                flame_length = self._fire_model.get_flame_length_at(position)
            else:
                continue

            if fire_intensity <= 0 and flame_length <= 0:
                continue

            protection = 1.0

            if agent_type == AgentType.CIVILIAN:
                protection = civilian_protection
                if fire_intensity <= 0 and flame_length <= 0:
                    continue
            elif agent_type == AgentType.FIRE_FORCE:
                protection = firefighter_protection
            elif agent_type == AgentType.MEDIC:
                protection = medic_protection
            elif agent_type == AgentType.POLICE:
                protection = police_protection

            effective_fire = fire_intensity * protection
            effective_flame = flame_length * protection

            if agent_type == AgentType.CIVILIAN:
                dist, _ = self._get_fire_distance(position)
                if dist > civilian_fire_range:
                    continue
                if dist < civilian_fire_range:
                    effective_fire *= 1.0 - dist / civilian_fire_range
                    effective_flame *= 1.0 - dist / civilian_fire_range

            if effective_fire > 0:
                state.take_fire_damage(effective_fire, 1.0)

            if effective_flame > 0.5:
                smoke_density = min(1.0, effective_flame / 5.0)
                state.take_smoke_damage(smoke_density, 1.0)

                heat_intensity = min(1.0, effective_flame / 3.0)
                state.take_heat_damage(heat_intensity, 1.0)

            state.update_status()

    def _get_fire_distance(
        self, position: tuple[float, float]
    ) -> tuple[float, tuple[float, float] | None]:
        """Get distance to nearest fire and its position."""
        if self._use_jax_fire and self._jax_fire_model is not None:
            return self._jax_fire_model.get_intensity_at(
                position[0], position[1]
            ), position

        if self._fire_model:
            return self._fire_model.get_fire_distance(position)

        return float("inf"), None

    def get_agent_status_counts(self) -> dict[AgentStatus, int]:
        """Get count of agents in each status.

        Returns:
            Dictionary mapping AgentStatus to count
        """
        counts: dict[AgentStatus, int] = {}
        for agent in self.agents:
            state = self._agent_states[agent]
            status = state.agent_status
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_survival_stats(self) -> dict[str, int]:
        """Get survival statistics.

        Returns:
            Dictionary with survival stats
        """
        total = len(self.agents)
        alive = sum(1 for a in self.agents if self._agent_states[a].is_alive())
        deceased = sum(
            1
            for a in self.agents
            if self._agent_states[a].agent_status == AgentStatus.DECEASED
        )
        healthy = sum(
            1
            for a in self.agents
            if self._agent_states[a].agent_status == AgentStatus.HEALTHY
        )
        injured = sum(
            1
            for a in self.agents
            if self._agent_states[a].agent_status == AgentStatus.INJURED
        )
        affected = sum(
            1
            for a in self.agents
            if self._agent_states[a].agent_status == AgentStatus.AFFECTED
        )
        critical = sum(
            1
            for a in self.agents
            if self._agent_states[a].agent_status == AgentStatus.CRITICAL
        )
        rescued = sum(1 for a in self.agents if self._agent_states[a].rescued)
        evacuated = sum(1 for a in self.agents if self._agent_states[a].evacuated)

        return {
            "total": total,
            "alive": alive,
            "deceased": deceased,
            "healthy": healthy,
            "injured": injured,
            "affected": affected,
            "critical": critical,
            "rescued": rescued,
            "evacuated": evacuated,
        }

    def _is_within_bounds(self, point: tuple[float, float]) -> bool:
        """Check if point is within map bounds."""
        if self._emergency_map:
            return self._emergency_map.is_within_bounds(point)
        x, y = point
        return 0 <= x <= self.config.map_width and 0 <= y <= self.config.map_height

    def _get_danger_level(self, point: tuple[float, float]) -> float:
        """Get danger level at a point."""
        if self._emergency_map:
            return self._emergency_map.get_danger_level(point)
        return 0.0

    def _get_nearest_active_incident(
        self, position: tuple[float, float]
    ) -> tuple[Any | None, float]:
        """Get nearest active incident to a position."""
        if self._emergency_map:
            return self._emergency_map.get_nearest_active_incident(position)
        return None, float("inf")

    def _reduce_fire_intensity(
        self,
        position: tuple[float, float],
        amount: float,
        effective_range: float = 50.0,
        resource_type: str = "water",
    ) -> bool:
        """Reduce fire intensity at position using suppression physics.

        Args:
            position: Position of the firefighter
            amount: Base amount to reduce
            effective_range: Maximum range to affect fire
            resource_type: Type of suppression resource ("water", "foam", "retardant")

        Returns:
            True if fire was reduced
        """
        if self._use_jax_fire and self._jax_fire_model is not None:
            return self._reduce_fire_intensity_jax(
                position, amount, effective_range, resource_type
            )

        if self._fire_model is None:
            return False

        fire_intensity = self._fire_model.get_intensity_at(position)
        if fire_intensity <= 0:
            return False

        if resource_type == "water":
            effectiveness = self._suppression_physics.compute_water_effectiveness(
                amount,
                fire_intensity * 1000.0,
                self.config.water_temperature,
            )
            for fire in self._fire_model.fires:
                dist = np.sqrt(
                    (fire.position[0] - position[0]) ** 2
                    + (fire.position[1] - position[1]) ** 2
                )
                if dist < effective_range:
                    distance_factor = 1.0 - (dist / effective_range)
                    fire.intensity *= max(0.0, 1.0 - effectiveness * distance_factor)
                    fire.fire_line_intensity *= max(
                        0.0, 1.0 - effectiveness * distance_factor
                    )

            self._suppression_applications.append(
                {
                    "position": position,
                    "type": resource_type,
                    "amount": amount,
                    "effectiveness": effectiveness,
                    "time": self._current_step,
                }
            )

            return effectiveness > 0

        elif resource_type == "foam":
            slope_angle = 0.0
            effectiveness = self._foam_physics.compute_foam_effectiveness(
                amount,
                fire_intensity * 1000.0,
                slope_angle,
                age=0.0,
            )
            for fire in self._fire_model.fires:
                dist = np.sqrt(
                    (fire.position[0] - position[0]) ** 2
                    + (fire.position[1] - position[1]) ** 2
                )
                if dist < effective_range:
                    distance_factor = 1.0 - (dist / effective_range)
                    fire.intensity *= max(
                        0.0, 1.0 - effectiveness * distance_factor * 0.8
                    )

            self._suppression_applications.append(
                {
                    "position": position,
                    "type": resource_type,
                    "amount": amount,
                    "effectiveness": effectiveness,
                    "time": self._current_step,
                }
            )

            return effectiveness > 0

        return False

    def _reduce_fire_intensity_jax(
        self,
        position: tuple[float, float],
        amount: float,
        effective_range: float = 50.0,
        resource_type: str = "water",
    ) -> bool:
        """Reduce fire intensity using JAX fire model."""
        if self._jax_fire_model is None:
            return False

        fire_intensity = self._jax_fire_model.get_intensity_at(position[0], position[1])

        if fire_intensity <= 0:
            return False

        if resource_type == "water":
            effectiveness = self._suppression_physics.compute_water_effectiveness(
                amount,
                fire_intensity * 1000.0,
                self.config.water_temperature,
            )

            self._suppression_applications.append(
                {
                    "position": position,
                    "type": resource_type,
                    "amount": amount,
                    "effectiveness": effectiveness,
                    "time": self._current_step,
                }
            )

            return effectiveness > 0

        elif resource_type == "foam":
            effectiveness = self._foam_physics.compute_foam_effectiveness(
                amount,
                fire_intensity * 1000.0,
                slope_angle=0.0,
                age=0.0,
            )

            self._suppression_applications.append(
                {
                    "position": position,
                    "type": resource_type,
                    "amount": amount,
                    "effectiveness": effectiveness,
                    "time": self._current_step,
                }
            )

            return effectiveness > 0

        return False

    def _initialize_agents(self) -> None:
        """Initialize agent configs and states."""
        for agent in self.possible_agents:
            agent_type = self._get_agent_type(agent)
            start_pos = self._get_random_safe_position()

            self._agent_configs[agent] = AgentConfig(
                agent_type=agent_type,
                agent_id=agent,
                position=start_pos,
            )

            self._agent_states[agent] = AgentState(
                position=start_pos,
                health=100.0,
                stamina=100.0,
            )

    def _get_random_safe_position(self) -> tuple[float, float]:
        """Get a random safe position on the map."""
        margin = self.config.movement.get("safe_position_margin", 50.0)
        max_width = self.config.map_width
        max_height = self.config.map_height
        while True:
            x = np.random.uniform(margin, max_width - margin)
            y = np.random.uniform(margin, max_height - margin)
            danger = self._get_danger_level((x, y))
            if danger < 0.3:
                return (x, y)

    def step(self, action: Any) -> None:
        """Execute one step of the environment."""
        if self.agent_selection not in self.agents:
            return

        agent = self.agent_selection
        self._execute_action(agent, action)

        self._update_fire_dynamics()
        self._apply_fire_damage_to_agents()

        reward = self._compute_reward(agent)
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        self._check_terminations()

        if self._current_step >= self.config.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
            return

        if self._agent_selector is not None:
            self.agent_selection = self._agent_selector.next()

        self._current_step += 1

        self._record_metrics()

    def _record_metrics(self) -> None:
        """Record current metrics for visualization."""
        active_incidents = 0

        if self._emergency_map:
            active_incidents = sum(
                1 for inc in self._emergency_map.incidents if inc.active
            )

        self._episode_metrics.record(
            step=self._current_step,
            agent_states=self._agent_states,
            active_incidents=active_incidents,
            resolved_incidents=self._resolved_incidents,
        )

    def get_metrics(self) -> EpisodeMetrics:
        """Get the current episode metrics.

        Returns:
            EpisodeMetrics object with recorded data
        """
        return self._episode_metrics

    def _execute_action(self, agent: str, action: Any) -> None:
        """Execute an action for an agent."""
        state = self._agent_states[agent]
        config = self._agent_configs[agent]

        if not state.is_alive():
            return

        self._update_movement(agent, state, config, action)

        state.action_taken = True

        agent_type = self._get_agent_type(agent)
        self._handle_type_specific_actions(agent, agent_type, action, state)

    def _update_movement(
        self, agent: str, state: AgentState, config: AgentConfig, action: Any
    ) -> None:
        """Update agent movement based on action and physics."""
        move_config = config.movement_config

        move_input = action.get("move", np.zeros(2, dtype=np.float32))
        run_input = action.get("run", 0)

        state.is_running = bool(
            run_input == 1 and move_config.can_run and not state.is_exhausted()
        )

        terrain_multiplier = self._emergency_map.get_speed_multiplier(state.position)
        speed_multiplier = (
            move_config.run_multiplier if state.is_running else 1.0
        ) * terrain_multiplier
        effective_max_speed = move_config.max_speed * speed_multiplier

        desired_vx = move_input[0] * effective_max_speed
        desired_vy = move_input[1] * effective_max_speed

        current_speed = state.get_speed()
        if current_speed > 0.01:
            current_heading = state.get_heading()
            desired_heading = (
                np.arctan2(desired_vy, desired_vx)
                if abs(desired_vx) > 0.01 or abs(desired_vy) > 0.01
                else current_heading
            )
            heading_diff = desired_heading - current_heading
            heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
            turn_rate = self.config.movement.get("turn_rate", 0.2)
            new_heading = current_heading + heading_diff * turn_rate
            speed = min(current_speed, effective_max_speed)
            new_vx = np.cos(new_heading) * speed
            new_vy = np.sin(new_heading) * speed
        else:
            new_vx = desired_vx
            new_vy = desired_vy

        state.velocity = (new_vx, new_vy)
        state.is_moving = state.get_speed() > 0.5

        prev_pos = state.position
        new_x = state.position[0] + state.velocity[0]
        new_y = state.position[1] + state.velocity[1]

        if self._is_within_bounds((new_x, new_y)):
            can_move = True

            if not self._emergency_map.is_passable((new_x, new_y)):
                can_move = False

            if can_move:
                danger = self._get_danger_level((new_x, new_y))
                danger_threshold = self.config.movement.get("danger_threshold", 0.8)
                if danger < danger_threshold:
                    state.position = (new_x, new_y)

                    dist_moved = np.sqrt(
                        (state.position[0] - prev_pos[0]) ** 2
                        + (state.position[1] - prev_pos[1]) ** 2
                    )
                    state.distance_traveled += dist_moved

                    stamina_cost = move_config.stamina_cost_per_step
                    if state.is_running:
                        stamina_cost *= 1.5
                    state.use_stamina(stamina_cost)

        if not state.is_moving and state.stamina < move_config.stamina_cost_per_step:
            stamina_restore_rate = self.config.movement.get("stamina_restore_rate", 0.5)
            state.restore_stamina(stamina_restore_rate)

    def _handle_type_specific_actions(
        self, agent: str, agent_type: AgentType, action: Any, state: AgentState
    ) -> None:
        """Handle type-specific actions for each agent."""
        if agent_type == AgentType.MEDIC:
            if action.get("heal", 0) == 1:
                self._medic_heal(agent, state)
            if action.get("use_medication", 0) == 1:
                self._medic_use_medication(agent, state)

        elif agent_type == AgentType.FIRE_FORCE:
            if action.get("extinguish", 0) == 1:
                self._firefighter_extinguish(agent, state)
            if action.get("use_foam", 0) == 1:
                self._firefighter_use_foam(agent, state)

        elif agent_type == AgentType.POLICE:
            if action.get("control_crowd", 0) == 1:
                self._police_control_crowd(agent, state)
            if action.get("place_barrier", 0) == 1:
                self._police_place_barrier(agent, state)

        elif agent_type == AgentType.CIVILIAN:
            if action.get("seek_help", 0) == 1:
                self._civilian_seek_help(agent, state)

    def _medic_heal(self, agent: str, state: AgentState) -> None:
        """Medic heals nearby agents or civilians."""
        config = self._agent_configs[agent]
        if not config.has_resource("medkits", 1):
            return

        for other_agent in self.agents:
            if other_agent == agent:
                continue
            other_state = self._agent_states[other_agent]
            dist = np.sqrt(
                (state.position[0] - other_state.position[0]) ** 2
                + (state.position[1] - other_state.position[1]) ** 2
            )
            if (
                dist < self.config.action_ranges.get("heal_range", 30.0)
                and other_state.health < 100
            ):
                if config.consume_resource("medkits", 1):
                    other_state.heal(30)
                    break

    def _medic_use_medication(self, agent: str, state: AgentState) -> None:
        """Medic uses medication on nearby critical agents."""
        config = self._agent_configs[agent]
        if not config.has_resource("medication", 1):
            return

        for other_agent in self.agents:
            if other_agent == agent:
                continue
            other_state = self._agent_states[other_agent]
            dist = np.sqrt(
                (state.position[0] - other_state.position[0]) ** 2
                + (state.position[1] - other_state.position[1]) ** 2
            )
            if (
                dist < self.config.action_ranges.get("medication_range", 20.0)
                and other_state.health < 50
            ):
                if config.consume_resource("medication", 1):
                    other_state.heal(50)
                    break

    def _firefighter_extinguish(self, agent: str, state: AgentState) -> None:
        """Firefighter extinguishes nearby fire incidents.

        Firefighters must be within effective range of the fire to extinguish it.
        Uses realistic suppression physics: water effectiveness decreases with fire intensity.
        """
        config = self._agent_configs[agent]
        if not config.has_resource("water", 10):
            return

        firefighter_range = self.config.action_ranges.get("firefighter_range", 50.0)

        if self._fire_model is not None:
            fire_reduced = self._reduce_fire_intensity(
                state.position,
                amount=10.0,
                effective_range=firefighter_range,
                resource_type="water",
            )
            if fire_reduced:
                config.consume_resource("water", 10)

        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if (
            incident
            and dist < firefighter_range
            and incident.incident_type == ZoneType.FIRE
        ):
            if config.has_resource("water", 10):
                config.consume_resource("water", 10)
                effectiveness = 1.0 - (dist / firefighter_range)
                water_amount = 10.0 * self.config.suppression.get(
                    "water_effectiveness", 1.0
                )

                if self._fire_model is not None:
                    fire_intensity = self._fire_model.get_intensity_at(state.position)
                    suppression_eff = (
                        self._suppression_physics.compute_water_effectiveness(
                            water_amount,
                            fire_intensity * 1000.0,
                            self.config.water_temperature,
                        )
                    )
                    effectiveness *= suppression_eff

                incident.reduce_severity(0.2 * effectiveness)
                if incident.is_resolved():
                    self._resolved_incidents += 1

    def _firefighter_use_foam(self, agent: str, state: AgentState) -> None:
        """Firefighter uses foam on hazardous incidents.

        Foam is more effective than water but has shorter range.
        Uses foam physics for persistence and downhill creep effects.
        """
        config = self._agent_configs[agent]
        if not config.has_resource("foam", 5):
            return

        foam_range = self.config.action_ranges.get("foam_range", 40.0)

        if self._fire_model is not None:
            fire_reduced = self._reduce_fire_intensity(
                state.position,
                amount=5.0,
                effective_range=foam_range,
                resource_type="foam",
            )
            if fire_reduced:
                config.consume_resource("foam", 5)

        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if (
            incident
            and dist < foam_range
            and incident.incident_type
            in (
                ZoneType.FIRE,
                ZoneType.HAZMAT,
            )
        ):
            if config.has_resource("foam", 5):
                config.consume_resource("foam", 5)
                effectiveness = 1.0 - (dist / foam_range)
                foam_amount = 5.0 * self.config.suppression.get(
                    "foam_effectiveness", 1.5
                )

                if self._fire_model is not None:
                    fire_intensity = self._fire_model.get_intensity_at(state.position)
                    foam_eff = self._foam_physics.compute_foam_effectiveness(
                        foam_amount,
                        fire_intensity * 1000.0,
                        slope_angle=0.0,
                        age=0.0,
                    )
                    effectiveness *= foam_eff

                incident.reduce_severity(0.3 * effectiveness)
                if incident.is_resolved():
                    self._resolved_incidents += 1

    def _police_control_crowd(self, agent: str, state: AgentState) -> None:
        """Police controls crowded areas.

        Police must be at the scene (within range) to respond to the event.
        """
        police_range = self.config.action_ranges.get("police_range", 60.0)

        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if (
            incident
            and dist < police_range
            and incident.incident_type == ZoneType.CROWDED
        ):
            effectiveness = 1.0 - (dist / police_range)
            incident.reduce_severity(0.15 * effectiveness)
            if incident.is_resolved():
                self._resolved_incidents += 1

    def _police_place_barrier(self, agent: str, state: AgentState) -> None:
        """Police places barriers to control area."""
        config = self._agent_configs[agent]
        if config.consume_resource("barriers", 1):
            pass

    def _perform_aerial_suppression(
        self,
        target_position: tuple[float, float],
        drop_type: str = "water",
        num_drops: int = 1,
        pattern: str = "spot",
    ) -> bool:
        """Perform aerial suppression drop.

        Args:
            target_position: Target position for drops
            drop_type: Type of drop ("water", "retardant")
            num_drops: Number of drops
            pattern: Drop pattern ("spot", "line")

        Returns:
            True if suppression was effective
        """
        if self._fire_model is None:
            return False

        wind_speed = self.config.wind_speed
        wind_direction = self.config.wind_direction

        if pattern == "line":
            drop_positions = self._aerial_physics.compute_line_drop_coverage(
                target_position,
                (target_position[0] + 100, target_position[1]),
                drop_spacing=20.0,
            )
        else:
            drop_positions = self._aerial_physics.compute_spot_drop_coverage(
                target_position,
                num_drops=num_drops,
                spread_radius=30.0,
            )

        total_effectiveness = 0.0

        for drop_pos in drop_positions:
            actual_pos = self._aerial_physics.compute_drop_accuracy(
                drop_pos,
                target_position,
                wind_speed,
                wind_direction,
            )

            drop_error = self._aerial_physics.compute_drop_error(
                wind_speed, release_height=100.0
            )

            for fire in self._fire_model.fires:
                dist = np.sqrt(
                    (fire.position[0] - actual_pos[0]) ** 2
                    + (fire.position[1] - actual_pos[1]) ** 2
                )

                if dist < drop_error + 20.0:
                    distance_factor = max(0.0, 1.0 - dist / (drop_error + 20.0))

                    amount = 100.0 if drop_type == "water" else 50.0
                    effectiveness = (
                        self._suppression_physics.compute_water_effectiveness(
                            amount,
                            fire.fire_line_intensity,
                            self.config.water_temperature,
                        )
                    )

                    fire.intensity *= max(
                        0.0, 1.0 - effectiveness * distance_factor * 0.3
                    )
                    fire.fire_line_intensity *= max(
                        0.0, 1.0 - effectiveness * distance_factor * 0.3
                    )

                    total_effectiveness += effectiveness * distance_factor

        return total_effectiveness > 0.1

    def _create_suppression_line(
        self,
        start_position: tuple[float, float],
        end_position: tuple[float, float],
        tool_type: str = "hand",
        num_workers: int = 1,
    ) -> list[tuple[float, float]]:
        """Create a suppression line between two points.

        Args:
            start_position: Start of line (x, y)
            end_position: End of line (x, y)
            tool_type: Type of tool ("hand", "chainsaw", "bulldozer")
            num_workers: Number of workers

        Returns:
            List of line segment positions
        """
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        line_length = np.sqrt(dx * dx + dy * dy)

        if line_length < 1.0:
            return [start_position]

        construction_rate = self._line_physics.compute_construction_rate(
            tool_type, num_workers, slope_angle=0.0
        )

        num_segments = max(1, int(line_length / (construction_rate * 10)))

        line_positions = []
        for i in range(num_segments + 1):
            t = i / max(1, num_segments)
            x = start_position[0] + dx * t
            y = start_position[1] + dy * t
            line_positions.append((x, y))

        if self._fire_model is not None:
            self._fire_model.add_containment_line(line_positions)

        return line_positions

    def _civilian_seek_help(self, agent: str, state: AgentState) -> None:
        """Civilian seeks help from responders."""
        nearest_incident, dist = self._get_nearest_active_incident(state.position)
        seek_help_distance = self.config.action_ranges.get(
            "civilian_seek_help_distance", 100.0
        )
        if nearest_incident and dist > seek_help_distance:
            nearest_responder = self._find_nearest_responder(state.position)
            if nearest_responder:
                direction = (
                    nearest_responder[0] - state.position[0],
                    nearest_responder[1] - state.position[1],
                )
                norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
                if norm > 0:
                    state.position = (
                        state.position[0]
                        + direction[0] / norm * self.config.agent_speed,
                        state.position[1]
                        + direction[1] / norm * self.config.agent_speed,
                    )

    def _find_nearest_responder(
        self, position: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Find nearest responder to a position."""
        nearest = None
        min_dist = float("inf")
        for agent in self.agents:
            if self._get_agent_type(agent) == AgentType.CIVILIAN:
                continue
            state = self._agent_states[agent]
            if state.is_alive():
                dist = np.sqrt(
                    (state.position[0] - position[0]) ** 2
                    + (state.position[1] - position[1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = state.position
        return nearest

    def _compute_reward(self, agent: str) -> float:
        """Compute reward for an agent."""
        reward = 0.0

        reward += self.config.reward_weights.get("time_penalty", 0)

        state = self._agent_states[agent]
        if state.is_alive():
            reward += (
                self.config.reward_weights.get("stamina_penalty", 0)
                * (100 - state.stamina)
                / 100
            )

            if state.agent_status == AgentStatus.AFFECTED:
                reward += self.config.reward_weights.get("survived", 0) * 0.1

        if state.agent_status == AgentStatus.DECEASED:
            reward += self.config.reward_weights.get("died", 0)

        return reward

    def _check_terminations(self) -> None:
        """Check if any agents or the episode should terminate."""
        for agent in self.agents:
            state = self._agent_states[agent]
            if not state.is_alive():
                self.terminations[agent] = True

        if self._resolved_incidents >= self._total_incidents:
            for agent in self.agents:
                self.terminations[agent] = True

    def observe(self, agent: str) -> np.ndarray:
        """Get observation for an agent."""
        if agent not in self.agents:
            return np.zeros(self._observation_spaces[agent].shape, dtype=np.float32)

        state = self._agent_states[agent]
        vision_radius = int(self.config.agent_vision_radius)
        obs_size = vision_radius * 2

        obs = np.zeros((obs_size, obs_size, 4), dtype=np.float32)

        for other_agent in self.agents:
            if other_agent == agent:
                continue

            other_state = self._agent_states[other_agent]
            if not other_state.is_alive():
                continue

            dx = other_state.position[0] - state.position[0]
            dy = other_state.position[1] - state.position[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < self.config.agent_vision_radius:
                ox = int(dx + vision_radius)
                oy = int(dy + vision_radius)
                if 0 <= ox < obs_size and 0 <= oy < obs_size:
                    agent_type = self._get_agent_type(other_agent)
                    obs[oy, ox, agent_type.value - 1] = 1.0

        for incident in self._emergency_map.incidents:
            if not incident.active:
                continue
            dx = incident.position[0] - state.position[0]
            dy = incident.position[1] - state.position[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < self.config.agent_vision_radius:
                ox = int(dx + vision_radius)
                oy = int(dy + vision_radius)
                if 0 <= ox < obs_size and 0 <= oy < obs_size:
                    obs[oy, ox, 3] = incident.severity

        return obs

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self._observation_spaces[agent]

    def render(
        self,
        mode: str = "human",
        graph_filter: GraphFilter | None = None,
    ) -> np.ndarray | None:
        """Render the environment.

        Args:
            mode: Rendering mode. Options: "human", "map", "graph", "rgb_array"
            graph_filter: Filter for graph visualization

        Returns:
            Rendered image as numpy array if mode is "rgb_array", else None
        """
        if mode == "rgb_array":
            render_mode = RenderMode.BOTH
            interactive = False
        elif mode == "map":
            render_mode = RenderMode.MAP
            interactive = False
        elif mode == "graph":
            render_mode = RenderMode.GRAPH
            interactive = False
        else:
            render_mode = RenderMode.BOTH
            interactive = True

        if self._renderer is None:
            self._renderer = FireSimRenderer(self._render_config)

        agent_types = {agent: self._get_agent_type(agent) for agent in self.agents}

        return self._renderer.render(
            emergency_map=self._emergency_map,
            agent_states=self._agent_states,
            agent_types=agent_types,
            agent_configs=self._agent_configs,
            mode=render_mode,
            graph_filter=graph_filter,
            interactive=interactive,
            episode_metrics=self._episode_metrics,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
