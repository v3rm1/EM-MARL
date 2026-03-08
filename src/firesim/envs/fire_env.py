"""FireSim: PettingZoo environment for emergency response multi-agent simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from firesim.envs.agent import AgentConfig, AgentState, AgentType
from firesim.envs.map import EmergencyMap, ZoneType, create_default_map
from firesim.envs.gis_map import GISMap, create_default_gis_map
from firesim.envs.render import (
    FireSimRenderer,
    RenderConfig,
    RenderMode,
    GraphFilter,
)


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
    use_gis: bool = False
    gis_map: GISMap | None = None
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "incident_resolved": 100.0,
            "casualty_prevented": 50.0,
            "damage_reduced": 25.0,
            "stamina_penalty": -0.1,
            "time_penalty": -0.01,
        }
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

    metadata: dict = {"render_modes": ["human"], "name": "firesim_v0"}

    def __init__(self, config: FireEnvConfig | None = None) -> None:
        """Initialize the FireSim environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or FireEnvConfig()

        self.possible_agents = self._create_agent_ids()
        self.agents = self.possible_agents.copy()

        self._agent_configs: dict[str, AgentConfig] = {}
        self._agent_states: dict[str, AgentState] = {}

        if self.config.use_gis:
            self._gis_map: GISMap | None = (
                self.config.gis_map or create_default_gis_map()
            )
            self._emergency_map: EmergencyMap | None = None
        else:
            self._emergency_map = create_default_map(
                width=self.config.map_width,
                height=self.config.map_height,
            )
            self._gis_map = None

        self._current_step = 0
        self._agent_selector: AgentSelector | None = None
        self._cumulative_rewards: dict[str, float] = {}

        self._action_spaces = self._create_action_spaces()
        self._observation_spaces = self._create_observation_spaces()

        if self._emergency_map:
            self._total_incidents = len(self._emergency_map.incidents)
        else:
            self._total_incidents = (
                len(self._gis_map.fire_zones) if self._gis_map else 0
            )
        self._resolved_incidents = 0

        self._renderer: FireSimRenderer | None = None
        self._render_config = RenderConfig()

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

        self._emergency_map = create_default_map(
            width=self.config.map_width,
            height=self.config.map_height,
        )
        self._resolved_incidents = 0
        self._total_incidents = len(self._emergency_map.incidents)

        self._initialize_agents()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

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
        margin = 50.0
        while True:
            x = np.random.uniform(margin, self.config.map_width - margin)
            y = np.random.uniform(margin, self.config.map_height - margin)
            danger = self._emergency_map.get_danger_level((x, y))
            if danger < 0.3:
                return (x, y)

    def step(self, action: Any) -> None:
        """Execute one step of the environment."""
        if self.agent_selection not in self.agents:
            return

        agent = self.agent_selection
        self._execute_action(agent, action)

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

        speed_multiplier = move_config.run_multiplier if state.is_running else 1.0
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
            turn_rate = 0.2
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

        if self._emergency_map.is_within_bounds((new_x, new_y)):
            danger = self._emergency_map.get_danger_level((new_x, new_y))
            if danger < 0.8:
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
            state.restore_stamina(0.5)

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
            if dist < 30 and other_state.health < 100:
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
            if dist < 20 and other_state.health < 50:
                if config.consume_resource("medication", 1):
                    other_state.heal(50)
                    break

    def _firefighter_extinguish(self, agent: str, state: AgentState) -> None:
        """Firefighter extinguishes nearby fire incidents."""
        config = self._agent_configs[agent]
        if not config.has_resource("water", 10):
            return

        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if incident and dist < 50 and incident.incident_type == ZoneType.FIRE:
            if config.consume_resource("water", 10):
                incident.reduce_severity(0.2)
                if incident.is_resolved():
                    self._resolved_incidents += 1

    def _firefighter_use_foam(self, agent: str, state: AgentState) -> None:
        """Firefighter uses foam on hazardous incidents."""
        config = self._agent_configs[agent]
        if not config.has_resource("foam", 5):
            return

        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if (
            incident
            and dist < 40
            and incident.incident_type
            in (
                ZoneType.FIRE,
                ZoneType.HAZMAT,
            )
        ):
            if config.consume_resource("foam", 5):
                incident.reduce_severity(0.3)
                if incident.is_resolved():
                    self._resolved_incidents += 1

    def _police_control_crowd(self, agent: str, state: AgentState) -> None:
        """Police controls crowded areas."""
        incident, dist = self._emergency_map.get_nearest_active_incident(state.position)
        if incident and dist < 60 and incident.incident_type == ZoneType.CROWDED:
            incident.reduce_severity(0.15)
            if incident.is_resolved():
                self._resolved_incidents += 1

    def _police_place_barrier(self, agent: str, state: AgentState) -> None:
        """Police places barriers to control area."""
        config = self._agent_configs[agent]
        if config.consume_resource("barriers", 1):
            pass

    def _civilian_seek_help(self, agent: str, state: AgentState) -> None:
        """Civilian seeks help from responders."""
        nearest_incident, dist = self._emergency_map.get_nearest_active_incident(
            state.position
        )
        if nearest_incident and dist > 100:
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

        render_map = self._gis_map if self._gis_map else self._emergency_map

        return self._renderer.render(
            emergency_map=render_map,
            agent_states=self._agent_states,
            agent_types=agent_types,
            agent_configs=self._agent_configs,
            mode=render_mode,
            graph_filter=graph_filter,
            interactive=interactive,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
