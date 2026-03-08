"""Tests for FireSim environment."""

import numpy as np

from firesim.envs import FireEnv, AgentType
from firesim.envs.fire_env import FireEnvConfig
from firesim.envs.agent import AgentConfig, AgentState, MovementConfig, AgentTypeConfig
from firesim.envs.map import EmergencyMap, Incident, Zone, ZoneType, create_default_map


class TestFireEnvConfig:
    """Tests for FireEnvConfig."""

    def test_default_config(self):
        config = FireEnvConfig()
        assert config.num_medics == 2
        assert config.num_fire_force == 2
        assert config.num_police == 2
        assert config.num_civilians == 5
        assert config.max_steps == 1000

    def test_custom_config(self):
        config = FireEnvConfig(
            num_medics=3,
            num_fire_force=4,
            num_police=2,
            num_civilians=10,
            max_steps=500,
        )
        assert config.num_medics == 3
        assert config.num_fire_force == 4
        assert config.num_police == 2
        assert config.num_civilians == 10
        assert config.max_steps == 500


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_types_exist(self):
        assert AgentType.MEDIC is not None
        assert AgentType.FIRE_FORCE is not None
        assert AgentType.POLICE is not None
        assert AgentType.CIVILIAN is not None


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_create_medic_config(self):
        config = AgentConfig(agent_type=AgentType.MEDIC, agent_id="medic_0")
        assert config.agent_type == AgentType.MEDIC
        assert config.agent_id == "medic_0"
        assert config.health == 100.0
        assert config.resources["medkits"] == 5
        assert config.resources["medication"] == 3

    def test_create_fireforce_config(self):
        config = AgentConfig(agent_type=AgentType.FIRE_FORCE, agent_id="firefighter_0")
        assert config.agent_type == AgentType.FIRE_FORCE
        assert config.resources["water"] == 100.0
        assert config.resources["foam"] == 50.0

    def test_create_police_config(self):
        config = AgentConfig(agent_type=AgentType.POLICE, agent_id="police_0")
        assert config.agent_type == AgentType.POLICE
        assert config.resources["barriers"] == 10
        assert config.resources["flare"] == 5

    def test_resource_management(self):
        config = AgentConfig(agent_type=AgentType.MEDIC, agent_id="medic_0")
        assert config.has_resource("medkits", 1)
        assert config.consume_resource("medkits", 1)
        assert config.resources["medkits"] == 4
        assert not config.consume_resource("medkits", 10)


class TestAgentState:
    """Tests for AgentState."""

    def test_create_state(self):
        state = AgentState(position=(100.0, 200.0))
        assert state.position == (100.0, 200.0)
        assert state.health == 100.0
        assert state.stamina == 100.0
        assert state.is_alive()

    def test_take_damage(self):
        state = AgentState(position=(0.0, 0.0))
        state.take_damage(30)
        assert state.health == 70
        assert state.is_alive()
        state.take_damage(80)
        assert state.health == 0
        assert not state.is_alive()

    def test_use_stamina(self):
        state = AgentState(position=(0.0, 0.0))
        assert state.use_stamina(30)
        assert state.stamina == 70
        assert not state.use_stamina(80)

    def test_heal(self):
        state = AgentState(position=(0.0, 0.0))
        state.take_damage(50)
        state.heal(30)
        assert state.health == 80


class TestEmergencyMap:
    """Tests for EmergencyMap."""

    def test_create_map(self):
        emergency_map = EmergencyMap(width=1000.0, height=1000.0)
        assert emergency_map.width == 1000.0
        assert emergency_map.height == 1000.0

    def test_add_zone(self):
        emergency_map = EmergencyMap(width=1000.0, height=1000.0)
        zone = Zone(
            zone_id="fire_zone",
            zone_type=ZoneType.FIRE,
            position=(500.0, 500.0),
            size=(100.0, 100.0),
        )
        emergency_map.add_zone(zone)
        assert len(emergency_map.zones) == 1
        assert emergency_map.get_zone_by_id("fire_zone") == zone

    def test_zone_contains_point(self):
        zone = Zone(
            zone_id="test",
            zone_type=ZoneType.FIRE,
            position=(500.0, 500.0),
            size=(100.0, 100.0),
        )
        assert zone.contains_point((500.0, 500.0))
        assert zone.contains_point((450.0, 550.0))
        assert not zone.contains_point((700.0, 700.0))

    def test_add_incident(self):
        emergency_map = EmergencyMap(width=1000.0, height=1000.0)
        incident = Incident(
            incident_id="fire_1",
            incident_type=ZoneType.FIRE,
            position=(300.0, 300.0),
            severity=0.8,
        )
        emergency_map.add_incident(incident)
        assert len(emergency_map.incidents) == 1
        assert emergency_map.get_incident_at((300.0, 300.0), radius=10) is not None

    def test_is_within_bounds(self):
        emergency_map = EmergencyMap(width=1000.0, height=1000.0)
        assert emergency_map.is_within_bounds((500.0, 500.0))
        assert emergency_map.is_within_bounds((0.0, 0.0))
        assert emergency_map.is_within_bounds((1000.0, 1000.0))
        assert not emergency_map.is_within_bounds((-1.0, 500.0))
        assert not emergency_map.is_within_bounds((500.0, 1001.0))


class TestCreateDefaultMap:
    """Tests for create_default_map function."""

    def test_default_map_creation(self):
        emergency_map = create_default_map()
        assert emergency_map.width == 1000.0
        assert emergency_map.height == 1000.0
        assert len(emergency_map.zones) > 0
        assert len(emergency_map.incidents) > 0


class TestFireEnv:
    """Tests for FireEnv."""

    def test_env_creation(self):
        env = FireEnv()
        assert len(env.possible_agents) == 11

    def test_env_reset(self):
        env = FireEnv()
        env.reset()
        assert len(env.agents) == 11
        assert env._current_step == 0

    def test_env_reset_with_seed(self):
        env = FireEnv()
        env.reset(seed=42)
        assert env._current_step == 0

    def test_observation_shape(self):
        env = FireEnv()
        env.reset()
        obs = env.observe(env.agents[0])
        assert obs.shape == (200, 200, 4)

    def test_action_spaces(self):
        env = FireEnv()
        for agent in env.possible_agents:
            action_space = env.action_space(agent)
            assert action_space is not None

    def test_observation_spaces(self):
        env = FireEnv()
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            assert obs_space is not None

    def test_env_step(self):
        env = FireEnv()
        env.reset()
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        env.step(action)
        assert env._current_step == 1

    def test_env_step_updates_agent_selection(self):
        env = FireEnv()
        env.reset()
        first_agent = env.agent_selection
        action = env.action_space(first_agent).sample()
        env.step(action)
        assert env.agent_selection != first_agent

    def test_multiple_steps(self):
        env = FireEnv()
        env.reset()
        for _ in range(10):
            agent = env.agent_selection
            action = env.action_space(agent).sample()
            env.step(action)
        assert env._current_step == 10

    def test_rewards_initialized(self):
        env = FireEnv()
        env.reset()
        for agent in env.agents:
            assert agent in env.rewards

    def test_terminations_truncations_initialized(self):
        env = FireEnv()
        env.reset()
        for agent in env.agents:
            assert env.terminations[agent] is False
            assert env.truncations[agent] is False

    def test_env_close(self):
        env = FireEnv()
        env.reset()
        env.close()


class TestFireEnvCustomConfig:
    """Tests for FireEnv with custom configuration."""

    def test_custom_agent_counts(self):
        config = FireEnvConfig(
            num_medics=1,
            num_fire_force=1,
            num_police=1,
            num_civilians=2,
        )
        env = FireEnv(config)
        assert len(env.possible_agents) == 5

    def test_custom_map_size(self):
        config = FireEnvConfig(
            map_width=500.0,
            map_height=600.0,
        )
        env = FireEnv(config)
        assert env._emergency_map.width == 500.0
        assert env._emergency_map.height == 600.0


class TestMovementConfig:
    """Tests for MovementConfig."""

    def test_movement_config_creation(self):
        config = MovementConfig(
            max_speed=10.0,
            max_acceleration=5.0,
            stamina_cost_per_step=1.0,
        )
        assert config.max_speed == 10.0
        assert config.max_acceleration == 5.0

    def test_agent_type_configs(self):
        assert AgentTypeConfig.MEDIC.max_speed == 8.0
        assert AgentTypeConfig.FIRE_FORCE.max_speed == 10.0
        assert AgentTypeConfig.POLICE.max_speed == 11.0
        assert AgentTypeConfig.CIVILIAN.max_speed == 7.0

    def test_agent_type_can_run(self):
        assert AgentTypeConfig.MEDIC.can_run is True
        assert AgentTypeConfig.FIRE_FORCE.can_run is True
        assert AgentTypeConfig.POLICE.can_run is True
        assert AgentTypeConfig.CIVILIAN.can_run is True

    def test_fireforce_can_climb(self):
        assert AgentTypeConfig.FIRE_FORCE.can_climb is True
        assert AgentTypeConfig.MEDIC.can_climb is False


class TestAgentStateMovement:
    """Tests for AgentState movement methods."""

    def test_get_speed(self):
        state = AgentState(position=(0.0, 0.0), velocity=(3.0, 4.0))
        assert state.get_speed() == 5.0

    def test_get_speed_zero_velocity(self):
        state = AgentState(position=(0.0, 0.0), velocity=(0.0, 0.0))
        assert state.get_speed() == 0.0

    def test_get_heading(self):
        state = AgentState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        assert abs(state.get_heading()) < 0.01

    def test_get_heading_diagonal(self):
        state = AgentState(position=(0.0, 0.0), velocity=(1.0, 1.0))
        expected = np.pi / 4
        assert abs(state.get_heading() - expected) < 0.01


class TestFireEnvMovement:
    """Tests for FireEnv movement."""

    def test_agent_velocity_update(self):
        env = FireEnv()
        env.reset()
        agent = env.agents[0]
        action = {"move": np.array([1.0, 0.0], dtype=np.float32), "run": 0}
        env.step(action)
        state = env._agent_states[agent]
        assert state.velocity[0] >= 0

    def test_agent_running(self):
        env = FireEnv()
        env.reset()
        agent = env.agents[0]
        initial_stamina = env._agent_states[agent].stamina
        action = {"move": np.array([1.0, 0.0], dtype=np.float32), "run": 1}
        env.step(action)
        final_stamina = env._agent_states[agent].stamina
        assert final_stamina < initial_stamina

    def test_agent_distance_traveled(self):
        env = FireEnv()
        env.reset()
        agent = env.agents[0]
        for _ in range(10):
            action = {"move": np.array([1.0, 0.0], dtype=np.float32), "run": 0}
            env.step(action)
        state = env._agent_states[agent]
        assert state.distance_traveled > 0

    def test_movement_stamina_cost(self):
        env = FireEnv()
        env.reset()
        agent = env.agents[0]
        initial_stamina = env._agent_states[agent].stamina
        action = {"move": np.array([0.0, 0.0], dtype=np.float32), "run": 0}
        env.step(action)
        final_stamina = env._agent_states[agent].stamina
        assert final_stamina < initial_stamina
