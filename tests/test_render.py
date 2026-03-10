"""Test rendering functionality."""

from emmarl.envs import FireEnv, AgentType
from emmarl.envs.agent import AgentState
from emmarl.envs.metrics import EpisodeMetrics
from emmarl.envs.render import (
    FireSimRenderer,
    GraphFilter,
    RenderConfig,
)


def test_render_basic():
    """Test basic rendering."""
    env = FireEnv()
    env.reset()

    img = env.render(mode="rgb_array")
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_map_only():
    """Test map-only rendering."""
    env = FireEnv()
    env.reset()

    img = env.render(mode="map")
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_graph_only():
    """Test graph-only rendering."""
    env = FireEnv()
    env.reset()

    img = env.render(mode="graph")
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_with_filter():
    """Test rendering with graph filter."""
    env = FireEnv()
    env.reset()

    graph_filter = GraphFilter(
        agent_types=[AgentType.MEDIC, AgentType.FIRE_FORCE],
        show_collaboration=True,
        show_hierarchy=True,
        show_proximity=False,
        min_stamina=50.0,
    )

    img = env.render(mode="graph", graph_filter=graph_filter)
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_all_agent_types():
    """Test filtering by different agent types."""
    env = FireEnv()
    env.reset()

    for agent_type in [
        AgentType.MEDIC,
        AgentType.FIRE_FORCE,
        AgentType.POLICE,
        AgentType.CIVILIAN,
    ]:
        graph_filter = GraphFilter(agent_types=[agent_type])
        img = env.render(mode="graph", graph_filter=graph_filter)
        assert img is not None

    env.close()


def test_renderer_direct():
    """Test FireSimRenderer directly."""
    from emmarl.envs.map import create_default_map

    renderer = FireSimRenderer()

    emergency_map = create_default_map()
    agent_states = {
        "medic_0": AgentState(
            position=(100.0, 100.0),
            velocity=(1.0, 0.0),
            health=100.0,
            stamina=80.0,
            status="active",
        ),
    }
    agent_types = {"medic_0": AgentType.MEDIC}
    agent_configs = {}

    img = renderer.render(emergency_map, agent_states, agent_types, agent_configs)
    assert img is not None
    assert img.shape[2] == 3

    renderer.close()


def test_render_with_steps():
    """Test rendering after several steps."""
    env = FireEnv()
    env.reset()

    for _ in range(10):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)

    img = env.render(mode="rgb_array")
    assert img is not None

    env.close()


if __name__ == "__main__":
    test_render_basic()
    test_render_map_only()
    test_render_graph_only()
    test_render_with_filter()
    test_render_all_agent_types()
    test_renderer_direct()
    test_render_with_steps()
    test_episode_metrics_initialization()
    test_episode_metrics_record()
    test_env_metrics_tracking()
    test_render_with_metrics()
    test_render_graph_with_metrics()
    test_render_metrics_direct()
    test_render_config_metrics_option()
    print("All rendering tests passed!")


def test_episode_metrics_initialization():
    """Test that EpisodeMetrics initializes correctly."""
    metrics = EpisodeMetrics()
    assert len(metrics.steps) == 0
    assert len(metrics.avg_health) == 0
    assert len(metrics.avg_stamina) == 0


def test_episode_metrics_record():
    """Test recording metrics."""
    metrics = EpisodeMetrics()
    agent_states = {
        "medic_0": AgentState(position=(100.0, 100.0), health=80.0, stamina=60.0),
        "firefighter_0": AgentState(
            position=(200.0, 200.0), health=100.0, stamina=90.0
        ),
    }

    metrics.record(
        step=0, agent_states=agent_states, active_incidents=2, resolved_incidents=0
    )

    assert len(metrics.steps) == 1
    assert metrics.steps[0] == 0
    assert len(metrics.avg_health) == 1
    assert metrics.avg_health[0] == 90.0
    assert metrics.active_incidents[0] == 2
    assert metrics.resolved_incidents[0] == 0


def test_env_metrics_tracking():
    """Test that FireEnv tracks metrics over episodes."""
    env = FireEnv()
    env.reset()

    for _ in range(5):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)

    metrics = env.get_metrics()
    assert len(metrics.steps) > 0
    assert len(metrics.avg_health) == len(metrics.steps)
    assert len(metrics.avg_stamina) == len(metrics.steps)

    env.close()


def test_render_with_metrics():
    """Test rendering with metrics."""
    env = FireEnv()
    env.reset()

    for _ in range(10):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)

    img = env.render(mode="rgb_array")
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_graph_with_metrics():
    """Test rendering graph mode with metrics."""
    env = FireEnv()
    env.reset()

    for _ in range(10):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)

    img = env.render(mode="graph")
    assert img is not None
    assert img.shape[2] == 3

    env.close()


def test_render_metrics_direct():
    """Test rendering metrics directly via FireSimRenderer."""
    from emmarl.envs.map import create_default_map

    renderer = FireSimRenderer()

    emergency_map = create_default_map()
    agent_states = {
        "medic_0": AgentState(position=(100.0, 100.0), health=100.0, stamina=80.0),
    }
    agent_types = {"medic_0": AgentType.MEDIC}
    agent_configs = {}

    metrics = EpisodeMetrics()
    metrics.record(
        step=0,
        agent_states=agent_states,
        active_incidents=1,
        resolved_incidents=0,
    )
    metrics.record(
        step=1,
        agent_states=agent_states,
        active_incidents=1,
        resolved_incidents=0,
    )

    img = renderer.render(
        emergency_map, agent_states, agent_types, agent_configs, episode_metrics=metrics
    )
    assert img is not None

    renderer.close()


def test_render_config_metrics_option():
    """Test RenderConfig with metrics options."""
    config = RenderConfig(show_metrics=False)
    assert config.show_metrics is False

    config = RenderConfig(show_metrics=True)
    assert config.show_metrics is True
