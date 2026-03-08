"""Test rendering functionality."""

from firesim.envs import FireEnv, AgentType
from firesim.envs.agent import AgentState
from firesim.envs.render import (
    FireSimRenderer,
    GraphFilter,
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
    from firesim.envs.map import create_default_map

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
    print("All rendering tests passed!")
