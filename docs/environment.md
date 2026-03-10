# Environment

## FireEnv Overview

`FireEnv` is the main environment class implementing the PettingZoo AEC (Agent Environment Cycle) API. It simulates an emergency response scenario where multiple agent types work together to handle various incidents.

## Basic Usage

```python
from emmarl.envs import FireEnv

# Create environment with default config
env = FireEnv()

# Initialize
env.reset()

# Run for one episode
for agent in env.agent_iter(max_iter=1000):
    observation = env.observe(agent)
    action = env.action_space(agent).sample()  # Your policy here
    env.step(action)

env.close()
```

## Visualization

The environment provides two rendering modes: the **Emergency Map** (grid-based) and **GIS Map** (geographic information system with terrain, roads, and buildings).

### Default EmergencyMap

![EmergencyMap Default](../images/emergency_map_default.png)

### GIS-Based Map

![GIS Map Default](../images/gis_map_default.png)

### Live Rendering with Metrics

When running simulations, you can enable live rendering with real-time metrics:

![EmergencyMap with Metrics](../images/emergency_map_with_metrics.png)

The metrics panel shows:
- **Agent Status Over Time**: Number of agents in each status category (Healthy, Injured, Affected, Critical, Deceased)
- **Health, Stamina & Incidents**: Average health and stamina of agents, plus active and resolved incidents

To enable live rendering:

```python
import matplotlib.pyplot as plt

env = FireEnv()
env.reset()

plt.ion()
env.render(mode="human")
plt.show()

for _ in range(1000):
    for agent in env.agents:
        action = env.action_space(agent).sample()
        env.step(action)
    env.render(mode="human")
    plt.pause(0.01)

env.close()
```

Use `--gis` flag with `random_agents.py` to use the GIS map:

```bash
python random_agents.py --gis
```

## Configuration

### FireEnvConfig

Customize the environment with `FireEnvConfig`:

```python
from emmarl.envs.fire_env import FireEnvConfig

config = FireEnvConfig(
    num_medics=3,
    num_fire_force=4,
    num_police=2,
    num_civilians=10,
    map_width=2000.0,
    map_height=2000.0,
    max_steps=2000,
    agent_speed=15.0,
    agent_vision_radius=150.0,
    reward_weights={
        "incident_resolved": 100.0,
        "casualty_prevented": 50.0,
        "stamina_penalty": -0.1,
    }
)

env = FireEnv(config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_medics` | int | 2 | Number of medic agents |
| `num_fire_force` | int | 2 | Number of firefighter agents |
| `num_police` | int | 2 | Number of police agents |
| `num_civilians` | int | 5 | Number of civilian agents |
| `map_width` | float | 1000.0 | Map width in units |
| `map_height` | float | 1000.0 | Map height in units |
| `max_steps` | int | 1000 | Maximum episode length |
| `agent_speed` | float | 10.0 | Base agent movement speed |
| `agent_vision_radius` | float | 100.0 | Agent observation radius |
| `use_gis` | bool | False | Use GIS-based map system |
| `gis_map` | GISMap | None | Custom GIS map (when use_gis=True) |
| `reward_weights` | dict | {...} | Reward function weights |

## Agent Types

The environment supports four agent types:

1. **Medic**: Heals agents, uses medication
2. **FireForce**: Extinguishes fires, uses foam
3. **Police**: Controls crowds, places barriers
4. **Civilian**: Seeks help, can be rescued

Each agent type has unique:
- Action space
- Movement properties
- Resources

See [Agents](agents.md) for details.

## Observations

Agents observe a local view of the environment:

```python
observation = env.observe(agent)
# Shape: (vision_radius*2, vision_radius*2, 4)
# Channels: agent types (medic, fire, police, civilian) + incidents
```

The observation is a 2D grid centered on the agent, containing:
- Other agents (color-coded by type)
- Active incidents (showing severity)
- Safe/danger zones

## Actions

Each agent type has its own action space. Common actions:

- **move**: 2D movement vector [-1, 1]
- **run**: Toggle running (1) or walking (0)

Type-specific actions:
- Medic: `heal`, `use_medication`, `communicate`
- FireForce: `extinguish`, `use_foam`, `communicate`
- Police: `control_crowd`, `place_barrier`, `communicate`
- Civilian: `seek_help`

## Rewards

The reward function considers:

- **incident_resolved**: Points for resolving incidents
- **casualty_prevented**: Points for saving lives
- **damage_reduced**: Points for reducing incident severity
- **stamina_penalty**: Penalty for stamina depletion
- **time_penalty**: Small penalty per step

## Episode Termination

An episode ends when:
1. All incidents are resolved
2. All agents are dead
3. Maximum steps reached (truncation)

## PettingZoo API

FireEnv follows the PettingZoo AEC API:

```python
# Reset environment
env.reset(seed=42, options={})

# Iterate through agents
for agent in env.agent_iter(max_iter=1000):
    obs = env.observe(agent)
    action = policy(obs)
    env.step(action)

# Check termination
if env.terminations[agent] or env.truncations[agent]:
    # Agent done

# Access info
info = env.infos[agent]
```

See [PettingZoo documentation](https://pettingzoo.farama.org/) for more details.
