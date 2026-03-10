# Examples

## Basic Usage

### Simple Episode

```python
import numpy as np
from emmarl.envs import FireEnv

env = FireEnv()
env.reset()

for agent in env.agent_iter(max_iter=1000):
    obs = env.observe(agent)
    action = env.action_space(agent).sample()
    env.step(action)

env.close()
```

### With Custom Configuration

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig

config = FireEnvConfig(
    num_medics=3,
    num_fire_force=4,
    num_police=2,
    num_civilians=10,
    map_width=2000.0,
    map_height=2000.0,
    max_steps=2000,
)

env = FireEnv(config)
env.reset()
```

### Random Agent Policy

```python
import numpy as np
from emmarl.envs import FireEnv

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    
    def act(self, agent):
        action_space = self.env.action_space(agent)
        return action_space.sample()

env = FireEnv()
policy = RandomPolicy(env)
env.reset()

steps = 0
for agent in env.agent_iter(max_iter=1000):
    action = policy.act(agent)
    env.step(action)
    steps += 1
    
    if all(env.terminations.values()):
        break

print(f"Episode lasted {steps} steps")
env.close()
```

## Multi-Agent Training

### Gathering Observations

```python
from emmarl.envs import FireEnv
import numpy as np

env = FireEnv()
env.reset()

# Collect observations for all agents
observations = {}
for agent in env.possible_agents:
    observations[agent] = env.observe(agent)

# Now iterate
for agent in env.agent_iter():
    obs = env.observe(agent)
    observations[agent] = obs
    
    action = your_policy(agent, obs)
    env.step(action)
```

### Reward Analysis

```python
from emmarl.envs import FireEnv
from collections import defaultdict

env = FireEnv()
env.reset()

episode_rewards = defaultdict(list)

for agent in env.agent_iter(max_iter=1000):
    reward = env.rewards[agent]
    episode_rewards[agent].append(reward)
    
    action = env.action_space(agent).sample()
    env.step(action)

# Summary
for agent, rewards in episode_rewards.items():
    print(f"{agent}: total={sum(rewards):.2f}, mean={np.mean(rewards):.2f}")

env.close()
```

## Custom Map

### Creating a Scenario

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig
from emmarl.envs.map import (
    EmergencyMap, Zone, Incident, ZoneType
)

# Create custom map
custom_map = EmergencyMap(width=1500.0, height=1500.0)

# Add fire zone
custom_map.add_zone(Zone(
    zone_id="industrial_fire",
    zone_type=ZoneType.FIRE,
    position=(400.0, 400.0),
    size=(150.0, 150.0),
    intensity=0.9,
))

# Add incident
custom_map.add_incident(Incident(
    incident_id="fire_1",
    incident_type=ZoneType.FIRE,
    position=(400.0, 400.0),
    severity=0.9,
    resources_required={"water": 80.0, "foam": 40.0},
))

# Note: Custom map integration requires modifying FireEnv
# or creating a custom environment subclass
```

## Agent Analysis

### Inspect Agent State

```python
from emmarl.envs import FireEnv
from emmarl.envs import AgentType

env = FireEnv()
env.reset()

# Run some steps
for _ in range(100):
    for agent in env.agents:
        action = env.action_space(agent).sample()
        env.step(action)

# Inspect states
for agent in env.possible_agents:
    state = env._agent_states[agent]
    config = env._agent_configs[agent]
    
    print(f"\n{agent} ({config.agent_type.name}):")
    print(f"  Position: {state.position}")
    print(f"  Health: {state.health:.1f}")
    print(f"  Stamina: {state.stamina:.1f}")
    print(f"  Velocity: {state.velocity}")
    print(f"  Status: {state.status}")
    print(f"  Resources: {config.resources}")

env.close()
```

### Tracking Incidents

```python
from emmarl.envs import FireEnv

env = FireEnv()
env.reset()

initial_incidents = len(env._emergency_map.incidents)
resolved = 0

for agent in env.agent_iter(max_iter=1000):
    current_resolved = sum(
        1 for i in env._emergency_map.incidents 
        if i.is_resolved()
    )
    
    if current_resolved > resolved:
        print(f"Incident resolved! Total: {current_resolved}/{initial_incidents}")
        resolved = current_resolved
    
    action = env.action_space(agent).sample()
    env.step(action)
    
    if resolved >= initial_incidents:
        print("All incidents resolved!")
        break

env.close()
```

## Visualization

### Basic Rendering

```python
from emmarl.envs import FireEnv

env = FireEnv()
env.reset()

# Render a single frame
img = env.render(mode="rgb_array")
# img shape: (height, width, 3)

env.close()
```

### Live Interactive Rendering

```python
import matplotlib.pyplot as plt
from emmarl.envs import FireEnv

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

### Rendering with Metrics

Enable metrics visualization to track agent status and performance over time:

```python
import matplotlib.pyplot as plt
from emmarl.envs import FireEnv
from emmarl.envs.render import RenderConfig

config = RenderConfig(show_metrics=True)

env = FireEnv()
env._render_config = config
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

### Using GIS Maps

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig
from emmarl.envs.gis_map import create_default_gis_map

gis_map = create_default_gis_map()
config = FireEnvConfig(use_gis=True, gis_map=gis_map)

env = FireEnv(config)
env.reset()
# ... run simulation
```

Or use the command-line tool:

```bash
python random_agents.py --gis
```

## Integration with RL Libraries

### Basic Training Loop (Pseudo-code)

```python
from emmarl.envs import FireEnv
import torch

env = FireEnv()

# Initialize your agents/policy
policy = YourPolicy()

for episode in range(1000):
    env.reset()
    episode_rewards = {agent: 0 for agent in env.possible_agents}
    
    for agent in env.agent_iter():
        obs = env.observe(agent)
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action = policy(obs_tensor)
        
        env.step(action)
        
        reward = env.rewards[agent]
        episode_rewards[agent] += reward
    
    # Log episode rewards
    print(f"Episode {episode}: {episode_rewards}")

env.close()
```

### Converting Observations

```python
import numpy as np
from emmarl.envs import FireEnv

env = FireEnv()
env.reset()

for agent in env.agent_iter():
    obs = env.observe(agent)
    
    # obs shape: (200, 200, 4)
    # Flatten for MLP
    obs_flat = obs.flatten()
    
    # Or keep spatial for CNN
    # obs shape: (4, 200, 200) - channels first
    obs_cf = np.transpose(obs, (2, 0, 1))
    
    env.step(env.action_space(agent).sample())
```
