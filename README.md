# EM-MARL

Emergency response multi-agent reinforcement learning simulator built on PettingZoo.

## Overview

EM-MARL simulates emergency response scenarios where multiple agent types (medics, fire force, police, civilians) collaborate to handle various incidents like fires, medical emergencies, and crowd control situations.

## Features

- **Multi-Agent RL**: PettingZoo-based AEC environment for multi-agent learning
- **4 Agent Types**: Medic, FireForce, Police, Civilian - each with unique actions
- **Map-Based Layout**: Continuous coordinate system with zones and incidents
- **Movement Physics**: Velocity-based movement with stamina and running
- **Resource Management**: Agents have limited resources (medkits, water, barriers, etc.)
- **Fire Dynamics**: Rothermel-based fire propagation model

## Quick Start

```python
from emmarl.envs import FireEnv

env = FireEnv()
env.reset()

for agent in env.agent_iter(max_iter=1000):
    obs = env.observe(agent)
    action = env.action_space(agent).sample()
    env.step(action)

env.close()
```

## Installation

```bash
# Install dependencies
pixi install

# Run tests
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest tests/
```

## Requirements

- Python 3.12
- pettingzoo >=1.25.0
- gymnasium >=1.0.0
- numpy >=2.4.2
- torch >=2.10.0

See `pixi.toml` for full dependencies.
