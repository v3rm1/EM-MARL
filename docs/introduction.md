# Introduction

## What is FireSim?

FireSim is an emergency response multi-agent reinforcement learning simulator that provides a flexible framework for studying coordination and cooperation among heterogeneous agents in disaster response scenarios.

## Motivation

Emergency response scenarios present unique challenges for multi-agent systems:

1. **Heterogeneous Agents**: Different agent types have different capabilities, limitations, and objectives
2. **Resource Constraints**: Limited resources (medkits, water, barriers) must be allocated strategically
3. **Dynamic Environment**: Incidents evolve over time, requiring adaptive responses
4. **Partial Observability**: Agents have limited vision and must make decisions with incomplete information
5. **Temporal Dependencies**: Tasks often require sustained effort over multiple time steps

## Design Principles

### PettingZoo-Based

Built on [PettingZoo](https://pettingzoo.farama.org/), FireSim follows the standard multi-agent environment API:

- **AEC (Agent Environment Cycle)**: Turn-based execution where agents act sequentially
- **Parallel**: Support for parallel execution where needed
- **Standard Interface**: Compatible with popular MARL libraries

### Map-Based vs Gridworld

Unlike traditional gridworld environments, FireSim uses a continuous coordinate system:

- **Continuous Space**: Agents move in 2D space with float coordinates
- **Zones**: Geographic areas with different properties (fire, medical, crowd)
- **Incidents**: Dynamic events that can be resolved through agent actions
- **Physics-Based Movement**: Velocity, acceleration, and momentum

### Agent Specialization

Each agent type has distinct:

- **Action Space**: Different available actions
- **Movement Properties**: Speed, stamina costs, terrain capabilities
- **Resources**: Unique inventory items
- **Objectives**: Different goals within the scenario

## Comparison with Other Environments

| Feature | FireSim | MAgent | MultiAgentHumanoid |
|---------|---------|--------|-------------------|
| Map-based | Yes | No | Limited |
| Agent types | 4+ | 2 | 1 |
| Resource system | Yes | Limited | No |
| Physics movement | Yes | No | Yes |
| PettingZoo | Yes | Yes | Yes |

## Getting Started

See [Installation](installation.md) for setup instructions, or jump to [Environment](environment.md) to understand the simulation mechanics.
