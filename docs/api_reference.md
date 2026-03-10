# API Reference

## Core Classes

### FireEnv

Main environment class implementing PettingZoo AEC API.

```python
from emmarl.envs import FireEnv
```

#### Constructor

```python
FireEnv(config: FireEnvConfig | None = None)
```

#### Methods

##### reset

```python
def reset(self, seed: int | None = None, options: dict | None = None) -> None
```

Reset the environment to initial state.

##### step

```python
def step(self, action: Any) -> None
```

Execute one step for the current agent.

##### observe

```python
def observe(self, agent: str) -> np.ndarray
```

Get observation for an agent.

##### action_space

```python
def action_space(self, agent: str) -> spaces.Space
```

Get action space for an agent.

##### observation_space

```python
def observation_space(self, agent: str) -> spaces.Space
```

Get observation space for an agent.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `agents` | list[str] | Current active agents |
| `possible_agents` | list[str] | All possible agents |
| `rewards` | dict[str, float] | Current rewards |
| `terminations` | dict[str, bool] | Termination flags |
| `truncations` | dict[str, bool] | Truncation flags |
| `infos` | dict[str, dict] | Additional info |

---

### FireEnvConfig

Configuration for FireEnv.

```python
from emmarl.envs.fire_env import FireEnvConfig
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_medics` | int | 2 | Number of medics |
| `num_fire_force` | int | 2 | Number of firefighters |
| `num_police` | int | 2 | Number of police |
| `num_civilians` | int | 5 | Number of civilians |
| `map_width` | float | 1000.0 | Map width |
| `map_height` | float | 1000.0 | Map height |
| `max_steps` | int | 1000 | Max episode steps |
| `agent_speed` | float | 10.0 | Base agent speed |
| `agent_vision_radius` | float | 100.0 | Observation radius |
| `reward_weights` | dict | {...} | Reward components |

---

## Agent Classes

### AgentType

Enum for agent types.

```python
from emmarl.envs import AgentType

AgentType.MEDIC
AgentType.FIRE_FORCE
AgentType.POLICE
AgentType.CIVILIAN
```

### AgentConfig

Configuration for a single agent.

```python
from emmarl.envs.agent import AgentConfig

config = AgentConfig(
    agent_type=AgentType.MEDIC,
    agent_id="medic_0",
    health=100.0,
    stamina=100.0,
    position=(0.0, 0.0),
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `movement_config` | MovementConfig | Movement parameters |
| `resources` | dict[str, float] | Agent resources |

#### Methods

- `has_resource(name, amount)` - Check if resource available
- `consume_resource(name, amount)` - Use resource (returns success)
- `add_resource(name, amount)` - Add resources

### AgentState

Mutable state for an agent.

```python
from emmarl.envs.agent import AgentState

state = AgentState(
    position=(100.0, 200.0),
    velocity=(0.0, 0.0),
    health=100.0,
    stamina=100.0,
)
```

#### Methods

- `is_alive()` - Check if alive
- `is_exhausted()` - Check if stamina depleted
- `take_damage(amount)` - Apply damage
- `use_stamina(amount)` - Use stamina
- `restore_stamina(amount)` - Recover stamina
- `heal(amount)` - Restore health
- `get_speed()` - Current velocity magnitude
- `get_heading()` - Direction in radians

### MovementConfig

Movement parameters.

```python
from emmarl.envs.agent import MovementConfig

config = MovementConfig(
    max_speed=10.0,
    max_acceleration=5.0,
    stamina_cost_per_step=1.0,
    can_run=True,
    run_multiplier=2.0,
    can_climb=False,
    can_swim=False,
)
```

### AgentTypeConfig

Pre-defined configs for each agent type.

```python
from emmarl.envs.agent import AgentTypeConfig

AgentTypeConfig.MEDIC
AgentTypeConfig.FIRE_FORCE
AgentTypeConfig.POLICE
AgentTypeConfig.CIVILIAN
```

---

## Map Classes

### EmergencyMap

The simulation map.

```python
from emmarl.envs.map import EmergencyMap

emap = EmergencyMap(width=1000.0, height=1000.0)
```

#### Methods

- `add_zone(zone)` - Add zone
- `add_incident(incident)` - Add incident
- `get_zones_at(point)` - Get zones containing point
- `get_zone_by_id(id)` - Get zone by ID
- `get_incident_at(point, radius)` - Get incident near point
- `get_nearest_active_incident(point)` - Nearest incident
- `is_within_bounds(point)` - Check bounds
- `is_path_clear(start, end)` - Check path
- `get_danger_level(point)` - Danger at point

### Zone

A geographic zone.

```python
from emmarl.envs.map import Zone, ZoneType

zone = Zone(
    zone_id="fire_1",
    zone_type=ZoneType.FIRE,
    position=(500.0, 500.0),
    size=(100.0, 100.0),
    intensity=0.8,
)
```

#### Properties

- `bounds` - Bounding box
- `contains_point(point)` - Point in zone
- `distance_to(point)` - Distance to point

### Incident

A dynamic event.

```python
from emmarl.envs.map import Incident, ZoneType

incident = Incident(
    incident_id="fire_1",
    incident_type=ZoneType.FIRE,
    position=(500.0, 500.0),
    severity=0.8,
    resources_required={"water": 50.0},
)
```

#### Methods

- `is_resolved()` - Check if resolved
- `reduce_severity(amount)` - Reduce severity

### ZoneType

Enum for zone/incident types.

```python
from emmarl.envs.map import ZoneType

ZoneType.SAFE
ZoneType.FIRE
ZoneType.FLOODED
ZoneType.COLLAPSED
ZoneType.HAZMAT
ZoneType.CROWDED
ZoneType.MEDICAL_EMERGENCY
ZoneType.ROAD
ZoneType.BUILDING
```

### create_default_map

Create a default emergency scenario.

```python
from emmarl.envs.map import create_default_map

emap = create_default_map(width=1000.0, height=1000.0)
```
