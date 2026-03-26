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

---

## Fire Dynamics Classes

### FireModel

Main fire propagation model managing multiple fire fronts with advanced physics.

```python
from emmarl.envs.fire_dynamics import FireModel

fire_model = FireModel(dynamics=FireDynamics(weather=Weather()))
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_ignition(position, intensity)` | None | Add ignition point |
| `add_containment_line(line)` | None | Add fire containment line |
| `update(fuel_map)` | None | Update fire propagation |
| `get_intensity_at(point)` | float | Fire intensity at point |
| `get_rate_of_spread_at(point)` | float | ROS at point |
| `get_flame_length_at(point)` | float | Flame length at point |
| `get_temperature_at(point)` | float | Temperature at point |
| `get_preheat_at(point)` | float | Pre-heat level at point |
| `get_fire_area()` | float | Total fire area |
| `get_fire_perimeter_length()` | float | Perimeter length |
| `is_point_in_fire(point)` | bool | Point inside fire perimeter |
| `get_active_ember_count()` | int | Number of active embers |
| `get_fire_distance(point)` | tuple | Distance to nearest fire |

### FireDynamics

Rothermel (1972) fire spread model.

```python
from emmarl.envs.fire_dynamics import FireDynamics, Weather

dynamics = FireDynamics(
    weather=Weather(wind_speed=15.0, wind_direction=0.5),
    slope_angle=10.0,
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `compute_rate_of_spread(fuel, direction)` | float | Rate of spread (m/s) |
| `compute_fire_intensity(fuel, ros)` | float | Fire intensity (kW/m) |
| `compute_flame_length(intensity)` | float | Flame length (m) |
| `compute_crown_fire_transition(fuel, ros, intensity)` | bool | Crown fire check |
| `compute_crown_fire_ros(fuel, ros)` | float | Crown fire ROS |
| `compute_spotting_probability(intensity, wind)` | float | Ember probability |
| `compute_max_spotting_distance(wind)` | float | Max ember distance (m) |
| `compute_containment_effectiveness(line_pos, fire_pos)` | float | Line effectiveness |
| `compute_preheat_effect(cell_temp)` | float | Pre-heat ignition effect |

### FuelProperties

Fuel properties for fire calculations.

```python
from emmarl.envs.fire_dynamics import FuelProperties, FuelModel

fuel = FuelProperties(
    fuel_model=FuelModel.BOREAL_FOREST,
    fuel_load=0.9,
    fuel_depth=1.0,
    moisture_content=0.1,
    bulk_density=0.1,
    surface_area_ratio=1500.0,
    mineral_content=0.055,
    particle_density=32.0,
    heat_diffusion_coefficient=0.005,
    canopy_base_height=3.0,
    canopy_fuel_load=0.8,
)
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fuel_model` | FuelModel | - | Fuel model type |
| `fuel_load` | float | 1.0 | Fuel load (kg/m²) |
| `fuel_depth` | float | 1.0 | Fuel depth (m) |
| `moisture_content` | float | 0.1 | Fuel moisture (0-1) |
| `bulk_density` | float | 0.1 | Bulk density (kg/m³) |
| `surface_area_ratio` | float | 1500.0 | Surface area ratio |
| `mineral_content` | float | 0.055 | Mineral content |
| `particle_density` | float | 32.0 | Particle density |
| `heat_diffusion_coefficient` | float | 0.005 | Heat diffusion rate |
| `canopy_base_height` | float | 0.0 | Canopy base height (m) |
| `canopy_fuel_load` | float | 0.0 | Canopy fuel load |

### FuelModel

Standard fuel models from Rothermel (1972).

```python
from emmarl.envs.fire_dynamics import FuelModel

FuelModel.GRASS
FuelModel.GRASS_MIX
FuelModel.TIMBER_GRASS
FuelModel.BOREAL_FOREST
FuelModel.CHAPARRAL
FuelModel.URBAN
FuelModel.WATER
FuelModel.BURNED
```

### Weather

Weather conditions affecting fire spread.

```python
from emmarl.envs.fire_dynamics import Weather

weather = Weather(
    wind_speed=10.0,
    wind_direction=0.0,
    temperature=25.0,
    humidity=0.4,
)
```

### FireState

Dynamic fire state at a location.

```python
from emmarl.envs.fire_dynamics import FireState

fire = FireState(
    position=(100.0, 200.0),
    intensity=0.8,
    rate_of_spread=5.0,
    fire_line_intensity=400.0,
    flame_length=2.5,
    flame_angle=0.3,
    fuel_consumed=0.5,
    is_spreading=True,
    spread_direction=0.0,
    temperature=800.0,
    pre_heating=0.3,
    is_crown_fire=False,
    ember_count=0,
)
```

### HeatMap

Grid-based heat map for thermal diffusion.

```python
from emmarl.envs.fire_dynamics import HeatMap

heat_map = HeatMap(width=100, height=100)

# Get temperature at grid cell
temp = heat_map.get_temp_at(grid_x, grid_y)

# Get pre-heat level
preheat = heat_map.get_preheat_at(grid_x, grid_y)

# Apply thermal diffusion
heat_map.diffuse_heat(diffusion_coef=0.05, dt=1.0, fire_positions=[...], fire_temperatures=[...])
```

### FirePerimeter

Polygon-based fire perimeter tracking.

```python
from emmarl.envs.fire_dynamics import FirePerimeter

perimeter = FirePerimeter()
perimeter.add_point(x, y)
perimeter.compute_metrics()

# Get metrics
area = perimeter.area
perimeter_len = perimeter.perimeter

# Point in polygon check
is_inside = perimeter.is_point_inside(x, y)
```

### Ember

Ember for fire spotting.

```python
from emmarl.envs.fire_dynamics import Ember

ember = Ember(
    position=(100.0, 200.0),
    velocity=(1.0, 0.5),
    lifetime=0.0,
    max_lifetime=10.0,
)

# Update position with wind
ember.update(dt=1.0, wind_speed=10.0, wind_dir=0.0)

# Check if active
is_active = ember.is_active()
```
