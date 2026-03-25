# Map System

FireSim uses a **grid-based EmergencyMap** system with terrain for managing the simulation space.

## Coordinate System

- **Origin**: (0, 0) at bottom-left
- **Grid**: Continuous space divided into cells (default cell_size=10.0)
- **Agents**: Position defined as (x, y) tuple

```python
# Example positions
(0, 0)       # Bottom-left corner
(500, 500)   # Center of 1000x1000 map
(1000, 1000) # Top-right corner
```

## Grid-Based Terrain

The map is divided into a grid of cells, each with a terrain type:

```python
from emmarl.envs.map import EmergencyMap

emergency_map = EmergencyMap(
    width=1000.0,
    height=1000.0,
    cell_size=10.0,  # Creates 100x100 grid
)
```

## Terrain Types

```python
from emmarl.envs.map import TerrainType

OPEN          # Open ground (default)
FOREST        # Forest, high fuel, slow movement (0.6x)
GRASS         # Grassland, medium fuel (0.8x)
URBAN         # Urban area (0.9x)
ROAD          # Roads, fast movement (1.2x), no fuel
WATER         # Water, impassable (0.0x)
BUILDING      # Building footprint (0.5x)
BURNED        # Burned area (0.7x)
```

### Terrain Properties

Each terrain type affects movement speed and fire behavior:

| Terrain   | Speed Multiplier | Fuel Load | Fire Resistance |
|-----------|-----------------|-----------|-----------------|
| OPEN      | 1.0             | 0.1       | 0.9             |
| FOREST    | 0.6             | 0.9       | 0.1             |
| GRASS     | 0.8             | 0.6       | 0.3             |
| URBAN     | 0.9             | 0.4       | 0.5             |
| ROAD      | 1.2             | 0.0       | 1.0             |
| WATER     | 0.0 (impassable)| 0.0       | 1.0             |
| BUILDING  | 0.5             | 0.5       | 0.6             |
| BURNED    | 0.7             | 0.0       | 1.0             |

### Querying Terrain

```python
# Get terrain at position
terrain = emergency_map.get_terrain_at((300.0, 300.0))

# Get speed multiplier (affects agent movement)
speed_mult = emergency_map.get_speed_multiplier((300.0, 300.0))

# Get fuel load (affects fire spread)
fuel = emergency_map.get_fuel_load((300.0, 300.0))

# Check if position is passable
can_pass = emergency_map.is_passable((300.0, 300.0))
```

## GridTerrain Class

For direct grid manipulation:

```python
from emmarl.envs.map import GridTerrain, TerrainType

grid = GridTerrain(width=100, height=100, cell_size=10.0)

# Set terrain at world position
grid.set_terrain_at(300.0, 300.0, TerrainType.FOREST)

# Set terrain at grid coordinates
grid.set_terrain_at_grid(30, 30, TerrainType.BUILDING)

# Fill rectangular region
grid.fill_rectangle(100, 100, 300, 300, TerrainType.GRASS)

# Convert between coordinates
grid_x, grid_y = grid.to_grid_coords(350.0, 250.0)
world_x, world_y = grid.to_world_coords(35, 25)
```

## Wildland-Urban Interface (WUI)

FireSim includes procedural WUI terrain generation:

```python
from emmarl.envs.map import EmergencyMap

emergency_map = EmergencyMap(width=1000.0, height=1000.0)
emergency_map.generate_wui_terrain(seed=42)
emergency_map.generate_road_grid()
```

This creates:
- **Urban core**: Dense buildings and roads in the center
- **WUI zone**: Mix of buildings, vegetation around urban area
- **Wildland**: Forest and grass in outer areas
- **Roads**: Grid pattern connecting all areas

## Procedural Road Generation

```python
# Generate road between two points
emergency_map.generate_road(x1, y1, x2, y2, width=20.0)

# Generate road grid across the map
emergency_map.generate_road_grid(seed=None)
```

## Zones

Zones represent geographic areas with specific properties:

### Zone Types

```python
from emmarl.envs.map import ZoneType

SAFE            # Safe area
FIRE            # Fire hazard
FLOODED         # Flood hazard
COLLAPSED       # Structural collapse
HAZMAT          # Hazardous materials
CROWDED         # Large crowd
MEDICAL_EMERGENCY  # Medical incident
ROAD            # Road/vehicle area
BUILDING        # Building structure
```

### Creating Zones

```python
from emmarl.envs.map import Zone, ZoneType

fire_zone = Zone(
    zone_id="fire_1",
    zone_type=ZoneType.FIRE,
    position=(300.0, 300.0),  # Center
    size=(100.0, 100.0),      # Width, height
    intensity=0.8,            # 0-1 severity
)
```

## Incidents

Incidents are dynamic events that agents must resolve:

```python
from emmarl.envs.map import Incident, ZoneType

fire_incident = Incident(
    incident_id="fire_1",
    incident_type=ZoneType.FIRE,
    position=(300.0, 300.0),
    severity=0.8,
    active=True,
    resources_required={"water": 50.0, "foam": 20.0},
)

# Check if resolved
fire_incident.is_resolved()  # False

# Reduce severity
fire_incident.reduce_severity(0.3)
fire_incident.is_resolved()  # True if severity <= 0
```

## Danger Level

Calculate danger at any point:

```python
danger = emergency_map.get_danger_level((300.0, 300.0))
# Returns 0-1 based on:
# - Zone type and intensity
# - Active incident severity
# - Distance from hazards
```

## Default Map

Create a default WUI emergency scenario:

```python
from emmarl.envs.map import create_default_map

emergency_map = create_default_map(
    width=1000.0,
    height=1000.0,
    seed=42  # Optional: reproducible terrain
)

# Contains:
# - WUI terrain with urban core, WUI zone, wildland
# - Procedural road grid
# - 2 fire incidents at wildland-urban interface
# - 1 medical incident in urban core
```

## Simple Map (for testing)

Create a simple map with basic rectangular terrain:

```python
from emmarl.envs.map import create_simple_map

emergency_map = create_simple_map(width=500.0, height=500.0)
```

## Custom Scenarios

Build custom scenarios:

```python
from emmarl.envs.map import EmergencyMap, Zone, Incident, ZoneType, TerrainType

custom_map = EmergencyMap(width=2000.0, height=2000.0)

# Add terrain
custom_map.generate_wui_terrain(seed=123)
custom_map.generate_road_grid()

# Add zones
custom_map.add_zone(Zone(
    zone_id="hazmat_zone",
    zone_type=ZoneType.HAZMAT,
    position=(500.0, 500.0),
    size=(200.0, 200.0),
    intensity=0.9,
))

# Add incidents
custom_map.add_incident(Incident(
    incident_id="hazmat_1",
    incident_type=ZoneType.HAZMAT,
    position=(500.0, 500.0),
    severity=0.9,
    resources_required={"foam": 30.0},
))
```

## Integration with FireEnv

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig

# Use default WUI terrain
config = FireEnvConfig(map_width=500.0, map_height=500.0)
env = FireEnv(config)

# Access terrain
env._emergency_map.terrain
env._emergency_map.get_terrain_at((250.0, 250.0))
```
