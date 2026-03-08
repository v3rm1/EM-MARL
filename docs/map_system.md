# Map System

FireSim uses a continuous coordinate map system rather than a gridworld, allowing for more realistic movement and spatial relationships.

## Coordinate System

- **Origin**: (0, 0) at bottom-left
- **Units**: Arbitrary (default map is 1000x1000 units)
- **Agents**: Position defined as (x, y) tuple

```python
# Example positions
(0, 0)       # Bottom-left corner
(500, 500)   # Center of 1000x1000 map
(1000, 1000) # Top-right corner
```

## EmergencyMap

The `EmergencyMap` class manages the simulation space:

```python
from firesim.envs.map import EmergencyMap

emergency_map = EmergencyMap(
    width=1000.0,
    height=1000.0,
    zones=[],
    incidents=[],
    obstacles=[],
)
```

## Zones

Zones represent geographic areas with specific properties:

### Zone Types

```python
from firesim.envs.map import ZoneType

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
from firesim.envs.map import Zone, ZoneType

fire_zone = Zone(
    zone_id="fire_1",
    zone_type=ZoneType.FIRE,
    position=(300.0, 300.0),  # Center
    size=(100.0, 100.0),      # Width, height
    intensity=0.8,            # 0-1 severity
    properties={},            # Custom properties
)
```

### Zone Properties

```python
# Check if point is in zone
fire_zone.contains_point((300.0, 300.0))  # True

# Get bounding box
fire_zone.bounds  # ((250, 250), (350, 350))

# Distance to zone center
fire_zone.distance_to((400.0, 400.0))  # ~141.4
```

## Incidents

Incidents are dynamic events that agents must resolve:

### Creating Incidents

```python
from firesim.envs.map import Incident, ZoneType

fire_incident = Incident(
    incident_id="fire_1",
    incident_type=ZoneType.FIRE,
    position=(300.0, 300.0),
    severity=0.8,
    active=True,
    casualties=0,
    resources_required={"water": 50.0, "foam": 20.0},
)
```

### Incident Properties

```python
# Check if resolved
fire_incident.is_resolved()  # False

# Reduce severity
fire_incident.reduce_severity(0.3)  # severity now 0.5
fire_incident.is_resolved()  # True if severity <= 0
```

### Managing Incidents

```python
# Add incident to map
emergency_map.add_incident(fire_incident)

# Get incident at location
incident = emergency_map.get_incident_at((300.0, 300.0), radius=10)

# Get nearest active incident
nearest, distance = emergency_map.get_nearest_active_incident((0.0, 0.0))
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

## Path Finding

### Bounds Checking

```python
# Check if point is within map
emergency_map.is_within_bounds((500.0, 500.0))  # True
emergency_map.is_within_bounds((-10.0, 500.0))  # False
```

### Obstacles

Add rectangular obstacles:

```python
# Add obstacle (start_corner, end_corner)
emergency_map.obstacles.append(((100, 100), (200, 200)))

# Check path
emergency_map.is_path_clear((50, 150), (250, 150))  # False
```

## Default Map

Create a default emergency scenario:

```python
from firesim.envs.map import create_default_map

emergency_map = create_default_map(
    width=1000.0,
    height=1000.0
)

# Contains:
# - 3 zones (fire, medical, crowd)
# - 3 incidents
```

## Custom Scenarios

Build custom scenarios:

```python
from firesim.envs.map import EmergencyMap, Zone, Incident, ZoneType

custom_map = EmergencyMap(width=2000.0, height=2000.0)

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

Pass custom maps to FireEnv:

```python
from firesim.envs import FireEnv
from firesim.envs.fire_env import FireEnvConfig

config = FireEnvConfig(map_width=2000.0, map_height=2000.0)
env = FireEnv(config)

# Access the map
env._emergency_map
```
