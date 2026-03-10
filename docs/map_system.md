# Map System

FireSim supports two map systems: a simple **EmergencyMap** and a GIS-based **GISMap** with advanced geospatial features.

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

## Map Selection

Choose between map types using the `use_gis` configuration flag:

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig

# Default: Use EmergencyMap
config = FireEnvConfig(use_gis=False)
env = FireEnv(config)

# GIS-based map
config = FireEnvConfig(use_gis=True)
env = FireEnv(config)
```

## EmergencyMap

The `EmergencyMap` class manages the simulation space:

```python
from emmarl.envs.map import EmergencyMap

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
from emmarl.envs.map import Incident, ZoneType

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
from emmarl.envs.map import create_default_map

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
from emmarl.envs.map import EmergencyMap, Zone, Incident, ZoneType

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
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig

config = FireEnvConfig(map_width=2000.0, map_height=2000.0)
env = FireEnv(config)

# Access the map
env._emergency_map
```

---

# GIS-Based Map System

The `GISMap` class provides advanced GIS features using Shapely for geometry operations.

## Overview

GISMap uses geographic coordinate-based geometry (points, lines, polygons) instead of simple rectangular zones:

```python
from emmarl.envs.gis_map import GISMap

gis_map = GISMap(
    bounds=(0.0, 0.0, 1000.0, 1000.0),
    buildings=[],
    roads=[],
    terrain_zones=[],
    fire_zones=[],
)
```

## Terrain Types

```python
from emmarl.envs.gis_map import TerrainType

WATER          # Water bodies
FOREST         # Forest areas (high fuel load)
GRASS          # Grassland
URBAN          # Urban areas
ROAD           # Roads
BUILDING       # Buildings
PARKLAND       # Parks
AGRICULTURAL   # Farmland
BURNED         # Burned areas
```

## Road Types

```python
from emmarl.envs.gis_map import RoadType

HIGHWAY        # Highway/motorway
PRIMARY        # Primary roads
SECONDARY      # Secondary roads
RESIDENTIAL    # Residential streets
PATH           # Footpaths/trails
```

## Buildings

Create building footprints with Shapely polygons:

```python
from shapely.geometry import Polygon
from emmarl.envs.gis_map import Building

building = Building(
    building_id="building_1",
    polygon=Polygon([(150, 150), (200, 150), (200, 200), (150, 200)]),
    height=15.0,
    floors=3,
    building_type="residential",
    fire_resistance=0.5,
)
```

## Roads

Create road segments as line strings:

```python
from shapely.geometry import LineString
from emmarl.envs.gis_map import Road, RoadType

road = Road(
    road_id="road_1",
    line=LineString([(50, 200), (350, 200)]),
    road_type=RoadType.PRIMARY,
    name="Main Street",
    lanes=2,
    one_way=False,
)
```

## Terrain Zones

Create terrain with fuel and moisture properties:

```python
from shapely.geometry import Polygon
from emmarl.envs.gis_map import TerrainZone, TerrainType

terrain = TerrainZone(
    zone_id="forest_1",
    polygon=Polygon([(100, 100), (300, 100), (300, 300), (100, 300)]),
    terrain_type=TerrainType.FOREST,
    fuel_load=0.9,
    moisture_content=0.2,
)
```

## Fire Zones

Fire zones track active fires with intensity:

```python
from shapely.geometry import Polygon
from emmarl.envs.gis_map import FireZone

fire = FireZone(
    zone_id="fire_1",
    polygon=Polygon([(120, 120), (180, 120), (180, 180), (120, 180)]),
    fire_intensity=0.8,
    is_contained=False,
)
```

## Query Methods

### Terrain at Point

```python
terrain = gis_map.get_terrain_at((150.0, 150.0))
# Returns: TerrainType or None
```

### Building at Point

```python
building = gis_map.get_building_at((175.0, 175.0))
# Returns: Building or None
```

### On Road Check

```python
is_on_road = gis_map.is_on_road((200.0, 200.0), buffer=5.0)
# Returns: bool
```

### Fire Intensity

```python
intensity = gis_map.get_fire_intensity_at((150.0, 150.0))
# Returns: float (0.0 - 1.0)
```

### Danger Level

```python
danger = gis_map.get_danger_level((150.0, 150.0))
# Returns: float (0.0 - 1.0)
# Considers: fire intensity, terrain type, building presence
```

### Nearest Road/Building

```python
road, dist = gis_map.get_nearest_road((500.0, 500.0))
building, dist = gis_map.get_nearest_building((500.0, 500.0))
```

## Default GIS Map

Create a default GIS map with preset terrain, buildings, roads, and fires:

```python
from emmarl.envs.gis_map import create_default_gis_map

gis_map = create_default_gis_map()
# Contains:
# - 3 terrain zones (forest, grass, parkland)
# - 3 buildings
# - 3 roads
# - 1 fire zone
```

## Custom GIS Map

Build custom GIS scenarios:

```python
from shapely.geometry import Polygon
from emmarl.envs.gis_map import (
    GISMap, Building, Road, TerrainZone, FireZone,
    TerrainType, RoadType
)

gis_map = GISMap(bounds=(0.0, 0.0, 2000.0, 2000.0))

# Add terrain
gis_map.add_terrain_zone(TerrainZone(
    zone_id="forest_1",
    polygon=Polygon([(100, 100), (500, 100), (500, 500), (100, 500)]),
    terrain_type=TerrainType.FOREST,
    fuel_load=0.9,
))

# Add building
gis_map.add_building(Building(
    building_id="building_1",
    polygon=Polygon([(200, 200), (250, 200), (250, 250), (200, 250)]),
    height=20.0,
))

# Add road
gis_map.add_road(Road(
    road_id="road_1",
    line=LineString([(0, 300), (1000, 300)]),
    road_type=RoadType.PRIMARY,
))

# Add fire
gis_map.add_fire_zone(FireZone(
    zone_id="fire_1",
    polygon=Polygon([(150, 150), (200, 150), (200, 200), (150, 200)]),
    fire_intensity=0.8,
))
```

## GeoJSON Support

### Export to GeoJSON

```python
geojson = gis_map.to_geojson()
# Returns dict with FeatureCollection
```

### Load from GeoJSON

```python
from emmarl.envs.gis_map import load_geojson

gis_map = load_geojson("scenario.geojson")
```

## OpenStreetMap Integration

You can create GIS maps directly from OpenStreetMap data for real-world locations:

### From Place Name

```python
from emmarl.envs.gis_map import create_gis_map_from_osm

# Query by city/region name
gis_map = create_gis_map_from_osm("Athens, Greece")
gis_map = create_gis_map_from_osm("Attica, Greece")
```

### From Bounding Box

```python
# North, South, East, West coordinates
gis_map = create_gis_map_from_osm(
    north=38.3,
    south=37.7,
    east=24.0,
    west=23.4,
)
```

### Attica WUI Preset

For the Wildland-Urban Interface of Attica region:

```python
from emmarl.envs.gis_map import create_wui_attica_map

gis_map = create_wui_attica_map()
# Downloads OSM data for: bounds (23.4, 37.7) to (24.0, 38.3)
```

### Integration with FireEnv

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig
from emmarl.envs.gis_map import create_gis_map_from_osm

gis_map = create_gis_map_from_osm("Attica, Greece")
config = FireEnvConfig(use_gis=True, gis_map=gis_map)
env = FireEnv(config)
env.reset()
```

**Note:** The first time you call OSM functions, it will download data which may take a while. Subsequent calls use cached data.

## Integration with FireEnv

Pass custom GIS maps to FireEnv:

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig
from emmarl.envs.gis_map import create_default_gis_map

gis_map = create_default_gis_map()
config = FireEnvConfig(use_gis=True, gis_map=gis_map)
env = FireEnv(config)

# Access the GIS map
env._gis_map
env._emergency_map  # None when use_gis=True
```
