"""Map-based layout system for FireSim emergency response environment."""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class TerrainType(Enum):
    """Terrain types for grid cells."""

    OPEN = auto()  # Open ground, default
    FOREST = auto()  # Forest, high fuel, slow movement
    GRASS = auto()  # Grassland, medium fuel
    URBAN = auto()  # Urban area, buildings
    ROAD = auto()  # Roads, fast movement
    WATER = auto()  # Water, impassable
    BUILDING = auto()  # Building footprint
    BURNED = auto()  # Burned area


class ZoneType(Enum):
    """Types of zones in the emergency scenario."""

    SAFE = auto()
    FIRE = auto()
    FLOODED = auto()
    COLLAPSED = auto()
    HAZMAT = auto()
    CROWDED = auto()
    MEDICAL_EMERGENCY = auto()
    ROAD = auto()
    BUILDING = auto()


TERRAIN_PROPERTIES = {
    TerrainType.OPEN: {
        "speed_multiplier": 1.0,
        "fuel_load": 0.1,
        "fire_resistance": 0.9,
    },
    TerrainType.FOREST: {
        "speed_multiplier": 0.6,
        "fuel_load": 0.9,
        "fire_resistance": 0.1,
    },
    TerrainType.GRASS: {
        "speed_multiplier": 0.8,
        "fuel_load": 0.6,
        "fire_resistance": 0.3,
    },
    TerrainType.URBAN: {
        "speed_multiplier": 0.9,
        "fuel_load": 0.4,
        "fire_resistance": 0.5,
    },
    TerrainType.ROAD: {
        "speed_multiplier": 1.2,
        "fuel_load": 0.0,
        "fire_resistance": 1.0,
    },
    TerrainType.WATER: {
        "speed_multiplier": 0.0,
        "fuel_load": 0.0,
        "fire_resistance": 1.0,
    },
    TerrainType.BUILDING: {
        "speed_multiplier": 0.5,
        "fuel_load": 0.5,
        "fire_resistance": 0.6,
    },
    TerrainType.BURNED: {
        "speed_multiplier": 0.7,
        "fuel_load": 0.0,
        "fire_resistance": 1.0,
    },
}


@dataclass
class GridTerrain:
    """Grid-based terrain system."""

    width: int
    height: int
    cell_size: float = 10.0
    terrain: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive")
        self.terrain = np.full(
            (self.height, self.width), TerrainType.OPEN.value, dtype=np.int32
        )

    def to_world_coords(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = grid_x * self.cell_size + self.cell_size / 2
        y = grid_y * self.cell_size + self.cell_size / 2
        return (x, y)

    def to_grid_coords(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(world_x / self.cell_size)
        grid_y = int(world_y / self.cell_size)
        return (grid_x, grid_y)

    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height

    def get_terrain_at(self, world_x: float, world_y: float) -> TerrainType:
        """Get terrain type at world position."""
        grid_x, grid_y = self.to_grid_coords(world_x, world_y)
        if self.is_valid_cell(grid_x, grid_y):
            return TerrainType(self.terrain[grid_y, grid_x])
        return TerrainType.OPEN

    def get_terrain_at_grid(self, grid_x: int, grid_y: int) -> TerrainType:
        """Get terrain type at grid position."""
        if self.is_valid_cell(grid_x, grid_y):
            return TerrainType(self.terrain[grid_y, grid_x])
        return TerrainType.OPEN

    def set_terrain_at(
        self, world_x: float, world_y: float, terrain_type: TerrainType
    ) -> None:
        """Set terrain type at world position."""
        grid_x, grid_y = self.to_grid_coords(world_x, world_y)
        if self.is_valid_cell(grid_x, grid_y):
            self.terrain[grid_y, grid_x] = terrain_type.value

    def set_terrain_at_grid(
        self, grid_x: int, grid_y: int, terrain_type: TerrainType
    ) -> None:
        """Set terrain type at grid position."""
        if self.is_valid_cell(grid_x, grid_y):
            self.terrain[grid_y, grid_x] = terrain_type.value

    def fill_rectangle(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        terrain_type: TerrainType,
    ) -> None:
        """Fill rectangular region with terrain."""
        gx1, gy1 = self.to_grid_coords(x1, y1)
        gx2, gy2 = self.to_grid_coords(x2, y2)
        gx1, gx2 = max(0, min(gx1, gx2)), min(self.width - 1, max(gx1, gx2))
        gy1, gy2 = max(0, min(gy1, gy2)), min(self.height - 1, max(gy1, gy2))
        self.terrain[gy1 : gy2 + 1, gx1 : gx2 + 1] = terrain_type.value

    def get_terrain_properties(self, terrain_type: TerrainType) -> dict:
        """Get properties for terrain type."""
        return TERRAIN_PROPERTIES.get(
            terrain_type, TERRAIN_PROPERTIES[TerrainType.OPEN]
        )

    def get_speed_multiplier(self, world_x: float, world_y: float) -> float:
        """Get movement speed multiplier at position."""
        terrain = self.get_terrain_at(world_x, world_y)
        return self.get_terrain_properties(terrain)["speed_multiplier"]

    def get_fuel_load(self, world_x: float, world_y: float) -> float:
        """Get fuel load at position."""
        terrain = self.get_terrain_at(world_x, world_y)
        return self.get_terrain_properties(terrain)["fuel_load"]

    def get_fire_resistance(self, world_x: float, world_y: float) -> float:
        """Get fire resistance at position."""
        terrain = self.get_terrain_at(world_x, world_y)
        return self.get_terrain_properties(terrain)["fire_resistance"]

    def is_passable(self, world_x: float, world_y: float) -> bool:
        """Check if position is passable."""
        terrain = self.get_terrain_at(world_x, world_y)
        return self.get_terrain_properties(terrain)["speed_multiplier"] > 0

    def get_neighbors(
        self, grid_x: int, grid_y: int, radius: int = 1
    ) -> list[tuple[int, int]]:
        """Get neighboring cell coordinates."""
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = grid_x + dx, grid_y + dy
                if self.is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors


@dataclass
class Zone:
    """A zone in the map representing an area of interest."""

    zone_id: str
    zone_type: ZoneType
    position: tuple[float, float]
    size: tuple[float, float]
    intensity: float = 1.0
    properties: dict = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get bounding box of zone."""
        x, y = self.position
        w, h = self.size
        return ((x - w / 2, y - h / 2), (x + w / 2, y + h / 2))

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Check if a point is within this zone."""
        (min_x, min_y), (max_x, max_y) = self.bounds
        px, py = point
        return min_x <= px <= max_x and min_y <= py <= max_y

    def distance_to(self, point: tuple[float, float]) -> float:
        """Calculate distance from zone center to a point."""
        px, py = point
        zx, zy = self.position
        return np.sqrt((px - zx) ** 2 + (py - zy) ** 2)


@dataclass
class Incident:
    """An incident/event in the emergency scenario."""

    incident_id: str
    incident_type: ZoneType
    position: tuple[float, float]
    severity: float = 1.0
    active: bool = True
    casualties: int = 0
    resources_required: dict[str, float] = field(default_factory=dict)
    resolved: bool = False

    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.resolved or self.severity <= 0

    def reduce_severity(self, amount: float) -> None:
        """Reduce incident severity."""
        self.severity = max(0.0, self.severity - amount)
        if self.severity <= 0:
            self.resolved = True
            self.active = False


@dataclass
class EmergencyMap:
    """Map representing the emergency scenario area."""

    width: float
    height: float
    grid_width: int = 100
    grid_height: int = 100
    cell_size: float = 10.0
    zones: list[Zone] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    obstacles: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )
    terrain: GridTerrain | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map dimensions must be positive")
        self.grid_width = int(self.width / self.cell_size)
        self.grid_height = int(self.height / self.cell_size)
        self.terrain = GridTerrain(
            width=self.grid_width,
            height=self.grid_height,
            cell_size=self.cell_size,
        )

    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the map."""
        self.zones.append(zone)

    def add_incident(self, incident: Incident) -> None:
        """Add an incident to the map."""
        self.incidents.append(incident)

    def get_zones_at(self, point: tuple[float, float]) -> list[Zone]:
        """Get all zones containing a point."""
        return [z for z in self.zones if z.contains_point(point)]

    def get_zone_by_id(self, zone_id: str) -> Zone | None:
        """Get zone by ID."""
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def get_incident_at(
        self, point: tuple[float, float], radius: float = 10.0
    ) -> Incident | None:
        """Get incident at or near a point."""
        for incident in self.incidents:
            if incident.active:
                dist = np.sqrt(
                    (incident.position[0] - point[0]) ** 2
                    + (incident.position[1] - point[1]) ** 2
                )
                if dist <= radius:
                    return incident
        return None

    def get_nearest_active_incident(
        self, point: tuple[float, float]
    ) -> tuple[Incident | None, float]:
        """Get nearest active incident to a point."""
        nearest = None
        min_dist = float("inf")
        for incident in self.incidents:
            if incident.active:
                dist = np.sqrt(
                    (incident.position[0] - point[0]) ** 2
                    + (incident.position[1] - point[1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = incident
        return nearest, min_dist

    def is_within_bounds(self, point: tuple[float, float]) -> bool:
        """Check if point is within map bounds."""
        x, y = point
        return 0 <= x <= self.width and 0 <= y <= self.height

    def is_passable(self, point: tuple[float, float]) -> bool:
        """Check if point is passable (not blocked by terrain)."""
        if self.terrain is None:
            return True
        return self.terrain.is_passable(point[0], point[1])

    def get_terrain_at(self, point: tuple[float, float]) -> TerrainType:
        """Get terrain type at point."""
        if self.terrain is None:
            return TerrainType.OPEN
        return self.terrain.get_terrain_at(point[0], point[1])

    def get_speed_multiplier(self, point: tuple[float, float]) -> float:
        """Get terrain speed multiplier at point."""
        if self.terrain is None:
            return 1.0
        return self.terrain.get_speed_multiplier(point[0], point[1])

    def get_fuel_load(self, point: tuple[float, float]) -> float:
        """Get terrain fuel load at point."""
        if self.terrain is None:
            return 0.1
        return self.terrain.get_fuel_load(point[0], point[1])

    def is_path_clear(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> bool:
        """Check if path between two points is clear (simple implementation)."""
        for obs_start, obs_end in self.obstacles:
            if self._line_intersects_rect(start, end, obs_start, obs_end):
                return False
        return True

    def _line_intersects_rect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        r1: tuple[float, float],
        r2: tuple[float, float],
    ) -> bool:
        """Simple line-rectangle intersection check."""
        min_x, min_y = min(r1[0], r2[0]), min(r1[1], r2[1])
        max_x, max_y = max(r1[0], r2[0]), max(r1[1], r2[1])
        px, py = p1
        dx, dy = p2[0] - px, p2[1] - py
        if dx == 0:
            return min_x <= px <= max_x and min(min_y, py) <= max(max_y, py)
        if dy == 0:
            return min_y <= py <= max_y and min(max_x, px) <= min(min_x, px)
        t1 = (min_x - px) / dx
        t2 = (max_x - px) / dx
        t3 = (min_y - py) / dy
        t4 = (max_y - py) / dy
        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))
        return tmax >= tmin and tmax >= 0 and tmin <= 1

    def get_danger_level(self, point: tuple[float, float]) -> float:
        """Get danger level at a point based on zones and incidents."""
        danger = 0.0
        for zone in self.get_zones_at(point):
            if zone.zone_type in (
                ZoneType.FIRE,
                ZoneType.HAZMAT,
                ZoneType.FLOODED,
                ZoneType.COLLAPSED,
            ):
                danger = max(danger, zone.intensity)
        for incident in self.incidents:
            if incident.active:
                dist = np.sqrt(
                    (incident.position[0] - point[0]) ** 2
                    + (incident.position[1] - point[1]) ** 2
                )
                if dist < 50:
                    danger = max(danger, incident.severity * (1 - dist / 50))
        return danger

    def generate_road(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float = 20.0,
    ) -> None:
        """Generate a road between two points."""
        if self.terrain is None:
            return

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        nx = -dy / length * width / 2
        ny = dx / length * width / 2

        steps = int(length / self.terrain.cell_size)
        for i in range(steps + 1):
            t = i / max(steps, 1)
            cx = x1 + dx * t
            cy = y1 + dy * t

            for ox in [-width / 2, width / 2]:
                for oy in [-width / 2, width / 2]:
                    px = cx + ox + nx * 0.5
                    py = cy + oy + ny * 0.5
                    if self.is_within_bounds((px, py)):
                        self.terrain.set_terrain_at(px, py, TerrainType.ROAD)

    def generate_road_grid(self, seed: int | None = None) -> None:
        """Generate a grid of roads across the map."""
        if self.terrain is None:
            return

        if seed is not None:
            np.random.seed(seed)

        width = self.width
        height = self.height

        num_horizontal = 3
        num_vertical = 3

        horizontal_spacing = height / (num_horizontal + 1)
        for i in range(1, num_horizontal + 1):
            y = horizontal_spacing * i
            self.generate_road(0, y, width, y, width=20)

        vertical_spacing = width / (num_vertical + 1)
        for i in range(1, num_vertical + 1):
            x = vertical_spacing * i
            self.generate_road(x, 0, x, height, width=20)

    def generate_wui_terrain(self, seed: int | None = None) -> None:
        """Generate Wildland-Urban Interface terrain layout.

        Creates:
        - Urban core in the center
        - WUI zone (mix of buildings and vegetation) around urban area
        - Wildland (forest/grass) in outer areas
        - Roads connecting everything
        """
        if self.terrain is None:
            return

        if seed is not None:
            np.random.seed(seed)

        width = self.width
        height = self.height
        terrain = self.terrain

        urban_center_x = width / 2
        urban_center_y = height / 2
        urban_radius = min(width, height) * 0.2
        wui_inner_radius = min(width, height) * 0.35
        wui_outer_radius = min(width, height) * 0.5

        for gy in range(terrain.height):
            for gx in range(terrain.width):
                world_x, world_y = terrain.to_world_coords(gx, gy)
                dist_from_center = np.sqrt(
                    (world_x - urban_center_x) ** 2 + (world_y - urban_center_y) ** 2
                )

                if dist_from_center < urban_radius:
                    if np.random.random() < 0.7:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.URBAN)
                    else:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.OPEN)
                elif dist_from_center < wui_inner_radius:
                    rand = np.random.random()
                    if rand < 0.3:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.BUILDING)
                    elif rand < 0.5:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.URBAN)
                    elif rand < 0.8:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.GRASS)
                    else:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.FOREST)
                elif dist_from_center < wui_outer_radius:
                    rand = np.random.random()
                    if rand < 0.4:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.FOREST)
                    elif rand < 0.7:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.GRASS)
                    else:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.OPEN)
                else:
                    rand = np.random.random()
                    if rand < 0.6:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.FOREST)
                    else:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.GRASS)

        self.generate_road_grid()

        for gy in range(terrain.height):
            for gx in range(terrain.width):
                world_x, world_y = terrain.to_world_coords(gx, gy)
                terrain_type = terrain.get_terrain_at_grid(gx, gy)

                if terrain_type == TerrainType.URBAN:
                    if np.random.random() < 0.3:
                        terrain.set_terrain_at_grid(gx, gy, TerrainType.BUILDING)


def create_default_map(
    width: float = 1000.0,
    height: float = 1000.0,
    cell_size: float = 10.0,
    seed: int | None = None,
) -> EmergencyMap:
    """Create a default emergency map with WUI terrain and incidents."""
    emergency_map = EmergencyMap(
        width=width,
        height=height,
        cell_size=cell_size,
    )

    if emergency_map.terrain:
        emergency_map.generate_wui_terrain(seed=seed)

    urban_center_x = width / 2
    urban_center_y = height / 2
    urban_radius = min(width, height) * 0.2

    emergency_map.add_zone(
        Zone(
            zone_id="fire_zone_1",
            zone_type=ZoneType.FIRE,
            position=(
                urban_center_x + urban_radius * 1.2,
                urban_center_y - urban_radius * 0.5,
            ),
            size=(80.0, 80.0),
            intensity=0.9,
        )
    )

    emergency_map.add_zone(
        Zone(
            zone_id="fire_zone_2",
            zone_type=ZoneType.FIRE,
            position=(
                urban_center_x - urban_radius * 0.8,
                urban_center_y + urban_radius * 1.1,
            ),
            size=(60.0, 60.0),
            intensity=0.7,
        )
    )

    emergency_map.add_zone(
        Zone(
            zone_id="medical_zone_1",
            zone_type=ZoneType.MEDICAL_EMERGENCY,
            position=(urban_center_x, urban_center_y),
            size=(60.0, 60.0),
            intensity=0.5,
        )
    )

    fire1_x = urban_center_x + urban_radius * 1.2
    fire1_y = urban_center_y - urban_radius * 0.5
    emergency_map.add_incident(
        Incident(
            incident_id="fire_1",
            incident_type=ZoneType.FIRE,
            position=(fire1_x, fire1_y),
            severity=0.9,
            resources_required={"water": 50.0, "foam": 20.0},
        )
    )

    fire2_x = urban_center_x - urban_radius * 0.8
    fire2_y = urban_center_y + urban_radius * 1.1
    emergency_map.add_incident(
        Incident(
            incident_id="fire_2",
            incident_type=ZoneType.FIRE,
            position=(fire2_x, fire2_y),
            severity=0.7,
            resources_required={"water": 40.0, "foam": 15.0},
        )
    )

    emergency_map.add_incident(
        Incident(
            incident_id="medical_1",
            incident_type=ZoneType.MEDICAL_EMERGENCY,
            position=(urban_center_x, urban_center_y),
            severity=0.5,
            resources_required={"medkits": 2, "medication": 1},
        )
    )

    return emergency_map


def create_simple_map(
    width: float = 1000.0,
    height: float = 1000.0,
    cell_size: float = 10.0,
) -> EmergencyMap:
    """Create a simple map with basic terrain layout (for testing)."""
    emergency_map = EmergencyMap(
        width=width,
        height=height,
        cell_size=cell_size,
    )

    if emergency_map.terrain:
        emergency_map.terrain.fill_rectangle(100, 100, 300, 300, TerrainType.FOREST)
        emergency_map.terrain.fill_rectangle(400, 400, 700, 700, TerrainType.GRASS)
        emergency_map.terrain.fill_rectangle(600, 100, 800, 250, TerrainType.URBAN)

        emergency_map.terrain.fill_rectangle(50, 150, 350, 250, TerrainType.ROAD)
        emergency_map.terrain.fill_rectangle(150, 50, 250, 350, TerrainType.ROAD)

        emergency_map.terrain.fill_rectangle(150, 150, 200, 200, TerrainType.BUILDING)
        emergency_map.terrain.fill_rectangle(250, 150, 290, 220, TerrainType.BUILDING)
        emergency_map.terrain.fill_rectangle(500, 500, 600, 600, TerrainType.BUILDING)

    emergency_map.add_zone(
        Zone(
            zone_id="fire_zone_1",
            zone_type=ZoneType.FIRE,
            position=(200.0, 200.0),
            size=(100.0, 100.0),
            intensity=0.8,
        )
    )

    emergency_map.add_zone(
        Zone(
            zone_id="medical_zone_1",
            zone_type=ZoneType.MEDICAL_EMERGENCY,
            position=(600.0, 400.0),
            size=(80.0, 80.0),
            intensity=0.6,
        )
    )

    emergency_map.add_zone(
        Zone(
            zone_id="crowd_zone_1",
            zone_type=ZoneType.CROWDED,
            position=(200.0, 700.0),
            size=(120.0, 100.0),
            intensity=0.5,
        )
    )

    emergency_map.add_incident(
        Incident(
            incident_id="fire_1",
            incident_type=ZoneType.FIRE,
            position=(200.0, 200.0),
            severity=0.8,
            resources_required={"water": 50.0, "foam": 20.0},
        )
    )

    emergency_map.add_incident(
        Incident(
            incident_id="medical_1",
            incident_type=ZoneType.MEDICAL_EMERGENCY,
            position=(600.0, 400.0),
            severity=0.6,
            resources_required={"medkits": 2, "medication": 1},
        )
    )

    emergency_map.add_incident(
        Incident(
            incident_id="crowd_1",
            incident_type=ZoneType.CROWDED,
            position=(200.0, 700.0),
            severity=0.5,
            resources_required={"barriers": 5},
        )
    )

    return emergency_map
