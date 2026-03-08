"""GIS-based map system for FireSim."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiPolygon,
    shape,
    mapping,
)
from shapely.ops import unary_union


class TerrainType(Enum):
    """Terrain types for GIS map."""

    WATER = auto()
    FOREST = auto()
    GRASS = auto()
    URBAN = auto()
    ROAD = auto()
    BUILDING = auto()
    PARKLAND = auto()
    AGRICULTURAL = auto()
    BURNED = auto()


class RoadType(Enum):
    """Types of roads."""

    HIGHWAY = auto()
    PRIMARY = auto()
    SECONDARY = auto()
    RESIDENTIAL = auto()
    PATH = auto()


@dataclass
class Building:
    """A building footprint."""

    building_id: str
    polygon: Polygon
    height: float = 0.0
    floors: int = 1
    building_type: str = "generic"
    address: str = ""
    fire_resistance: float = 0.5


@dataclass
class Road:
    """A road segment."""

    road_id: str
    line: LineString
    road_type: RoadType = RoadType.RESIDENTIAL
    name: str = ""
    lanes: int = 2
    one_way: bool = False


@dataclass
class TerrainZone:
    """A terrain zone polygon."""

    zone_id: str
    polygon: Polygon
    terrain_type: TerrainType
    fuel_load: float = 1.0
    moisture_content: float = 0.3


@dataclass
class FireZone:
    """A fire zone with dynamic fire state."""

    zone_id: str
    polygon: Polygon
    fire_intensity: float = 0.0
    fire_fronts: list[LineString] = field(default_factory=list)
    is_contained: bool = False
    containment_line: LineString | None = None


@dataclass
class GISMap:
    """GIS-based map with terrain, buildings, and roads."""

    bounds: tuple[float, float, float, float]
    buildings: list[Building] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    terrain_zones: list[TerrainZone] = field(default_factory=list)
    fire_zones: list[FireZone] = field(default_factory=list)
    terrain_raster: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]:
            raise ValueError("Invalid bounds: min must be less than max")

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2,
        )

    def add_building(self, building: Building) -> None:
        self.buildings.append(building)

    def add_road(self, road: Road) -> None:
        self.roads.append(road)

    def add_terrain_zone(self, zone: TerrainZone) -> None:
        self.terrain_zones.append(zone)

    def add_fire_zone(self, zone: FireZone) -> None:
        self.fire_zones.append(zone)

    def is_within_bounds(self, point: tuple[float, float]) -> bool:
        x, y = point
        return (
            self.bounds[0] <= x <= self.bounds[2]
            and self.bounds[1] <= y <= self.bounds[3]
        )

    def get_terrain_at(self, point: tuple[float, float]) -> TerrainType | None:
        p = Point(point)
        for zone in self.terrain_zones:
            if zone.polygon.contains(p):
                return zone.terrain_type
        return None

    def get_building_at(self, point: tuple[float, float]) -> Building | None:
        p = Point(point)
        for building in self.buildings:
            if building.polygon.contains(p):
                return building
        return None

    def is_on_road(self, point: tuple[float, float], buffer: float = 5.0) -> bool:
        p = Point(point).buffer(buffer)
        for road in self.roads:
            if road.line.intersects(p):
                return True
        return False

    def is_in_building(self, point: tuple[float, float]) -> bool:
        p = Point(point)
        for building in self.buildings:
            if building.polygon.contains(p):
                return True
        return False

    def get_fire_at(self, point: tuple[float, float]) -> FireZone | None:
        p = Point(point)
        for zone in self.fire_zones:
            if zone.polygon.contains(p):
                return zone
        return None

    def get_fire_intensity_at(self, point: tuple[float, float]) -> float:
        fire = self.get_fire_at(point)
        if fire:
            return fire.fire_intensity
        return 0.0

    def get_nearest_road(self, point: tuple[float, float]) -> tuple[Road | None, float]:
        min_dist = float("inf")
        nearest = None
        p = Point(point)
        for road in self.roads:
            dist = p.distance(road.line)
            if dist < min_dist:
                min_dist = dist
                nearest = road
        return nearest, min_dist

    def get_nearest_building(
        self, point: tuple[float, float]
    ) -> tuple[Building | None, float]:
        min_dist = float("inf")
        nearest = None
        p = Point(point)
        for building in self.buildings:
            dist = p.distance(building.polygon)
            if dist < min_dist:
                min_dist = dist
                nearest = building
        return nearest, min_dist

    def distance_to_nearest_building(self, point: tuple[float, float]) -> float:
        _, dist = self.get_nearest_building(point)
        return dist

    def get_danger_level(self, point: tuple[float, float]) -> float:
        danger = self.get_fire_intensity_at(point)
        terrain = self.get_terrain_at(point)
        if terrain in (TerrainType.FOREST, TerrainType.GRASS):
            danger = max(danger, 0.3)
        elif terrain == TerrainType.URBAN:
            danger = max(danger, 0.2)
        building = self.get_building_at(point)
        if building:
            danger = max(danger, 0.1)
        return danger

    def to_geojson(self) -> dict:
        features = []
        for building in self.buildings:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "building",
                        "id": building.building_id,
                        "height": building.height,
                        "floors": building.floors,
                    },
                    "geometry": mapping(building.polygon),
                }
            )
        for road in self.roads:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "road",
                        "id": road.road_id,
                        "road_type": road.road_type.name,
                    },
                    "geometry": mapping(road.line),
                }
            )
        for zone in self.terrain_zones:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "terrain",
                        "id": zone.zone_id,
                        "terrain_type": zone.terrain_type.name,
                    },
                    "geometry": mapping(zone.polygon),
                }
            )
        for zone in self.fire_zones:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "fire",
                        "id": zone.zone_id,
                        "intensity": zone.fire_intensity,
                    },
                    "geometry": mapping(zone.polygon),
                }
            )
        return {"type": "FeatureCollection", "features": features}


def create_default_gis_map() -> GISMap:
    """Create a default GIS-style map with terrain and buildings."""
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    gis_map = GISMap(bounds=bounds)

    forest_polygon = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
    gis_map.add_terrain_zone(
        TerrainZone(
            zone_id="forest_1",
            polygon=forest_polygon,
            terrain_type=TerrainType.FOREST,
            fuel_load=0.9,
            moisture_content=0.2,
        )
    )

    grass_polygon = Polygon([(400, 400), (700, 400), (700, 700), (400, 700)])
    gis_map.add_terrain_zone(
        TerrainZone(
            zone_id="grass_1",
            polygon=grass_polygon,
            terrain_type=TerrainType.GRASS,
            fuel_load=0.6,
            moisture_content=0.4,
        )
    )

    park_polygon = Polygon([(600, 100), (800, 100), (800, 250), (600, 250)])
    gis_map.add_terrain_zone(
        TerrainZone(
            zone_id="park_1",
            polygon=park_polygon,
            terrain_type=TerrainType.PARKLAND,
            fuel_load=0.3,
            moisture_content=0.5,
        )
    )

    gis_map.add_building(
        Building(
            building_id="building_1",
            polygon=Polygon([(150, 150), (200, 150), (200, 200), (150, 200)]),
            height=15.0,
            floors=3,
            building_type="residential",
        )
    )

    gis_map.add_building(
        Building(
            building_id="building_2",
            polygon=Polygon([(250, 150), (290, 150), (290, 220), (250, 220)]),
            height=25.0,
            floors=5,
            building_type="commercial",
        )
    )

    gis_map.add_building(
        Building(
            building_id="building_3",
            polygon=Polygon([(500, 500), (600, 500), (600, 600), (500, 600)]),
            height=20.0,
            floors=4,
            building_type="industrial",
        )
    )

    gis_map.add_road(
        Road(
            road_id="road_1",
            line=LineString([(50, 200), (350, 200)]),
            road_type=RoadType.PRIMARY,
            name="Main Street",
            lanes=2,
        )
    )

    gis_map.add_road(
        Road(
            road_id="road_2",
            line=LineString([(200, 50), (200, 350)]),
            road_type=RoadType.SECONDARY,
            name="Cross Street",
            lanes=2,
        )
    )

    gis_map.add_road(
        Road(
            road_id="road_3",
            line=LineString([(400, 450), (900, 450)]),
            road_type=RoadType.PRIMARY,
            name="Highway",
            lanes=4,
        )
    )

    fire_polygon = Polygon([(120, 120), (180, 120), (180, 180), (120, 180)])
    gis_map.add_fire_zone(
        FireZone(
            zone_id="fire_1",
            polygon=fire_polygon,
            fire_intensity=0.8,
        )
    )

    return gis_map


def load_geojson(filepath: str) -> GISMap:
    """Load a GIS map from a GeoJSON file."""
    import json

    with open(filepath, "r") as f:
        data = json.load(f)

    bounds = (float("inf"), float("inf"), float("-inf"), float("-inf"))
    buildings = []
    roads = []
    terrain_zones = []
    fire_zones = []

    for feature in data.get("features", []):
        geom_type = feature.get("geometry", {}).get("type")
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        if geom_type == "Polygon":
            polygon = shape(geom)
            minx, miny, maxx, maxy = polygon.bounds
            bounds = (
                min(bounds[0], minx),
                min(bounds[1], miny),
                max(bounds[2], maxx),
                max(bounds[3], maxy),
            )

            feat_type = props.get("type", "")
            if feat_type == "building":
                buildings.append(
                    Building(
                        building_id=props.get("id", f"building_{len(buildings)}"),
                        polygon=polygon,
                        height=props.get("height", 0.0),
                        floors=props.get("floors", 1),
                        building_type=props.get("building_type", "generic"),
                    )
                )
            elif feat_type == "terrain":
                terrain_type_str = props.get("terrain_type", "GRASS")
                try:
                    terrain_type = TerrainType[terrain_type_str]
                except KeyError:
                    terrain_type = TerrainType.GRASS
                terrain_zones.append(
                    TerrainZone(
                        zone_id=props.get("id", f"terrain_{len(terrain_zones)}"),
                        polygon=polygon,
                        terrain_type=terrain_type,
                        fuel_load=props.get("fuel_load", 0.5),
                        moisture_content=props.get("moisture_content", 0.3),
                    )
                )
            elif feat_type == "fire":
                fire_zones.append(
                    FireZone(
                        zone_id=props.get("id", f"fire_{len(fire_zones)}"),
                        polygon=polygon,
                        fire_intensity=props.get("intensity", 0.5),
                    )
                )

        elif geom_type == "LineString":
            line = shape(geom)
            minx, miny, maxx, maxy = line.bounds
            bounds = (
                min(bounds[0], minx),
                min(bounds[1], miny),
                max(bounds[2], maxx),
                max(bounds[3], maxy),
            )

            feat_type = props.get("type", "")
            if feat_type == "road":
                road_type_str = props.get("road_type", "RESIDENTIAL")
                try:
                    road_type = RoadType[road_type_str]
                except KeyError:
                    road_type = RoadType.RESIDENTIAL
                roads.append(
                    Road(
                        road_id=props.get("id", f"road_{len(roads)}"),
                        line=line,
                        road_type=road_type,
                        name=props.get("name", ""),
                        lanes=props.get("lanes", 2),
                        one_way=props.get("one_way", False),
                    )
                )

    if bounds[0] == float("inf"):
        bounds = (0, 0, 1000, 1000)

    return GISMap(
        bounds=bounds,
        buildings=buildings,
        roads=roads,
        terrain_zones=terrain_zones,
        fire_zones=fire_zones,
    )
