"""Map-based layout system for FireSim emergency response environment."""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


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
    zones: list[Zone] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    obstacles: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map dimensions must be positive")

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


def create_default_map(width: float = 1000.0, height: float = 1000.0) -> EmergencyMap:
    """Create a default emergency map with some preset zones and incidents."""
    emergency_map = EmergencyMap(width=width, height=height)

    emergency_map.add_zone(
        Zone(
            zone_id="fire_zone_1",
            zone_type=ZoneType.FIRE,
            position=(300.0, 300.0),
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
            position=(300.0, 300.0),
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
