"""Fire dynamics simulation using the Rothermel model and physical fire propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class FuelModel(Enum):
    """Standard fuel models from Rothermel (1972)."""

    GRASS = 1
    GRASS_MIX = 2
    TIMBER_GRASS = 3
    TIMBER_MIX = 4
    HARDWOOD_LITTER = 5
    CHAPARRAL = 6
    SAGEBRUSH = 7
    BOREAL_FOREST = 8
    CUT_BLOCK = 9
    BURNED = 10
    URBAN = 11
    WATER = 12


@dataclass
class FuelProperties:
    """Fuel properties for fire spread calculations."""

    fuel_model: FuelModel
    fuel_load: float = 1.0
    fuel_depth: float = 1.0
    moisture_content: float = 0.1
    bulk_density: float = 0.1
    surface_area_ratio: float = 1500.0
    mineral_content: float = 0.055
    effective_heating_number: float = 0.01
    particle_density: float = 32.0

    @staticmethod
    def from_terrain(
        terrain_type: str, fuel_load: float, moisture: float
    ) -> FuelProperties:
        """Create fuel properties from terrain type."""
        model_map = {
            "forest": FuelModel.BOREAL_FOREST,
            "grass": FuelModel.GRASS,
            "shrub": FuelModel.CHAPARRAL,
            "urban": FuelModel.URBAN,
            "water": FuelModel.WATER,
        }
        model = model_map.get(terrain_type.lower(), FuelModel.GRASS)
        return FuelProperties(
            fuel_model=model,
            fuel_load=fuel_load,
            moisture_content=moisture,
            bulk_density=fuel_load / 2.0,
        )


@dataclass
class Weather:
    """Weather conditions affecting fire spread."""

    wind_speed: float = 10.0
    wind_direction: float = 0.0
    temperature: float = 25.0
    humidity: float = 0.4

    def get_wind_vector(self) -> tuple[float, float]:
        """Get wind velocity vector."""
        rad = np.radians(self.wind_direction)
        return (self.wind_speed * np.cos(rad), self.wind_speed * np.sin(rad))


@dataclass
class FireState:
    """Dynamic fire state at a location."""

    position: tuple[float, float]
    intensity: float = 0.0
    rate_of_spread: float = 0.0
    fire_line_intensity: float = 0.0
    flame_length: float = 0.0
    flame_angle: float = 0.0
    fuel_consumed: float = 0.0
    is_spreading: bool = False
    spread_direction: float = 0.0


@dataclass
class FireDynamics:
    """Fire dynamics model based on Rothermel (1972)."""

    weather: Weather = field(default_factory=Weather)
    slope_angle: float = 0.0

    GRAVITY: float = 9.81
    AIR_DENSITY: float = 1.2
    HEAT_OF_PYROLYSIS: float = 2500000.0

    def compute_reaction_intensity(self, fuel: FuelProperties) -> float:
        """Compute reaction intensity (kW/m²)."""
        particle_density = fuel.particle_density
        fuel_load = fuel.fuel_load
        bulk_density = fuel.bulk_density
        moisture = fuel.moisture_content
        mineral = fuel.mineral_content

        moisture_ratio = moisture / (1 + moisture)
        mineral_factor = 1 - mineral

        reaction = (
            particle_density
            * bulk_density
            * fuel_load
            * np.exp(-138.0 / particle_density)
            * np.exp(-0.0173 * particle_density)
            * mineral_factor
            * (1 - moisture_ratio)
        )

        return max(0.0, reaction * 1000.0)

    def compute_wind_factor(self, fuel: FuelProperties) -> float:
        """Compute wind factor from Rothermel model."""
        sar = fuel.surface_area_ratio
        wind_speed = self.weather.wind_speed

        if wind_speed < 0.1:
            return 0.0

        wind_factor = sar * wind_speed / 1000.0
        return wind_factor

    def compute_slope_factor(self, fuel: FuelProperties) -> float:
        """Compute slope factor from Rothermel model."""
        slope_rad = np.radians(self.slope_angle)
        return np.exp(slope_rad * 3.54)

    def compute_rate_of_spread(
        self, fuel: FuelProperties, direction: float | None = None
    ) -> float:
        """Compute rate of spread (m/s)."""
        reaction_intensity = self.compute_reaction_intensity(fuel)
        wind_factor = self.compute_wind_factor(fuel)
        slope_factor = self.compute_slope_factor(fuel)

        effective_intensity = reaction_intensity * (1 + wind_factor + slope_factor - 1)

        if effective_intensity <= 0:
            return 0.0

        bulk_density = fuel.bulk_density
        if bulk_density < 0.01:
            return 0.0

        heat_sink = (
            250.0
            + 1116.0 * fuel.moisture_content
            - 0.0614
            * fuel.moisture_content
            * fuel.moisture_content
            * fuel.particle_density
        )

        ros = effective_intensity / (heat_sink * bulk_density * fuel.fuel_depth)

        if direction is not None:
            wind_dir = self.weather.wind_direction
            angle_diff = abs(direction - wind_dir)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            wind_effect = np.cos(angle_diff) * wind_factor
            ros *= max(0.1, 1.0 + wind_effect)

        return max(0.0, min(ros, 100.0))

    def compute_fire_intensity(self, fuel: FuelProperties, ros: float) -> float:
        """Compute fire intensity (kW/m)."""
        reaction = self.compute_reaction_intensity(fuel)
        fire_intensity = reaction * ros / 100.0
        return min(fire_intensity, 1000.0)

    def compute_flame_length(self, fire_intensity: float) -> float:
        """Compute flame length from fire intensity (m)."""
        if fire_intensity <= 0:
            return 0.0
        return 0.0775 * (fire_intensity**0.46)

    def compute_flame_angle(self, wind_speed: float) -> float:
        """Compute flame angle from wind speed (radians)."""
        return np.arctan(0.93 * wind_speed / 3.0)

    def compute_spread_direction(
        self, fuel: FuelProperties, position: tuple[float, float]
    ) -> float:
        """Compute direction of fire spread."""
        wind_dir = self.weather.wind_direction

        if self.slope_angle > 0:
            slope_dir = np.arctan2(self.slope_angle, 1.0)
            return 0.7 * wind_dir + 0.3 * slope_dir

        return wind_dir

    def get_fire_state_at(
        self, position: tuple[float, float], fuel: FuelProperties | None = None
    ) -> FireState:
        """Get fire state at a position."""
        if fuel is None:
            fuel = FuelProperties(FuelModel.GRASS)

        ros = self.compute_rate_of_spread(fuel)
        intensity = self.compute_fire_intensity(fuel, ros)
        flame_length = self.compute_flame_length(intensity)
        flame_angle = self.compute_flame_angle(self.weather.wind_speed)

        return FireState(
            position=position,
            intensity=intensity / 1000.0,
            rate_of_spread=ros,
            fire_line_intensity=intensity,
            flame_length=flame_length,
            flame_angle=flame_angle,
            is_spreading=ros > 0.1,
            spread_direction=self.compute_spread_direction(fuel, position),
        )


@dataclass
class FireModel:
    """Full fire propagation model managing multiple fire fronts."""

    dynamics: FireDynamics = field(default_factory=FireDynamics)
    fires: list[FireState] = field(default_factory=list)
    ignition_points: list[tuple[float, float, float]] = field(default_factory=list)
    containment_lines: list[list[tuple[float, float]]] = field(default_factory=list)
    time: float = 0.0
    dt: float = 1.0

    def add_ignition(
        self, position: tuple[float, float], intensity: float = 1.0
    ) -> None:
        """Add an ignition point."""
        self.ignition_points.append((position[0], position[1], intensity))
        self.fires.append(
            FireState(
                position=position,
                intensity=intensity,
                is_spreading=True,
            )
        )

    def add_containment_line(self, line: list[tuple[float, float]]) -> None:
        """Add a fire containment line."""
        self.containment_lines.append(line)

    def update(self, fuel_map: dict[tuple[float, float], FuelProperties]) -> None:
        """Update fire propagation."""
        new_fires: list[FireState] = []

        for fire in self.fires:
            if not fire.is_spreading:
                continue

            fuel = fuel_map.get(
                (int(fire.position[0]), int(fire.position[1])),
                FuelProperties(FuelModel.GRASS, 0.5, 0.3),
            )

            ros = self.dynamics.compute_rate_of_spread(fuel, fire.spread_direction)
            fire.rate_of_spread = ros

            intensity = self.dynamics.compute_fire_intensity(fuel, ros)
            fire.fire_line_intensity = intensity
            fire.intensity = min(intensity / 1000.0, 1.0)

            fire.flame_length = self.dynamics.compute_flame_length(intensity)
            fire.flame_angle = self.dynamics.compute_flame_angle(
                self.dynamics.weather.wind_speed
            )

            if ros > 0.01:
                spread_dist = ros * self.dt
                fire.position = (
                    fire.position[0] + spread_dist * np.cos(fire.spread_direction),
                    fire.position[1] + spread_dist * np.sin(fire.spread_direction),
                )
                new_fires.append(fire)
            else:
                fire.is_spreading = False

            fire.fuel_consumed = min(1.0, fire.fuel_consumed + ros * self.dt / 10.0)

        self.fires.extend(new_fires)
        self.time += self.dt

    def get_intensity_at(self, point: tuple[float, float]) -> float:
        """Get fire intensity at a point."""
        max_intensity = 0.0
        for fire in self.fires:
            dist = np.sqrt(
                (fire.position[0] - point[0]) ** 2 + (fire.position[1] - point[1]) ** 2
            )
            if dist < 50:
                intensity = fire.intensity * (1 - dist / 50)
                max_intensity = max(max_intensity, intensity)
        return max_intensity

    def get_rate_of_spread_at(self, point: tuple[float, float]) -> float:
        """Get rate of spread at a point."""
        for fire in self.fires:
            dist = np.sqrt(
                (fire.position[0] - point[0]) ** 2 + (fire.position[1] - point[1]) ** 2
            )
            if dist < 10:
                return fire.rate_of_spread
        return 0.0

    def get_flame_length_at(self, point: tuple[float, float]) -> float:
        """Get flame length at a point."""
        for fire in self.fires:
            dist = np.sqrt(
                (fire.position[0] - point[0]) ** 2 + (fire.position[1] - point[1]) ** 2
            )
            if dist < 20:
                return fire.flame_length * (1 - dist / 20)
        return 0.0

    def get_fire_distance(
        self, point: tuple[float, float]
    ) -> tuple[float, tuple[float, float] | None]:
        """Get distance to nearest fire and its position.

        Returns:
            Tuple of (distance, fire_position)
        """
        if not self.fires:
            return float("inf"), None

        min_dist = float("inf")
        nearest_fire = None

        for fire in self.fires:
            dist = np.sqrt(
                (fire.position[0] - point[0]) ** 2 + (fire.position[1] - point[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_fire = fire.position

        return min_dist, nearest_fire


def create_default_fire_model(
    ignition_points: list[tuple[float, float]] | None = None,
    wind_speed: float = 10.0,
    wind_direction: float = 0.0,
) -> FireModel:
    """Create a default fire model."""
    weather = Weather(wind_speed=wind_speed, wind_direction=wind_direction)
    dynamics = FireDynamics(weather=weather)
    model = FireModel(dynamics=dynamics)

    if ignition_points:
        for point in ignition_points:
            model.add_ignition(point)

    return model
