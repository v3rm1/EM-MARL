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
    heat_diffusion_coefficient: float = 0.005
    canopy_base_height: float = 0.0
    canopy_fuel_load: float = 0.0

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

        canopy_base = 3.0 if terrain_type.lower() == "forest" else 0.0
        canopy_fuel = 0.8 if terrain_type.lower() == "forest" else 0.0

        return FuelProperties(
            fuel_model=model,
            fuel_load=fuel_load,
            moisture_content=moisture,
            bulk_density=fuel_load / 2.0,
            canopy_base_height=canopy_base,
            canopy_fuel_load=canopy_fuel,
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
    temperature: float = 300.0
    pre_heating: float = 0.0
    is_crown_fire: bool = False
    ember_count: int = 0


@dataclass
class HeatMap:
    """Heat map for thermal diffusion between cells."""

    width: int
    height: int
    temperatures: np.ndarray | None = None
    preheat_levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.temperatures = np.full((self.height, self.width), 300.0, dtype=np.float32)
        self.preheat_levels = np.zeros((self.height, self.width), dtype=np.float32)

    def get_temp_at(self, grid_x: int, grid_y: int) -> float:
        """Get temperature at grid cell."""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return float(self.temperatures[grid_y, grid_x])
        return 300.0

    def get_preheat_at(self, grid_x: int, grid_y: int) -> float:
        """Get pre-heat level at grid cell."""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return float(self.preheat_levels[grid_y, grid_x])
        return 0.0

    def set_temp_at(self, grid_x: int, grid_y: int, temp: float) -> None:
        """Set temperature at grid cell."""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            self.temperatures[grid_y, grid_x] = temp

    def set_preheat_at(self, grid_x: int, grid_y: int, preheat: float) -> None:
        """Set pre-heat level at grid cell."""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            self.preheat_levels[grid_y, grid_x] = preheat

    def diffuse_heat(
        self,
        diffusion_coef: float,
        dt: float,
        fire_positions: list[tuple[int, int]],
        fire_temperatures: list[float],
    ) -> None:
        """Apply thermal diffusion to adjacent cells."""
        new_temps = self.temperatures.copy()

        for gy in range(self.height):
            for gx in range(self.width):
                current_temp = self.temperatures[gy, gx]

                if (gx, gy) in fire_positions:
                    continue

                neighbor_sum = 0.0
                neighbor_count = 0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            neighbor_sum += self.temperatures[ny, nx]
                            neighbor_count += 1

                if neighbor_count > 0:
                    avg_neighbor = neighbor_sum / neighbor_count
                    new_temps[gy, gx] = (
                        current_temp
                        + diffusion_coef * (avg_neighbor - current_temp) * dt
                    )

        self.temperatures = new_temps

    def apply_fire_heat(
        self,
        grid_x: int,
        grid_y: int,
        fire_temp: float,
        radius: int = 1,
    ) -> None:
        """Apply heat from fire to nearby cells."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 0:
                        falloff = 1.0 / (1.0 + dist * 0.5)
                        self.temperatures[ny, nx] = max(
                            self.temperatures[ny, nx], fire_temp * falloff
                        )


@dataclass
class FirePerimeter:
    """Fire perimeter tracking for polygon-based evolution."""

    points: list[tuple[float, float]] = field(default_factory=list)
    area: float = 0.0
    perimeter: float = 0.0

    def add_point(self, x: float, y: float) -> None:
        """Add a point to the perimeter."""
        self.points.append((x, y))

    def compute_metrics(self) -> None:
        """Compute area and perimeter from points."""
        if len(self.points) < 3:
            self.area = 0.0
            self.perimeter = 0.0
            return

        area = 0.0
        perimeter = 0.0
        n = len(self.points)

        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
            perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        self.area = abs(area) / 2.0
        self.perimeter = perimeter

    def is_point_inside(self, x: float, y: float) -> bool:
        """Ray casting algorithm for point in polygon."""
        if len(self.points) < 3:
            return False

        inside = False
        n = len(self.points)
        j = n - 1

        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside


@dataclass
class Ember:
    """Ember for fire spotting."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    lifetime: float = 0.0
    max_lifetime: float = 10.0

    def update(self, dt: float, wind_speed: float, wind_dir: float) -> None:
        """Update ember position and lifetime."""
        self.lifetime += dt

        rad = np.radians(wind_dir)
        base_vx = wind_speed * 0.3 * np.cos(rad)
        base_vy = wind_speed * 0.3 * np.sin(rad)

        self.position = (
            self.position[0] + (self.velocity[0] + base_vx) * dt,
            self.position[1] + (self.velocity[1] + base_vy) * dt,
        )

    def is_active(self) -> bool:
        """Check if ember is still active."""
        return self.lifetime < self.max_lifetime


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

    def compute_crown_fire_transition(
        self, fuel: FuelProperties, surface_ros: float, surface_intensity: float
    ) -> bool:
        """Determine if crown fire transition should occur.

        Uses Rothermel crown fire transition criteria:
        - Critical surface fire intensity > 300 kW/m
        - Crown base height < 6m
        - Canopy bulk density > 0.1 kg/m³
        """
        critical_intensity = 300.0
        can_transition = (
            surface_intensity > critical_intensity
            and fuel.canopy_base_height > 0
            and fuel.canopy_base_height < 6.0
            and fuel.canopy_fuel_load > 0.1
        )
        return can_transition

    def compute_crown_fire_ros(self, fuel: FuelProperties, surface_ros: float) -> float:
        """Compute crown fire rate of spread (m/s).

        Crown fire spreads faster than surface fire.
        """
        if fuel.canopy_base_height <= 0 or fuel.canopy_fuel_load <= 0:
            return surface_ros

        crown_ros_mult = 1.5 + (self.weather.wind_speed / 20.0)
        return surface_ros * crown_ros_mult

    def compute_spotting_probability(
        self, fire_intensity: float, wind_speed: float
    ) -> float:
        """Compute probability of fire spotting.

        Higher fire intensity and wind speed increase spotting.
        """
        if fire_intensity <= 0 or wind_speed < 1.0:
            return 0.0

        base_prob = fire_intensity * wind_speed * 0.0001
        return min(base_prob, 0.3)

    def compute_max_spotting_distance(self, wind_speed: float) -> float:
        """Compute maximum spotting distance in meters.

        Embers can travel up to ~2.5x wind speed in km.
        """
        return 2.5 * wind_speed * 1000.0

    def compute_containment_effectiveness(
        self,
        line_position: tuple[float, float],
        fire_position: tuple[float, float],
        line_width: float = 3.0,
    ) -> float:
        """Compute containment line effectiveness (0-1).

        Closer fire to line = higher containment probability.
        """
        dist = np.sqrt(
            (fire_position[0] - line_position[0]) ** 2
            + (fire_position[1] - line_position[1]) ** 2
        )

        if dist < line_width:
            return 0.95
        elif dist < 50.0:
            return 0.8 * (1.0 - dist / 50.0)
        return 0.0

    def compute_preheat_effect(
        self, cell_temp: float, ambient_temp: float = 300.0
    ) -> float:
        """Compute pre-heating effect on ignition time.

        Higher temperatures reduce time to ignition.
        """
        temp_excess = max(0.0, cell_temp - ambient_temp)
        ignition_time_reduction = 1.0 - min(temp_excess / 500.0, 0.8)
        return ignition_time_reduction

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
    embers: list[Ember] = field(default_factory=list)
    perimeter: FirePerimeter = field(default_factory=FirePerimeter)
    heat_map: HeatMap | None = None
    time: float = 0.0
    dt: float = 1.0

    grid_width: int = 100
    grid_height: int = 100

    def __post_init__(self) -> None:
        if self.heat_map is None:
            self.heat_map = HeatMap(width=self.grid_width, height=self.grid_height)

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
                temperature=800.0,
            )
        )
        self.perimeter.add_point(position[0], position[1])

    def add_containment_line(self, line: list[tuple[float, float]]) -> None:
        """Add a fire containment line."""
        self.containment_lines.append(line)

    def _update_heat_diffusion(
        self, fuel_map: dict[tuple[float, float], FuelProperties]
    ) -> None:
        """Update heat transfer between cells."""
        if self.heat_map is None:
            return

        fire_positions = []
        fire_temps = []

        for fire in self.fires:
            if fire.is_spreading and fire.intensity > 0.1:
                gx = int(fire.position[0] / 10)
                gy = int(fire.position[1] / 10)
                fire_positions.append((gx, gy))
                fire_temps.append(500.0 + fire.fire_line_intensity)

        if not fire_positions:
            return

        diffusion_coef = 0.05
        self.heat_map.diffuse_heat(diffusion_coef, self.dt, fire_positions, fire_temps)

        for (gx, gy), temp in zip(fire_positions, fire_temps):
            self.heat_map.apply_fire_heat(gx, gy, temp, radius=2)

    def _update_spotting(
        self, fuel_map: dict[tuple[float, float], FuelProperties]
    ) -> None:
        """Update fire spotting/ember transport."""
        wind_speed = self.dynamics.weather.wind_speed
        wind_dir = self.dynamics.weather.wind_direction

        for ember in self.embers:
            if ember.is_active():
                ember.update(self.dt, wind_speed, wind_dir)

                gx = int(ember.position[0] / 10)
                gy = int(ember.position[1] / 10)

                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    fuel_key = (gx * 10, gy * 10)
                    fuel = fuel_map.get(fuel_key, FuelProperties(FuelModel.GRASS))

                    if fuel.fuel_load > 0.1 and fuel.fuel_load < 0.9:
                        self.add_ignition(ember.position, 0.3)

        self.embers = [e for e in self.embers if e.is_active()]

        for fire in self.fires:
            if not fire.is_spreading:
                continue

            spotting_prob = self.dynamics.compute_spotting_probability(
                fire.fire_line_intensity, wind_speed
            )

            if np.random.random() < spotting_prob * self.dt:
                max_dist = self.dynamics.compute_max_spotting_distance(wind_speed)

                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(50, min(max_dist, 500))

                ember_pos = (
                    fire.position[0] + distance * np.cos(angle),
                    fire.position[1] + distance * np.sin(angle),
                )

                ember_vel = (
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                )

                self.embers.append(Ember(position=ember_pos, velocity=ember_vel))

    def _update_crown_fire(
        self, fuel_map: dict[tuple[float, float], FuelProperties]
    ) -> None:
        """Update crown fire dynamics."""
        for fire in self.fires:
            if not fire.is_spreading:
                continue

            fuel_key = (int(fire.position[0]), int(fire.position[1]))
            fuel = fuel_map.get(fuel_key, FuelProperties(FuelModel.GRASS))

            if fuel.canopy_base_height > 0:
                is_crown = self.dynamics.compute_crown_fire_transition(
                    fuel, fire.rate_of_spread, fire.fire_line_intensity
                )
                fire.is_crown_fire = is_crown

                if is_crown:
                    fire.rate_of_spread = self.dynamics.compute_crown_fire_ros(
                        fuel, fire.rate_of_spread
                    )
                    fire.rate_of_spread = min(fire.rate_of_spread, 150.0)

    def _update_containment(
        self, fuel_map: dict[tuple[float, float], FuelProperties]
    ) -> None:
        """Update containment line effectiveness."""
        for fire in self.fires:
            if not fire.is_spreading:
                continue

            for line in self.containment_lines:
                for line_point in line:
                    effectiveness = self.dynamics.compute_containment_effectiveness(
                        line_point, fire.position
                    )

                    if np.random.random() < effectiveness * 0.1 * self.dt:
                        fire.rate_of_spread *= 0.5

    def _update_pyrolysis(
        self, fuel_map: dict[tuple[float, float], FuelProperties]
    ) -> None:
        """Update pyrolysis/pre-heating zones."""
        if self.heat_map is None:
            return

        for fire in self.fires:
            if not fire.is_spreading:
                continue

            gx = int(fire.position[0] / 10)
            gy = int(fire.position[1] / 10)

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            preheat = 0.3 / (dist * 0.5)
                            current_preheat = self.heat_map.get_preheat_at(nx, ny)
                            self.heat_map.set_preheat_at(
                                nx, ny, min(current_preheat + preheat, 1.0)
                            )

    def _update_perimeter(self) -> None:
        """Update fire perimeter tracking."""
        self.perimeter.points.clear()

        for fire in self.fires:
            if fire.is_spreading:
                self.perimeter.add_point(fire.position[0], fire.position[1])

        self.perimeter.compute_metrics()

    def update(self, fuel_map: dict[tuple[float, float], FuelProperties]) -> None:
        """Update fire propagation with advanced physics."""
        self._update_heat_diffusion(fuel_map)
        self._update_pyrolysis(fuel_map)
        self._update_crown_fire(fuel_map)
        self._update_spotting(fuel_map)
        self._update_containment(fuel_map)

        new_fires: list[FireState] = []

        for fire in self.fires:
            if not fire.is_spreading:
                continue

            fuel_key = (int(fire.position[0]), int(fire.position[1]))
            fuel = fuel_map.get(fuel_key, FuelProperties(FuelModel.GRASS, 0.5, 0.3))

            if self.heat_map is not None:
                gx = int(fire.position[0] / 10)
                gy = int(fire.position[1] / 10)
                preheat = self.heat_map.get_preheat_at(gx, gy)

                if preheat > 0.2:
                    fuel = FuelProperties(
                        fuel_model=fuel.fuel_model,
                        fuel_load=fuel.fuel_load,
                        fuel_depth=fuel.fuel_depth,
                        moisture_content=fuel.moisture_content * (1 - preheat * 0.5),
                        bulk_density=fuel.bulk_density,
                        surface_area_ratio=fuel.surface_area_ratio,
                        mineral_content=fuel.mineral_content,
                        effective_heating_number=fuel.effective_heating_number,
                        particle_density=fuel.particle_density,
                        heat_diffusion_coefficient=fuel.heat_diffusion_coefficient,
                        canopy_base_height=fuel.canopy_base_height,
                        canopy_fuel_load=fuel.canopy_fuel_load,
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

            if fire.is_crown_fire:
                fire.temperature = 1000.0
            else:
                fire.temperature = 500.0 + intensity * 0.5

            containment_reduction = 1.0
            for line in self.containment_lines:
                for line_point in line:
                    containment_reduction *= (
                        1.0
                        - self.dynamics.compute_containment_effectiveness(
                            line_point, fire.position
                        )
                        * 0.5
                    )

            effective_ros = ros * containment_reduction

            if effective_ros > 0.01:
                spread_dist = effective_ros * self.dt

                angle_variation = np.random.uniform(-0.1, 0.1)
                new_direction = fire.spread_direction + angle_variation

                fire.position = (
                    fire.position[0] + spread_dist * np.cos(new_direction),
                    fire.position[1] + spread_dist * np.sin(new_direction),
                )
                fire.spread_direction = new_direction
                new_fires.append(fire)
            else:
                fire.is_spreading = False

            fire.fuel_consumed = min(
                1.0, fire.fuel_consumed + effective_ros * self.dt / 10.0
            )

        self.fires.extend(new_fires)
        self._update_perimeter()
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

    def get_temperature_at(self, point: tuple[float, float]) -> float:
        """Get temperature at a point from heat map."""
        if self.heat_map is None:
            return 300.0

        gx = int(point[0] / 10)
        gy = int(point[1] / 10)
        return self.heat_map.get_temp_at(gx, gy)

    def get_preheat_at(self, point: tuple[float, float]) -> float:
        """Get pre-heat level at a point."""
        if self.heat_map is None:
            return 0.0

        gx = int(point[0] / 10)
        gy = int(point[1] / 10)
        return self.heat_map.get_preheat_at(gx, gy)

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

    def get_fire_area(self) -> float:
        """Get total fire area from perimeter."""
        return self.perimeter.area

    def get_fire_perimeter_length(self) -> float:
        """Get fire perimeter length."""
        return self.perimeter.perimeter

    def is_point_in_fire(self, point: tuple[float, float]) -> bool:
        """Check if point is inside fire perimeter."""
        return self.perimeter.is_point_inside(point[0], point[1])

    def get_active_ember_count(self) -> int:
        """Get number of active embers."""
        return len(self.embers)


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


@dataclass
class SuppressionResource:
    """Suppression resource type."""

    WATER: str = "water"
    FOAM: str = "foam"
    RETARDANT: str = "retardant"
    AERIAL_WATER: str = "aerial_water"


@dataclass
class SuppressionPhysics:
    """Physics model for fire suppression effectiveness.

    Models water/foam/retardant application effectiveness based on:
    - Fire intensity at target location
    - Water temperature and evaporation rates
    - Chemical persistence for foam/retardant
    - Downhill creep effects for liquids
    """

    MAX_FIRE_INTENSITY: float = 1000.0
    WATER_TEMPERATURE: float = 20.0
    WATER_BOILING_POINT: float = 100.0
    LATENT_HEAT_VAPORIZATION: float = 2260000.0

    def compute_water_effectiveness(
        self,
        water_amount: float,
        fire_intensity: float,
        water_temperature: float = 20.0,
    ) -> float:
        """Compute water suppression effectiveness.

        Args:
            water_amount: Amount of water applied (gallons)
            fire_intensity: Fire intensity at target (kW/m)
            water_temperature: Temperature of water (Celsius)

        Returns:
            Effective suppression value (0-1)
        """
        if water_amount <= 0 or fire_intensity <= 0:
            return 0.0

        if fire_intensity >= self.MAX_FIRE_INTENSITY:
            return 0.0

        temp_factor = max(
            0.5, 1.0 - (water_temperature - self.WATER_TEMPERATURE) / 100.0
        )

        intensity_factor = 1.0 - (fire_intensity / self.MAX_FIRE_INTENSITY)

        effective_suppression = water_amount * intensity_factor * temp_factor

        return min(effective_suppression / 100.0, 1.0)

    def compute_evaporation_rate(
        self,
        fire_intensity: float,
        water_temperature: float = 20.0,
    ) -> float:
        """Compute water evaporation rate based on fire intensity.

        Args:
            fire_intensity: Fire intensity at target (kW/m)
            water_temperature: Temperature of water (Celsius)

        Returns:
            Evaporation rate (gallons/second)
        """
        if fire_intensity <= 0:
            return 0.0

        temp_above_boiling = max(0.0, water_temperature - self.WATER_BOILING_POINT)

        evaporation = (fire_intensity / self.LATENT_HEAT_VAPORIZATION) * 0.001

        evaporation *= 1.0 + temp_above_boiling / 100.0

        return min(evaporation, 1.0)


@dataclass
class FoamPhysics:
    """Physics model for foam/retardant suppression behavior.

    Models:
    - Chemical persistence factor
    - Downhill creep effect
    - Coverage area
    """

    PERSISTENCE_DECAY_RATE: float = 0.02
    DOWNSLOPE_CREEP_FACTOR: float = 0.15
    COVERAGE_RADIUS_FACTOR: float = 2.0

    def compute_foam_effectiveness(
        self,
        foam_amount: float,
        fire_intensity: float,
        slope_angle: float = 0.0,
        age: float = 0.0,
    ) -> float:
        """Compute foam/retardant suppression effectiveness.

        Args:
            foam_amount: Amount of foam applied (gallons)
            fire_intensity: Fire intensity at target (kW/m)
            slope_angle: Slope of terrain (degrees, positive = downhill)
            age: Time since application (seconds)

        Returns:
            Effective suppression value (0-1)
        """
        if foam_amount <= 0 or fire_intensity <= 0:
            return 0.0

        persistence_factor = max(0.1, 1.0 - age * self.PERSISTENCE_DECAY_RATE)

        slope_factor = 1.0
        if slope_angle > 0:
            slope_factor = 1.0 + slope_angle * self.DOWNSLOPE_CREEP_FACTOR / 10.0

        intensity_factor = max(0.2, 1.0 - fire_intensity / 1500.0)

        effectiveness = (
            foam_amount * persistence_factor * slope_factor * intensity_factor
        )

        return min(effectiveness / 50.0, 1.0)

    def compute_coverage_area(self, foam_amount: float) -> float:
        """Compute coverage area from foam amount.

        Args:
            foam_amount: Amount of foam applied (gallons)

        Returns:
            Coverage area in square meters
        """
        return foam_amount * self.COVERAGE_RADIUS_FACTOR

    def apply_downhill_creep(
        self,
        position: tuple[float, float],
        slope_angle: float,
        slope_direction: float,
        dt: float,
    ) -> tuple[float, float]:
        """Apply downhill creep to foam position.

        Args:
            position: Current position (x, y)
            slope_angle: Slope angle in degrees
            slope_direction: Direction of slope in radians
            dt: Time step (seconds)

        Returns:
            New position after creep
        """
        if slope_angle <= 0:
            return position

        creep_distance = slope_angle * self.DOWNSLOPE_CREEP_FACTOR * dt / 10.0

        new_x = position[0] + np.cos(slope_direction) * creep_distance
        new_y = position[1] + np.sin(slope_direction) * creep_distance

        return (new_x, new_y)


@dataclass
class AerialSuppressionPhysics:
    """Physics model for aerial water/retardant drops.

    Models:
    - Wind drift effects on drop accuracy
    - Coverage pattern (line vs spot drops)
    - Drop dispersion
    """

    DROP_VELOCITY: float = 20.0
    DISPERSION_FACTOR: float = 5.0

    def compute_drop_accuracy(
        self,
        drop_position: tuple[float, float],
        target_position: tuple[float, float],
        wind_speed: float,
        wind_direction: float,
        release_height: float = 100.0,
    ) -> tuple[float, float]:
        """Compute actual drop position accounting for wind.

        Args:
            drop_position: Intended drop position (x, y)
            target_position: Target fire position (x, y)
            wind_speed: Wind speed (m/s)
            wind_direction: Wind direction (degrees)
            release_height: Aircraft release height (meters)

        Returns:
            Actual drop position (x, y)
        """
        fall_time = release_height / self.DROP_VELOCITY

        wind_rad = np.radians(wind_direction)
        wind_drift_x = wind_speed * np.cos(wind_rad) * fall_time
        wind_drift_y = wind_speed * np.sin(wind_rad) * fall_time

        actual_x = drop_position[0] + wind_drift_x * 0.5
        actual_y = drop_position[1] + wind_drift_y * 0.5

        return (actual_x, actual_y)

    def compute_drop_error(
        self,
        wind_speed: float,
        release_height: float = 100.0,
    ) -> float:
        """Compute expected drop error due to wind.

        Args:
            wind_speed: Wind speed (m/s)
            release_height: Aircraft release height (meters)

        Returns:
            Error radius in meters
        """
        fall_time = release_height / self.DROP_VELOCITY

        base_error = wind_speed * fall_time * 0.3

        return base_error + self.DISPERSION_FACTOR

    def compute_line_drop_coverage(
        self,
        start_position: tuple[float, float],
        end_position: tuple[float, float],
        drop_spacing: float = 20.0,
    ) -> list[tuple[float, float]]:
        """Compute drop positions for line coverage pattern.

        Args:
            start_position: Start of fire line (x, y)
            end_position: End of fire line (x, y)
            drop_spacing: Spacing between drops (meters)

        Returns:
            List of drop positions
        """
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        line_length = np.sqrt(dx * dx + dy * dy)

        if line_length < 1.0:
            return [start_position]

        num_drops = max(1, int(line_length / drop_spacing))

        positions = []
        for i in range(num_drops + 1):
            t = i / max(1, num_drops)
            x = start_position[0] + dx * t
            y = start_position[1] + dy * t
            positions.append((x, y))

        return positions

    def compute_spot_drop_coverage(
        self,
        target_position: tuple[float, float],
        num_drops: int = 3,
        spread_radius: float = 30.0,
    ) -> list[tuple[float, float]]:
        """Compute drop positions for spot coverage pattern.

        Args:
            target_position: Target fire center (x, y)
            num_drops: Number of drops
            spread_radius: Radius to spread drops (meters)

        Returns:
            List of drop positions
        """
        positions = [target_position]

        if num_drops <= 1:
            return positions

        for i in range(num_drops - 1):
            angle = (2 * np.pi * i) / (num_drops - 1)
            distance = spread_radius * np.random.uniform(0.5, 1.0)

            x = target_position[0] + np.cos(angle) * distance
            y = target_position[1] + np.sin(angle) * distance

            positions.append((x, y))

        return positions


@dataclass
class SuppressionLinePhysics:
    """Physics model for progressive suppression line construction.

    Models:
    - Tool effectiveness factors
    - Line construction rate
    - Containment effectiveness
    """

    HAND_TOOL_RATE: float = 0.5
    CHAINSAW_RATE: float = 1.5
    BULDOZER_RATE: float = 3.0
    MINERAL_SOIL_EFFECTIVENESS: float = 0.8

    def compute_construction_rate(
        self,
        tool_type: str,
        num_workers: int = 1,
        slope_angle: float = 0.0,
    ) -> float:
        """Compute suppression line construction rate.

        Args:
            tool_type: Type of tool ("hand", "chainsaw", "bulldozer")
            num_workers: Number of workers
            slope_angle: Terrain slope (degrees)

        Returns:
            Construction rate (meters/second)
        """
        base_rate = self.HAND_TOOL_RATE

        if tool_type.lower() == "chainsaw":
            base_rate = self.CHAINSAW_RATE
        elif tool_type.lower() == "bulldozer":
            base_rate = self.BULDOZER_RATE

        slope_factor = max(0.3, 1.0 - slope_angle * 0.05)

        return base_rate * num_workers * slope_factor

    def compute_line_effectiveness(
        self,
        line_age: float,
        line_width: float,
        mineral_applied: bool = False,
    ) -> float:
        """Compute containment line effectiveness.

        Args:
            line_age: Time since line creation (seconds)
            line_width: Width of containment line (meters)
            mineral_applied: Whether mineral soil was applied

        Returns:
            Effectiveness value (0-1)
        """
        base_effectiveness = min(line_width / 10.0, 1.0)

        if mineral_applied:
            base_effectiveness *= self.MINERAL_SOIL_EFFECTIVENESS

        age_factor = min(1.0, line_age / 60.0)

        return base_effectiveness * (0.5 + 0.5 * age_factor)

    def compute_fire_line_contact(
        self,
        fire_position: tuple[float, float],
        line_positions: list[tuple[float, float]],
        line_width: float = 3.0,
    ) -> float:
        """Compute fire contact with containment line.

        Args:
            fire_position: Fire front position (x, y)
            line_positions: List of line segment positions
            line_width: Width of line (meters)

        Returns:
            Contact intensity (0-1)
        """
        if not line_positions:
            return 0.0

        min_dist = float("inf")
        for pos in line_positions:
            dist = np.sqrt(
                (fire_position[0] - pos[0]) ** 2 + (fire_position[1] - pos[1]) ** 2
            )
            min_dist = min(min_dist, dist)

        if min_dist < line_width:
            return 1.0 - (min_dist / line_width)

        return 0.0
