"""JAX-based fire physics for GPU-accelerated fire simulation.

This module provides JAX implementations of fire dynamics for significantly
improved computational performance on GPU hardware.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class FireStateJAX:
    """JAX-compatible fire state for GPU acceleration.

    All arrays use float32 for performance on GPU.
    """

    intensity: jax.Array
    temperature: jax.Array
    fuel_moisture: jax.Array
    preheat: jax.Array
    fire_mask: jax.Array
    ember_pos: jax.Array
    ember_vel: jax.Array
    ember_active: jax.Array
    wind_speed: float
    wind_direction: float
    time: float
    dt: float = 1.0
    grid_height: int = 100
    grid_width: int = 100


@struct.dataclass
class WeatherJAX:
    """Weather conditions for JAX fire model."""

    wind_speed: float = 10.0
    wind_direction: float = 0.0
    temperature: float = 25.0
    humidity: float = 0.4


def create_fire_state_jax(
    grid_height: int = 100,
    grid_width: int = 100,
    wind_speed: float = 10.0,
    wind_direction: float = 0.0,
    max_embers: int = 1000,
) -> FireStateJAX:
    """Create initial JAX fire state.

    Args:
        grid_height: Grid height in cells
        grid_width: Grid width in cells
        wind_speed: Initial wind speed (m/s)
        wind_direction: Wind direction (degrees)
        max_embers: Maximum number of embers

    Returns:
        FireStateJAX instance
    """
    return FireStateJAX(
        intensity=jnp.zeros((grid_height, grid_width), dtype=jnp.float32),
        temperature=jnp.full((grid_height, grid_width), 300.0, dtype=jnp.float32),
        fuel_moisture=jnp.full((grid_height, grid_width), 0.1, dtype=jnp.float32),
        preheat=jnp.zeros((grid_height, grid_width), dtype=jnp.float32),
        fire_mask=jnp.zeros((grid_height, grid_width), dtype=jnp.bool_),
        ember_pos=jnp.zeros((max_embers, 2), dtype=jnp.float32),
        ember_vel=jnp.zeros((max_embers, 2), dtype=jnp.float32),
        ember_active=jnp.zeros(max_embers, dtype=jnp.bool_),
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        time=0.0,
        dt=1.0,
        grid_height=grid_height,
        grid_width=grid_width,
    )


def add_fire_ignition(
    state: FireStateJAX,
    position: tuple[int, int],
    intensity: float = 1.0,
    radius: int = 2,
) -> FireStateJAX:
    """Add fire ignition at position.

    Args:
        state: Current fire state
        position: Grid position (y, x)
        intensity: Fire intensity (0-1)
        radius: Ignition radius in cells

    Returns:
        Updated fire state
    """
    gy, gx = position
    H, W = state.grid_height, state.grid_width

    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    dist = jnp.sqrt((y - gy) ** 2 + (x - gx) ** 2)
    mask = dist <= radius

    new_intensity = state.intensity * (1 - mask) + intensity * mask * (
        1 - dist / (radius + 1)
    )
    new_temp = jnp.where(mask, 800.0 + intensity * 200.0, state.temperature)
    new_fire_mask = state.fire_mask | mask

    return state.replace(
        intensity=new_intensity,
        temperature=new_temp,
        fire_mask=new_fire_mask,
    )


def jax_diffuse_heat(
    temperature: jax.Array,
    fire_mask: jax.Array,
    diffusion_coef: float,
    dt: float,
) -> jax.Array:
    """GPU-accelerated thermal diffusion using convolution.

    Args:
        temperature: Temperature array (H, W)
        fire_mask: Boolean mask for fire cells (H, W)
        diffusion_coef: Diffusion coefficient
        dt: Time step

    Returns:
        Updated temperature array
    """
    kernel = jnp.array(
        [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=jnp.float32
    )

    kernel = kernel[:, :, None, None]

    padded = jnp.pad(temperature, 1, mode="edge")

    neighbor_avg = jax.lax.conv_general_dilated(
        padded[None, :, :, None],
        kernel,
        window_strides=(1, 1),
        padding="valid",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0, :, :, 0]

    new_temp = temperature + diffusion_coef * (neighbor_avg - temperature) * dt

    return jnp.where(fire_mask, temperature, new_temp)


def jax_apply_fire_heat(
    temperature: jax.Array,
    fire_mask: jax.Array,
    fire_intensity: jax.Array,
    radius: int = 2,
) -> jax.Array:
    """Apply heat from fire to nearby cells.

    Args:
        temperature: Temperature array (H, W)
        fire_mask: Boolean mask for fire cells (H, W)
        fire_intensity: Fire intensity (H, W)
        radius: Heat application radius

    Returns:
        Updated temperature array
    """
    H, W = temperature.shape
    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

    max_temp = jnp.max(temperature)

    y_exp = y[:, :, None, None]
    x_exp = x[:, :, None, None]
    y_center = y[None, None, :, :]
    x_center = x[None, None, :, :]

    dist = jnp.sqrt((y_exp - y_center) ** 2 + (x_exp - x_center) ** 2)

    falloff = jnp.where(dist > 0, 1.0 / (1.0 + dist * 0.5), 0.0)
    falloff = jnp.where(dist <= radius, falloff, 0.0)

    fire_exp = fire_intensity[None, None, :, :]
    heat_contribution = jnp.max(fire_exp * falloff * 500.0, axis=(2, 3))

    new_temp = jnp.maximum(temperature, heat_contribution + max_temp * 0.5)

    return jnp.where(fire_mask, new_temp, temperature)


def jax_compute_reaction_intensity(
    fuel_load: jax.Array,
    bulk_density: jax.Array,
    moisture_content: jax.Array,
    mineral_content: jax.Array,
    particle_density: float = 32.0,
) -> jax.Array:
    """Compute reaction intensity for fire spread (vectorized).

    Based on Rothermel model.

    Args:
        fuel_load: Fuel load (kg/m²)
        bulk_density: Bulk density (kg/m³)
        moisture_content: Moisture content (0-1)
        mineral_content: Mineral content fraction
        particle_density: Particle density

    Returns:
        Reaction intensity (kW/m²)
    """
    moisture_ratio = moisture_content / (1.0 + moisture_content)
    mineral_factor = 1.0 - mineral_content

    reaction = (
        particle_density
        * bulk_density
        * fuel_load
        * jnp.exp(-138.0 / particle_density)
        * jnp.exp(-0.0173 * particle_density)
        * mineral_factor
        * (1.0 - moisture_ratio)
    )

    return jnp.maximum(0.0, reaction * 1000.0)


def jax_compute_ros(
    reaction_intensity: jax.Array,
    wind_speed: float,
    wind_direction: float,
    slope_angle: jax.Array,
    fuel_load: jax.Array,
    bulk_density: jax.Array,
    moisture_content: jax.Array,
    fuel_depth: float = 1.0,
    surface_area_ratio: float = 1500.0,
    particle_density: float = 32.0,
) -> jax.Array:
    """Compute rate of spread for entire grid (vectorized).

    Args:
        reaction_intensity: Reaction intensity (H, W)
        wind_speed: Wind speed (m/s)
        wind_direction: Wind direction (degrees)
        slope_angle: Slope angle per cell (H, W)
        fuel_load: Fuel load (H, W)
        bulk_density: Bulk density (H, W)
        moisture_content: Moisture content (H, W)
        fuel_depth: Fuel depth (m)
        surface_area_ratio: Surface area ratio
        particle_density: Particle density

    Returns:
        Rate of spread (m/s)
    """
    wind_factor = surface_area_ratio * wind_speed / 1000.0
    slope_factor = jnp.exp(slope_angle * 3.54 / (1.0 + slope_angle))

    effective_intensity = reaction_intensity * (1.0 + wind_factor + slope_factor - 1.0)
    effective_intensity = jnp.maximum(0.0, effective_intensity)

    bulk_density_safe = jnp.maximum(bulk_density, 0.01)

    heat_sink = (
        250.0
        + 1116.0 * moisture_content
        - 0.0614 * moisture_content * moisture_content * particle_density
    )

    ros = effective_intensity / (heat_sink * bulk_density_safe * fuel_depth)

    wind_rad = jnp.radians(wind_direction)
    wind_effect = jnp.cos(wind_rad) * wind_factor
    ros = ros * jnp.maximum(0.1, 1.0 + wind_effect)

    return jnp.clip(ros, 0.0, 100.0)


def jax_compute_fire_intensity(
    reaction_intensity: jax.Array,
    ros: jax.Array,
) -> jax.Array:
    """Compute fire intensity from reaction intensity and ROS.

    Args:
        reaction_intensity: Reaction intensity (H, W)
        ros: Rate of spread (H, W)

    Returns:
        Fire intensity (kW/m)
    """
    fire_intensity = reaction_intensity * ros / 100.0
    return jnp.minimum(fire_intensity, 1000.0)


def jax_compute_flame_length(fire_intensity: jax.Array) -> jax.Array:
    """Compute flame length from fire intensity.

    Args:
        fire_intensity: Fire intensity (H, W)

    Returns:
        Flame length (m)
    """
    return 0.0775 * (jnp.maximum(fire_intensity, 0.0) ** 0.46)


def jax_update_fire_spread(
    state: FireStateJAX,
    fuel_map: jax.Array,
    moisture_map: jax.Array,
    slope_map: jax.Array,
    dt: float,
) -> FireStateJAX:
    """Update fire spread across entire grid.

    Args:
        state: Current fire state
        fuel_map: Fuel load grid (H, W)
        moisture_map: Moisture content grid (H, W)
        slope_map: Slope angle grid (H, W)
        dt: Time step

    Returns:
        Updated fire state
    """
    bulk_density = fuel_map / 2.0
    bulk_density = jnp.maximum(bulk_density, 0.01)

    reaction = jax_compute_reaction_intensity(
        fuel_map, bulk_density, moisture_map, 0.055
    )

    ros = jax_compute_ros(
        reaction,
        state.wind_speed,
        state.wind_direction,
        slope_map,
        fuel_map,
        bulk_density,
        moisture_map,
    )

    fire_intensity = jax_compute_fire_intensity(reaction, ros)

    fire_mask_new = (fire_intensity > 10.0) & (fuel_map > 0.1)

    intensity_update = state.intensity * 0.9 + fire_intensity / 1000.0 * 0.1

    temp_update = 500.0 + fire_intensity * 0.5
    new_temp = jnp.where(fire_mask_new, temp_update, state.temperature)

    return state.replace(
        intensity=jnp.clip(intensity_update, 0.0, 1.0),
        temperature=new_temp,
        fire_mask=state.fire_mask | fire_mask_new,
    )


def jax_update_embers(
    state: FireStateJAX,
    dt: float,
    key: jax.random.PRNGKey,
) -> FireStateJAX:
    """Update ember positions using JAX.

    Args:
        state: Current fire state
        dt: Time step
        key: Random key

    Returns:
        Updated fire state
    """
    active_indices = jnp.where(state.ember_active)[0]

    if len(active_indices) == 0:
        return state

    active_pos = state.ember_pos[active_indices]
    active_vel = state.ember_vel[active_indices]

    wind_rad = jnp.radians(state.wind_direction)
    wind_vec = jnp.array([jnp.cos(wind_rad), jnp.sin(wind_rad)])

    key, subkey = jax.random.split(key)
    turbulence = jax.random.normal(subkey, active_pos.shape) * 2.0

    new_vel = active_vel + (wind_vec * state.wind_speed * 0.3 + turbulence) * dt
    new_pos = active_pos + new_vel * dt

    H, W = state.grid_width, state.grid_height
    bounds_mask = (
        (new_pos[:, 0] >= 0)
        & (new_pos[:, 0] < W)
        & (new_pos[:, 1] >= 0)
        & (new_pos[:, 1] < H)
    )

    new_active = bounds_mask

    all_pos = state.ember_pos.at[active_indices].set(new_pos)
    all_vel = state.ember_vel.at[active_indices].set(new_vel)
    all_active = state.ember_active.at[active_indices].set(new_active)

    return state.replace(
        ember_pos=all_pos,
        ember_vel=all_vel,
        ember_active=all_active,
    )


def jax_spawn_embers(
    state: FireStateJAX,
    key: jax.random.PRNGKey,
    probability: float = 0.01,
    max_new_embers: int = 50,
) -> FireStateJAX:
    """Spawn new embers from fire front.

    Args:
        state: Current fire state
        key: Random key
        probability: Spawn probability per fire cell
        max_new_embers: Maximum new embers to spawn

    Returns:
        Updated fire state
    """
    fire_cells = jnp.where(state.fire_mask & (state.intensity > 0.3))
    num_fire_cells = len(fire_cells[0])

    if num_fire_cells == 0:
        return state

    key, subkey = jax.random.split(key)
    should_spawn = jax.random.uniform(subkey) < probability * num_fire_cells

    if not should_spawn:
        return state

    num_to_spawn = min(max_new_embers, num_fire_cells)
    indices = jax.random.choice(subkey, num_fire_cells, (num_to_spawn,), replace=False)

    spawn_y = fire_cells[0][indices].astype(jnp.float32)
    spawn_x = fire_cells[1][indices].astype(jnp.float32)

    key, subkey = jax.random.split(key)
    offset_y = jax.random.uniform(subkey, (num_to_spawn,), minval=-5.0, maxval=5.0)
    key, subkey = jax.random.split(key)
    offset_x = jax.random.uniform(subkey, (num_to_spawn,), minval=-5.0, maxval=5.0)

    new_y = jnp.clip(spawn_y + offset_y, 0, state.grid_height - 1)
    new_x = jnp.clip(spawn_x + offset_x, 0, state.grid_width - 1)

    new_positions = jnp.stack([new_x, new_y], axis=1)

    key, subkey = jax.random.split(key)
    new_velocities = jax.random.uniform(
        subkey, (num_to_spawn, 2), minval=-2.0, maxval=2.0
    )

    inactive_indices = jnp.where(~state.ember_active)[0][:num_to_spawn]

    if len(inactive_indices) == 0:
        return state

    num_actual = min(len(inactive_indices), num_to_spawn)

    all_pos = state.ember_pos.at[inactive_indices[:num_actual]].set(
        new_positions[:num_actual]
    )
    all_vel = state.ember_vel.at[inactive_indices[:num_actual]].set(
        new_velocities[:num_actual]
    )
    all_active = state.ember_active.at[inactive_indices[:num_actual]].set(True)

    return state.replace(
        ember_pos=all_pos,
        ember_vel=all_vel,
        ember_active=all_active,
    )


def jax_compute_water_suppression(
    fire_intensity: jax.Array,
    water_amount: float,
    fire_intensity_max: float = 1000.0,
) -> jax.Array:
    """Compute water suppression effectiveness (vectorized).

    Args:
        fire_intensity: Fire intensity grid (H, W)
        water_amount: Amount of water applied
        fire_intensity_max: Maximum fire intensity

    Returns:
        Suppression effectiveness grid (H, W)
    """
    intensity_factor = 1.0 - (fire_intensity / fire_intensity_max)
    effectiveness = water_amount * jnp.maximum(intensity_factor, 0.0) / 100.0
    return jnp.clip(effectiveness, 0.0, 1.0)


def jax_apply_suppression(
    state: FireStateJAX,
    suppression_positions: jax.Array,
    suppression_amounts: jax.Array,
    suppression_radius: float = 5.0,
) -> FireStateJAX:
    """Apply suppression to fire at multiple positions.

    Args:
        state: Current fire state
        suppression_positions: Positions (N, 2) as [x, y]
        suppression_amounts: Suppression amounts (N,)
        suppression_radius: Effect radius

    Returns:
        Updated fire state
    """
    H, W = state.grid_height, state.grid_width
    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

    total_suppression = jnp.zeros((H, W), dtype=jnp.float32)

    def apply_single_suppression(suppression, pos):
        sx, sy = pos
        dist = jnp.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        effect = jnp.where(
            dist < suppression_radius,
            suppression * (1.0 - dist / suppression_radius),
            0.0,
        )
        return suppression + effect

    for i in range(len(suppression_positions)):
        total_suppression = jax.lax.fori_loop(
            0,
            1,
            lambda _, v: apply_single_suppression(v, suppression_positions[i]),
            total_suppression,
        )

    total_suppression = jnp.minimum(total_suppression, 1.0)

    new_intensity = state.intensity * (1.0 - total_suppression)
    new_temp = state.temperature * (1.0 - total_suppression * 0.5)

    return state.replace(
        intensity=jnp.maximum(new_intensity, 0.0),
        temperature=jnp.maximum(new_temp, 300.0),
    )


@struct.dataclass
class DiurnalCycleJAX:
    """Diurnal cycle for JAX (vectorized)."""

    time: float = 6.0
    day_temp: float = 30.0
    night_temp: float = 15.0
    day_humidity: float = 0.3
    night_humidity: float = 0.7

    def get_temperature(self, time: float | None = None) -> float:
        """Get temperature at given time."""
        t = time if time is not None else self.time

        if t < 6:
            return self.night_temp
        elif t < 12:
            progress = (t - 6) / 6
            return self.night_temp + (self.day_temp - self.night_temp) * jnp.sin(
                progress * jnp.pi / 2
            )
        elif t < 18:
            return self.day_temp
        else:
            progress = (t - 18) / 6
            return self.day_temp - (self.day_temp - self.night_temp) * jnp.sin(
                progress * jnp.pi / 2
            )

    def get_humidity(self, time: float | None = None) -> float:
        """Get humidity at given time."""
        t = time if time is not None else self.time

        if t < 6:
            return self.night_humidity
        elif t < 12:
            progress = (t - 6) / 6
            return self.night_humidity - (
                self.night_humidity - self.day_humidity
            ) * jnp.sin(progress * jnp.pi / 2)
        elif t < 18:
            return self.day_humidity
        else:
            progress = (t - 18) / 6
            return self.day_humidity + (
                self.night_humidity - self.day_humidity
            ) * jnp.sin(progress * jnp.pi / 2)

    def get_fire_danger_rating(self, time: float | None = None) -> float:
        """Get fire danger rating."""
        temp = self.get_temperature(time)
        humidity = self.get_humidity(time)

        temp_factor = (temp - 10) / 30
        humidity_factor = 1.0 - humidity

        return jnp.clip(temp_factor * 0.6 + humidity_factor * 0.4, 0.0, 1.0)

    def update(self, dt: float) -> DiurnalCycleJAX:
        """Update diurnal cycle."""
        return self.replace(time=(self.time + dt) % 24.0)


class JAXFireModel:
    """Main JAX fire model for GPU-accelerated simulation."""

    def __init__(
        self,
        grid_height: int = 100,
        grid_width: int = 100,
        wind_speed: float = 10.0,
        wind_direction: float = 0.0,
        max_embers: int = 1000,
        enable_diurnal: bool = True,
    ):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.enable_diurnal = enable_diurnal

        self._state = create_fire_state_jax(
            grid_height=grid_height,
            grid_width=grid_width,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            max_embers=max_embers,
        )

        if enable_diurnal:
            self._diurnal = DiurnalCycleJAX()

        self._key = jax.random.PRNGKey(0)

    @property
    def state(self) -> FireStateJAX:
        return self._state

    def add_ignition(
        self,
        grid_position: tuple[int, int],
        intensity: float = 1.0,
        radius: int = 2,
    ) -> None:
        self._state = add_fire_ignition(self._state, grid_position, intensity, radius)

    def step(
        self,
        fuel_map: jax.Array | None = None,
        slope_map: jax.Array | None = None,
        dt: float = 1.0,
    ) -> FireStateJAX:
        """Step the fire model forward.

        Args:
            fuel_map: Optional fuel load grid (H, W)
            slope_map: Optional slope angle grid (H, W)
            dt: Time step

        Returns:
            Updated fire state
        """
        H, W = self.grid_height, self.grid_width

        if fuel_map is None:
            fuel_map = jnp.full((H, W), 0.5, dtype=jnp.float32)
        if slope_map is None:
            slope_map = jnp.zeros((H, W), dtype=jnp.float32)

        moisture_map = self._state.fuel_moisture

        self._state = self._state.replace(time=self._state.time + dt)

        diffusion_coef = 0.05
        temp = jax_diffuse_heat(
            self._state.temperature,
            self._state.fire_mask,
            diffusion_coef,
            dt,
        )
        temp = jax_apply_fire_heat(
            temp,
            self._state.fire_mask,
            self._state.intensity,
            radius=2,
        )

        self._state = self._state.replace(temperature=temp)

        self._state = jax_update_fire_spread(
            self._state,
            fuel_map,
            moisture_map,
            slope_map,
            dt,
        )

        self._key, subkey = jax.random.split(self._key)
        self._state = jax_spawn_embers(self._state, subkey, probability=0.01)

        self._key, subkey = jax.random.split(self._key)
        self._state = jax_update_embers(self._state, dt, subkey)

        if self.enable_diurnal:
            self._diurnal = self._diurnal.update(dt / 3600.0)
            humidity_diff = self._diurnal.get_humidity() - 0.4

            new_moisture = self._state.fuel_moisture - humidity_diff * 0.01 * dt
            new_moisture = jnp.clip(new_moisture, 0.01, 1.0)

            self._state = self._state.replace(fuel_moisture=new_moisture)

        return self._state

    def get_intensity_at(self, world_x: float, world_y: float) -> float:
        """Get fire intensity at world position."""
        cell_size = 10.0
        gx = int(world_x / cell_size)
        gy = int(world_y / cell_size)

        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return float(self._state.intensity[gy, gx])
        return 0.0

    def get_temperature_at(self, world_x: float, world_y: float) -> float:
        """Get temperature at world position."""
        cell_size = 10.0
        gx = int(world_x / cell_size)
        gy = int(world_y / cell_size)

        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return float(self._state.temperature[gy, gx])
        return 300.0

    def to_numpy(self) -> dict:
        """Convert JAX state to NumPy for compatibility."""
        return {
            "intensity": jnp.array(self._state.intensity),
            "temperature": jnp.array(self._state.temperature),
            "fuel_moisture": jnp.array(self._state.fuel_moisture),
            "preheat": jnp.array(self._state.preheat),
            "fire_mask": jnp.array(self._state.fire_mask),
            "time": float(self._state.time),
        }


def create_jax_fire_model(
    grid_height: int = 100,
    grid_width: int = 100,
    wind_speed: float = 10.0,
    wind_direction: float = 0.0,
    enable_diurnal: bool = True,
) -> JAXFireModel:
    """Create a JAX fire model.

    Args:
        grid_height: Grid height
        grid_width: Grid width
        wind_speed: Wind speed (m/s)
        wind_direction: Wind direction (degrees)
        enable_diurnal: Enable diurnal cycle

    Returns:
        JAXFireModel instance
    """
    return JAXFireModel(
        grid_height=grid_height,
        grid_width=grid_width,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        enable_diurnal=enable_diurnal,
    )
