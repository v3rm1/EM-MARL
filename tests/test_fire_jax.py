"""Tests for JAX fire physics module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from emmarl.envs.fire_jax import (
    FireStateJAX,
    create_fire_state_jax,
    add_fire_ignition,
    jax_diffuse_heat,
    jax_apply_fire_heat,
    jax_compute_reaction_intensity,
    jax_compute_ros,
    jax_compute_fire_intensity,
    jax_compute_water_suppression,
    JAXFireModel,
    create_jax_fire_model,
    DiurnalCycleJAX,
)


class TestFireStateJAX:
    """Tests for FireStateJAX creation."""

    def test_create_fire_state(self):
        state = create_fire_state_jax(grid_height=50, grid_width=50)
        assert state.grid_height == 50
        assert state.grid_width == 50
        assert state.intensity.shape == (50, 50)
        assert state.temperature.shape == (50, 50)
        assert jnp.all(state.temperature == 300.0)

    def test_add_ignition(self):
        state = create_fire_state_jax(50, 50)
        state = add_fire_ignition(state, (25, 25), intensity=1.0, radius=2)
        assert jnp.any(state.intensity > 0)
        assert jnp.any(state.fire_mask)


class TestThermalDiffusion:
    """Tests for JAX thermal diffusion."""

    def test_diffuse_heat(self):
        H, W = 20, 20
        temperature = jnp.full((H, W), 300.0, dtype=jnp.float32)
        fire_mask = jnp.zeros((H, W), dtype=jnp.bool_)

        fire_mask = fire_mask.at[10, 10].set(True)
        temperature = temperature.at[10, 10].set(800.0)

        new_temp = jax_diffuse_heat(temperature, fire_mask, 0.05, 1.0)

        assert new_temp[10, 10] == 800.0
        assert jnp.any(new_temp > 300.0)

    def test_diffuse_heat_no_fire(self):
        H, W = 10, 10
        temperature = jnp.ones((H, W), dtype=jnp.float32) * 300.0
        fire_mask = jnp.zeros((H, W), dtype=jnp.bool_)

        new_temp = jax_diffuse_heat(temperature, fire_mask, 0.1, 1.0)

        assert jnp.allclose(new_temp, 300.0, atol=1.0)


class TestFireSpread:
    """Tests for JAX fire spread calculations."""

    def test_reaction_intensity(self):
        fuel_load = jnp.array([0.5, 1.0, 0.2], dtype=jnp.float32)
        bulk_density = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
        moisture = jnp.array([0.1, 0.15, 0.05], dtype=jnp.float32)
        mineral = 0.055

        reaction = jax_compute_reaction_intensity(
            fuel_load, bulk_density, moisture, mineral
        )

        assert jnp.all(reaction >= 0)

    def test_ros_computation(self):
        H, W = 10, 10
        reaction = jnp.full((H, W), 50.0, dtype=jnp.float32)
        wind_speed = 10.0
        wind_dir = 0.0
        slope = jnp.zeros((H, W), dtype=jnp.float32)
        fuel = jnp.full((H, W), 0.5, dtype=jnp.float32)
        bulk = jnp.full((H, W), 0.1, dtype=jnp.float32)
        moisture = jnp.full((H, W), 0.1, dtype=jnp.float32)

        ros = jax_compute_ros(
            reaction, wind_speed, wind_dir, slope, fuel, bulk, moisture
        )

        assert jnp.all(ros >= 0)
        assert jnp.all(ros <= 100.0)

    def test_fire_intensity(self):
        reaction = jnp.array([100.0, 500.0, 1000.0], dtype=jnp.float32)
        ros = jnp.array([5.0, 10.0, 15.0], dtype=jnp.float32)

        intensity = jax_compute_fire_intensity(reaction, ros)

        assert jnp.all(intensity >= 0)


class TestSuppression:
    """Tests for JAX suppression physics."""

    def test_water_suppression(self):
        fire_intensity = jnp.array([100.0, 500.0, 900.0], dtype=jnp.float32)
        water_amount = 50.0

        effectiveness = jax_compute_water_suppression(fire_intensity, water_amount)

        assert effectiveness[0] > effectiveness[1] > effectiveness[2]

    def test_max_intensity_suppression(self):
        fire_intensity = jnp.full((10, 10), 1000.0, dtype=jnp.float32)
        water_amount = 100.0

        effectiveness = jax_compute_water_suppression(fire_intensity, water_amount)

        assert jnp.all(effectiveness == 0.0)


class TestJAXFireModel:
    """Tests for JAXFireModel."""

    def test_create_model(self):
        model = create_jax_fire_model(grid_height=50, grid_width=50)
        assert model.grid_height == 50
        assert model.grid_width == 50

    def test_add_ignition(self):
        model = create_jax_fire_model(50, 50)
        model.add_ignition((25, 25), intensity=1.0)

        assert jnp.any(model.state.intensity > 0)

    def test_model_step(self):
        model = create_jax_fire_model(30, 30)
        model.add_ignition((15, 15), intensity=1.0)

        fuel_map = jnp.full((30, 30), 0.5, dtype=jnp.float32)
        slope_map = jnp.zeros((30, 30), dtype=jnp.float32)

        state = model.step(fuel_map, slope_map, dt=1.0)

        assert state.time > 0

    def test_get_intensity_at(self):
        model = create_jax_fire_model(50, 50)
        model.add_ignition((25, 25), intensity=1.0, radius=3)
        model.step(dt=1.0)

        intensity = model.get_intensity_at(250.0, 250.0)
        assert intensity >= 0

    def test_get_temperature_at(self):
        model = create_jax_fire_model(50, 50)
        model.add_ignition((25, 25), intensity=1.0)
        model.step(dt=1.0)

        temp = model.get_temperature_at(250.0, 250.0)
        assert temp >= 300.0


class TestDiurnalCycleJAX:
    """Tests for DiurnalCycleJAX."""

    def test_create_diurnal(self):
        cycle = DiurnalCycleJAX()
        assert cycle.time == 6.0

    def test_daytime_temperature(self):
        cycle = DiurnalCycleJAX(day_temp=30.0, night_temp=15.0)
        temp = cycle.get_temperature(10.0)
        assert 15.0 < temp <= 30.0

    def test_nighttime_temperature(self):
        cycle = DiurnalCycleJAX(day_temp=30.0, night_temp=15.0)
        temp = cycle.get_temperature(2.0)
        assert temp < 20.0

    def test_daytime_humidity(self):
        cycle = DiurnalCycleJAX(day_humidity=0.3, night_humidity=0.7)
        humidity = cycle.get_humidity(14.0)
        assert humidity < 0.5

    def test_fire_danger_rating(self):
        cycle = DiurnalCycleJAX()
        danger = cycle.get_fire_danger_rating(14.0)
        assert 0.0 <= danger <= 1.0

    def test_update(self):
        cycle = DiurnalCycleJAX()
        initial_time = cycle.time
        new_cycle = cycle.update(1.0)
        assert new_cycle.time == (initial_time + 1.0) % 24.0


class TestJAXPerformance:
    """Basic performance tests for JAX operations."""

    def test_diffusion_performance(self):
        import time

        H, W = 100, 100
        temperature = jnp.full((H, W), 300.0, dtype=jnp.float32)
        temperature = temperature.at[50, 50].set(800.0)
        fire_mask = jnp.zeros((H, W), dtype=jnp.bool_)

        start = time.time()
        for _ in range(100):
            temperature = jax_diffuse_heat(temperature, fire_mask, 0.05, 1.0)
        elapsed = time.time() - start

        print(f"100 iterations of 100x100 diffusion: {elapsed:.3f}s")

    @pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="Performance test requires GPU"
    )
    def test_model_step_performance_gpu(self):
        import time

        model = create_jax_fire_model(100, 100)
        model.add_ignition((50, 50), intensity=1.0)

        fuel_map = jnp.full((100, 100), 0.5, dtype=jnp.float32)
        slope_map = jnp.zeros((100, 100), dtype=jnp.float32)

        start = time.time()
        for _ in range(50):
            model.step(fuel_map, slope_map, dt=1.0)
        elapsed = time.time() - start

        print(f"50 iterations of 100x100 fire model: {elapsed:.3f}s")
        assert elapsed < 10.0
