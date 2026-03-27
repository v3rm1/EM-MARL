# JAX Hybrid Architecture Plan for EM-MARL

## Overview

This document outlines the implementation plan to add JAX-based fire physics and batch parallel environments to EM-MARL for improved computational performance. The hybrid approach keeps the PettingZoo AEC loop in Python while accelerating fire simulations and enabling parallel batch environments with JAX.

## Target Hardware

- **Cluster**: 10x NVIDIA V100 GPUs (32GB each)
- **Expected Speedup**: 50-200x for fire physics, enabling large-scale MARL training

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PettingZoo AEC Loop (Python)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Agent Step  │  │  Rewards    │  │  Metrics    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│              JAX Fire Physics (GPU Accelerated)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Thermal      │  │ Fire Spread  │  │ Ember       │             │
│  │ Diffusion    │  │ (Rothermel)  │  │ Transport   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Diurnal      │  │ Fuel         │  │ Suppression  │             │
│  │ Cycle        │  │ Consumption  │  │ Physics      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Batch Parallel Environments (JAX)                    │
│  ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐                          │
│  │Env 1│ │Env 2│ │Env 3│ ...  │Env N│  ← Vectorized step      │
│  └─────┘ └─────┘ └─────┘      └─────┘                          │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PyTorch RL Training (CPU/GPU)                        │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │ Env Wrapper  │  │ Agent Policy │  ← Bridge: JAX → PyTorch   │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: JAX Fire Physics Core ✅ COMPLETE

### 1.1 Dependencies ✅

Added to `pixi.toml`:

```toml
jax = ">=0.4.20"
jaxlib = ">=0.4.20"
flax = ">=0.8.0"
```

### 1.2 JAX-based Fire State Arrays ✅

**File**: `src/emmarl/envs/fire_jax.py` (new - 800 lines)

Implemented classes:
- `FireStateJAX` - Flax dataclass for GPU-compatible state
- `WeatherJAX` - Weather conditions for JAX
- `DiurnalCycleJAX` - JAX diurnal cycle
- `JAXFireModel` - Main model class

| State Array | Shape | Dtype | Description |
|-------------|-------|-------|-------------|
| `fire_intensity` | `(grid_h, grid_w)` | `float32` | Fire intensity per cell |
| `fire_temperature` | `(grid_h, grid_w)` | `float32` | Temperature (K) |
| `fuel_moisture` | `(grid_h, grid_w)` | `float32` | Fuel moisture 0-1 |
| `preheat_level` | `(grid_h, grid_w)` | `float32` | Pre-heating factor |
| `fire_front_mask` | `(grid_h, grid_w)` | `bool` | Active fire cells |
| `ember_positions` | `(num_embers, 2)` | `float32` | Ember particle positions |
| `ember_velocities` | `(num_embers, 2)` | `float32` | Ember velocities |

```python
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class FireStateJAX:
    """JAX-compatible fire state for GPU acceleration."""
    intensity: jax.Array      # (H, W)
    temperature: jax.Array    # (H, W)
    fuel_moisture: jax.Array # (H, W)
    preheat: jax.Array       # (H, W)
    fire_mask: jax.Array      # (H, W) bool
    ember_pos: jax.Array     # (N, 2)
    ember_vel: jax.Array     # (N, 2)
    ember_active: jax.Array  # (N,) bool
    wind_speed: float
    wind_direction: float
    time: float
    dt: float = 1.0
```

### 1.3 Vectorized Fire Operations ✅

| Operation | Implementation | Status |
|-----------|---------------|--------|
| Thermal diffusion | `jax.lax.conv_general_dilated` | ✅ IMPLEMENTED |
| Rothermel ROS | `jax.vmap` over grid | ✅ IMPLEMENTED |
| Fire spread | Masked array ops | ✅ IMPLEMENTED |
| Ember transport | JAX particle system | ✅ IMPLEMENTED |
| Diurnal cycle | `DiurnalCycleJAX` | ✅ IMPLEMENTED |
| Suppression | Vectorized physics | ✅ IMPLEMENTED |

### 1.4 Key Functions Implemented

- `jax_diffuse_heat()` - GPU thermal diffusion
- `jax_compute_reaction_intensity()` - Vectorized Rothermel
- `jax_compute_ros()` - Grid ROS calculation
- `jax_update_fire_spread()` - Full fire spread step
- `jax_update_embers()` - Ember particle transport
- `jax_spawn_embers()` - Probabilistic ember generation
- `jax_compute_water_suppression()` - Water effectiveness

### 1.5 Usage

```python
from emmarl.envs.fire_jax import create_jax_fire_model
import jax.numpy as jnp

# Create JAX fire model
model = create_jax_fire_model(
    grid_height=100,
    grid_width=100,
    wind_speed=10.0,
    wind_direction=0.0,
)

# Add ignition
model.add_ignition((50, 50), intensity=1.0)

# Step the model
fuel_map = jnp.full((100, 100), 0.5, dtype=jnp.float32)
slope_map = jnp.zeros((100, 100), dtype=jnp.float32)
state = model.step(fuel_map, slope_map, dt=1.0)

# Get fire info
intensity = model.get_intensity_at(250.0, 250.0)
```

Or via FireEnv:

```python
from emmarl.envs import FireEnv
from emmarl.envs.fire_env import FireEnvConfig

config = FireEnvConfig(enable_fire_dynamics=True)
env = FireEnv(config, use_jax_fire=True)  # JAX-powered!
env.reset()
```

### 1.6 Tests

```bash
pixi run test tests/test_fire_jax.py
# 21 passed, 1 skipped (GPU required)
```

### 1.4 Key Functions to Implement (Reference)

#### Thermal Diffusion (Priority: HIGH)

Current implementation (fire_dynamics.py:140-176):
```python
def diffuse_heat(self, diffusion_coef, dt, fire_positions, fire_temps):
    new_temps = self.temperatures.copy()
    for gy in range(self.height):
        for gx in range(self.width):
            # Python loop over neighbors
            ...
```

JAX implementation:
```python
def jax_diffuse_heat(
    temperature: jax.Array,      # (H, W)
    fire_mask: jax.Array,       # (H, W)
    fire_temps: jax.Array,      # (H, W)
    diffusion_coef: float,
    dt: float
) -> jax.Array:
    """GPU-accelerated thermal diffusion using convolution."""
    kernel = jnp.array([
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0]
    ])[:, :, None, None]
    
    # Apply convolution with padding
    padded = jnp.pad(temperature, 1, mode='edge')
    neighbor_avg = jax.lax.conv_general_dilated(
        padded[None, :, :, None],
        kernel,
        window_strides=(1, 1),
        padding='valid'
    )[0, :, :, 0]
    
    # Update only non-fire cells
    new_temp = temperature + diffusion_coef * (neighbor_avg - temperature) * dt
    return jnp.where(fire_mask, fire_temps, new_temp)
```

#### Fire Spread with Rothermel (Priority: HIGH)

```python
def jax_compute_ros_grid(
    fuel_map: jax.Array,        # (H, W) fuel load
    moisture_map: jax.Array,    # (H, W) moisture
    wind_speed: float,
    wind_dir: float,
    slope_angle: jax.Array     # (H, W)
) -> jax.Array:
    """Vectorized ROS computation for entire grid."""
    # Vectorize the Rothermel calculations across all cells
    ...
    return ros_grid

def jax_fire_spread(
    fire_state: FireStateJAX,
    fuel_map: jax.Array,
    dt: float
) -> FireStateJAX:
    """Update fire spread across grid using JAX."""
    ros = jax_compute_ros_grid(
        fuel_map, 
        fire_state.fuel_moisture,
        fire_state.wind_speed,
        fire_state.wind_direction,
        fire_state.slope_angle
    )
    # Compute spread using marching squares or similar
    ...
```

#### Ember Transport (Priority: MEDIUM)

```python
def jax_update_embers(
    ember_pos: jax.Array,      # (N, 2)
    ember_vel: jax.Array,      # (N, 2)
    ember_active: jax.Array,   # (N,)
    wind_speed: float,
    wind_dir: float,
    dt: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Update ember positions using JAX scan."""
    
    def update_single(emb_state, _):
        pos, vel, active = emb_state
        
        # Wind effect
        wind_rad = jnp.radians(wind_dir)
        wind_vec = jnp.array([jnp.cos(wind_rad), jnp.sin(wind_rad)])
        
        # Random turbulence
        key = ...
        turbulence = random.normal(key, (2,)) * 2.0
        
        # Update
        new_vel = vel + (wind_vec * wind_speed * 0.3 + turbulence) * dt
        new_pos = pos + new_vel * dt
        new_active = active & (new_pos[..., 0] >= 0)  # Boundary check
        
        return (new_pos, new_vel, new_active), None
    
    (new_pos, new_vel, new_active), _ = jax.lax.scan(
        update_single,
        (ember_pos, ember_vel, ember_active),
        None,
        length=len(ember_pos)
    )
    
    return new_pos, new_vel, new_active
```

### 1.5 Custom JAX Operations (Priority: LOW - Future Enhancement)

Some fire behaviors may require custom JAX operations not easily expressed with standard primitives:

1. **Complex Containment Lines**
   - Current: Python list of line segments with distance calculations
   - JAX: Custom kernel for polygon-fire intersection

2. **Terrain-Adaptive Diffusion**
   - Current: Per-cell diffusion coefficient lookup
   - JAX: Custom stencil with spatially-varying coefficients

3. **Multi-front Fire Merging**
   - Current: Python logic to merge fire polygons
   - JAX: Connected components algorithm

4. **Agent-Fire Interaction**
   - Current: Sequential Python loop over agents
   - JAX: Batched distance computation with JIT

---

## Phase 2: Batch Parallel Environments

### 2.1 Vectorized Environment Step

**File**: `src/emmarl/envs/parallel_jax.py` (new)

```python
class FireEnvBatch:
    """Batch of N environments running in parallel via JAX.
    
    Enables efficient RL training by running N envs in parallel
    on GPU, amortizing Python overhead.
    """
    
    def __init__(
        self, 
        num_envs: int, 
        grid_size: tuple[int, int] = (100, 100),
        num_agents: int = 11
    ):
        self.num_envs = num_envs
        self.grid_size = grid_size
        
        # Fire state arrays: (num_envs, H, W)
        self.fire_states = self._init_fire_states()
        
        # Agent states: (num_envs, num_agents, state_dim)
        self.agent_states = self._init_agent_states(num_agents)
        
    def batch_step(
        self, 
        actions: jax.Array  # (num_envs, num_agents, action_dim)
    ) -> tuple[
        jax.Array,  # observations (num_envs, num_agents, obs_dim)
        jax.Array,  # rewards (num_envs, num_agents)
        jax.Array,  # dones (num_envs,)
    ]:
        """Single vectorized step for all N environments."""
        
        # 1. Update agent positions (vectorized)
        self.agent_states = self._update_agents_vectorized(actions)
        
        # 2. Update fire (JAX-compiled)
        self.fire_states = self._jax_fire_step(self.fire_states)
        
        # 3. Compute rewards (vectorized)
        rewards = self._compute_rewards_vectorized()
        
        # 4. Check terminations
        dones = self._check_dones_vectorized()
        
        return self._get_observations(), rewards, dones
    
    @jax.jit
    def _jax_fire_step(self, fire_states):
        """JIT-compiled fire update."""
        # All fire physics in a single compiled function
        return jax.fori_loop(0, self.num_envs, fire_step_single, fire_states)
```

### 2.2 Performance Targets

| Metric | Current (NumPy) | Target (JAX + V100) |
|--------|-----------------|---------------------|
| Single env step | ~10ms | ~1ms |
| 64 env batch | N/A | ~10ms |
| 256 env batch | N/A | ~30ms |
| Fire spread (200x200) | ~50ms | ~1ms |
| Fire spread (1000x1000) | ~500ms | ~10ms |
| Thermal diffusion (100x100) | ~20ms | ~0.5ms |

### 2.3 Memory Budget (V100 32GB)

| Component | Per Env | 64 Envs | 256 Envs |
|-----------|---------|---------|----------|
| Fire state (float32) | 1.6 MB | 100 MB | 400 MB |
| Agent states | 0.5 MB | 32 MB | 128 MB |
| Observations | 2 MB | 128 MB | 512 MB |
| **Total** | ~4 MB | ~260 MB | ~1 GB |

---

## Phase 3: PyTorch-JAX Bridge

### 3.1 Integration with FireEnv

Modify `src/emmarl/envs/fire_env.py`:

```python
class FireEnv(AECEnv):
    def __init__(self, config, use_jax_fire: bool = True):
        # Existing initialization
        ...
        
        if use_jax_fire:
            from emmarl.envs.fire_jax import JAXFireModel
            self._jax_fire = JAXFireModel(
                grid_size=(grid_h, grid_w),
                dt=config.dt
            )
            # Sync initial state
            self._sync_to_jax()
            
    def _update_fire_dynamics(self):
        if self._jax_fire:
            # Run JAX fire step (compiled)
            self._jax_fire.step()
            # Sync back to NumPy for agent observations
            self._sync_from_jax()
        else:
            # Fallback to NumPy
            ...
```

### 3.2 Gymnasium/PettingZoo Adapter

**File**: `src/emmarl/envs/wrapper.py` (new)

```python
import torch
import jax
import numpy as np
from gymnasium import Space
from typing import Any

class JAXToTorchWrapper:
    """Bridge JAX environment observations to PyTorch tensors.
    
    Enables use with PyTorch-based RL libraries (stable-baselines3, SBX, etc.)
    """
    
    def __init__(self, env: AECEnv, device: str = 'cuda'):
        self.env = env
        self.device = device
        
    def reset(self, **kwargs) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._to_torch(obs), info
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Convert action to numpy
        action_np = action.cpu().numpy()
        
        # Step environment
        obs, reward, done, trunc, info = self.env.step(action_np)
        
        return (
            self._to_torch(obs),
            torch.tensor(reward, device=self.device),
            torch.tensor(done, device=self.device),
            torch.tensor(trunc, device=self.device),
            info
        )
    
    def _to_torch(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs).float().to(self.device)
    
    def observation_space(self) -> Space:
        return self.env.observation_space(self.env.agent_selection)
    
    def action_space(self) -> Space:
        return self.env.action_space(self.env.agent_selection)
```

### 3.3 Usage with Stable-Baselines3

```python
from stable_baselines3 import PPO
from emmarl.envs.wrapper import JAXToTorchWrapper
from emmarl.envs import FireEnv

# Create JAX-accelerated environment
base_env = FireEnv(config, use_jax_fire=True)
env = JAXToTorchWrapper(base_env, device='cuda')

# Train with SB3
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

---

## Implementation Order & Timeline

### Phase 1: JAX Fire Physics Core ✅ COMPLETE

| Phase | Task | Status |
|-------|------|--------|
| 1.1 | Add JAX to dependencies | ✅ COMPLETE |
| 1.2 | Create `FireStateJAX` dataclass | ✅ COMPLETE |
| 1.3 | Implement `jax_diffuse_heat` | ✅ COMPLETE |
| 1.4 | Vectorize Rothermel ROS | ✅ COMPLETE |
| 1.5 | Add ember transport (JAX) | ✅ COMPLETE |
| 1.6 | Add suppression physics (JAX) | ✅ COMPLETE |
| 1.7 | Add diurnal cycle (JAX) | ✅ COMPLETE |

**Phase 1 Complete: 22 tests passing**

### Phase 2: Batch Parallel Environments

| Phase | Task | Priority | Est. Time |
|-------|------|----------|-----------|
| 2.1 | Create `FireEnvBatch` class | HIGH | 3 days |
| 2.2 | Vectorized reward computation | HIGH | 2 days |
| 2.3 | Batch agent-fire interactions | HIGH | 2 days |

### Phase 3: PyTorch-JAX Bridge

| Phase | Task | Priority | Est. Time |
|-------|------|----------|-----------|
| 3.1 | Integrate JAX fire into `FireEnv` | ✅ COMPLETE | (partial) |
| 3.2 | PyTorch bridge wrapper | HIGH | 1 day |
| 3.3 | Test + benchmark | HIGH | 2 days |
| 3.4 | Documentation updates | MEDIUM | 1 day |

**Remaining: ~11 days**

---

## File Structure After Implementation

```
src/emmarl/envs/
├── __init__.py
├── agent.py              # Agent logic (Python)
├── map.py                # Map/terrain (Python)
├── fire_dynamics.py      # Keep for compatibility, reference only
├── fire_jax.py           # NEW: JAX fire physics
├── parallel_jax.py       # NEW: Batch parallel envs
├── wrapper.py           # NEW: JAX ↔ PyTorch bridge
├── fire_env.py           # MODIFIED: Add JAX option
├── render.py             # Rendering (Python)
├── metrics.py            # Metrics (Python)
├── config_loader.py      # Config loading (Python)
└── config/
    └── default_config.json
```

---

## Testing Strategy

### Unit Tests
- Test JAX functions against NumPy reference implementations
- Use `jax.test_util.check_grads` for gradient checking
- Property-based testing with Hypothesis

### Integration Tests
- End-to-end environment step with JAX fire
- Compare observations with NumPy baseline
- Benchmark on V100 vs CPU

### Performance Tests
- Fire spread timing: 100x100, 200x200, 500x500, 1000x1000
- Batch env throughput: 1, 16, 64, 256 envs
- Memory usage profiling

---

## Known Limitations

1. **Custom Python Control Flow**: Complex branching (e.g., multiple containment line types) may not JIT well
2. **Debugging**: JAX-traced arrays are harder to inspect than NumPy
3. **Dynamic Shapes**: Variable number of embers/agents requires padding or dynamic dispatch
4. **GPU Memory**: Large grids + many envs need careful memory management

---

## Future Enhancements

1. **Distributed Training**: Use JAX pmap for multi-GPU training
2. **JAX-based Agents**: Implement neural networks in Flax for on-GPU policy
3. **Real-time Rendering**: Stream JAX state to GPU renderer
4. **Custom Stencil Kernels**: Write CUDA kernels for specialized fire behaviors
