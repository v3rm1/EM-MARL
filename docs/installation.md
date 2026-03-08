# Installation

## Requirements

- Python 3.12
- macOS (currently tested on arm64)

## Quick Start

### 1. Install Pixi

FireSim uses [pixi](https://pixi.sh/) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Install Dependencies

```bash
pixi install
```

### 3. Verify Installation

```bash
# Run tests
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest tests/

# Run with verbose output
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest -v tests/
```

## Running Code

### Basic Usage

```bash
PYTHONPATH=src .pixi/envs/default/bin/python your_script.py
```

### Interactive Development

Activate the environment:

```bash
source .pixi/envs/default/bin/activate
```

Then run Python scripts normally:

```bash
python your_script.py
```

## Development Tools

### Running Tests

```bash
# All tests
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest tests/

# Single test file
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest tests/test_env.py

# Single test function
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest tests/test_env.py::TestFireEnv::test_env_reset

# With pattern matching
PYTHONPATH=src .pixi/envs/default/bin/python -m pytest -k "test_reset" tests/
```

### Code Quality

```bash
# Format code
PYTHONPATH=src .pixi/envs/default/bin/ruff format src/

# Lint code
PYTHONPATH=src .pixi/envs/default/bin/ruff check src/

# Auto-fix issues
PYTHONPATH=src .pixi/envs/default/bin/ruff check --fix src/
```

## Dependencies

Core dependencies (from `pixi.toml`):

- `pettingzoo >=1.25.0, <2` - Multi-agent environment
- `gymnasium >=1.0.0, <2` - RL environment base
- `numpy >=2.4.2, <3` - Numerical computing
- `torch >=2.10.0, <3` - Deep learning
- `matplotlib >=3.10.8, <4` - Visualization
- `seaborn >=0.13.2, <0.14` - Statistical graphics
- `scipy >=1.17.0, <2` - Scientific computing
- `scikit-learn >=1.8.0, <2` - Machine learning
- `tensorboardx >=2.6.4, <3` - TensorBoard logging
- `aim >=3.29.1, <4` - Experiment tracking
