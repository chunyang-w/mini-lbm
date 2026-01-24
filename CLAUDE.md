# CLAUDE.md

This file provides guidance for AI assistants working with this codebase.

## Project Overview

**mini-lbm** is a minimal, high-performance Lattice Boltzmann Method (LBM) implementation in Python for educational purposes. It implements the D2Q9 scheme (2 dimensions, 9 velocity directions) for computational fluid dynamics (CFD) simulations.

- **Author:** Chunyang Wang (GitHub: chunyang-w)
- **License:** MIT
- **Python Version:** >=3.10

### What is LBM?

The Lattice Boltzmann Method is a computational fluid dynamics technique that simulates fluid flow by tracking particle distribution functions on a discrete lattice. The D2Q9 scheme uses a 2D lattice with 9 velocity directions.

## Repository Structure

```
mini-lbm/
├── main.py              # Main entry point with simulation parameters and loop
├── lbm2d.py             # High-performance D2Q9 LBM implementation (Numba JIT)
├── visualization.py     # Non-blocking visualization and animation export
├── tests/
│   ├── __init__.py
│   └── test_lbm.py      # Comprehensive test suite (33 tests)
├── pyproject.toml       # Project configuration (PEP 517)
├── uv.lock              # Dependency lock file
├── .python-version      # Python version specification (3.10)
├── README.md            # Project readme
└── LICENSE              # MIT License
```

## Development Setup

This project uses **uv** as the package manager.

### Install Dependencies

```bash
# Core dependencies only
uv sync

# With dev dependencies (pytest, benchmarks)
uv sync --all-extras
```

### Run the Simulation

```bash
# With live visualization (default)
uv run python main.py

# Without live visualization (headless, still saves animation)
uv run python main.py --no-visualize

# Custom output file
uv run python main.py -o my_simulation.gif
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## Architecture

### Performance Optimization Strategy

The code is optimized for CPU performance using:

1. **Numba JIT Compilation** - All hot paths are decorated with `@njit` for near-C performance
2. **Parallel Execution** - `prange` for automatic parallelization across CPU cores
3. **Contiguous Memory Layout** - NumPy arrays with efficient memory access patterns
4. **Pre-computed Constants** - Lattice weights, velocities, and opposite directions as global arrays

### Key Components in lbm2d.py

**D2Q9 Lattice Constants** (lines 14-28):
- `EX`, `EY`: Lattice velocity components
- `WEIGHTS`: D2Q9 weights (4/9, 1/9, 1/36)
- `OPPOSITE`: Opposite direction mapping for bounce-back
- `CS2`: Lattice speed of sound squared (1/3)

**JIT-Compiled Functions**:
- `compute_equilibrium()`: Vectorized equilibrium distribution
- `collide_bgk()`: BGK collision operator
- `stream()`: Streaming step with periodic boundaries
- `compute_macroscopic()`: Density and velocity computation
- `apply_bounce_back_*()`: Boundary condition functions
- `apply_zou_he_*()`: Inlet/outlet pressure boundaries

**LBM2D Class** (lines 183-390):
- Encapsulates solver state and provides high-level API
- Methods: `step()`, `run()`, `set_cylinder_obstacle()`, `compute_drag_lift()`

### main.py Structure

1. **Physical Parameters** (lines 12-24): DFG Benchmark 2 configuration
2. **Unit Conversion** (lines 26-42): Physical to lattice units
3. **`sanity_check()`** (lines 49-91): Parameter validation and stability checks
4. **`main()`** (lines 109-154): Simulation loop with performance metrics

## Dependencies

**Core:**
- **numpy** (>=2.2.6): Array operations and numerical computing
- **numba** (>=0.60.0): JIT compilation for CPU performance
- **matplotlib** (>=3.8.0): Visualization and animation export

**Development:**
- **pytest** (>=8.0.0): Test framework
- **pytest-benchmark** (>=4.0.0): Performance benchmarking

## Visualization

The simulation includes real-time visualization and automatic animation export.

### Features

- **Live display**: Real-time velocity magnitude field (default: enabled)
- **Animation export**: Automatically saved to GIF/MP4 (always enabled)
- **Performance-optimized**: Frame capture decoupled from simulation loop

### Usage

```bash
# Live visualization + save animation
uv run python main.py

# Headless mode (no display window, still saves animation)
uv run python main.py --no-visualize

# Custom output filename
uv run python main.py -o flow_animation.mp4
```

### Configuration (main.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `VIS_INTERVAL` | 100 | Steps between frame captures |
| `OUTPUT_FILE` | "lbm_simulation.gif" | Output animation filename |

### Performance Impact

Visualization is designed to have minimal impact on simulation performance:

1. **Sparse frame capture**: Only every `VIS_INTERVAL` steps (default: 100)
2. **Non-blocking display**: Uses `plt.pause(0.001)` for responsive UI
3. **Buffered saving**: Frames stored in memory, written at end
4. **Separate rendering**: Animation saving uses dedicated figure instance

## Testing

The test suite (`tests/test_lbm.py`) contains 33 tests covering:

### D2Q9 Lattice Properties
- Weight normalization (sum to 1)
- Velocity isotropy (first and second moments)
- Opposite direction symmetry

### Equilibrium Distribution
- Mass conservation (sum equals density)
- Momentum conservation (first moment equals momentum)
- Vectorized/single-cell consistency

### Collision Operator
- Mass conservation
- Momentum conservation
- Relaxation to equilibrium

### Streaming
- Mass conservation
- Direction correctness

### Physical Validation
- **Poiseuille flow**: Parabolic velocity profile (R² > 0.85)
- Mass conservation in simulations
- Numerical stability across tau values (0.51 to 1.5)

### Integration Tests
- Solver initialization
- Obstacle creation
- Drag/lift computation
- Performance (MLUPS metric)

## Running Benchmarks

```bash
# Run with benchmark output
uv run pytest tests/ -v --benchmark-only

# Quick performance test
uv run python -c "from lbm2d import LBM2D; s = LBM2D(100, 50, 0.7, 0.05); s.run(1000)"
```

## Key Conventions

### Variable Naming

Physics-aware naming:
- `Lx`, `Ly`: Domain dimensions (physical units)
- `Nx`, `Ny`: Lattice node counts
- `tau`: Relaxation time
- `omega`: Relaxation parameter (1/tau)
- `cs`, `cs2`: Lattice speed of sound
- `feq`: Equilibrium distribution
- `rho`, `ux`, `uy`: Macroscopic fields

### Code Organization

1. Global constants at module level
2. JIT-compiled functions before class
3. Class encapsulates state and high-level methods
4. Main script handles parameter setup and execution

### Performance Guidelines

- Use Numba `@njit(parallel=True, cache=True)` for hot loops
- Avoid Python objects in JIT functions (use primitive types)
- Pre-allocate arrays; avoid allocation in loops
- Use `prange` for embarrassingly parallel loops

## Test Case: DFG Benchmark 2 (Re=100)

The default simulation implements the DFG Benchmark for 2D flow around a cylinder.

Reference: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html

| Parameter | Value | Description |
|-----------|-------|-------------|
| Domain | 2.2m × 0.41m | Channel dimensions |
| Cylinder | (0.2, 0.2), D=0.1m | Obstacle position and diameter |
| Re | 100 | Reynolds number |
| u_mean | 1 m/s | Mean flow velocity |
| T | 10s | Total simulation time |

## Performance Metrics

The solver reports MLUPS (Million Lattice Updates Per Second) as the performance metric. Typical performance:

- Single-threaded: 10-30 MLUPS
- Multi-threaded (4 cores): 30-100 MLUPS

Performance depends on CPU, memory bandwidth, and problem size.

## Common Tasks

### Adding New Boundary Conditions

1. Create JIT-compiled function in `lbm2d.py`
2. Call from `LBM2D.step()` method
3. Add tests in `tests/test_lbm.py`

### Modifying the Test Case

1. Update parameters in `main.py` (lines 12-46)
2. Run `sanity_check()` to validate stability
3. Adjust `n_steps` based on physical time required

### Running Validation

```bash
# Full test suite
uv run pytest tests/ -v

# Specific test class
uv run pytest tests/test_lbm.py::TestPoiseuilleFlow -v

# With coverage
uv run pytest tests/ --cov=lbm2d
```

## Notes for AI Assistants

- All performance-critical code uses Numba JIT - maintain this pattern
- Tests validate physical correctness, not just code execution
- The DFG Benchmark provides reference data for validation
- Keep educational clarity in comments while maintaining performance
- When modifying JIT functions, ensure type consistency (use explicit dtypes)
