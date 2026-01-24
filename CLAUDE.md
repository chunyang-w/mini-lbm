# CLAUDE.md

This file provides guidance for AI assistants working with this codebase.

## Project Overview

**mini-lbm** is a minimal Lattice Boltzmann Method (LBM) implementation in Python for educational purposes. It implements the D2Q9 scheme (2 dimensions, 9 velocity directions) for computational fluid dynamics (CFD) simulations.

- **Author:** Chunyang Wang (GitHub: chunyang-w)
- **License:** MIT
- **Python Version:** >=3.10

### What is LBM?

The Lattice Boltzmann Method is a computational fluid dynamics technique that simulates fluid flow by tracking particle distribution functions on a discrete lattice. The D2Q9 scheme uses a 2D lattice with 9 velocity directions.

## Repository Structure

```
mini-lbm/
├── main.py              # Main entry point with LBM parameters and simulation setup
├── lbm2d.py             # 2D LBM implementation (to be developed)
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
uv sync
```

### Run the Simulation

```bash
uv run python main.py
```

## Key Files

### main.py

The main entry point containing:

- **LBM Parameters** (lines 9-14):
  - `tau`: Relaxation time (0.6, typically 0.5-1.0)
  - `omega`: Relaxation parameter (1/tau)
  - `cs`: Lattice speed of sound (1/√3)
  - `cs2`, `cs4`: Powers of lattice speed of sound

- **Simulation Geometry** (lines 17-30):
  - Domain size: 2.2m × 0.41m
  - Lattice resolution: 440 × 82 nodes
  - Obstacle: Cylinder at (0.2, 0.2) with radius 0.05m
  - Test case: DFG Benchmark 2 (Re=100)

- **Functions**:
  - `sanity_check()`: Parameter validation
  - `main()`: Simulation entry point

### lbm2d.py

Placeholder for the 2D LBM algorithm implementation. Expected to contain:
- Distribution function initialization
- Collision operator (BGK)
- Streaming step
- Boundary conditions
- Macroscopic variable computation (density, velocity)

## Dependencies

- **numpy** (>=2.2.6): Numerical computing and array operations
- **piglet** (>=1.0.0): Templating library for output generation

## Coding Conventions

### Documentation Style

- Module-level docstrings with author, date, and GitHub username
- Inline comments explaining physical quantities and formulas
- URLs to reference benchmarks included in comments

### Variable Naming

Physics-aware naming conventions:
- `Lx`, `Ly`: Domain dimensions (physical units)
- `Nx`, `Ny`: Lattice node counts
- `Bx`, `By`: Obstacle position
- `tau`: Relaxation time
- `cs`: Lattice speed of sound
- `mu_lbm`: Kinematic viscosity in LBM units

### Code Organization

1. Imports at the top
2. LBM parameters and constants
3. Simulation geometry configuration
4. Helper functions
5. Main function

## Test Case

The simulation is configured for the **DFG Benchmark 2 (Re=100)** - a standard CFD validation case involving 2D channel flow around a circular cylinder.

Reference: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html

### Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Domain | 2.2m × 0.41m | Channel dimensions |
| Cylinder | (0.2, 0.2), r=0.05m | Obstacle position and radius |
| Re | 100 | Reynolds number |
| u_mean | 1 m/s | Mean flow velocity |
| T | 10s | Total simulation time |

## Current Development Status

This is an early-stage educational project. The parameter setup and project structure are complete, but the core LBM algorithm implementation is pending.

### Implemented

- Project structure and configuration
- LBM parameter definitions
- Dependency management with uv

### To Be Implemented

- Distribution function management
- Collision operator (BGK model)
- Streaming step
- Boundary conditions (inlet, outlet, walls, obstacle)
- Macroscopic variable calculations
- Output/visualization
- Complete simulation loop

## Common Tasks

### Adding New LBM Features

1. Implement in `lbm2d.py`
2. Import and use in `main.py`
3. Add appropriate parameter validation in `sanity_check()`

### Running Validation

```bash
uv run python main.py
```

### Updating Dependencies

```bash
uv add <package>
uv sync
```

## Notes for AI Assistants

- This is a scientific computing project - maintain precision in numerical operations
- Use numpy vectorization for performance in LBM operations
- Preserve the educational nature with clear comments
- Reference standard LBM literature for algorithm implementations
- The DFG Benchmark 2 provides validation data - implementations should aim to match published results
