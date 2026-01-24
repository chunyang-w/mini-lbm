"""
Minimal yet practical LBM code in Python - D2Q9 scheme.
Author:             Chunyang Wang
Date:               5th Nov 2025
Github username:    chunyang-w
"""
import time
import argparse
import numpy as np
from lbm2d import LBM2D, warmup_jit, CS2


# Physical parameters for DFG Benchmark 2 (Re=100)
# Reference: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html  # noqa

# Domain geometry (physical units: meters)
Lx, Ly = 2.2, 0.41         # Domain size: 2.2m x 0.41m
Bx, By = 0.2, 0.2          # Obstacle center at (0.2, 0.2)
D = 0.1                    # Cylinder diameter: 0.1m
r = D / 2                  # Cylinder radius: 0.05m

# Flow parameters
Re = 100                   # Reynolds number
u_mean_phys = 1.0          # Mean inlet velocity (m/s)
nu_phys = u_mean_phys * D / Re  # Kinematic viscosity from Re definition

# Lattice parameters
Nx, Ny = 440, 82           # Lattice nodes in x and y directions
dx = Lx / Nx               # Lattice spacing (m)

# Convert physical to lattice units
u_inlet_lbm = 0.1          # Inlet velocity in lattice units (keep small for stability)
dt = u_inlet_lbm * dx / u_mean_phys  # Time step from velocity scaling

# Compute tau from viscosity matching
# nu_lbm = cs2 * (tau - 0.5) = nu_phys * dt / dx^2
nu_lbm = nu_phys * dt / (dx * dx)
tau = nu_lbm / CS2 + 0.5

# Cylinder position in lattice units
cx_lbm = int(Bx / dx)
cy_lbm = int(By / dx)
r_lbm = r / dx

# Simulation time
T_phys = 10.0              # Total physical time (seconds)
n_steps = int(T_phys / dt)

# Visualization settings
VIS_INTERVAL = 100         # Steps between frame captures (adjust for performance)
OUTPUT_FILE = "lbm_simulation.gif"  # Output animation file


def sanity_check():
    """Validate simulation parameters for stability."""
    print("=" * 60)
    print("LBM Simulation Parameters")
    print("=" * 60)
    print(f"Domain: {Lx}m x {Ly}m -> {Nx} x {Ny} lattice nodes")
    print(f"Cylinder: center=({Bx}, {By})m, diameter={D}m")
    print(f"Reynolds number: {Re}")
    print(f"Physical viscosity: {nu_phys:.6f} m^2/s")
    print(f"Lattice spacing: dx = {dx:.6f} m")
    print(f"Time step: dt = {dt:.6e} s")
    print(f"Relaxation time: tau = {tau:.4f}")
    print(f"Lattice viscosity: nu_lbm = {nu_lbm:.6f}")
    print(f"Inlet velocity (lattice): u_inlet = {u_inlet_lbm}")
    print(f"Total steps: {n_steps}")
    print("=" * 60)

    # Stability checks
    errors = []

    if tau <= 0.5:
        errors.append(f"ERROR: tau={tau:.4f} <= 0.5 (unstable)")

    if tau > 2.0:
        errors.append(f"WARNING: tau={tau:.4f} > 2.0 (may be inaccurate)")

    if u_inlet_lbm > 0.3:
        errors.append(f"WARNING: u_inlet={u_inlet_lbm} > 0.3 (compressibility errors)")

    # Check Mach number
    Ma = u_inlet_lbm / np.sqrt(CS2)
    if Ma > 0.3:
        errors.append(f"WARNING: Mach number Ma={Ma:.3f} > 0.3 (compressibility effects)")

    if errors:
        print("\nStability warnings/errors:")
        for e in errors:
            print(f"  {e}")
        print()
    else:
        print("\nAll stability checks passed.\n")

    return len([e for e in errors if e.startswith("ERROR")]) == 0


def create_combined_callback(solver, visualizer=None, show_live=True, print_interval=1000):
    """
    Create a callback that handles both progress printing and visualization.

    This approach ensures visualization doesn't slow down the simulation by:
    1. Only capturing frames at VIS_INTERVAL
    2. Using non-blocking display updates
    3. Separating frame storage from display

    Parameters
    ----------
    solver : LBM2D
        The solver instance
    visualizer : LBMVisualizer, optional
        Visualizer for frame capture and display
    show_live : bool
        Whether to show live visualization
    print_interval : int
        Steps between progress prints

    Returns
    -------
    callback : callable
        Combined callback function
    """
    last_print = [0]
    first_vis = [True]

    def callback(step, solver):
        # Progress printing (sparse)
        if step - last_print[0] >= print_interval or step == n_steps:
            vel_mag = solver.get_velocity_magnitude()
            max_vel = np.max(vel_mag)
            mean_rho = np.mean(solver.rho)
            cd, cl = solver.compute_drag_lift(cx_lbm, cy_lbm, r_lbm)

            print(f"Step {step:6d}/{n_steps}: "
                  f"max|u|={max_vel:.4f}, "
                  f"<rho>={mean_rho:.6f}, "
                  f"Cd={cd:.4f}, Cl={cl:.4f}")
            last_print[0] = step

        # Visualization (at VIS_INTERVAL)
        if visualizer is not None and step % VIS_INTERVAL == 0:
            vel_mag = solver.get_velocity_magnitude()
            phys_time = step * dt

            # Initialize display on first visualization call
            if first_vis[0] and show_live:
                visualizer.show(vel_mag, solver.obstacle)
                first_vis[0] = False

            visualizer.update_frame(vel_mag, step, phys_time, store_frame=True)

            if show_live:
                import matplotlib.pyplot as plt
                plt.pause(0.001)  # Minimal pause for display update

    return callback


def main(visualize: bool = True):
    """
    Run the LBM simulation.

    Parameters
    ----------
    visualize : bool
        If True, show real-time animation window (default: True)
        Animation is always saved to file regardless of this setting.
    """
    if not sanity_check():
        print("Simulation aborted due to parameter errors.")
        return

    print("Warming up JIT compilation...")
    warmup_jit()

    # Set up visualization (import here to avoid matplotlib import if not needed for tests)
    from visualization import LBMVisualizer
    visualizer = LBMVisualizer(
        Nx, Ny,
        figsize=(14, 4),
        cmap='coolwarm',
        vmin=0.0,
        vmax=u_inlet_lbm * 2.0,
        title=f'LBM Flow Around Cylinder (Re={Re})'
    )

    print("Initializing solver...")
    solver = LBM2D(Nx, Ny, tau, u_inlet_lbm)

    # Set cylinder obstacle
    solver.set_cylinder_obstacle(cx_lbm, cy_lbm, r_lbm)
    print(f"Cylinder set at lattice position ({cx_lbm}, {cy_lbm}) with radius {r_lbm:.1f}")

    # Create combined callback
    callback = create_combined_callback(
        solver,
        visualizer=visualizer,
        show_live=visualize,
        print_interval=1000
    )

    # Run simulation
    print(f"\nStarting simulation for {n_steps} steps...")
    print(f"Visualization: {'ENABLED (live display)' if visualize else 'DISABLED (saving only)'}")
    print(f"Frame capture interval: every {VIS_INTERVAL} steps")
    print(f"Output file: {OUTPUT_FILE}\n")

    start_time = time.perf_counter()

    # Run with callback at every VIS_INTERVAL for frame capture
    solver.run(n_steps, callback=callback, callback_interval=VIS_INTERVAL)

    elapsed = time.perf_counter() - start_time
    mlups = (Nx * Ny * n_steps) / elapsed / 1e6

    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    print(f"Performance: {mlups:.2f} MLUPS (Million Lattice Updates Per Second)")

    # Final statistics
    vel_mag = solver.get_velocity_magnitude()
    cd, cl = solver.compute_drag_lift(cx_lbm, cy_lbm, r_lbm)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Max velocity magnitude: {np.max(vel_mag):.6f}")
    print(f"Mean density: {np.mean(solver.rho):.6f}")
    print(f"Drag coefficient Cd: {cd:.4f}")
    print(f"Lift coefficient Cl: {cl:.4f}")
    print("=" * 60)

    # Save animation (always, regardless of visualize flag)
    print(f"\nSaving animation to {OUTPUT_FILE}...")
    visualizer.save_animation(
        OUTPUT_FILE,
        fps=30,
        dpi=150,
        obstacle_mask=solver.obstacle
    )

    # Close visualization window
    visualizer.close()

    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LBM D2Q9 Flow Simulation')
    parser.add_argument(
        '--no-visualize', '-nv',
        action='store_true',
        help='Disable live visualization (animation still saved to file)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output animation filename (default: {OUTPUT_FILE})'
    )
    args = parser.parse_args()

    # Update output file if specified
    if args.output != OUTPUT_FILE:
        OUTPUT_FILE = args.output

    main(visualize=not args.no_visualize)
