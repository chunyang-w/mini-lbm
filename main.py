"""
Minimal yet practical LBM code in Python - D2Q9 scheme.
Author:             Chunyang Wang
Date:               5th Nov 2025
Github username:    chunyang-w
"""
import numpy as np

# LBM parameters
tau = 0.6                  # Relaxation time (usually choosen from 0.5 to 1.0)
omega = 1 / tau            # Relaxation parameter
cs = 1 / np.sqrt(3)        # Lattice speed of sound
cs2 = cs ** 2              # Square of lattice speed of sound
cs4 = cs2 ** 2             # Fourth power of lattice speed of sound

# Simulation setup
# Simulation Space-Time Geometry:
# Test case setup based on:
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html  # noqa
Lx, Ly = 2.2, 0.41         # Domain size is 2.2m x 0.41m
Bx, By = 0.2, 0.2          # Obstacle location is located at (0.2, 0.2)
r = 0.05
T = 10                     # Total simulation time is 10s
u_mean = 1                 # Mean velocity is 1 m/s

Nx, Ny = 440, 82           # Lattice nodes in x and y directions

mu_lbm = 1/3 * (tau - 0.5)  # Kinematic viscosity in LBM units
dx = Lx / Nx                # Lattice spacing
dt = 


def sanity_check():
    
    pass

def main():
    sanity_check()
    # Initialize distribution functions, macroscopic variables, etc.
    # Main LBM loop: streaming, collision, boundary conditions, etc.
    # Output results
