"""
High-performance 2D Lattice Boltzmann Method (D2Q9) implementation.
Optimized for CPU using Numba JIT compilation.

Author:             Chunyang Wang
Date:               5th Nov 2025
Github username:    chunyang-w
"""
import numpy as np
from numba import njit, prange


# D2Q9 lattice velocities (x, y components)
# Direction ordering: 0=rest, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

# D2Q9 weights
WEIGHTS = np.array([
    4.0/9.0,                           # rest (0)
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,  # cardinal (1-4)
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0  # diagonal (5-8)
], dtype=np.float64)

# Opposite direction indices for bounce-back
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Lattice constants
CS2 = 1.0 / 3.0  # Speed of sound squared
CS4 = CS2 * CS2


@njit(cache=True)
def compute_equilibrium_single(rho, ux, uy, weights, ex, ey, cs2):
    """Compute equilibrium distribution for a single cell."""
    feq = np.empty(9, dtype=np.float64)
    u_sq = ux * ux + uy * uy

    for i in range(9):
        eu = ex[i] * ux + ey[i] * uy
        feq[i] = weights[i] * rho * (1.0 + eu / cs2 +
                                      0.5 * eu * eu / (cs2 * cs2) -
                                      0.5 * u_sq / cs2)
    return feq


@njit(parallel=True, cache=True)
def compute_equilibrium(rho, ux, uy, feq, weights, ex, ey, cs2):
    """Compute equilibrium distribution for all cells (vectorized with Numba)."""
    Nx, Ny = rho.shape

    for i in prange(Nx):
        for j in range(Ny):
            u_sq = ux[i, j] * ux[i, j] + uy[i, j] * uy[i, j]
            for k in range(9):
                eu = ex[k] * ux[i, j] + ey[k] * uy[i, j]
                feq[i, j, k] = weights[k] * rho[i, j] * (
                    1.0 + eu / cs2 +
                    0.5 * eu * eu / (cs2 * cs2) -
                    0.5 * u_sq / cs2
                )


@njit(parallel=True, cache=True)
def collide_bgk(f, feq, omega):
    """BGK collision operator (in-place)."""
    Nx, Ny, Q = f.shape

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Q):
                f[i, j, k] = f[i, j, k] - omega * (f[i, j, k] - feq[i, j, k])


@njit(parallel=True, cache=True)
def stream(f, f_new, ex, ey):
    """Streaming step - propagate distributions to neighboring cells."""
    Nx, Ny, Q = f.shape

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Q):
                # Compute source indices with periodic boundary
                i_src = (i - ex[k]) % Nx
                j_src = (j - ey[k]) % Ny
                f_new[i, j, k] = f[i_src, j_src, k]


@njit(parallel=True, cache=True)
def compute_macroscopic(f, rho, ux, uy, ex, ey):
    """Compute macroscopic quantities (density, velocity) from distributions."""
    Nx, Ny, Q = f.shape

    for i in prange(Nx):
        for j in range(Ny):
            rho_local = 0.0
            ux_local = 0.0
            uy_local = 0.0

            for k in range(Q):
                rho_local += f[i, j, k]
                ux_local += f[i, j, k] * ex[k]
                uy_local += f[i, j, k] * ey[k]

            rho[i, j] = rho_local
            if rho_local > 1e-10:
                ux[i, j] = ux_local / rho_local
                uy[i, j] = uy_local / rho_local
            else:
                ux[i, j] = 0.0
                uy[i, j] = 0.0


@njit(cache=True)
def apply_bounce_back_obstacle(f, obstacle, opposite):
    """Apply bounce-back boundary condition on obstacle nodes."""
    Nx, Ny, Q = f.shape

    for i in range(Nx):
        for j in range(Ny):
            if obstacle[i, j]:
                for k in range(Q):
                    f[i, j, k], f[i, j, opposite[k]] = f[i, j, opposite[k]], f[i, j, k]


@njit(cache=True)
def apply_zou_he_inlet(f, rho, ux, uy, ux_inlet, weights, j_indices):
    """Apply Zou-He velocity boundary condition at inlet (left wall, i=0)."""
    i = 0
    for j in j_indices:
        # Known: ux, uy=0; Unknown: f[1], f[5], f[8], rho
        # From mass conservation and momentum equations
        rho_wall = (f[i, j, 0] + f[i, j, 2] + f[i, j, 4] +
                   2.0 * (f[i, j, 3] + f[i, j, 6] + f[i, j, 7])) / (1.0 - ux_inlet)

        ru = rho_wall * ux_inlet

        f[i, j, 1] = f[i, j, 3] + (2.0/3.0) * ru
        f[i, j, 5] = f[i, j, 7] - 0.5 * (f[i, j, 2] - f[i, j, 4]) + (1.0/6.0) * ru
        f[i, j, 8] = f[i, j, 6] + 0.5 * (f[i, j, 2] - f[i, j, 4]) + (1.0/6.0) * ru

        rho[i, j] = rho_wall
        ux[i, j] = ux_inlet
        uy[i, j] = 0.0


@njit(cache=True)
def apply_zou_he_outlet(f, rho, ux, uy, Nx):
    """Apply Zou-He pressure boundary condition at outlet (right wall, i=Nx-1)."""
    i = Nx - 1
    rho_out = 1.0  # Reference density

    for j in range(f.shape[1]):
        # Known: rho; Unknown: f[3], f[6], f[7], ux
        ux_wall = -1.0 + (f[i, j, 0] + f[i, j, 2] + f[i, j, 4] +
                         2.0 * (f[i, j, 1] + f[i, j, 5] + f[i, j, 8])) / rho_out

        ru = rho_out * ux_wall

        f[i, j, 3] = f[i, j, 1] - (2.0/3.0) * ru
        f[i, j, 6] = f[i, j, 8] - 0.5 * (f[i, j, 2] - f[i, j, 4]) - (1.0/6.0) * ru
        f[i, j, 7] = f[i, j, 5] + 0.5 * (f[i, j, 2] - f[i, j, 4]) - (1.0/6.0) * ru

        rho[i, j] = rho_out
        ux[i, j] = ux_wall
        uy[i, j] = 0.0


@njit(cache=True)
def apply_bounce_back_walls(f, opposite, Ny):
    """Apply bounce-back on top and bottom walls."""
    Nx = f.shape[0]

    # Bottom wall (j=0)
    for i in range(Nx):
        # Directions pointing into wall: 4, 7, 8
        # After streaming, these came from inside and should bounce back
        f[i, 0, 2] = f[i, 0, 4]  # 4 -> 2
        f[i, 0, 5] = f[i, 0, 7]  # 7 -> 5
        f[i, 0, 6] = f[i, 0, 8]  # 8 -> 6

    # Top wall (j=Ny-1)
    for i in range(Nx):
        # Directions pointing into wall: 2, 5, 6
        f[i, Ny-1, 4] = f[i, Ny-1, 2]  # 2 -> 4
        f[i, Ny-1, 7] = f[i, Ny-1, 5]  # 5 -> 7
        f[i, Ny-1, 8] = f[i, Ny-1, 6]  # 6 -> 8


class LBM2D:
    """
    High-performance 2D Lattice Boltzmann solver using D2Q9 scheme.

    Uses Numba JIT compilation for CPU optimization.
    """

    def __init__(self, Nx, Ny, tau, u_inlet=0.1):
        """
        Initialize the LBM solver.

        Parameters
        ----------
        Nx : int
            Number of lattice nodes in x direction
        Ny : int
            Number of lattice nodes in y direction
        tau : float
            Relaxation time (should be > 0.5 for stability)
        u_inlet : float
            Inlet velocity in lattice units
        """
        self.Nx = Nx
        self.Ny = Ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.u_inlet = u_inlet

        # Kinematic viscosity in lattice units
        self.nu = CS2 * (tau - 0.5)

        # Allocate arrays (contiguous memory layout for performance)
        self.f = np.zeros((Nx, Ny, 9), dtype=np.float64)
        self.f_new = np.zeros((Nx, Ny, 9), dtype=np.float64)
        self.feq = np.zeros((Nx, Ny, 9), dtype=np.float64)

        # Macroscopic fields
        self.rho = np.ones((Nx, Ny), dtype=np.float64)
        self.ux = np.zeros((Nx, Ny), dtype=np.float64)
        self.uy = np.zeros((Nx, Ny), dtype=np.float64)

        # Obstacle mask
        self.obstacle = np.zeros((Nx, Ny), dtype=np.bool_)

        # Fluid node indices for inlet (excluding walls)
        self.inlet_j = np.arange(1, Ny - 1, dtype=np.int64)

        # Store lattice constants as instance attributes
        self.ex = EX.astype(np.float64)
        self.ey = EY.astype(np.float64)
        self.weights = WEIGHTS
        self.opposite = OPPOSITE

        # Initialize to equilibrium
        self._initialize_equilibrium()

    def _initialize_equilibrium(self):
        """Initialize distribution functions to equilibrium."""
        # Set initial velocity field (parabolic profile for Poiseuille-like flow)
        for j in range(self.Ny):
            y_norm = j / (self.Ny - 1)
            # Parabolic profile: u(y) = 4 * u_max * y * (1 - y)
            self.ux[:, j] = 4.0 * self.u_inlet * y_norm * (1.0 - y_norm)

        # Compute equilibrium
        compute_equilibrium(self.rho, self.ux, self.uy, self.feq,
                           self.weights, self.ex, self.ey, CS2)

        # Set f = feq
        self.f[:] = self.feq[:]

    def set_cylinder_obstacle(self, cx, cy, r):
        """
        Set a cylindrical obstacle in the domain.

        Parameters
        ----------
        cx, cy : float
            Center of cylinder in lattice units
        cr : float
            Radius of cylinder in lattice units
        """
        for i in range(self.Nx):
            for j in range(self.Ny):
                if (i - cx)**2 + (j - cy)**2 <= r**2:
                    self.obstacle[i, j] = True
                    self.ux[i, j] = 0.0
                    self.uy[i, j] = 0.0

    def step(self):
        """Perform one LBM time step."""
        # 1. Compute equilibrium
        compute_equilibrium(self.rho, self.ux, self.uy, self.feq,
                           self.weights, self.ex, self.ey, CS2)

        # 2. Collision (BGK)
        collide_bgk(self.f, self.feq, self.omega)

        # 3. Apply bounce-back on obstacle before streaming
        if np.any(self.obstacle):
            apply_bounce_back_obstacle(self.f, self.obstacle, self.opposite)

        # 4. Streaming
        stream(self.f, self.f_new, EX, EY)
        self.f, self.f_new = self.f_new, self.f

        # 5. Boundary conditions
        apply_bounce_back_walls(self.f, self.opposite, self.Ny)
        apply_zou_he_inlet(self.f, self.rho, self.ux, self.uy,
                          self.u_inlet, self.weights, self.inlet_j)
        apply_zou_he_outlet(self.f, self.rho, self.ux, self.uy, self.Nx)

        # 6. Compute macroscopic quantities
        compute_macroscopic(self.f, self.rho, self.ux, self.uy, self.ex, self.ey)

        # Zero velocity inside obstacle
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0

    def run(self, n_steps, callback=None, callback_interval=100):
        """
        Run simulation for n_steps.

        Parameters
        ----------
        n_steps : int
            Number of time steps to run
        callback : callable, optional
            Function to call periodically: callback(step, solver)
        callback_interval : int
            Interval between callback calls
        """
        for step in range(n_steps):
            self.step()

            if callback is not None and (step + 1) % callback_interval == 0:
                callback(step + 1, self)

    def get_velocity_magnitude(self):
        """Return velocity magnitude field."""
        return np.sqrt(self.ux**2 + self.uy**2)

    def get_vorticity(self):
        """Compute vorticity field (duy/dx - dux/dy)."""
        # Use central differences
        duy_dx = np.zeros_like(self.ux)
        dux_dy = np.zeros_like(self.ux)

        duy_dx[1:-1, :] = (self.uy[2:, :] - self.uy[:-2, :]) / 2.0
        dux_dy[:, 1:-1] = (self.ux[:, 2:] - self.ux[:, :-2]) / 2.0

        return duy_dx - dux_dy

    def compute_drag_lift(self, cx, cy, r):
        """
        Compute drag and lift coefficients on cylinder using momentum exchange.

        Parameters
        ----------
        cx, cy : float
            Center of cylinder in lattice units
        r : float
            Radius of cylinder in lattice units

        Returns
        -------
        cd, cl : float
            Drag and lift coefficients
        """
        fx, fy = 0.0, 0.0

        # Loop over fluid nodes adjacent to obstacle
        for i in range(self.Nx):
            for j in range(self.Ny):
                if not self.obstacle[i, j]:  # Fluid node
                    for k in range(1, 9):  # Skip rest particle
                        i_neighbor = i + int(self.ex[k])
                        j_neighbor = j + int(self.ey[k])

                        if (0 <= i_neighbor < self.Nx and
                            0 <= j_neighbor < self.Ny and
                            self.obstacle[i_neighbor, j_neighbor]):
                            # Momentum exchange: force on obstacle from fluid
                            # f_k goes into wall, f_opposite comes back
                            opp = self.opposite[k]
                            fx += self.ex[k] * (self.f[i, j, k] + self.f[i, j, opp])
                            fy += self.ey[k] * (self.f[i, j, k] + self.f[i, j, opp])

        # Normalize by reference values
        u_ref = self.u_inlet
        rho_ref = 1.0
        D = 2.0 * r  # Cylinder diameter

        cd = 2.0 * fx / (rho_ref * u_ref**2 * D)
        cl = 2.0 * fy / (rho_ref * u_ref**2 * D)

        return cd, cl


def warmup_jit():
    """Pre-compile Numba functions with a small test run."""
    solver = LBM2D(10, 10, 0.6, 0.05)
    solver.step()
