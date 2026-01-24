"""
Comprehensive test suite for the LBM D2Q9 implementation.

Tests cover:
- Mass conservation
- Momentum conservation
- Equilibrium distribution properties
- Poiseuille flow (analytical solution comparison)
- Collision operator correctness
- Streaming step correctness
- Boundary conditions
- Numerical stability
"""
import numpy as np
import pytest
from lbm2d import (
    LBM2D,
    compute_equilibrium,
    compute_equilibrium_single,
    collide_bgk,
    stream,
    compute_macroscopic,
    EX, EY, WEIGHTS, OPPOSITE, CS2,
    warmup_jit,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def jit_warmup():
    """Warm up JIT compilation once per test module."""
    warmup_jit()


@pytest.fixture
def small_solver(jit_warmup):
    """Create a small LBM solver for quick tests."""
    return LBM2D(Nx=50, Ny=20, tau=0.8, u_inlet=0.05)


@pytest.fixture
def channel_solver(jit_warmup):
    """Create a solver for channel flow tests."""
    return LBM2D(Nx=100, Ny=30, tau=0.7, u_inlet=0.05)


# ============================================================================
# D2Q9 Lattice Property Tests
# ============================================================================

class TestD2Q9Lattice:
    """Test D2Q9 lattice properties and constants."""

    def test_weights_sum_to_one(self):
        """Weights must sum to 1 for mass conservation."""
        assert np.isclose(np.sum(WEIGHTS), 1.0), f"Weights sum to {np.sum(WEIGHTS)}, not 1.0"

    def test_weights_positive(self):
        """All weights must be positive."""
        assert np.all(WEIGHTS > 0), "Some weights are non-positive"

    def test_velocity_isotropy_first_moment(self):
        """First moment of lattice velocities weighted by w should be zero."""
        ex_sum = np.sum(WEIGHTS * EX)
        ey_sum = np.sum(WEIGHTS * EY)
        assert np.isclose(ex_sum, 0.0, atol=1e-14), f"Ex moment = {ex_sum}"
        assert np.isclose(ey_sum, 0.0, atol=1e-14), f"Ey moment = {ey_sum}"

    def test_velocity_isotropy_second_moment(self):
        """Second moment should satisfy isotropy: sum(w*ei*ej) = cs2 * delta_ij."""
        exx = np.sum(WEIGHTS * EX * EX)
        eyy = np.sum(WEIGHTS * EY * EY)
        exy = np.sum(WEIGHTS * EX * EY)

        assert np.isclose(exx, CS2, atol=1e-14), f"Exx = {exx}, expected {CS2}"
        assert np.isclose(eyy, CS2, atol=1e-14), f"Eyy = {eyy}, expected {CS2}"
        assert np.isclose(exy, 0.0, atol=1e-14), f"Exy = {exy}, expected 0"

    def test_opposite_directions(self):
        """Opposite direction mapping should be symmetric."""
        for i in range(9):
            opp = OPPOSITE[i]
            assert OPPOSITE[opp] == i, f"OPPOSITE not symmetric for direction {i}"
            assert EX[opp] == -EX[i], f"Ex not opposite for direction {i}"
            assert EY[opp] == -EY[i], f"Ey not opposite for direction {i}"


# ============================================================================
# Equilibrium Distribution Tests
# ============================================================================

class TestEquilibrium:
    """Test equilibrium distribution function properties."""

    def test_equilibrium_mass_conservation(self, jit_warmup):
        """Sum of equilibrium distributions should equal density."""
        rho = 1.5
        ux, uy = 0.1, 0.05

        feq = compute_equilibrium_single(rho, ux, uy, WEIGHTS,
                                          EX.astype(np.float64),
                                          EY.astype(np.float64), CS2)

        assert np.isclose(np.sum(feq), rho, rtol=1e-10), \
            f"feq sum = {np.sum(feq)}, expected {rho}"

    def test_equilibrium_momentum_conservation(self, jit_warmup):
        """First moment of feq should give momentum."""
        rho = 1.2
        ux, uy = 0.08, -0.03

        feq = compute_equilibrium_single(rho, ux, uy, WEIGHTS,
                                          EX.astype(np.float64),
                                          EY.astype(np.float64), CS2)

        px = np.sum(feq * EX)
        py = np.sum(feq * EY)

        assert np.isclose(px, rho * ux, rtol=1e-10), \
            f"px = {px}, expected {rho * ux}"
        assert np.isclose(py, rho * uy, rtol=1e-10), \
            f"py = {py}, expected {rho * uy}"

    def test_equilibrium_at_rest(self, jit_warmup):
        """Equilibrium at rest should equal weights times density."""
        rho = 1.0
        ux, uy = 0.0, 0.0

        feq = compute_equilibrium_single(rho, ux, uy, WEIGHTS,
                                          EX.astype(np.float64),
                                          EY.astype(np.float64), CS2)

        expected = WEIGHTS * rho
        np.testing.assert_allclose(feq, expected, rtol=1e-10)

    def test_equilibrium_vectorized_consistency(self, jit_warmup):
        """Vectorized equilibrium should match single-cell computation."""
        Nx, Ny = 10, 8
        rho = np.random.uniform(0.9, 1.1, (Nx, Ny))
        ux = np.random.uniform(-0.1, 0.1, (Nx, Ny))
        uy = np.random.uniform(-0.1, 0.1, (Nx, Ny))
        feq = np.zeros((Nx, Ny, 9))

        compute_equilibrium(rho, ux, uy, feq, WEIGHTS,
                           EX.astype(np.float64), EY.astype(np.float64), CS2)

        # Check a few random cells
        for _ in range(5):
            i, j = np.random.randint(0, Nx), np.random.randint(0, Ny)
            feq_single = compute_equilibrium_single(rho[i, j], ux[i, j], uy[i, j],
                                                     WEIGHTS,
                                                     EX.astype(np.float64),
                                                     EY.astype(np.float64), CS2)
            np.testing.assert_allclose(feq[i, j], feq_single, rtol=1e-10)


# ============================================================================
# Collision Operator Tests
# ============================================================================

class TestCollision:
    """Test BGK collision operator properties."""

    def test_collision_mass_conservation(self, jit_warmup):
        """Collision should conserve mass."""
        Nx, Ny = 20, 15
        f = np.random.uniform(0.1, 0.2, (Nx, Ny, 9))
        f_original_mass = np.sum(f)

        # Compute equilibrium from current state
        rho = np.sum(f, axis=2)
        ux = np.sum(f * EX, axis=2) / rho
        uy = np.sum(f * EY, axis=2) / rho
        feq = np.zeros_like(f)
        compute_equilibrium(rho, ux, uy, feq, WEIGHTS,
                           EX.astype(np.float64), EY.astype(np.float64), CS2)

        # Apply collision
        omega = 1.0 / 0.7
        collide_bgk(f, feq, omega)

        assert np.isclose(np.sum(f), f_original_mass, rtol=1e-10), \
            "Collision did not conserve mass"

    def test_collision_momentum_conservation(self, jit_warmup):
        """Collision should conserve momentum."""
        Nx, Ny = 20, 15
        f = np.random.uniform(0.1, 0.2, (Nx, Ny, 9))
        px_original = np.sum(f * EX)
        py_original = np.sum(f * EY)

        # Compute equilibrium
        rho = np.sum(f, axis=2)
        ux = np.sum(f * EX, axis=2) / rho
        uy = np.sum(f * EY, axis=2) / rho
        feq = np.zeros_like(f)
        compute_equilibrium(rho, ux, uy, feq, WEIGHTS,
                           EX.astype(np.float64), EY.astype(np.float64), CS2)

        # Apply collision
        omega = 1.0 / 0.8
        collide_bgk(f, feq, omega)

        px_new = np.sum(f * EX)
        py_new = np.sum(f * EY)

        assert np.isclose(px_new, px_original, rtol=1e-10), \
            f"X momentum not conserved: {px_original} -> {px_new}"
        assert np.isclose(py_new, py_original, rtol=1e-10), \
            f"Y momentum not conserved: {py_original} -> {py_new}"

    def test_collision_relaxes_to_equilibrium(self, jit_warmup):
        """Multiple collisions should relax distribution to equilibrium."""
        Nx, Ny = 10, 10
        f = np.random.uniform(0.1, 0.2, (Nx, Ny, 9))

        # Compute equilibrium
        rho = np.sum(f, axis=2)
        ux = np.sum(f * EX, axis=2) / rho
        uy = np.sum(f * EY, axis=2) / rho
        feq = np.zeros_like(f)
        compute_equilibrium(rho, ux, uy, feq, WEIGHTS,
                           EX.astype(np.float64), EY.astype(np.float64), CS2)

        # Apply many collision steps (omega=1 means full relaxation)
        omega = 1.0
        for _ in range(10):
            collide_bgk(f, feq, omega)

        np.testing.assert_allclose(f, feq, rtol=1e-10)


# ============================================================================
# Streaming Tests
# ============================================================================

class TestStreaming:
    """Test streaming step properties."""

    def test_streaming_mass_conservation(self, jit_warmup):
        """Streaming should conserve total mass (periodic boundaries)."""
        Nx, Ny = 30, 20
        f = np.random.uniform(0.1, 0.2, (Nx, Ny, 9))
        f_new = np.zeros_like(f)
        mass_before = np.sum(f)

        stream(f, f_new, EX, EY)

        mass_after = np.sum(f_new)
        assert np.isclose(mass_before, mass_after, rtol=1e-14), \
            f"Streaming changed mass: {mass_before} -> {mass_after}"

    def test_streaming_direction(self, jit_warmup):
        """Test that streaming moves distributions in correct directions."""
        Nx, Ny = 20, 20
        f = np.zeros((Nx, Ny, 9))
        f_new = np.zeros_like(f)

        # Place a marker in the center
        cx, cy = Nx // 2, Ny // 2
        f[cx, cy, :] = 1.0

        stream(f, f_new, EX, EY)

        # Check that distributions moved to correct neighbors
        for k in range(9):
            new_x = (cx + EX[k]) % Nx
            new_y = (cy + EY[k]) % Ny
            assert f_new[new_x, new_y, k] == 1.0, \
                f"Direction {k}: expected at ({new_x}, {new_y}), not found"


# ============================================================================
# Macroscopic Quantities Tests
# ============================================================================

class TestMacroscopic:
    """Test computation of macroscopic quantities."""

    def test_macroscopic_from_equilibrium(self, jit_warmup):
        """Macroscopic quantities from equilibrium should recover input."""
        Nx, Ny = 15, 10
        rho_input = np.random.uniform(0.9, 1.1, (Nx, Ny))
        ux_input = np.random.uniform(-0.1, 0.1, (Nx, Ny))
        uy_input = np.random.uniform(-0.1, 0.1, (Nx, Ny))

        # Compute equilibrium
        feq = np.zeros((Nx, Ny, 9))
        compute_equilibrium(rho_input, ux_input, uy_input, feq, WEIGHTS,
                           EX.astype(np.float64), EY.astype(np.float64), CS2)

        # Recover macroscopic
        rho_out = np.zeros((Nx, Ny))
        ux_out = np.zeros((Nx, Ny))
        uy_out = np.zeros((Nx, Ny))
        compute_macroscopic(feq, rho_out, ux_out, uy_out,
                           EX.astype(np.float64), EY.astype(np.float64))

        np.testing.assert_allclose(rho_out, rho_input, rtol=1e-10)
        np.testing.assert_allclose(ux_out, ux_input, rtol=1e-10)
        np.testing.assert_allclose(uy_out, uy_input, rtol=1e-10)


# ============================================================================
# Physical Validation Tests
# ============================================================================

class TestPoiseuilleFlow:
    """Test against analytical Poiseuille flow solution."""

    def test_poiseuille_velocity_profile(self, jit_warmup):
        """
        Poiseuille flow between two parallel plates should develop
        a parabolic velocity profile.

        Analytical solution: u(y) = (dp/dx) * y * (H - y) / (2 * mu)
        For pressure-driven flow with specified inlet/outlet.
        """
        # Set up a long channel for fully developed flow
        Nx, Ny = 300, 31  # Long channel, odd Ny for center node
        tau = 0.9  # Higher tau for more accurate flow
        u_inlet = 0.04

        solver = LBM2D(Nx, Ny, tau, u_inlet)

        # Run until steady state (no obstacle, just channel flow)
        n_steps = 10000
        solver.run(n_steps)

        # Check velocity profile at outlet (should be parabolic)
        # For Poiseuille flow: u(y) = u_max * 4 * y/H * (1 - y/H)
        # where y is distance from bottom wall

        # Sample velocity at x = 2/3 of domain length (well developed)
        x_sample = int(0.66 * Nx)
        u_profile = solver.ux[x_sample, 1:-1]  # Exclude wall nodes

        # Fit to parabola: u(y) = a * y * (1 - y)
        y_norm = np.linspace(0, 1, len(u_profile))
        parabola = 4 * y_norm * (1 - y_norm)

        # Normalize both profiles
        u_norm = u_profile / np.max(u_profile) if np.max(u_profile) > 0 else u_profile
        parabola_norm = parabola / np.max(parabola)

        # Check correlation (R^2) - LBM with Zou-He BCs has some deviation
        ss_res = np.sum((u_norm - parabola_norm)**2)
        ss_tot = np.sum((u_norm - np.mean(u_norm))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        assert r_squared > 0.85, \
            f"Poiseuille profile R^2 = {r_squared:.4f}, expected > 0.85"


class TestMassConservation:
    """Test mass conservation in complete simulations."""

    def test_mass_conservation_no_obstacle(self, channel_solver):
        """Total mass should be conserved in closed system."""
        solver = channel_solver
        initial_mass = np.sum(solver.rho)

        solver.run(500)

        final_mass = np.sum(solver.rho)

        # Allow small tolerance for boundary effects
        rel_error = abs(final_mass - initial_mass) / initial_mass
        assert rel_error < 0.01, \
            f"Mass changed by {rel_error*100:.2f}%"

    def test_mass_conservation_with_obstacle(self, small_solver):
        """Mass conservation with obstacle (open boundaries allow some mass flux)."""
        solver = small_solver

        # Add obstacle
        solver.set_cylinder_obstacle(solver.Nx // 4, solver.Ny // 2, 3)

        initial_mass = np.sum(solver.rho)
        solver.run(200)
        final_mass = np.sum(solver.rho)

        # Open inlet/outlet BCs exchange mass, so tolerance is higher
        rel_error = abs(final_mass - initial_mass) / initial_mass
        assert rel_error < 0.10, \
            f"Mass changed by {rel_error*100:.2f}% with obstacle (allowed up to 10%)"


class TestNumericalStability:
    """Test numerical stability properties."""

    def test_density_positivity(self, small_solver):
        """Density should remain positive throughout simulation."""
        solver = small_solver
        solver.set_cylinder_obstacle(solver.Nx // 4, solver.Ny // 2, 2)

        for _ in range(100):
            solver.step()
            assert np.all(solver.rho > 0), "Negative density detected"

    def test_density_bounded(self, small_solver):
        """Density should not deviate too far from unity."""
        solver = small_solver
        solver.run(500)

        rho_min, rho_max = np.min(solver.rho), np.max(solver.rho)

        assert rho_min > 0.5, f"Density too low: {rho_min}"
        assert rho_max < 2.0, f"Density too high: {rho_max}"

    def test_velocity_bounded(self, small_solver):
        """Velocity should remain below speed of sound."""
        solver = small_solver
        solver.run(500)

        vel_mag = solver.get_velocity_magnitude()
        max_vel = np.max(vel_mag)

        # Maximum velocity should be well below speed of sound
        cs = np.sqrt(CS2)
        assert max_vel < cs, f"Velocity {max_vel} exceeds speed of sound {cs}"

    @pytest.mark.parametrize("tau", [0.51, 0.6, 0.8, 1.0, 1.5])
    def test_stability_various_tau(self, tau, jit_warmup):
        """Simulation should remain stable for various tau values."""
        solver = LBM2D(30, 15, tau, u_inlet=0.05)

        # Should not raise or produce NaN
        solver.run(100)

        assert not np.any(np.isnan(solver.rho)), f"NaN in density for tau={tau}"
        assert not np.any(np.isnan(solver.ux)), f"NaN in ux for tau={tau}"
        assert not np.any(np.isnan(solver.uy)), f"NaN in uy for tau={tau}"


# ============================================================================
# Solver Integration Tests
# ============================================================================

class TestSolverIntegration:
    """Integration tests for the LBM2D solver class."""

    def test_solver_initialization(self, jit_warmup):
        """Solver should initialize correctly."""
        solver = LBM2D(40, 20, tau=0.7, u_inlet=0.08)

        assert solver.Nx == 40
        assert solver.Ny == 20
        assert solver.tau == 0.7
        assert np.isclose(solver.omega, 1.0 / 0.7)
        assert solver.f.shape == (40, 20, 9)
        assert solver.rho.shape == (40, 20)

    def test_cylinder_obstacle_creation(self, small_solver):
        """Cylinder obstacle should be correctly placed."""
        solver = small_solver
        cx, cy, r = 15, 10, 3

        solver.set_cylinder_obstacle(cx, cy, r)

        # Center should be obstacle
        assert solver.obstacle[cx, cy], "Center not marked as obstacle"

        # Points inside should be obstacle
        assert solver.obstacle[cx + 1, cy], "Point inside not obstacle"

        # Points outside should not be obstacle
        assert not solver.obstacle[cx + r + 2, cy], "Point outside marked as obstacle"

    def test_vorticity_computation(self, channel_solver):
        """Vorticity should be computable without errors."""
        solver = channel_solver
        solver.run(100)

        vorticity = solver.get_vorticity()

        assert vorticity.shape == (solver.Nx, solver.Ny)
        assert not np.any(np.isnan(vorticity))

    def test_drag_lift_computation(self, jit_warmup):
        """Drag and lift should be computable on obstacle."""
        solver = LBM2D(60, 30, tau=0.8, u_inlet=0.05)
        cx, cy, r = 15, 15, 3
        solver.set_cylinder_obstacle(cx, cy, r)

        solver.run(200)

        cd, cl = solver.compute_drag_lift(cx, cy, r)

        # Drag should be positive (resistance to flow)
        assert cd > 0, f"Drag coefficient {cd} should be positive"

        # Lift should be small for symmetric setup
        assert abs(cl) < abs(cd), "Lift should be smaller than drag for symmetric flow"

    def test_callback_execution(self, small_solver):
        """Callback should be called at correct intervals."""
        solver = small_solver
        call_count = [0]
        steps_called = []

        def callback(step, s):
            call_count[0] += 1
            steps_called.append(step)

        solver.run(100, callback=callback, callback_interval=25)

        assert call_count[0] == 4, f"Callback called {call_count[0]} times, expected 4"
        assert steps_called == [25, 50, 75, 100]


# ============================================================================
# Performance Sanity Tests
# ============================================================================

class TestPerformance:
    """Basic performance sanity tests."""

    def test_jit_compilation_works(self, jit_warmup):
        """JIT compilation should produce working code."""
        solver = LBM2D(20, 10, tau=0.7, u_inlet=0.05)

        # First call compiles, should work
        solver.step()

        # Subsequent calls should also work
        solver.step()
        solver.step()

        assert not np.any(np.isnan(solver.rho))

    def test_reasonable_performance(self, jit_warmup):
        """Solver should achieve reasonable MLUPS."""
        import time

        Nx, Ny = 100, 50
        solver = LBM2D(Nx, Ny, tau=0.7, u_inlet=0.05)

        # Warmup
        solver.run(10)

        # Timed run
        n_steps = 500
        start = time.perf_counter()
        solver.run(n_steps)
        elapsed = time.perf_counter() - start

        mlups = (Nx * Ny * n_steps) / elapsed / 1e6

        # Should achieve at least 1 MLUPS (very conservative)
        assert mlups > 1.0, f"Performance too low: {mlups:.2f} MLUPS"
