"""
test_beam_analysis.py
=====================
Unit tests for the Beam Analysis Calculation Engine.

Test categories
---------------
1.  Three or more point loads (no UVL)
2.  Triangular UVL (one intensity is zero)
3.  Trapezoidal UVL (both intensities non-zero and different)
4.  UVL over the full span
5.  UVL over only part of the span
6.  Load positions at valid boundary values (0 and L)
7.  Invalid inputs → ValueError
    a. Non-positive beam length
    b. Fewer than three point loads
    c. Point load position outside [0, L]
    d. UVL region outside [0, L]
    e. UVL start ≥ UVL end
    f. Negative UVL intensity
8.  Vertical force equilibrium  R_A + R_B == total load
9.  Moment equilibrium  ΣM about A == 0
10. Boundary conditions  M(0) == 0  and  M(L) == 0
11. Shear-force jump at a point load equals the load magnitude
12. Sample beam sanity check

All numerical comparisons use an absolute tolerance of 1 × 10⁻⁹ kN (or kN·m)
unless otherwise noted.
"""

import math
import sys
import os
import pytest

# Allow running tests from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    validate_beam,
    uvl_resultant,
    calculate_reactions,
    total_resultant_load,
    shear_force,
    bending_moment,
    sample_beam,
)

ATOL = 1e-9   # absolute tolerance for floating-point comparisons


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_beam(L, loads, uvl=None):
    """Convenience constructor."""
    return BeamConfig(length=L, point_loads=loads, uvl=uvl)


# ===========================================================================
# 1.  Three or more point loads (no UVL)
# ===========================================================================

class TestThreeOrMorePointLoads:
    """Beam with exactly three and also more than three point loads, no UVL."""

    def test_exactly_three_point_loads_reactions(self):
        """
        Beam L = 8 m.
        Loads: 10 kN at 2 m, 20 kN at 5 m, 30 kN at 7 m.

        ΣM_A = 0:  R_B * 8 = 10*2 + 20*5 + 30*7 = 20 + 100 + 210 = 330
                   R_B = 330 / 8 = 41.25 kN
        ΣFy = 0:   R_A = (10+20+30) - 41.25 = 18.75 kN
        """
        L = 8.0
        loads = [
            PointLoad(10.0, 2.0),
            PointLoad(20.0, 5.0),
            PointLoad(30.0, 7.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 18.75) < ATOL
        assert abs(R_B - 41.25) < ATOL

    def test_five_point_loads_reactions(self):
        """
        Beam L = 10 m.
        Loads: 5 kN@1 m, 10 kN@3 m, 15 kN@5 m, 20 kN@7 m, 25 kN@9 m.

        ΣM_A: R_B * 10 = 5*1 + 10*3 + 15*5 + 20*7 + 25*9
                        = 5 + 30 + 75 + 140 + 225 = 475
              R_B = 47.5 kN
              R_A = (5+10+15+20+25) - 47.5 = 75 - 47.5 = 27.5 kN
        """
        L = 10.0
        loads = [
            PointLoad(5.0, 1.0),
            PointLoad(10.0, 3.0),
            PointLoad(15.0, 5.0),
            PointLoad(20.0, 7.0),
            PointLoad(25.0, 9.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 27.5) < ATOL
        assert abs(R_B - 47.5) < ATOL

    def test_total_resultant_no_uvl(self):
        """Total resultant with no UVL equals sum of point-load magnitudes."""
        loads = [PointLoad(10, 1), PointLoad(20, 3), PointLoad(30, 6)]
        config = make_beam(8.0, loads)
        assert abs(total_resultant_load(config) - 60.0) < ATOL

    def test_shear_between_loads(self):
        """
        Beam L = 6 m, single symmetric load pattern.
        Loads: 12 kN at 2 m, 12 kN at 4 m, 12 kN at 6 m (at roller).
        ΣM_A: R_B*6 = 12*2 + 12*4 + 12*6 = 24+48+72 = 144  →  R_B = 24 kN
              R_A = 36 - 24 = 12 kN

        V just right of x=0:  V(0⁺) = R_A = 12 kN
        V between 2 and 4:    V = 12 - 12 = 0 kN
        V between 4 and 6:    V = 12 - 12 - 12 = -12 kN
        """
        L = 6.0
        loads = [PointLoad(12, 2), PointLoad(12, 4), PointLoad(12, 6)]
        config = make_beam(L, loads)
        eps = 1e-9

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 12.0) < ATOL
        assert abs(R_B - 24.0) < ATOL

        assert abs(shear_force(config, eps) - 12.0) < ATOL
        assert abs(shear_force(config, 3.0)) < ATOL          # between loads
        assert abs(shear_force(config, 5.0) - (-12.0)) < ATOL


# ===========================================================================
# 2.  Triangular UVL (one intensity is zero)
# ===========================================================================

class TestTriangularUVL:
    """UVL with one zero intensity forms a triangular load."""

    def test_triangle_ascending_resultant(self):
        """
        w_a = 0, w_b = 20 kN/m over [2, 6] (4 m span).
        R = (0 + 20)/2 * 4 = 40 kN
        x̄ = [0*(2*2+6) + 20*(2+2*6)] / [3*(0+20)]
           = [0 + 20*14] / 60
           = 280 / 60
           ≈ 4.6667 m  (= 2 + 2/3 * 4)
        """
        uvl = UVL(a=2.0, b=6.0, w_a=0.0, w_b=20.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 40.0) < ATOL
        assert abs(x_bar - (2.0 + 2.0 / 3.0 * 4.0)) < ATOL

    def test_triangle_descending_resultant(self):
        """
        w_a = 20 kN/m, w_b = 0 over [1, 5] (4 m span).
        R = 40 kN
        x̄ = [20*(2*1+5) + 0*(1+2*5)] / [3*(20+0)]
           = [20*7] / 60 = 140/60 ≈ 2.3333 m  (= 1 + 1/3 * 4)
        """
        uvl = UVL(a=1.0, b=5.0, w_a=20.0, w_b=0.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 40.0) < ATOL
        assert abs(x_bar - (1.0 + 1.0 / 3.0 * 4.0)) < ATOL

    def test_triangle_reactions(self):
        """
        Beam L = 10 m.
        Three point loads: 10 kN@2m, 10 kN@5m, 10 kN@8m
        Triangular UVL: 0→40 kN/m over [0, 10] m (ascending full span).
        R_uvl = 200 kN, x̄ = 10*2/3 = 6.667 m

        Total load = 30 + 200 = 230 kN
        ΣM_A: R_B*10 = 10*2 + 10*5 + 10*8 + 200*(20/3)
                     = 20 + 50 + 80 + 1333.333... = 1483.333...
              R_B = 148.333... kN
              R_A = 230 - 148.333... = 81.666... kN
        """
        L = 10.0
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=0.0, b=10.0, w_a=0.0, w_b=40.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_uvl, x_bar = uvl_resultant(uvl)
        assert abs(R_uvl - 200.0) < ATOL
        assert abs(x_bar - 20.0 / 3.0) < ATOL

        R_A, R_B = calculate_reactions(config)
        expected_moment = 10*2 + 10*5 + 10*8 + 200 * (20.0/3.0)
        expected_R_B = expected_moment / L
        expected_R_A = 230.0 - expected_R_B

        assert abs(R_A - expected_R_A) < ATOL
        assert abs(R_B - expected_R_B) < ATOL


# ===========================================================================
# 3.  Trapezoidal UVL (both non-zero, different)
# ===========================================================================

class TestTrapezoidalUVL:
    """UVL with both intensities non-zero and unequal."""

    def test_trapezoid_resultant(self):
        """
        w_a = 10, w_b = 30 kN/m over [1, 5] (4 m span).
        R = (10 + 30) / 2 * 4 = 80 kN
        x̄ = [10*(2*1+5) + 30*(1+2*5)] / [3*(10+30)]
           = [10*7 + 30*11] / 120
           = [70 + 330] / 120
           = 400 / 120 = 10/3 ≈ 3.3333 m
        """
        uvl = UVL(a=1.0, b=5.0, w_a=10.0, w_b=30.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 80.0) < ATOL
        assert abs(x_bar - 10.0 / 3.0) < ATOL

    def test_trapezoid_reactions(self):
        """
        Beam L = 12 m.
        Loads: 15 kN@3m, 25 kN@6m, 10 kN@9m
        Trapezoidal UVL: 8→24 kN/m over [2, 10] (8 m span).

        R_uvl = (8+24)/2 * 8 = 128 kN
        x̄ = [8*(2*2+10) + 24*(2+2*10)] / [3*(8+24)]
           = [8*14 + 24*22] / 96
           = [112 + 528] / 96
           = 640 / 96 = 20/3 ≈ 6.6667 m

        Total = 15 + 25 + 10 + 128 = 178 kN
        ΣM_A: R_B*12 = 15*3 + 25*6 + 10*9 + 128*(20/3)
                     = 45 + 150 + 90 + 853.333... = 1138.333...
              R_B = 94.861... kN
              R_A = 178 - 94.861... = 83.138... kN
        """
        L = 12.0
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)

        x_bar = (8 * (2*2 + 10) + 24 * (2 + 2*10)) / (3 * (8 + 24))
        expected_moment = 15*3 + 25*6 + 10*9 + 128 * x_bar
        expected_R_B = expected_moment / L
        expected_R_A = 178.0 - expected_R_B

        assert abs(R_A - expected_R_A) < ATOL
        assert abs(R_B - expected_R_B) < ATOL

    def test_uniform_special_case(self):
        """
        Uniform load (w_a == w_b) is a valid special case of the trapezoidal UVL.
        w = 10 kN/m over [0, 10] m.  Centroid at 5 m.
        """
        uvl = UVL(a=0.0, b=10.0, w_a=10.0, w_b=10.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 100.0) < ATOL
        assert abs(x_bar - 5.0) < ATOL


# ===========================================================================
# 4.  UVL over the full span
# ===========================================================================

class TestUVLFullSpan:
    """UVL that extends from x = 0 to x = L."""

    def test_uniform_full_span_reactions_symmetric(self):
        """
        Symmetric UDL: w = 20 kN/m over full 10 m span.
        R = 200 kN, x̄ = 5 m.
        R_A = R_B = 100 kN.
        """
        L = 10.0
        loads = [PointLoad(0, 0), PointLoad(0, 5), PointLoad(0, L)]  # dummy zero loads
        uvl = UVL(a=0.0, b=L, w_a=20.0, w_b=20.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 100.0) < ATOL
        assert abs(R_B - 100.0) < ATOL

    def test_triangular_full_span_reactions(self):
        """
        Triangular ascending UDL: 0 at x=0, w0 at x=L.
        R = w0*L/2, x̄ = 2L/3.
        R_B = R * (2L/3) / L = w0*L/3.
        R_A = w0*L/2 - w0*L/3 = w0*L/6.
        Using w0 = 12 kN/m, L = 6 m → R = 36 kN, R_B = 24 kN, R_A = 12 kN.
        """
        L = 6.0
        w0 = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 3), PointLoad(0, 6)]
        uvl = UVL(a=0.0, b=L, w_a=0.0, w_b=w0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - w0 * L / 6.0) < ATOL
        assert abs(R_B - w0 * L / 3.0) < ATOL


# ===========================================================================
# 5.  UVL over part of the span
# ===========================================================================

class TestUVLPartialSpan:
    """UVL confined to a subset [a, b] of [0, L]."""

    def test_partial_uvl_shear_piecewise(self):
        """
        Beam L = 12 m.
        UVL: 10 → 10 kN/m (uniform) over [3, 9] (6 m), so R_uvl = 60 kN @ x̄=6m.
        Three dummy point loads of 0 at 0, 6, 12.
        R_A = R_B = 30 kN.

        x < 3:   V(x) = 30 (no UVL contribution yet)
        x = 6:   UVL force on [3,6] = 10*3 = 30 kN  →  V(6) = 30 - 30 = 0
        x > 9:   UVL force = 60 kN  →  V(10) = 30 - 60 = -30 kN
        """
        L = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 6), PointLoad(0, 12)]
        uvl = UVL(a=3.0, b=9.0, w_a=10.0, w_b=10.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 30.0) < ATOL

        # Before UVL starts
        assert abs(shear_force(config, 1.0) - 30.0) < ATOL
        # Inside UVL midpoint
        assert abs(shear_force(config, 6.0)) < ATOL
        # After UVL ends
        assert abs(shear_force(config, 10.0) - (-30.0)) < ATOL

    def test_partial_uvl_moment_boundary(self):
        """M(0) == 0 and M(L) == 0 for the beam with partial UVL."""
        L = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 6), PointLoad(0, 12)]
        uvl = UVL(a=3.0, b=9.0, w_a=10.0, w_b=10.0)
        config = make_beam(L, loads, uvl)

        assert abs(bending_moment(config, 0.0)) < ATOL
        assert abs(bending_moment(config, L)) < ATOL


# ===========================================================================
# 6.  Load positions at valid boundary values
# ===========================================================================

class TestBoundaryPositions:
    """Point loads placed exactly at x = 0 and x = L are permissible."""

    def test_loads_at_supports(self):
        """
        Loads at the supports have zero effect on opposite reaction via moment.
        Load at x=0 does not enter the moment equation about A.
        Load at x=L does not affect R_A (it is directly taken by the roller).

        L = 5 m, loads: 20 kN@0m, 40 kN@3m, 20 kN@5m
        ΣM_A: R_B*5 = 20*0 + 40*3 + 20*5 = 0 + 120 + 100 = 220
              R_B = 44 kN
              R_A = (20+40+20) - 44 = 36 kN
        """
        L = 5.0
        loads = [
            PointLoad(20.0, 0.0),
            PointLoad(40.0, 3.0),
            PointLoad(20.0, 5.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 36.0) < ATOL
        assert abs(R_B - 44.0) < ATOL


# ===========================================================================
# 7.  Invalid inputs → ValueError
# ===========================================================================

class TestInvalidInputs:
    """All invalid configurations must raise ValueError."""

    def test_non_positive_length_zero(self):
        config = make_beam(0.0, [PointLoad(10, 0), PointLoad(10, 0), PointLoad(10, 0)])
        with pytest.raises(ValueError, match="length"):
            validate_beam(config)

    def test_non_positive_length_negative(self):
        config = make_beam(-5.0, [PointLoad(10, 0), PointLoad(10, 0), PointLoad(10, 0)])
        with pytest.raises(ValueError, match="length"):
            validate_beam(config)

    def test_fewer_than_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10, 2), PointLoad(20, 5)])
        with pytest.raises(ValueError, match="three"):
            validate_beam(config)

    def test_point_load_before_beam(self):
        loads = [PointLoad(10, -1), PointLoad(10, 5), PointLoad(10, 8)]
        config = make_beam(10.0, loads)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_point_load_after_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 11)]
        config = make_beam(10.0, loads)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_extends_before_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=-1.0, b=5.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_extends_after_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=5.0, b=11.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_start_equals_end(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=4.0, b=4.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="strictly less"):
            validate_beam(config)

    def test_uvl_start_greater_than_end(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=7.0, b=3.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="strictly less"):
            validate_beam(config)

    def test_negative_uvl_intensity_w_a(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=2.0, b=8.0, w_a=-5.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(config)

    def test_negative_uvl_intensity_w_b(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=2.0, b=8.0, w_a=10.0, w_b=-5.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(config)


# ===========================================================================
# 8.  Vertical force equilibrium  R_A + R_B == total load
# ===========================================================================

class TestVerticalEquilibrium:
    """R_A + R_B must equal total downward load for every configuration."""

    def _check_equilibrium(self, config):
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        W = total_resultant_load(config)
        assert abs(R_A + R_B - W) < ATOL, (
            f"Force equilibrium violated: R_A={R_A}, R_B={R_B}, W={W}"
        )

    def test_equilibrium_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,8)])
        self._check_equilibrium(config)

    def test_equilibrium_trapezoidal_uvl(self):
        loads = [PointLoad(5, 1), PointLoad(15, 4), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=8.0, w_a=6.0, w_b=18.0)
        config = make_beam(10.0, loads, uvl)
        self._check_equilibrium(config)

    def test_equilibrium_full_span_triangular(self):
        L = 8.0
        loads = [PointLoad(0, 0), PointLoad(0, 4), PointLoad(0, 8)]
        uvl = UVL(a=0.0, b=L, w_a=0.0, w_b=16.0)
        config = make_beam(L, loads, uvl)
        self._check_equilibrium(config)


# ===========================================================================
# 9.  Moment equilibrium  ΣM about A == 0
# ===========================================================================

class TestMomentEquilibrium:
    """Verify ΣM about A = 0: R_B*L == Σ(load * position)."""

    def _check_moment_equilibrium(self, config):
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        L = config.length

        moment_loads = sum(pl.magnitude * pl.position for pl in config.point_loads)
        if config.uvl is not None:
            R_uvl, x_bar = uvl_resultant(config.uvl)
            moment_loads += R_uvl * x_bar

        assert abs(R_B * L - moment_loads) < ATOL, (
            f"Moment equilibrium violated: R_B*L={R_B*L}, ΣM_loads={moment_loads}"
        )

    def test_moment_eq_point_loads_only(self):
        config = make_beam(8.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,7)])
        self._check_moment_equilibrium(config)

    def test_moment_eq_with_trapezoidal_uvl(self):
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(12.0, loads, uvl)
        self._check_moment_equilibrium(config)


# ===========================================================================
# 10. Boundary conditions M(0) == 0 and M(L) == 0
# ===========================================================================

class TestBoundaryConditions:
    """M(0) = 0 and M(L) = 0 must hold for simply supported beams."""

    def _check_moment_boundaries(self, config):
        validate_beam(config)
        L = config.length
        assert abs(bending_moment(config, 0.0)) < ATOL, "M(0) ≠ 0"
        assert abs(bending_moment(config, L)) < ATOL, "M(L) ≠ 0"

    def test_boundary_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,8)])
        self._check_moment_boundaries(config)

    def test_boundary_triangular_uvl(self):
        loads = [PointLoad(5, 1), PointLoad(10, 5), PointLoad(5, 9)]
        uvl = UVL(a=0.0, b=10.0, w_a=0.0, w_b=20.0)
        config = make_beam(10.0, loads, uvl)
        self._check_moment_boundaries(config)

    def test_boundary_trapezoidal_partial_uvl(self):
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(12.0, loads, uvl)
        self._check_moment_boundaries(config)

    def test_boundary_five_point_loads(self):
        loads = [
            PointLoad(5.0, 1.0), PointLoad(10.0, 3.0), PointLoad(15.0, 5.0),
            PointLoad(20.0, 7.0), PointLoad(25.0, 9.0),
        ]
        config = make_beam(10.0, loads)
        self._check_moment_boundaries(config)

    def test_boundary_sample_beam(self):
        config = sample_beam()
        self._check_moment_boundaries(config)


# ===========================================================================
# 11. Shear-force jump at a point load
# ===========================================================================

class TestShearJump:
    """
    At a point load position p, V(p⁺) - V(p⁻) = -P  (downward P > 0).

    Using the left-FBD convention:
      V(p⁺) = V(p⁻) - P
    so the jump (right minus left) equals -P.
    """

    def test_shear_jump_single_load(self):
        """
        Beam L = 6 m.  Single mid-span 30 kN load at x=3 m (plus two dummies).
        R_A = R_B = 15 kN (by symmetry, ignoring zero dummies).
        V(3⁻) = 15 kN,  V(3⁺) = 15 - 30 = -15 kN.
        Jump = -30 kN.
        """
        L = 6.0
        loads = [PointLoad(0, 0), PointLoad(30, 3), PointLoad(0, 6)]
        config = make_beam(L, loads)
        validate_beam(config)

        eps = 1e-8
        V_before = shear_force(config, 3.0 - eps)
        V_after  = shear_force(config, 3.0 + eps)
        jump = V_after - V_before

        assert abs(jump - (-30.0)) < 1e-5   # looser tol due to finite eps

    def test_bending_moment_continuous_at_load(self):
        """M is continuous at a point load; M(p⁻) ≈ M(p⁺) ≈ M(p)."""
        L = 6.0
        loads = [PointLoad(0, 0), PointLoad(30, 3), PointLoad(0, 6)]
        config = make_beam(L, loads)
        eps = 1e-8
        M_left  = bending_moment(config, 3.0 - eps)
        M_right = bending_moment(config, 3.0 + eps)
        M_at    = bending_moment(config, 3.0)

        assert abs(M_left - M_at) < 1e-4
        assert abs(M_right - M_at) < 1e-4


# ===========================================================================
# 12. Sample beam sanity check
# ===========================================================================

class TestSampleBeam:
    """Run all equilibrium and boundary checks on the built-in sample beam."""

    def test_sample_beam_is_valid(self):
        config = sample_beam()
        validate_beam(config)   # should not raise

    def test_sample_beam_equilibrium(self):
        config = sample_beam()
        R_A, R_B = calculate_reactions(config)
        W = total_resultant_load(config)
        assert abs(R_A + R_B - W) < ATOL

    def test_sample_beam_moment_boundaries(self):
        config = sample_beam()
        L = config.length
        assert abs(bending_moment(config, 0.0)) < ATOL
        assert abs(bending_moment(config, L)) < ATOL

    def test_sample_beam_shear_at_supports(self):
        """V(0⁺) ≈ R_A and V(L⁻) ≈ -R_B."""
        config = sample_beam()
        L = config.length
        R_A, R_B = calculate_reactions(config)
        eps = 1e-8

        assert abs(shear_force(config, eps) - R_A) < 1e-5
        assert abs(shear_force(config, L - eps) + R_B) < 1e-5
