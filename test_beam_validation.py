"""
test_beam_validation.py
=======================
Phase 4: Independent Engineering Verification and Regression Test Suite.

Tests:
1. Static equilibrium audits (ΣFy = 0, ΣMA = 0, ΣMB = 0) with independent formulation.
2. Centroid and resultant checks for zero UVL, triangular UVL, trapezoidal UVL.
3. Analytical Reference Cases:
   - Case A: Central point load P at L/2 (R_A=R_B=P/2, M_max=PL/4).
   - Case B: Full-span UDL w0 (R_A=R_B=w0*L/2, M_max=w0*L^2/8).
   - Case C: Triangular UVL 0->w0 (W=w0*L/2, x_bar=2L/3, R_A=w0*L/6, R_B=w0*L/3, M_max=w0*L^2/(9*sqrt(3))).
   - Case D: Asymmetric combination of 3 point loads + partial-span trapezoidal UVL.
4. SFD & BMD physical consistency:
   - Shear jump ΔV = -P at point loads.
   - Support reaction positive shear step at x=0.
   - Moment continuity across point loads (M(x-) == M(x+)).
   - Zero-moment boundaries M(0) = 0 and M(L) = 0.
   - Differential relationship dM/dx = V in smooth intervals via finite differences.
   - Shear integral relationship: V(x2) - V(x1) = -∫ w(t) dt.
5. Extrema precision verification against analytical reference values.
6. Edge case and boundary stress testing:
   - Zero-intensity UVL (w_a=0, w_b=0).
   - Point load at support boundaries (x=0, x=L).
   - Point loads at UVL boundaries (x=a, x=b).
   - Multiple point loads at the exact same position x.
   - Extremely short valid beams (e.g. L=0.01 m).
   - Very large load magnitudes (e.g. 100,000 kN).
   - Rejection of NaN, Inf, and negative magnitudes.
"""

import math
import sys
import os
import pytest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    calculate_reactions,
    shear_force,
    bending_moment,
    uvl_resultant,
    validate_beam,
    sample_beam,
)
from src.beam_plotter import find_bending_moment_extrema
from src.beam_validation import (
    independent_uvl_resultant,
    independent_calculate_reactions,
    verify_static_equilibrium,
    analytical_central_point_load,
    analytical_full_span_udl,
    analytical_triangular_uvl,
)


# ===========================================================================
# 1. Independent UVL and Static Equilibrium Audits
# ===========================================================================

class TestIndependentEquilibrium:
    """Audit static equilibrium against independent formulation."""

    def test_independent_reactions_match_engine_sample_beam(self):
        config = sample_beam()
        R_A_eng, R_B_eng = calculate_reactions(config)
        R_A_ind, R_B_ind = independent_calculate_reactions(config)

        assert pytest.approx(R_A_eng, abs=1e-10) == R_A_ind
        assert pytest.approx(R_B_eng, abs=1e-10) == R_B_ind

        audit = verify_static_equilibrium(config, R_A_eng, R_B_eng, tol=1e-9)
        assert audit["is_balanced"] is True
        assert abs(audit["residual_vertical"]) < 1e-9
        assert abs(audit["residual_moment_A"]) < 1e-9
        assert abs(audit["residual_moment_B"]) < 1e-9

    def test_uvl_resultant_and_centroid_zero_intensity(self):
        # Zero intensity UVL must not divide by zero and return R=0, x_bar=(a+b)/2
        R_eng, x_bar_eng = uvl_resultant(UVL(a=2.0, b=6.0, w_a=0.0, w_b=0.0))
        R_ind, x_bar_ind = independent_uvl_resultant(2.0, 6.0, 0.0, 0.0)

        assert R_eng == 0.0
        assert R_ind == 0.0
        assert x_bar_eng == 4.0
        assert x_bar_ind == 4.0

    def test_uvl_resultant_and_centroid_triangular(self):
        # Triangular UVL from a=0 to b=9, w_a=0, w_b=30
        R_ind, x_bar_ind = independent_uvl_resultant(0.0, 9.0, 0.0, 30.0)
        # R = 0.5 * 30 * 9 = 135 kN
        # x_bar = 2/3 * 9 = 6.0 m
        assert pytest.approx(R_ind, abs=1e-12) == 135.0
        assert pytest.approx(x_bar_ind, abs=1e-12) == 6.0


# ===========================================================================
# 2. Analytical Benchmark Cases
# ===========================================================================

class TestAnalyticalBenchmarks:
    """Verify calculation engine against textbook analytical solutions."""

    def test_case_A_central_point_load(self):
        """
        Benchmark A: Beam with central point load P=100 kN at midspan L/2 (L=10 m).
        To satisfy engine requirement of >= 3 point loads, we place dummy 0 kN loads at valid positions.
        """
        L = 10.0
        P = 100.0
        expected = analytical_central_point_load(L, P)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=P, position=5.0),
                PointLoad(magnitude=0.0, position=9.0),
            ],
            uvl=None,
        )

        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        # Midspan bending moment
        M_mid = bending_moment(config, 5.0)
        assert pytest.approx(M_mid, abs=1e-9) == expected["M_max"]

        # Shear left and right of midspan
        V_left = shear_force(config, 5.0 - 1e-9)
        V_right = shear_force(config, 5.0 + 1e-9)
        assert pytest.approx(V_left, abs=1e-7) == expected["V_left_mid"]
        assert pytest.approx(V_right, abs=1e-7) == expected["V_right_mid"]

        # Extrema algorithm check
        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_value"], abs=1e-7) == expected["M_max"]
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-5) == expected["x_M_max"]

    def test_case_B_full_span_udl(self):
        """
        Benchmark B: Full-span UDL w0=20 kN/m across L=8 m.
        Dummy 0 kN point loads are added to satisfy >= 3 point loads.
        """
        L = 8.0
        w0 = 20.0
        expected = analytical_full_span_udl(L, w0)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=0.0, position=4.0),
                PointLoad(magnitude=0.0, position=7.0),
            ],
            uvl=UVL(a=0.0, b=L, w_a=w0, w_b=w0),
        )

        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        # Midspan moment = w L^2 / 8 = 20 * 64 / 8 = 160 kN.m
        M_mid = bending_moment(config, 4.0)
        assert pytest.approx(M_mid, abs=1e-9) == expected["M_max"]

        # Extrema check: smooth parabolic peak at exact midspan
        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_value"], abs=1e-7) == expected["M_max"]
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-6) == expected["x_M_max"]

    def test_case_C_triangular_uvl(self):
        """
        Benchmark C: Full-span triangular load increasing 0 -> w0=36 kN/m across L=6 m.
        """
        L = 6.0
        w0 = 36.0
        expected = analytical_triangular_uvl(L, w0)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=0.0, position=2.0),
                PointLoad(magnitude=0.0, position=5.0),
            ],
            uvl=UVL(a=0.0, b=L, w_a=0.0, w_b=w0),
        )

        R_A, R_B = calculate_reactions(config)
        # R_A = 36 * 6 / 6 = 36 kN, R_B = 36 * 6 / 3 = 72 kN
        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        # Exact zero shear root at L / sqrt(3)
        x_expected = expected["x_M_max"]
        V_at_root = shear_force(config, x_expected)
        assert pytest.approx(V_at_root, abs=1e-6) == 0.0

        M_max_analytical = expected["M_max"]
        M_at_root = bending_moment(config, x_expected)
        assert pytest.approx(M_at_root, abs=1e-7) == M_max_analytical

        # Check bisection finds the exact root and peak moment
        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-5) == x_expected
        assert pytest.approx(extrema["max_positive_value"], abs=1e-6) == M_max_analytical

    def test_case_D_asymmetric_mixed_loading(self):
        """
        Benchmark D: Asymmetric combination of 3 active point loads and partial-span trapezoidal UVL.
        """
        config = BeamConfig(
            length=12.0,
            point_loads=[
                PointLoad(magnitude=18.0, position=2.5),
                PointLoad(magnitude=42.0, position=6.0),
                PointLoad(magnitude=24.0, position=10.0),
            ],
            uvl=UVL(a=4.0, b=9.0, w_a=10.0, w_b=35.0),
        )
        validate_beam(config)

        # Independent reactions
        R_A_ind, R_B_ind = independent_calculate_reactions(config)
        R_A_eng, R_B_eng = calculate_reactions(config)
        assert pytest.approx(R_A_eng, abs=1e-10) == R_A_ind
        assert pytest.approx(R_B_eng, abs=1e-10) == R_B_ind

        # Verify static balance
        audit = verify_static_equilibrium(config, R_A_eng, R_B_eng)
        assert audit["is_balanced"] is True


# ===========================================================================
# 3. SFD & BMD Consistency and Calculus Derivatives
# ===========================================================================

class TestCalculusAndConsistency:
    """Verify differential relations dM/dx = V and dV/dx = -w."""

    def test_differential_relation_dM_dx_equals_V(self):
        """Inside smooth intervals (away from point loads), (M(x+h) - M(x-h))/(2h) == V(x)."""
        config = sample_beam()
        # Test stations at smooth positions: x = 1.0 (before any load), 4.0 (inside UVL), 6.0 (inside UVL)
        test_stations = [1.0, 4.0, 6.0, 9.0]
        h = 1e-6

        for x in test_stations:
            M_plus = bending_moment(config, x + h)
            M_minus = bending_moment(config, x - h)
            dM_dx_num = (M_plus - M_minus) / (2.0 * h)
            V_exact = shear_force(config, x)

            # Central difference error is O(h^2); for smooth quadratic/cubic curve ~ 1e-6 tolerance
            assert pytest.approx(dM_dx_num, abs=1e-4) == V_exact

    def test_shear_integral_across_uvl(self):
        """Change in shear across a UVL interval must equal negative total distributed load."""
        config = sample_beam()
        # UVL is on [3.0, 7.0] with w_a=0, w_b=30, plus point load of 35 kN at x=5.0
        # Let's test on [3.1, 4.9] which is strictly inside the UVL with NO point loads!
        x1, x2 = 3.1, 4.9
        V1 = shear_force(config, x1)
        V2 = shear_force(config, x2)
        delta_V = V2 - V1

        # w(x) = 0 + (30 - 0)/(7 - 3) * (x - 3) = 7.5 * (x - 3)
        # Integral of w(x) from 3.1 to 4.9:
        # ∫_{3.1}^{4.9} 7.5*(x-3) dx = 7.5 * [ (x-3)^2 / 2 ]_{3.1}^{4.9}
        # = 3.75 * [ 1.9^2 - 0.1^2 ] = 3.75 * [ 3.61 - 0.01 ] = 3.75 * 3.6 = 13.5 kN
        expected_load_integral = 13.5
        assert pytest.approx(delta_V, abs=1e-8) == -expected_load_integral

    def test_shear_jump_at_concentrated_loads(self):
        """V(x-) - V(x+) == P for every downward concentrated load."""
        config = sample_beam()
        _EPS = 1e-9
        for pl in config.point_loads:
            V_left = shear_force(config, pl.position - _EPS)
            V_right = shear_force(config, pl.position + _EPS)
            jump = V_left - V_right
            assert pytest.approx(jump, abs=1e-6) == pl.magnitude

    def test_bending_moment_continuity_at_concentrated_loads(self):
        """M(x-) == M(x+) at point load locations."""
        config = sample_beam()
        _EPS = 1e-9
        for pl in config.point_loads:
            M_left = bending_moment(config, pl.position - _EPS)
            M_right = bending_moment(config, pl.position + _EPS)
            assert pytest.approx(M_left, abs=1e-6) == M_right


# ===========================================================================
# 4. Boundary and Edge-Case Stress Tests
# ===========================================================================

class TestBoundaryAndEdgeCases:
    """Stress test boundary conditions, extreme scales, and edge configurations."""

    def test_loads_at_exact_supports(self):
        """Point loads directly at x = 0 and x = L."""
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=50.0, position=0.0),
                PointLoad(magnitude=30.0, position=5.0),
                PointLoad(magnitude=40.0, position=10.0),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        # Total load = 120 kN. Moments: 50*0 + 30*5 + 40*10 = 550 kN.m => R_B = 55 kN => R_A = 65 kN
        assert pytest.approx(R_B, abs=1e-9) == 55.0
        assert pytest.approx(R_A, abs=1e-9) == 65.0
        assert bending_moment(config, 0.0) == 0.0
        assert bending_moment(config, 10.0) == 0.0

    def test_multiple_loads_at_same_position(self):
        """Two point loads at the exact same section position x=4.0."""
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=20.0, position=4.0),
                PointLoad(magnitude=30.0, position=4.0),
                PointLoad(magnitude=15.0, position=8.0),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        # Moment about A = 50*4 + 15*8 = 200 + 120 = 320 => R_B = 32 kN, R_A = 33 kN
        assert pytest.approx(R_B, abs=1e-9) == 32.0
        assert pytest.approx(R_A, abs=1e-9) == 33.0
        # Total shear drop at x=4.0 should be 50.0 kN
        V_left = shear_force(config, 4.0 - 1e-9)
        V_right = shear_force(config, 4.0 + 1e-9)
        assert pytest.approx(V_left - V_right, abs=1e-6) == 50.0

    def test_extremely_short_beam(self):
        """Very short beam L = 0.05 m."""
        config = BeamConfig(
            length=0.05,
            point_loads=[
                PointLoad(magnitude=10.0, position=0.01),
                PointLoad(magnitude=20.0, position=0.025),
                PointLoad(magnitude=10.0, position=0.04),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        # Symmetric => R_A = R_B = 20 kN
        assert pytest.approx(R_A, abs=1e-9) == 20.0
        assert pytest.approx(R_B, abs=1e-9) == 20.0
        assert pytest.approx(bending_moment(config, 0.0), abs=1e-9) == 0.0
        assert pytest.approx(bending_moment(config, 0.05), abs=1e-9) == 0.0

    def test_very_large_load_magnitudes(self):
        """Realistic heavy civil engineering load scale (100,000 kN)."""
        config = BeamConfig(
            length=50.0,
            point_loads=[
                PointLoad(magnitude=100000.0, position=10.0),
                PointLoad(magnitude=200000.0, position=25.0),
                PointLoad(magnitude=100000.0, position=40.0),
            ],
            uvl=UVL(a=0.0, b=50.0, w_a=5000.0, w_b=5000.0),
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        # Symmetrical loading => total = 400,000 + 250,000 = 650,000 kN => R_A = R_B = 325,000 kN
        assert pytest.approx(R_A, abs=1e-6) == 325000.0
        assert pytest.approx(R_B, abs=1e-6) == 325000.0

    def test_non_finite_inputs_raise_value_error(self):
        """Verify that NaN and Inf are rejected by validate_beam."""
        # NaN length
        with pytest.raises(ValueError, match="finite"):
            validate_beam(BeamConfig(length=float("nan"), point_loads=[
                PointLoad(10.0, 1.0), PointLoad(10.0, 2.0), PointLoad(10.0, 3.0)
            ]))

        # Inf load position
        with pytest.raises(ValueError, match="finite"):
            validate_beam(BeamConfig(length=10.0, point_loads=[
                PointLoad(10.0, 1.0), PointLoad(10.0, float("inf")), PointLoad(10.0, 3.0)
            ]))

        # Negative load magnitude
        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(BeamConfig(length=10.0, point_loads=[
                PointLoad(10.0, 1.0), PointLoad(-5.0, 2.0), PointLoad(10.0, 3.0)
            ]))
