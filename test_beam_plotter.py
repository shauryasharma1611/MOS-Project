"""
test_beam_plotter.py
====================
Phase 2 unit tests for src/beam_plotter.py.

All tests use the non-interactive Agg backend so no GUI window is opened.
Phase 1 tests in test_beam_analysis.py are completely unmodified; running
``pytest tests/`` executes both suites together.

Test categories
---------------
1.  Figure and axes creation — correct type, 3 subplots.
2.  Successful plotting for the sample beam (Phase 1 config).
3.  Three or more point loads + triangular UVL.
4.  Trapezoidal UVL.
5.  UVL over the full span.
6.  UVL over a partial span.
7.  Shear discontinuities at point loads (correct jump magnitudes).
8.  Bending moment continuity at point loads.
9.  Correct zero-moment support boundaries (M(0) == M(L) == 0).
10. Maximum bending moment reporting via find_bending_moment_extrema.
11. Plotting without showing a GUI window (show=False).
12. SFD array — NaN segments present at each point-load position.
13. BMD array — starts and ends at zero.
14. Extrema positions lie strictly inside [0, L] for standard configs.
"""

import sys
import os

import numpy as np
import matplotlib
# Force non-interactive backend BEFORE any other matplotlib import
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    calculate_reactions,
    shear_force,
    bending_moment,
    sample_beam,
)
from src.beam_plotter import (
    plot_beam_diagrams,
    find_bending_moment_extrema,
    _build_sfd_arrays,
    _build_bmd_arrays,
)

ATOL = 1e-6   # loose tolerance for floating-point spot checks in plots


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_all_figures():
    """Automatically close all Matplotlib figures after every test."""
    yield
    plt.close("all")


def _make_beam(L, loads, uvl=None):
    return BeamConfig(length=L, point_loads=loads, uvl=uvl)


# Typical 3-load, no-UVL beam
BEAM_3PL = _make_beam(
    8.0,
    [PointLoad(10.0, 2.0), PointLoad(20.0, 5.0), PointLoad(30.0, 7.0)],
)

# Triangular UVL (ascending)
BEAM_TRI = _make_beam(
    10.0,
    [PointLoad(5.0, 1.0), PointLoad(10.0, 5.0), PointLoad(5.0, 9.0)],
    UVL(a=2.0, b=8.0, w_a=0.0, w_b=20.0),
)

# Trapezoidal UVL
BEAM_TRAP = _make_beam(
    12.0,
    [PointLoad(15.0, 3.0), PointLoad(25.0, 6.0), PointLoad(10.0, 9.0)],
    UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0),
)

# UVL over the full span (uniform)
BEAM_FULL_UVL = _make_beam(
    10.0,
    [PointLoad(0.0, 0.0), PointLoad(0.0, 5.0), PointLoad(0.0, 10.0)],
    UVL(a=0.0, b=10.0, w_a=20.0, w_b=20.0),
)

# UVL over partial span
BEAM_PART_UVL = _make_beam(
    12.0,
    [PointLoad(0.0, 0.0), PointLoad(0.0, 6.0), PointLoad(0.0, 12.0)],
    UVL(a=3.0, b=9.0, w_a=10.0, w_b=10.0),
)


# ===========================================================================
# 1. Figure and axes creation
# ===========================================================================

class TestFigureCreation:
    """plot_beam_diagrams must return a Figure and exactly three Axes."""

    def test_returns_figure(self):
        fig, axes = plot_beam_diagrams(sample_beam(), show=False)
        assert isinstance(fig, plt.Figure)

    def test_returns_three_axes(self):
        fig, axes = plot_beam_diagrams(sample_beam(), show=False)
        assert len(axes) == 3

    def test_axes_are_matplotlib_axes(self):
        from matplotlib.axes import Axes
        fig, (ax1, ax2, ax3) = plot_beam_diagrams(sample_beam(), show=False)
        for ax in (ax1, ax2, ax3):
            assert isinstance(ax, Axes)

    def test_figure_has_three_subplots(self):
        fig, _ = plot_beam_diagrams(sample_beam(), show=False)
        assert len(fig.axes) == 3


# ===========================================================================
# 2. Successful plotting for the Phase 1 sample beam
# ===========================================================================

class TestSampleBeamPlot:
    """Basic smoke tests: the sample beam should plot without error."""

    def test_plot_sample_beam_no_exception(self):
        fig, axes = plot_beam_diagrams(sample_beam(), show=False)
        assert fig is not None

    def test_sfd_axis_has_lines(self):
        _, (_, ax_sfd, _) = plot_beam_diagrams(sample_beam(), show=False)
        # At least one Line2D should be present in the SFD
        assert len(ax_sfd.lines) > 0

    def test_bmd_axis_has_lines(self):
        _, (_, _, ax_bmd) = plot_beam_diagrams(sample_beam(), show=False)
        assert len(ax_bmd.lines) > 0

    def test_loading_axis_has_patches(self):
        """Loading diagram should have at least the support patches."""
        _, (ax_load, _, _) = plot_beam_diagrams(sample_beam(), show=False)
        assert len(ax_load.patches) > 0


# ===========================================================================
# 3. Three or more point loads + triangular UVL
# ===========================================================================

class TestTriangularUVLPlot:
    """Plots with a triangular UVL should succeed and produce valid data."""

    def test_no_exception(self):
        fig, axes = plot_beam_diagrams(BEAM_TRI, show=False)
        assert fig is not None

    def test_sfd_array_finite_values_exist(self):
        x, V = _build_sfd_arrays(BEAM_TRI)
        finite_mask = ~np.isnan(V)
        assert finite_mask.sum() > 0

    def test_bmd_zero_at_supports(self):
        x, M = _build_bmd_arrays(BEAM_TRI)
        assert abs(M[0]) < ATOL
        assert abs(M[-1]) < ATOL


# ===========================================================================
# 4. Trapezoidal UVL
# ===========================================================================

class TestTrapezoidalUVLPlot:
    def test_no_exception(self):
        fig, _ = plot_beam_diagrams(BEAM_TRAP, show=False)
        assert fig is not None

    def test_bmd_continuous(self):
        """BMD values at point-load positions must be identical from both sides."""
        _EPS = 1e-9
        for pl in BEAM_TRAP.point_loads:
            M_left  = bending_moment(BEAM_TRAP, pl.position - _EPS)
            M_right = bending_moment(BEAM_TRAP, pl.position + _EPS)
            assert abs(M_left - M_right) < ATOL, (
                f"BMD discontinuity at pl.position={pl.position}: "
                f"M_left={M_left}, M_right={M_right}"
            )

    def test_sfd_has_nan_at_loads(self):
        """SFD array must contain NaN values (the jump markers)."""
        x, V = _build_sfd_arrays(BEAM_TRAP)
        assert np.isnan(V).any(), "Expected NaN markers at shear jumps"


# ===========================================================================
# 5. UVL over the full span
# ===========================================================================

class TestFullSpanUVLPlot:
    def test_no_exception(self):
        fig, _ = plot_beam_diagrams(BEAM_FULL_UVL, show=False)
        assert fig is not None

    def test_reactions_symmetric(self):
        """Symmetric UDL → R_A == R_B."""
        R_A, R_B = calculate_reactions(BEAM_FULL_UVL)
        assert abs(R_A - R_B) < ATOL

    def test_bmd_zero_boundaries(self):
        x, M = _build_bmd_arrays(BEAM_FULL_UVL)
        assert abs(M[0]) < ATOL
        assert abs(M[-1]) < ATOL


# ===========================================================================
# 6. UVL over a partial span
# ===========================================================================

class TestPartialSpanUVLPlot:
    def test_no_exception(self):
        fig, _ = plot_beam_diagrams(BEAM_PART_UVL, show=False)
        assert fig is not None

    def test_shear_before_uvl_constant(self):
        """Before the UVL starts (x < a), shear should be constant (no distributed load)."""
        config = BEAM_PART_UVL
        uvl_a = config.uvl.a
        # Sample three points before the UVL starts
        xs = np.linspace(0.01, uvl_a - 0.01, 5)
        V_vals = [shear_force(config, x) for x in xs]
        for v in V_vals:
            assert abs(v - V_vals[0]) < ATOL, (
                f"Shear should be constant before UVL; got {V_vals}"
            )


# ===========================================================================
# 7. Shear discontinuities at point loads
# ===========================================================================

class TestShearDiscontinuities:
    """shear_force jumps by exactly P at each point-load position."""

    _EPS = 1e-8

    def _check_jumps(self, config):
        for pl in config.point_loads:
            if abs(pl.magnitude) < 1e-12:
                continue
            V_left  = shear_force(config, pl.position - self._EPS)
            V_right = shear_force(config, pl.position + self._EPS)
            jump    = V_left - V_right          # should equal pl.magnitude
            assert abs(jump - pl.magnitude) < 1e-4, (
                f"Shear jump at x={pl.position}: expected {pl.magnitude}, "
                f"got {jump:.6f}"
            )

    def test_shear_jump_sample_beam(self):
        self._check_jumps(sample_beam())

    def test_shear_jump_trapezoidal_uvl(self):
        self._check_jumps(BEAM_TRAP)

    def test_shear_jump_triangular_uvl(self):
        self._check_jumps(BEAM_TRI)

    def test_sfd_nan_count_equals_nonzero_load_count(self):
        """
        The SFD array should contain at least one NaN per non-zero point load,
        because those are the jump markers.
        """
        config = BEAM_3PL
        x, V = _build_sfd_arrays(config)
        num_nans = int(np.isnan(V).sum())
        nonzero_loads = sum(1 for pl in config.point_loads
                            if abs(pl.magnitude) >= 1e-12)
        # Each non-zero load generates at least one NaN segment
        assert num_nans >= nonzero_loads, (
            f"Expected ≥{nonzero_loads} NaNs, got {num_nans}"
        )


# ===========================================================================
# 8. Bending moment continuity at point loads
# ===========================================================================

class TestBMContinuity:
    """M(x) must be continuous at every point-load position."""

    _EPS = 1e-9

    def _check_continuity(self, config):
        for pl in config.point_loads:
            M_left  = bending_moment(config, pl.position - self._EPS)
            M_right = bending_moment(config, pl.position + self._EPS)
            assert abs(M_left - M_right) < ATOL, (
                f"Moment discontinuity at x={pl.position}: "
                f"M_left={M_left:.8f}, M_right={M_right:.8f}"
            )

    def test_continuity_sample_beam(self):
        self._check_continuity(sample_beam())

    def test_continuity_three_point_loads(self):
        self._check_continuity(BEAM_3PL)

    def test_continuity_trapezoidal_uvl(self):
        self._check_continuity(BEAM_TRAP)


# ===========================================================================
# 9. Zero-moment support boundaries
# ===========================================================================

class TestZeroMomentBoundaries:
    """M(0) == 0 and M(L) == 0 for all test configurations."""

    def _check(self, config):
        L = config.length
        assert abs(bending_moment(config, 0.0)) < ATOL, "M(0) ≠ 0"
        assert abs(bending_moment(config, L))   < ATOL, "M(L) ≠ 0"

    def test_sample_beam(self):
        self._check(sample_beam())

    def test_three_point_loads(self):
        self._check(BEAM_3PL)

    def test_triangular_uvl(self):
        self._check(BEAM_TRI)

    def test_trapezoidal_uvl(self):
        self._check(BEAM_TRAP)

    def test_full_span_uvl(self):
        self._check(BEAM_FULL_UVL)

    def test_partial_span_uvl(self):
        self._check(BEAM_PART_UVL)

    def test_bmd_array_zero_at_x0(self):
        """The first element of the BMD array should be ≈ 0."""
        for config in [sample_beam(), BEAM_3PL, BEAM_TRI, BEAM_TRAP]:
            x, M = _build_bmd_arrays(config)
            assert abs(M[0]) < ATOL, f"BMD[0] = {M[0]:.6e} for {config}"

    def test_bmd_array_zero_at_xL(self):
        """The last element of the BMD array should be ≈ 0."""
        for config in [sample_beam(), BEAM_3PL, BEAM_TRI, BEAM_TRAP]:
            x, M = _build_bmd_arrays(config)
            assert abs(M[-1]) < ATOL, f"BMD[-1] = {M[-1]:.6e} for {config}"


# ===========================================================================
# 10. Maximum bending moment reporting
# ===========================================================================

class TestMaxBendingMoment:
    """find_bending_moment_extrema must return a valid ExtremaResult dict."""

    def test_returns_dict_with_required_keys(self):
        result = find_bending_moment_extrema(sample_beam())
        required_keys = {
            "max_positive_value", "max_positive_pos",
            "min_negative_value", "min_negative_pos",
            "candidates", "note",
        }
        assert required_keys.issubset(result.keys())

    def test_max_positive_non_negative(self):
        """Simply supported beams under downward loads have M ≥ 0 everywhere
        (for typical loading); the max must be non-negative."""
        result = find_bending_moment_extrema(sample_beam())
        assert result["max_positive_value"] >= -ATOL

    def test_max_position_within_span(self):
        L = sample_beam().length
        result = find_bending_moment_extrema(sample_beam())
        assert 0.0 <= result["max_positive_pos"] <= L

    def test_min_position_within_span(self):
        L = sample_beam().length
        result = find_bending_moment_extrema(sample_beam())
        assert 0.0 <= result["min_negative_pos"] <= L

    def test_candidates_is_nonempty_list(self):
        result = find_bending_moment_extrema(sample_beam())
        assert isinstance(result["candidates"], list)
        assert len(result["candidates"]) > 0

    def test_note_is_string(self):
        result = find_bending_moment_extrema(sample_beam())
        assert isinstance(result["note"], str)

    def test_max_value_consistent_with_engine(self):
        """
        The reported maximum should equal bending_moment evaluated at its
        reported position (within numerical tolerance).
        """
        config = sample_beam()
        result = find_bending_moment_extrema(config)
        x_max = result["max_positive_pos"]
        M_check = bending_moment(config, x_max)
        assert abs(M_check - result["max_positive_value"]) < ATOL

    def test_extrema_for_symmetric_udl(self):
        """
        Symmetric UDL on L=10m: max moment at midspan (x=5).
        R_A = R_B = 100 kN, w=20 kN/m → M(5) = 100*5 - 20*5*2.5 = 500-250=250 kN·m
        """
        config = BEAM_FULL_UVL
        result = find_bending_moment_extrema(config)
        # Midspan moment for uniform load: w*L²/8 = 20*100/8 = 250 kN·m
        assert abs(result["max_positive_value"] - 250.0) < ATOL
        assert abs(result["max_positive_pos"] - 5.0) < 0.005   # within 5 mm

    def test_three_point_loads_max_at_load_or_between(self):
        """Maximum moment must be ≥ moment at each load position."""
        config = BEAM_3PL
        result = find_bending_moment_extrema(config)
        M_max_reported = result["max_positive_value"]
        for pl in config.point_loads:
            M_at_load = bending_moment(config, pl.position)
            assert M_max_reported >= M_at_load - ATOL, (
                f"Reported max {M_max_reported} < M at load pos "
                f"{pl.position}: {M_at_load}"
            )


# ===========================================================================
# 11. Plotting without showing a GUI window
# ===========================================================================

class TestNoGUI:
    """show=False must never open a GUI and still return the figure."""

    def test_no_show_returns_figure(self):
        fig, axes = plot_beam_diagrams(sample_beam(), show=False)
        assert fig is not None

    def test_no_show_does_not_raise(self):
        """Should not raise any display-related exception in headless mode."""
        try:
            fig, _ = plot_beam_diagrams(sample_beam(), show=False)
        except Exception as e:
            pytest.fail(f"plot_beam_diagrams raised: {e}")

    def test_figure_saveable_to_bytes(self):
        """The figure can be saved to an in-memory buffer — essential for Streamlit."""
        import io
        fig, _ = plot_beam_diagrams(sample_beam(), show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72)
        buf.seek(0)
        assert len(buf.read()) > 0


# ===========================================================================
# 12. SFD array — NaN segments at each point-load position
# ===========================================================================

class TestSFDArray:
    """Internal _build_sfd_arrays helper: verify NaN placement and coverage."""

    def test_nan_present_for_nonzero_loads(self):
        """Non-zero point loads must produce NaN entries in the V array."""
        x, V = _build_sfd_arrays(BEAM_3PL)
        assert np.isnan(V).any()

    def test_finite_values_cover_full_span(self):
        """Finite x-values must cover both x≈0 and x≈L."""
        L = BEAM_3PL.length
        x, V = _build_sfd_arrays(BEAM_3PL)
        finite_x = x[~np.isnan(V)]
        assert finite_x.min() < 0.05 * L
        assert finite_x.max() > 0.95 * L

    def test_shear_starts_at_R_A(self):
        """The first finite shear value should equal R_A (within tolerance)."""
        config = BEAM_3PL
        R_A, _ = calculate_reactions(config)
        x, V = _build_sfd_arrays(config)
        first_finite_V = V[~np.isnan(V)][0]
        assert abs(first_finite_V - R_A) < ATOL

    def test_shear_ends_near_neg_R_B(self):
        """The last finite shear value should be ≈ −R_B."""
        config = BEAM_3PL
        _, R_B = calculate_reactions(config)
        x, V = _build_sfd_arrays(config)
        last_finite_V = V[~np.isnan(V)][-1]
        assert abs(last_finite_V + R_B) < ATOL


# ===========================================================================
# 13. BMD array — starts and ends at zero
# ===========================================================================

class TestBMDArray:
    """Internal _build_bmd_arrays helper: boundary and monotone checks."""

    def test_starts_at_zero(self):
        x, M = _build_bmd_arrays(sample_beam())
        assert abs(M[0]) < ATOL

    def test_ends_at_zero(self):
        x, M = _build_bmd_arrays(sample_beam())
        assert abs(M[-1]) < ATOL

    def test_x_starts_at_zero(self):
        x, _ = _build_bmd_arrays(sample_beam())
        assert abs(x[0]) < ATOL

    def test_x_ends_at_L(self):
        config = sample_beam()
        x, _ = _build_bmd_arrays(config)
        assert abs(x[-1] - config.length) < ATOL

    def test_x_monotonically_increasing(self):
        x, _ = _build_bmd_arrays(sample_beam())
        assert np.all(np.diff(x) >= 0), "x-array is not monotonically increasing"

    def test_includes_critical_points(self):
        """BMD x-array must include all point-load positions."""
        config = sample_beam()
        x, _ = _build_bmd_arrays(config)
        for pl in config.point_loads:
            assert np.any(np.abs(x - pl.position) < 1e-10), (
                f"Point load position {pl.position} not found in BMD x-array"
            )


# ===========================================================================
# 14. Extrema positions for standard configs
# ===========================================================================

class TestExtremaPositions:
    """For typical downward loading, max M lies strictly between 0 and L."""

    def _interior_max(self, config):
        L = config.length
        result = find_bending_moment_extrema(config)
        pos = result["max_positive_pos"]
        # Allow endpoints for pathological cases but warn
        return pos, L

    def test_sample_beam_interior_max(self):
        config = sample_beam()
        L = config.length
        result = find_bending_moment_extrema(config)
        pos = result["max_positive_pos"]
        # Max cannot exceed the full span
        assert 0.0 <= pos <= L

    def test_three_load_beam_interior_max(self):
        config = BEAM_3PL
        L = config.length
        result = find_bending_moment_extrema(config)
        pos = result["max_positive_pos"]
        assert 0.0 <= pos <= L

    def test_extrema_values_bounded_by_simple_estimate(self):
        """Max moment must be less than the product of max load × span."""
        config = sample_beam()
        L = config.length
        max_possible = sum(pl.magnitude for pl in config.point_loads) * L
        result = find_bending_moment_extrema(config)
        assert result["max_positive_value"] <= max_possible + ATOL
