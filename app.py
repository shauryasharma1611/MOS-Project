"""
app.py
======
Beam Analysis Studio — SFD & BMD
Phase 3: Interactive Streamlit Dashboard for Simply Supported Beam Analysis.

Provides:
- Beam geometry and loading configuration (point loads, UVL, supports).
- Live validation with helpful error messages.
- Real-time reaction calculations (R_A, R_B) and bending moment extrema.
- Three-panel diagrams: Loading, SFD, and BMD.
- Numerical results table with sampling and custom section evaluation.
- Diagram (PNG) and data (CSV) export capabilities.
"""

from __future__ import annotations

import io
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless / server-safe figure rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    calculate_reactions,
    shear_force,
    bending_moment,
    validate_beam,
    sample_beam,
)
from src.beam_plotter import (
    plot_beam_diagrams,
    find_bending_moment_extrema,
)
from src.beam_report import generate_beam_pdf_report
from src.beam_validation import verify_static_equilibrium


# ===========================================================================
# Helper Functions (tested in test_app.py)
# ===========================================================================

def validate_inputs(
    length: float,
    point_loads: List[Dict[str, float]],
    uvl_enabled: bool,
    uvl_a: float,
    uvl_b: float,
    uvl_w_a: float,
    uvl_w_b: float,
) -> Tuple[bool, List[str], Optional[BeamConfig]]:
    """
    Validate user inputs and construct a BeamConfig if valid.

    Parameters
    ----------
    length : float
        Beam span in metres.
    point_loads : List[Dict[str, float]]
        List of dicts with keys 'magnitude' (kN) and 'position' (m).
    uvl_enabled : bool
        Whether a UVL is applied.
    uvl_a, uvl_b : float
        UVL start and end positions in metres.
    uvl_w_a, uvl_w_b : float
        UVL intensities in kN/m at positions a and b.

    Returns
    -------
    (is_valid, error_messages, config)
    """
    errors: List[str] = []

    # 1. Beam length validation
    if length <= 0.0:
        errors.append(f"Beam length must be strictly positive (L > 0). Got {length:.3f} m.")

    # 2. Point load count
    if len(point_loads) < 3:
        errors.append(
            f"At least three concentrated point loads are required. Currently configured: {len(point_loads)}."
        )

    # 3. Point load bounds
    for idx, pl in enumerate(point_loads):
        pos = pl.get("position", 0.0)
        mag = pl.get("magnitude", 0.0)
        if not (0.0 <= pos <= length):
            errors.append(
                f"Point load #{idx + 1} position ({pos:.3f} m) must be within beam span [0.0, {length:.3f}] m."
            )

    # 4. UVL validation if enabled
    constructed_uvl: Optional[UVL] = None
    if uvl_enabled:
        if not (0.0 <= uvl_a < uvl_b <= length):
            errors.append(
                f"UVL range [{uvl_a:.3f}, {uvl_b:.3f}] m must satisfy 0 ≤ a < b ≤ L ({length:.3f} m)."
            )
        if uvl_w_a < 0.0 or uvl_w_b < 0.0:
            errors.append(
                f"UVL intensities must be non-negative (downward positive). Got w_a={uvl_w_a:.2f}, w_b={uvl_w_b:.2f} kN/m."
            )
        if not errors:
            constructed_uvl = UVL(
                a=float(uvl_a),
                b=float(uvl_b),
                w_a=float(uvl_w_a),
                w_b=float(uvl_w_b),
            )

    if errors:
        return False, errors, None

    # Construct BeamConfig
    pl_objs = [
        PointLoad(magnitude=float(p["magnitude"]), position=float(p["position"]))
        for p in point_loads
    ]
    config = BeamConfig(
        length=float(length),
        point_loads=pl_objs,
        uvl=constructed_uvl,
    )

    # Run calculation engine's internal validator
    try:
        validate_beam(config)
    except ValueError as ve:
        return False, [str(ve)], None

    return True, [], config


def generate_numerical_table(
    config: BeamConfig,
    num_samples: int = 21,
) -> pd.DataFrame:
    """
    Generate a table of positions x, shear forces V(x), and bending moments M(x).
    Includes beam endpoints, all point-load locations, UVL endpoints, and regular samples.
    """
    L = config.length
    _EPS = 1e-9

    # Collect critical points
    critical_pts = {0.0, L}
    for pl in config.point_loads:
        critical_pts.add(pl.position)
    if config.uvl is not None:
        critical_pts.add(config.uvl.a)
        critical_pts.add(config.uvl.b)

    # Uniform sample grid
    grid_pts = set(np.linspace(0.0, L, max(num_samples, 2)))
    all_pts = sorted(critical_pts | grid_pts)

    rows = []
    for x in all_pts:
        # Check if x is at a point load location
        is_pl = any(abs(pl.position - x) < 1e-8 for pl in config.point_loads)
        
        # Left limit V(x-) and Right limit V(x+)
        if x == 0.0:
            v_left = np.nan
            v_right = shear_force(config, _EPS)
        elif x == L:
            v_left = shear_force(config, L - _EPS)
            v_right = np.nan
        elif is_pl:
            v_left = shear_force(config, x - _EPS)
            v_right = shear_force(config, x + _EPS)
        else:
            v_val = shear_force(config, x)
            v_left = v_val
            v_right = v_val

        m_val = bending_moment(config, x)

        rows.append({
            "Position x (m)": round(x, 4),
            "Shear V(x⁻) (kN)": round(v_left, 3) if not np.isnan(v_left) else "—",
            "Shear V(x⁺) (kN)": round(v_right, 3) if not np.isnan(v_right) else "—",
            "Bending Moment M(x) (kN·m)": round(m_val, 3),
        })

    return pd.DataFrame(rows)


def evaluate_custom_section(
    config: BeamConfig,
    x: float,
) -> Dict[str, Any]:
    """
    Evaluate shear force and bending moment at a custom section x with left/right conventions.
    """
    _EPS = 1e-9
    L = config.length
    if not (0.0 <= x <= L):
        raise ValueError(f"Section position {x} m is outside beam span [0, {L}] m.")

    v_left = shear_force(config, x - _EPS) if x > 0.0 else np.nan
    v_right = shear_force(config, x + _EPS) if x < L else np.nan
    v_engine = shear_force(config, x)  # engine convention (strict left)
    m = bending_moment(config, x)

    is_pl = any(abs(pl.position - x) < 1e-8 for pl in config.point_loads)

    return {
        "x": x,
        "V_left": v_left,
        "V_right": v_right,
        "V_engine": v_engine,
        "M": m,
        "is_point_load": is_pl,
    }


# ===========================================================================
# Streamlit Dashboard UI
# ===========================================================================

def init_session_state():
    """Initialize session state defaults matching sample_beam()"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        sample = sample_beam()
        st.session_state.beam_length = float(sample.length)
        st.session_state.point_loads = [
            {"magnitude": float(pl.magnitude), "position": float(pl.position)}
            for pl in sample.point_loads
        ]
        st.session_state.uvl_enabled = sample.uvl is not None
        if sample.uvl is not None:
            st.session_state.uvl_a = float(sample.uvl.a)
            st.session_state.uvl_b = float(sample.uvl.b)
            st.session_state.uvl_w_a = float(sample.uvl.w_a)
            st.session_state.uvl_w_b = float(sample.uvl.w_b)
        else:
            st.session_state.uvl_a = 0.0
            st.session_state.uvl_b = float(sample.length)
            st.session_state.uvl_w_a = 0.0
            st.session_state.uvl_w_b = 20.0


def reset_to_sample_beam():
    """Reset session state to sample_beam configuration."""
    sample = sample_beam()
    st.session_state.beam_length = float(sample.length)
    st.session_state.point_loads = [
        {"magnitude": float(pl.magnitude), "position": float(pl.position)}
        for pl in sample.point_loads
    ]
    st.session_state.uvl_enabled = True
    st.session_state.uvl_a = float(sample.uvl.a)
    st.session_state.uvl_b = float(sample.uvl.b)
    st.session_state.uvl_w_a = float(sample.uvl.w_a)
    st.session_state.uvl_w_b = float(sample.uvl.w_b)


def main():
    st.set_page_config(
        page_title="Beam Analysis Studio — SFD & BMD",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # ── Application Header ─────────────────────────────────────────────────
    st.title(" Beam Analysis Studio — SFD & BMD")
    st.markdown(
        """
        **Interactive Mechanical Engineering Dashboard** for simply supported beams subjected to 
        concentrated point loads and uniformly varying distributed loads (UVL).
        Calculates equilibrium support reactions, generates analytical Shear Force Diagrams (SFD) 
        and Bending Moment Diagrams (BMD), and locates moment extrema with sub-millimetre precision.
        """
    )
    st.divider()

    # ── Sidebar Configuration ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Beam Configuration")

        col_reset, _ = st.columns([1, 1])
        with col_reset:
            if st.button("🔄 Reset to Sample Beam", use_container_width=True):
                reset_to_sample_beam()
                st.rerun()

        st.subheader("1. Beam Geometry")
        length = st.number_input(
            "Beam Length L (m)",
            min_value=0.1,
            max_value=200.0,
            value=float(st.session_state.beam_length),
            step=0.5,
            key="input_beam_length",
            help="Total span of the simply supported beam between pin (x=0) and roller (x=L).",
        )
        st.session_state.beam_length = length

        st.divider()

        # ── Point Loads ────────────────────────────────────────────────────
        st.subheader("2. Concentrated Point Loads (≥ 3)")
        st.caption("Downward loads are positive (+kN). Positions must be within [0, L] m.")

        # Controls to add/remove loads
        col_add, col_rem = st.columns(2)
        with col_add:
            if st.button("➕ Add Load", use_container_width=True):
                # Add default load at midspan or 3/4 span
                new_pos = min(round(length * 0.5, 2), length)
                st.session_state.point_loads.append({"magnitude": 25.0, "position": new_pos})
                st.rerun()

        with col_rem:
            can_remove = len(st.session_state.point_loads) > 3
            if st.button("➖ Remove Last", disabled=not can_remove, use_container_width=True):
                if len(st.session_state.point_loads) > 3:
                    st.session_state.point_loads.pop()
                    st.rerun()

        # Dynamic load editors
        updated_point_loads = []
        for i, pl in enumerate(st.session_state.point_loads):
            with st.expander(f"Point Load #{i + 1}", expanded=(i < 3)):
                col_p, col_m = st.columns(2)
                with col_p:
                    pos = st.number_input(
                        f"Position (m) #{i + 1}",
                        min_value=0.0,
                        max_value=float(length),
                        value=float(min(pl["position"], length)),
                        step=0.25,
                        key=f"pl_pos_{i}",
                    )
                with col_m:
                    mag = st.number_input(
                        f"Magnitude (kN) #{i + 1}",
                        min_value=0.0,
                        max_value=5000.0,
                        value=float(pl["magnitude"]),
                        step=5.0,
                        key=f"pl_mag_{i}",
                    )
                updated_point_loads.append({"magnitude": mag, "position": pos})
        st.session_state.point_loads = updated_point_loads

        st.divider()

        # ── Distributed Load (UVL) ─────────────────────────────────────────
        st.subheader("3. Distributed Load (UVL)")
        st.caption("Supports triangular, trapezoidal, or full/partial span loads.")
        uvl_enabled = st.checkbox(
            "Enable Uniformly Varying Load",
            value=st.session_state.uvl_enabled,
            key="input_uvl_enabled",
        )
        st.session_state.uvl_enabled = uvl_enabled

        uvl_a = float(st.session_state.uvl_a)
        uvl_b = float(st.session_state.uvl_b)
        uvl_w_a = float(st.session_state.uvl_w_a)
        uvl_w_b = float(st.session_state.uvl_w_b)

        if uvl_enabled:
            col_a, col_b = st.columns(2)
            with col_a:
                uvl_a = st.number_input(
                    "Start Position a (m)",
                    min_value=0.0,
                    max_value=float(length),
                    value=float(min(st.session_state.uvl_a, length * 0.9)),
                    step=0.25,
                    key="input_uvl_a",
                )
            with col_b:
                uvl_b = st.number_input(
                    "End Position b (m)",
                    min_value=0.0,
                    max_value=float(length),
                    value=float(min(max(st.session_state.uvl_b, uvl_a + 0.1), length)),
                    step=0.25,
                    key="input_uvl_b",
                )

            col_wa, col_wb = st.columns(2)
            with col_wa:
                uvl_w_a = st.number_input(
                    "Intensity w_a (kN/m)",
                    min_value=0.0,
                    max_value=2000.0,
                    value=float(st.session_state.uvl_w_a),
                    step=5.0,
                    key="input_uvl_wa",
                )
            with col_wb:
                uvl_w_b = st.number_input(
                    "Intensity w_b (kN/m)",
                    min_value=0.0,
                    max_value=2000.0,
                    value=float(st.session_state.uvl_w_b),
                    step=5.0,
                    key="input_uvl_wb",
                )
            st.session_state.uvl_a = uvl_a
            st.session_state.uvl_b = uvl_b
            st.session_state.uvl_w_a = uvl_w_a
            st.session_state.uvl_w_b = uvl_w_b

        st.divider()

        # ── Sign Convention Box ────────────────────────────────────────────
        with st.expander("ℹEngineering Conventions & Units", expanded=False):
            st.markdown(
                """
                - **Length & Positions**: Metres (m).
                - **Point Loads**: Positive = Downward (kN).
                - **Reactions ($R_A, R_B$)**: Positive = Upward (kN).
                - **Distributed Load**: Intensity in kN/m (downward positive).
                - **Shear Force $V(x)$**: Left free body slides upward (+kN).
                - **Bending Moment $M(x)$**: Sagging tension at bottom fibres (+kN·m).
                """
            )

    # ── Input Validation ───────────────────────────────────────────────────
    is_valid, error_msgs, config = validate_inputs(
        length=length,
        point_loads=st.session_state.point_loads,
        uvl_enabled=uvl_enabled,
        uvl_a=uvl_a,
        uvl_b=uvl_b,
        uvl_w_a=uvl_w_a,
        uvl_w_b=uvl_w_b,
    )

    if not is_valid:
        st.error(" **Configuration Error**: Please resolve the following issues:")
        for err in error_msgs:
            st.markdown(f"- {err}")
        return

    # Valid Configuration Feedback
    st.success(" **Configuration Valid**: Engineering calculations and diagrams updated.")

    # ── Live Engineering Calculations ──────────────────────────────────────
    R_A, R_B = calculate_reactions(config)
    extrema = find_bending_moment_extrema(config)
    audit = verify_static_equilibrium(config, R_A, R_B)

    # ── Metric Cards ───────────────────────────────────────────────────────
    st.subheader(" Key Structural Results")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Left Reaction R_A (x = 0.00 m)",
            value=f"{R_A:.3f} kN",
            help="Vertical reaction at pin support A (upward positive).",
        )

    with m_col2:
        st.metric(
            label=f"Right Reaction R_B (x = {config.length:.2f} m)",
            value=f"{R_B:.3f} kN",
            help="Vertical reaction at roller support B (upward positive).",
        )

    with m_col3:
        max_pos = extrema["max_positive_value"]
        x_max_pos = extrema["max_positive_pos"]
        st.metric(
            label=f"Max Sagging Moment (+M) at x = {x_max_pos:.3f} m",
            value=f"{max_pos:.3f} kN·m",
            help="Maximum positive bending moment (sagging). Root-based analytical bisection / critical point.",
        )

    with m_col4:
        min_neg = extrema["min_negative_value"]
        x_min_neg = extrema["min_negative_pos"]
        if min_neg < -1e-6:
            st.metric(
                label=f"Max Hogging Moment (-M) at x = {x_min_neg:.3f} m",
                value=f"{min_neg:.3f} kN·m",
                help="Maximum negative bending moment (hogging). Tension in top fibre.",
            )
        else:
            st.metric(
                label="Min Negative Moment (-M)",
                value="None (0.000 kN·m)",
                help="No negative (hogging) moment exists on this span under downward loads.",
            )

    # Second row of metrics: Shear Extremes & Verification Audit
    s_col1, s_col2, s_col3 = st.columns([1.5, 1.5, 2.0])
    v_start = shear_force(config, 1e-9)
    v_end = shear_force(config, config.length - 1e-9)
    with s_col1:
        st.metric(
            label="Max Positive Shear (+V) at x = 0⁺ m",
            value=f"+{v_start:.3f} kN",
            help="Shear immediately right of left support (exact reaction step R_A).",
        )
    with s_col2:
        st.metric(
            label=f"Max Negative Shear (-V) at x = {config.length:.2f}⁻ m",
            value=f"{v_end:.3f} kN",
            help="Shear immediately left of right support (exact reaction step -R_B).",
        )
    with s_col3:
        if audit["is_balanced"]:
            st.success(
                f"⚖️ **Equilibrium Verified** (Residuals: $\\Delta F_y = {abs(audit['residual_vertical']):.1e}$ kN, "
                f"$\\Delta M_A = {abs(audit['residual_moment_A']):.1e}$ kN·m)"
            )
        else:
            st.error("⚠️ Static equilibrium residual check failed.")

    st.caption(
        "ℹ️ *Note*: Moment extrema are determined via analytical zero-shear root bisection and critical points. "
        "This software is an engineering analysis tool for educational and academic project verification; it does not replace professional structural safety certification."
    )

    st.divider()

    # ── Plots Section ──────────────────────────────────────────────────────
    st.subheader("📈 Structural Diagrams (Loading, SFD & BMD)")

    # Generate figure using existing plotting engine
    fig, axes = plot_beam_diagrams(config, show=False, figsize=(11.0, 13.0))

    # Display in Streamlit
    st.pyplot(fig, clear_figure=False)

    # Export Diagram & Report Buttons
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # Prevent Matplotlib memory accumulation

    # Generate PDF report from current valid configuration
    pdf_bytes = None
    pdf_error = None
    try:
        pdf_bytes = generate_beam_pdf_report(config)
    except Exception as e:
        pdf_error = str(e)

    # Generate CSV export data for download
    df_results = generate_numerical_table(config, num_samples=25)
    csv_buffer = df_results.to_csv(index=False).encode("utf-8")

    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.8, 1.5])
    with col_btn1:
        st.download_button(
            label="💾 Download Diagram (PNG)",
            data=buf,
            file_name="beam_analysis_diagrams.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_btn2:
        if pdf_bytes is not None:
            st.download_button(
                label="📄 Download Report (PDF)",
                data=pdf_bytes,
                file_name="beam_engineering_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.error(f"Report generation error: {pdf_error}")
    with col_btn3:
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv_buffer,
            file_name="beam_numerical_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
