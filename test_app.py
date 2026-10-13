"""
test_app.py
===========
Phase 3 unit tests for app.py helper and validation logic.

Tests:
1. validate_inputs with valid sample beam configuration.
2. validate_inputs catching L <= 0.
3. validate_inputs catching < 3 point loads.
4. validate_inputs catching point loads outside span [0, L].
5. validate_inputs catching invalid UVL ranges (a >= b, a < 0, b > L).
6. validate_inputs catching negative UVL intensities.
7. generate_numerical_table produces expected columns, spans, and critical points.
8. evaluate_custom_section correctly distinguishes point-load jump vs smooth section.
9. evaluate_custom_section raises ValueError on out-of-span probe.
"""

import io
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure MOS Project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beam_analysis import sample_beam, PointLoad, UVL, BeamConfig
from app import (
    validate_inputs,
    generate_numerical_table,
    evaluate_custom_section,
)


class TestValidateInputs:
    """Test the validation logic of app.py"""

    def test_valid_sample_configuration(self):
        sample = sample_beam()
        pl_dicts = [{"magnitude": pl.magnitude, "position": pl.position} for pl in sample.point_loads]
        is_valid, errors, config = validate_inputs(
            length=sample.length,
            point_loads=pl_dicts,
            uvl_enabled=True,
            uvl_a=sample.uvl.a,
            uvl_b=sample.uvl.b,
            uvl_w_a=sample.uvl.w_a,
            uvl_w_b=sample.uvl.w_b,
        )
        assert is_valid is True
        assert len(errors) == 0
        assert config is not None
        assert config.length == 10.0
        assert len(config.point_loads) == 3
        assert config.uvl is not None

    def test_negative_or_zero_length(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 3.0},
        ]
        is_valid, errors, config = validate_inputs(
            length=-5.0,
            point_loads=pl_dicts,
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is False
        assert any("length must be strictly positive" in e for e in errors)
        assert config is None

    def test_fewer_than_three_loads(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
        ]
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_dicts,
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is False
        assert any("At least three concentrated point loads" in e for e in errors)

    def test_point_load_outside_span(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 12.0},  # outside 10m
        ]
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_dicts,
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is False
        assert any("Point load #3 position" in e for e in errors)

    def test_uvl_range_invalid(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 3.0},
        ]
        # a >= b
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_dicts,
            uvl_enabled=True,
            uvl_a=6.0, uvl_b=4.0, uvl_w_a=10.0, uvl_w_b=20.0,
        )
        assert is_valid is False
        assert any("UVL range" in e for e in errors)

    def test_uvl_beyond_beam_length(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 3.0},
        ]
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_dicts,
            uvl_enabled=True,
            uvl_a=2.0, uvl_b=15.0, uvl_w_a=10.0, uvl_w_b=20.0,
        )
        assert is_valid is False
        assert any("UVL range" in e for e in errors)

    def test_negative_uvl_intensity(self):
        pl_dicts = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 3.0},
        ]
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_dicts,
            uvl_enabled=True,
            uvl_a=1.0, uvl_b=5.0, uvl_w_a=-5.0, uvl_w_b=10.0,
        )
        assert is_valid is False
        assert any("intensities must be non-negative" in e for e in errors)


class TestNumericalTable:
    """Test generating the numerical results table."""

    def test_table_structure_and_columns(self):
        config = sample_beam()
        df = generate_numerical_table(config, num_samples=21)
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "Position x (m)",
            "Shear V(x⁻) (kN)",
            "Shear V(x⁺) (kN)",
            "Bending Moment M(x) (kN·m)",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Verify x starts at 0 and ends at 10
        assert df["Position x (m)"].iloc[0] == 0.0
        assert df["Position x (m)"].iloc[-1] == 10.0

        # Verify critical points are included
        xs = set(df["Position x (m)"])
        assert 2.0 in xs
        assert 5.0 in xs
        assert 8.0 in xs
        assert 3.0 in xs
        assert 7.0 in xs


class TestCustomSectionProbe:
    """Test the single-section probe evaluation."""

    def test_smooth_section_probe(self):
        config = sample_beam()
        res = evaluate_custom_section(config, 1.0)
        assert res["x"] == 1.0
        assert res["is_point_load"] is False
        assert abs(res["V_left"] - res["V_right"]) < 1e-6
        assert isinstance(res["M"], float)

    def test_point_load_section_probe(self):
        config = sample_beam()
        res = evaluate_custom_section(config, 2.0)
        assert res["x"] == 2.0
        assert res["is_point_load"] is True
        # At x=2.0 there is a 20 kN downward load, so V drops by 20 kN
        assert pytest.approx(res["V_left"] - res["V_right"], abs=1e-5) == 20.0

    def test_out_of_bounds_section_probe(self):
        config = sample_beam()
        with pytest.raises(ValueError):
            evaluate_custom_section(config, -0.5)
        with pytest.raises(ValueError):
            evaluate_custom_section(config, 15.0)


class TestDashboardAndExportRegression:
    """Verify data exports and engine isolation on invalid configuration."""

    def test_invalid_input_does_not_invoke_engine(self):
        """Invalid configurations must fail validation and return config=None."""
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=[{"magnitude": 10.0, "position": 1.0}],  # only 1 load
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is False
        assert len(errors) > 0
        assert config is None

    def test_csv_export_reflects_current_config(self):
        """Verify CSV export contains correct headers and valid numerical content."""
        config = sample_beam()
        df = generate_numerical_table(config, num_samples=15)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        assert len(csv_bytes) > 0

        # Read back and verify
        df_read = pd.read_csv(io.BytesIO(csv_bytes))
        assert "Position x (m)" in df_read.columns
        assert "Shear V(x⁻) (kN)" in df_read.columns
        assert "Shear V(x⁺) (kN)" in df_read.columns
        assert "Bending Moment M(x) (kN·m)" in df_read.columns
        assert len(df_read) >= 15

    def test_png_export_from_current_figure(self):
        """Verify PNG image byte buffer can be created from plot_beam_diagrams without errors."""
        import matplotlib.pyplot as plt
        from src.beam_plotter import plot_beam_diagrams

        config = sample_beam()
        fig, axes = plot_beam_diagrams(config, show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic header

    def test_pdf_export_from_current_configuration(self):
        """Verify PDF export is generated from current configuration."""
        from src.beam_report import generate_beam_pdf_report

        config = sample_beam()
        pdf_bytes = generate_beam_pdf_report(config)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 50000

    def test_reset_to_sample_beam_workflow(self):
        """Verify reset_to_sample_beam helper restores default sample attributes."""
        import streamlit as st
        from app import reset_to_sample_beam, init_session_state

        init_session_state()
        # Alter state
        st.session_state.beam_length = 25.0
        st.session_state.point_loads = [{"magnitude": 100.0, "position": 12.0}]
        st.session_state.uvl_enabled = False

        # Reset
        reset_to_sample_beam()
        assert st.session_state.beam_length == 10.0
        assert len(st.session_state.point_loads) == 3
        assert st.session_state.uvl_enabled is True
        assert st.session_state.uvl_a == 3.0
        assert st.session_state.uvl_b == 7.0

    def test_dynamic_add_and_remove_point_loads(self):
        """Verify point load list manipulations maintain valid config."""
        pl_list = [
            {"magnitude": 10.0, "position": 1.0},
            {"magnitude": 20.0, "position": 2.0},
            {"magnitude": 30.0, "position": 3.0},
        ]
        # Add 4th load
        pl_list.append({"magnitude": 40.0, "position": 4.0})
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_list,
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is True
        assert len(config.point_loads) == 4

        # Remove down to 3
        pl_list.pop()
        is_valid, errors, config = validate_inputs(
            length=10.0,
            point_loads=pl_list,
            uvl_enabled=False,
            uvl_a=0.0, uvl_b=1.0, uvl_w_a=0.0, uvl_w_b=0.0,
        )
        assert is_valid is True
        assert len(config.point_loads) == 3
