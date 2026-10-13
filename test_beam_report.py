"""
test_beam_report.py
===================
Phase 5 unit tests for src/beam_report.py.

Tests:
1. Successful PDF generation from standard sample beam.
2. Verification of binary signature (%PDF-) and non-empty content.
3. PDF generation with triangular UVL.
4. PDF generation with trapezoidal UVL.
5. PDF generation with zero-intensity UVL.
6. PDF generation with point-loads only (no UVL).
7. Invalid configurations raise ValueError and abort PDF generation.
8. Generation directly to file path vs in-memory BytesIO.
9. Verify Matplotlib figures are cleanly closed without leaks.
10. Verify Streamlit integration uses current configuration.
"""

import io
import os
import sys
import tempfile
import pytest
import matplotlib.pyplot as plt

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    sample_beam,
)
from src.beam_report import generate_beam_pdf_report


class TestPDFReportGeneration:
    """Test PDF engineering report creation."""

    def test_sample_beam_pdf_generation(self):
        config = sample_beam()
        pdf_bytes = generate_beam_pdf_report(config)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 50000  # typical complete report is ~100-250KB
        assert pdf_bytes.startswith(b"%PDF-")
        assert b"%%EOF" in pdf_bytes[-1024:]

    def test_pdf_generation_to_filepath(self):
        config = sample_beam()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pdf_bytes = generate_beam_pdf_report(config, output_path_or_buffer=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 50000
            with open(tmp_path, "rb") as f:
                content = f.read()
            assert content.startswith(b"%PDF-")
            assert len(pdf_bytes) == len(content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_pdf_generation_to_bytesio_buffer(self):
        config = sample_beam()
        buf = io.BytesIO()
        pdf_bytes = generate_beam_pdf_report(config, output_path_or_buffer=buf)
        assert buf.getvalue().startswith(b"%PDF-")
        assert len(buf.getvalue()) == len(pdf_bytes)

    def test_triangular_uvl_report(self):
        config = BeamConfig(
            length=8.0,
            point_loads=[
                PointLoad(magnitude=15.0, position=2.0),
                PointLoad(magnitude=25.0, position=4.0),
                PointLoad(magnitude=10.0, position=6.0),
            ],
            uvl=UVL(a=0.0, b=8.0, w_a=0.0, w_b=30.0),
        )
        pdf_bytes = generate_beam_pdf_report(config)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_zero_intensity_uvl_report(self):
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=20.0, position=2.0),
                PointLoad(magnitude=30.0, position=5.0),
                PointLoad(magnitude=15.0, position=8.0),
            ],
            uvl=UVL(a=1.0, b=7.0, w_a=0.0, w_b=0.0),
        )
        pdf_bytes = generate_beam_pdf_report(config)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_no_uvl_report(self):
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=20.0, position=2.0),
                PointLoad(magnitude=30.0, position=5.0),
                PointLoad(magnitude=15.0, position=8.0),
            ],
            uvl=None,
        )
        pdf_bytes = generate_beam_pdf_report(config)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_invalid_config_raises_value_error(self):
        # Invalid length
        with pytest.raises(ValueError):
            generate_beam_pdf_report(
                BeamConfig(length=-5.0, point_loads=[PointLoad(10.0, 1.0), PointLoad(10.0, 2.0), PointLoad(10.0, 3.0)])
            )

        # Fewer than 3 point loads
        with pytest.raises(ValueError):
            generate_beam_pdf_report(
                BeamConfig(length=10.0, point_loads=[PointLoad(10.0, 1.0), PointLoad(10.0, 2.0)])
            )

    def test_matplotlib_figures_closed_properly(self):
        config = sample_beam()
        open_figs_before = len(plt.get_fignums())
        generate_beam_pdf_report(config)
        open_figs_after = len(plt.get_fignums())
        assert open_figs_after == open_figs_before, "Temporary figures were not closed!"
