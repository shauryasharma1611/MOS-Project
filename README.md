# Beam Analysis and Shear Force & Bending Moment Diagram Generator

> **Phase 1 to 6 Complete — Ready for College Demonstration & Project Submission**
> Run `pytest tests/ -v` (142 tests passing) or `streamlit run app.py` for the live dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Quick Start — Calculation Engine](#quick-start--calculation-engine)
5. [Quick Start — Generating Diagrams](#quick-start--generating-diagrams)
6. [Phase 2: Diagram Generation](#phase-2-diagram-generation)
   - [Three-Panel Figure](#three-panel-figure)
   - [Shear Force Diagram (SFD)](#shear-force-diagram-sfd)
   - [Bending Moment Diagram (BMD)](#bending-moment-diagram-bmd)
   - [Maximum Bending Moment Finder](#maximum-bending-moment-finder)
7. [Engineering Theory](#engineering-theory)
   - [Beam Model](#beam-model)
   - [Sign Convention](#sign-convention)
   - [UVL Resultant (analytical)](#uvl-resultant-analytical)
   - [Support Reactions](#support-reactions)
   - [Shear Force V(x)](#shear-force-vx)
   - [Bending Moment M(x)](#bending-moment-mx)
8. [API Reference](#api-reference)
9. [Sample Configuration](#sample-configuration)
10. [Running the Tests](#running-the-tests)
11. [Validation Checklist](#validation-checklist)

---

## Project Overview

This project provides a **pure-Python calculation engine** for analysing a
**simply supported beam** subject to:

- Any number of concentrated **point loads** (≥ 3).
- A single **Uniformly Varying Load (UVL)** which may be triangular,
  trapezoidal, or uniform, applied over part or the full span.

The engine computes:

- Equivalent resultant and centroid of the UVL (analytically).
- Left (pin) and right (roller) vertical support **reactions** via static
  equilibrium.
- Internal **shear force V(x)** at any section.
- Internal **bending moment M(x)** at any section.

---

## Directory Structure

```
MOS Project/
├── app.py                     ← Streamlit interactive dashboard (Phase 3)
├── src/
│   ├── __init__.py
│   ├── beam_analysis.py       ← calculation engine (Phase 1)
│   ├── beam_plotter.py        ← SFD / BMD diagram generation (Phase 2)
│   ├── beam_validation.py     ← independent static equilibrium & verification (Phase 4)
│   └── beam_report.py         ← PDF engineering report generation (Phase 5)
├── tests/
│   ├── __init__.py
│   ├── test_beam_analysis.py  ← Phase 1 pytest suite (42 tests)
│   ├── test_beam_plotter.py   ← Phase 2 pytest suite (59 tests)
│   ├── test_app.py            ← Phase 3 pytest suite (14 tests)
│   ├── test_beam_validation.py← Phase 4 verification suite (16 tests)
│   └── test_beam_report.py    ← Phase 5 report tests (8 tests)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick Start — Calculation Engine

```python
from src.beam_analysis import (
    BeamConfig, PointLoad, UVL,
    validate_beam, calculate_reactions,
    shear_force, bending_moment, sample_beam,
)

# Use the built-in sample or define your own
config = sample_beam()
validate_beam(config)

R_A, R_B = calculate_reactions(config)
print(f"R_A = {R_A:.4f} kN  |  R_B = {R_B:.4f} kN")

# Evaluate at x = 4 m
x = 4.0
print(f"V({x}) = {shear_force(config, x):.4f} kN")
print(f"M({x}) = {bending_moment(config, x):.4f} kN·m")
```

---

## Quick Start — Generating Diagrams

```python
from src.beam_analysis import sample_beam
from src.beam_plotter import plot_beam_diagrams

config = sample_beam()

# Display interactively
fig, (ax_load, ax_sfd, ax_bmd) = plot_beam_diagrams(config, show=True)

# Or save to file without showing a window
fig, axes = plot_beam_diagrams(config, show=False)
fig.savefig("beam_diagrams.png", dpi=150, bbox_inches="tight")
```

---

## Quick Start — Launching the Streamlit Dashboard (Phase 3)

Launch the interactive studio in your browser:

```bash
streamlit run app.py
```

Features in the dashboard:
- **Interactive Beam Controls**: Dynamic length, add/remove point loads (≥ 3), custom load positions and magnitudes.
- **UVL Configuration**: Full or partial span triangular and trapezoidal distributed loads.
- **Live Input Validation**: Guardrails checking length > 0, point load limits [0, L], and UVL boundaries.
- **Key Structural Metrics**: Real-time reaction indicators ($R_A, R_B$) and maximum sagging/hogging moments with exact positions.
- **Publication-Ready Figures**: High-resolution Matplotlib 3-panel diagram (Loading, SFD, BMD).
- **Interactive Section Probe**: Query shear force ($V^-, V^+$) and bending moment at any section $x$.
- **Exporting Options**:
  - **📄 PDF Engineering Report**: Publication-grade PDF containing executive summary, step-by-step reaction equilibrium equations, equilibrium residual audit, embedded diagrams, critical extrema table, numerical stations table, and scope limitations.
  - **💾 PNG Diagram Export**: Download high-resolution 3-panel diagram.
  - **📥 CSV Data Export**: Tabular station data ($x$, $V(x^-)$, $V(x^+)$, $M(x)$).

---

## College Demonstration & Live Evaluation Workflow (Phase 6)

Follow this 9-step guide for an evaluation or demonstration:

1. **Environment Setup**:
   ```bash
   # Activate project virtual environment
   source .venv_p2/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Automated Test Suite**:
   ```bash
   pytest tests/ -v
   # All 142 unit, calculus, benchmark, and regression tests pass cleanly.
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open the displayed URL (typically `http://localhost:8501`) in your browser.

4. **Examine Default Sample Configuration**:
   - Beam span: $L = 10.0\text{ m}$.
   - 3 Concentrated loads: $20\text{ kN}$ at $2.0\text{ m}$, $35\text{ kN}$ at $5.0\text{ m}$, $15\text{ kN}$ at $8.0\text{ m}$.
   - Trapezoidal UVL: $a = 3.0\text{ m}$ to $b = 7.0\text{ m}$, $w_a = 0.0\text{ kN/m}$, $w_b = 30.0\text{ kN/m}$.

5. **Verify Support Reactions & Equilibrium**:
   - Observe real-time metric cards for $R_A = 60.000\text{ kN}$ and $R_B = 70.000\text{ kN}$.
   - Note the **Equilibrium Verified** green status banner verifying $\Sigma F_y = 0$ and $\Sigma M_A = 0$ ($< 10^{-9}$ tolerance).

6. **Inspect SFD and BMD Diagrams**:
   - Panel A: Beam diagram with upward reaction arrows ($R_A, R_B$), downward load arrows, and shaded blue UVL polygon.
   - Panel B: Shear Force Diagram showing vertical jump steps ($\Delta V = -P$) at $x = 2.0\text{ m}, 5.0\text{ m}, 8.0\text{ m}$.
   - Panel C: Bending Moment Diagram showing smooth cubic curve under the UVL with callout marker at max sagging moment ($M_{\max} = 175.000\text{ kN}\cdot\text{m}$).

7. **Test Interactive Controls**:
   - In sidebar, click **➕ Add Load** to add a 4th point load, or adjust the beam span slider.
   - Observe live diagram and metric updates without full-page reloads.
   - Click **🔄 Reset to Sample Beam** to restore initial textbook values.

8. **Query Specific Section**:
   - Expand the **📋 Detailed Numerical Table & Section Evaluation** section.
   - Move the section slider to $x = 2.0\text{ m}$ to see both left-hand $V(2.0^-)$ and right-hand $V(2.0^+)$ shear values.

9. **Export Deliverables**:
   - Click **💾 Download Diagram (PNG)** for an immediate high-resolution diagram image.
   - Click **📄 Download Engineering Report (PDF)** to download a complete, formatted calculation report.
   - Click **📥 Download Data (CSV)** for tabular spreadsheets.

---

## Phase 2: Diagram Generation

### Three-Panel Figure

`plot_beam_diagrams(config, num_points=1001, show=True)` produces a
Matplotlib figure with three vertically stacked panels sharing a common
horizontal axis (position in metres):

| Panel | Title | Content |
|---|---|---|
| A (top) | Beam Loading Diagram | Beam centreline, pin/roller supports, point-load arrows (kN), UVL polygon (kN/m), reaction labels |
| B (middle) | Shear Force Diagram (SFD) | V(x) vs x with correct vertical jumps at point loads |
| C (bottom) | Bending Moment Diagram (BMD) | M(x) vs x with sagging/hogging fill; max/min labels |

The function returns `(fig, (ax_load, ax_sfd, ax_bmd))` so each axis can
be reused by a Streamlit dashboard or further customised.

### Shear Force Diagram (SFD)

- Evaluated using the Phase 1 `shear_force(config, x)` engine function.
- **Discontinuities** at each concentrated load are rendered as vertical
  line segments.  NaN sentinels are inserted into the array so Matplotlib
  does **not** draw a spurious diagonal through the jump.
- `num_smooth = max(200, num_points // 2)` points are sampled per smooth
  interval between critical x-values.
- Jump magnitude `ΔV = P` is annotated at each load position.

### Bending Moment Diagram (BMD)

- Evaluated using the Phase 1 `bending_moment(config, x)` engine function.
- Continuous everywhere — no NaN gaps.
- Critical x-values (point-load positions, UVL boundaries) are always
  included in the x-array so kinks are captured exactly.
- Sagging regions (M > 0) are shaded blue; hogging regions (M < 0) red.
- The maximum positive and minimum negative moments are located by
  `find_bending_moment_extrema()` and annotated with value and position.

### Maximum Bending Moment Finder

`find_bending_moment_extrema(config, num_sample=2001)` returns a
dictionary with:

| Key | Type | Description |
|---|---|---|
| `max_positive_value` | float | Largest positive M (kN·m) |
| `max_positive_pos` | float | Position of max M (m) |
| `min_negative_value` | float | Most negative M (kN·m) |
| `min_negative_pos` | float | Position of min M (m) |
| `candidates` | list | All (x, M) pairs evaluated |
| `note` | str | Numerical accuracy statement |

Zero-shear positions (potential moment extrema) are found by **bisection**
with convergence < 1 × 10⁻¹⁰ m.  A dense uniform grid of `num_sample`
points is used as a safety net; its positional accuracy is `L / num_sample`.

---

## Engineering Theory

### Beam Model

```
      P1      P2         P3
       ↓       ↓          ↓
───────●───────●──────────●──────────
△                                   ○
x=0                                x=L
(Pin)                          (Roller)
```

- The beam has length **L > 0**.
- The **pin** at `x = 0` provides a vertical reaction **R_A** (upward positive).
- The **roller** at `x = L` provides a vertical reaction **R_B** (upward positive).
- The structure is **statically determinate** (2 equations, 2 unknowns).

---

### Sign Convention

| Quantity | Positive direction |
|---|---|
| Applied point load `P` | **Downward** |
| UVL intensity `w(x)` | **Downward** |
| Reaction `R_A`, `R_B` | **Upward** |
| Shear force `V(x)` | Left free body resultant **upward** (sagging tendency) |
| Bending moment `M(x)` | **Sagging** (tension at bottom fibre) |

This matches the **structural / civil engineering** convention used in most
British and international textbooks.

---

### UVL Resultant (analytical)

A UVL on `[a, b]` has intensity:

```
w(x) = w_a + (w_b − w_a)/(b − a) · (x − a)
```

**Resultant force** (area of the load diagram):

```
R = ∫_a^b w(x) dx = (w_a + w_b)/2 · (b − a)          … (1)
```

**Line of action** (centroid of the trapezoidal/triangular area):

```
x̄ = [w_a·(2a + b) + w_b·(a + 2b)] / [3·(w_a + w_b)]  … (2)
```

Special cases:
- **Triangular (ascending):** `w_a = 0` → `x̄ = a + 2(b−a)/3`
- **Triangular (descending):** `w_b = 0` → `x̄ = a + (b−a)/3`
- **Uniform (UDL):** `w_a = w_b` → `x̄ = (a + b)/2`

---

### Support Reactions

Taking moments about the left support (**A**) — anti-clockwise positive:

```
ΣM_A = 0:  R_B · L = Σ(P_i · x_i) + R_uvl · x̄   … (3)
ΣFy  = 0:  R_A + R_B = Σ P_i + R_uvl             … (4)
```

Solving sequentially:

```
R_B = [Σ(P_i · x_i) + R_uvl · x̄] / L
R_A = Σ P_i + R_uvl − R_B
```

---

### Shear Force V(x)

Using the **left-hand free body diagram** at section `x`:

```
V(x) = R_A − Σ{ P_i : pos_i < x } − F_uvl(x)
```

where:

```
F_uvl(x) = ∫_a^{min(x,b)} w(t) dt   (UVL force on [0, x])
```

**Piecewise evaluation of F_uvl(x):**

| Region | F_uvl(x) |
|---|---|
| x ≤ a | 0 |
| a < x ≤ b | `(w_a + w(x)) / 2 · (x − a)` where `w(x)` is the intensity at x |
| x > b | `(w_a + w_b) / 2 · (b − a)` = full UVL resultant |

**Jump at a point load** (at position `p`):

```
V(p⁺) − V(p⁻) = −P     (downward P reduces shear)
```

---

### Bending Moment M(x)

Using the **left-hand free body diagram**:

```
M(x) = R_A · x − Σ{ P_i · (x − pos_i) : pos_i ≤ x } − M_uvl(x)
```

where the UVL moment contribution is:

```
M_uvl(x) = ∫_a^{min(x,b)} w(t) · (x − t) dt
```

Evaluated analytically with `s = min(x,b) − a` and `slope = (w_b − w_a)/(b − a)`:

```
M_uvl(x) = w_a·(x−a)·s − w_a·s²/2 + slope·(x−a)·s²/2 − slope·s³/3
```

**Boundary conditions:**

```
M(0) = 0    (pin — free to rotate)
M(L) = 0    (roller — free to rotate)
```

These are automatically satisfied because the reactions are derived from
equilibrium. They are verified in the test suite.

---

## API Reference

### Data Classes

| Class | Fields | Description |
|---|---|---|
| `PointLoad` | `magnitude` (kN), `position` (m) | Concentrated vertical load |
| `UVL` | `a`, `b` (m), `w_a`, `w_b` (kN/m) | Uniformly varying load |
| `BeamConfig` | `length` (m), `point_loads`, `uvl` | Complete beam definition |

### Functions

| Function | Returns | Description |
|---|---|---|
| `validate_beam(config)` | `None` / raises `ValueError` | Input validation |
| `uvl_resultant(uvl)` | `(R, x_bar)` | UVL resultant and centroid |
| `calculate_reactions(config)` | `(R_A, R_B)` kN | Support reactions |
| `total_resultant_load(config)` | `float` kN | Sum of all downward loads |
| `shear_force(config, x)` | `float` kN | Internal shear at position x |
| `bending_moment(config, x)` | `float` kN·m | Internal moment at position x |
| `sample_beam()` | `BeamConfig` | Built-in example configuration |

---

## Sample Configuration

```python
BeamConfig(
    length = 10.0,                         # m
    point_loads = [
        PointLoad(magnitude=20.0, position=2.0),   # 20 kN at 2 m
        PointLoad(magnitude=35.0, position=5.0),   # 35 kN at 5 m
        PointLoad(magnitude=15.0, position=8.0),   # 15 kN at 8 m
    ],
    uvl = UVL(a=3.0, b=7.0, w_a=0.0, w_b=30.0),  # 0→30 kN/m over [3,7]
)
```

**UVL properties:**

- Type: **Triangular (ascending)** — `w_a = 0`.
- Span: 4 m  →  `R_uvl = (0 + 30)/2 × 4 = 60 kN`
- Centroid: `x̄ = 3 + 2/3 × 4 = 5.667 m`

**Total load:** 20 + 35 + 15 + 60 = **130 kN**

---

## Running the Tests

```bash
# Activate the virtual environment first
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Run both Phase 1 and Phase 2 suites together
pytest tests/ -v

# Full test suite (Phases 1–6)
pytest tests/ -v

# Phase 5 engineering report tests only
pytest tests/test_beam_report.py -v

# Phase 4 independent verification only
pytest tests/test_beam_validation.py -v

# Phase 3 dashboard tests only
pytest tests/test_app.py -v
```

The automated test suites force the Matplotlib `Agg` backend at import time so no
GUI window is opened during testing.

Expected output: **all 142 tests in tests/ PASSED**.

---

## Validation Checklist

| Check | Method |
|---|---|
| `R_A + R_B = W_total` | `TestVerticalEquilibrium` |
| `R_B · L = ΣM_loads` | `TestMomentEquilibrium` |
| `M(0) = 0` | `TestBoundaryConditions` |
| `M(L) = 0` | `TestBoundaryConditions` |
| Shear jump = −P at point load | `TestShearJump` |
| M continuous at point load | `TestShearJump::test_bending_moment_continuous_at_load` |
| Triangular UVL centroid | `TestTriangularUVL` |
| Trapezoidal UVL centroid | `TestTrapezoidalUVL` |
| Invalid inputs raise ValueError | `TestInvalidInputs` |

---

*Phase 2 will add Matplotlib-based SFD/BMD diagram generation.*
