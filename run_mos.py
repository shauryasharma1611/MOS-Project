#!/usr/bin/env python3
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv_p2" / "bin" / "python"

subprocess.run(
    [str(PYTHON), "-m", "streamlit", "run", str(ROOT / "app.py")],
    cwd=ROOT
)
