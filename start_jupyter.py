"""Start JupyterLab for the AML RAG implementation notebooks.

Usage (from the implementation/ folder):
    python start_jupyter.py
    .venv/Scripts/python start_jupyter.py   # Windows — guarantees venv Python
    .venv/bin/python start_jupyter.py       # Unix
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Locate the venv jupyter-lab executable
if sys.platform == "win32":
    jupyter_lab = HERE / ".venv" / "Scripts" / "jupyter-lab.exe"
else:
    jupyter_lab = HERE / ".venv" / "bin" / "jupyter-lab"

if not jupyter_lab.exists():
    print(f"Error: {jupyter_lab} not found. Run setup first — see README.md.")
    sys.exit(1)

print("Starting JupyterLab at http://localhost:8888")
print("Token: aml_rag_2026")
print("Press Ctrl+C to stop.")
print()

subprocess.run([
    str(jupyter_lab),
    "--port", "8888",
    "--IdentityProvider.token", "aml_rag_2026",
    "--ip", "127.0.0.1",
    "--notebook-dir", str(HERE),
])


