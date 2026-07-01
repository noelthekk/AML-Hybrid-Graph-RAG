import subprocess
import sys
from pathlib import Path

hf = Path(__file__).parent / ".venv/Scripts/hf.exe"
subprocess.run([str(hf)] + sys.argv[1:])
