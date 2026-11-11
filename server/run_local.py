"""Start the local test console on http://127.0.0.1:8000 (localhost only).

  %USERPROFILE%\\venvs\\bronchnav\\Scripts\\python.exe server\\run_local.py

Needs the venv with `pip install -e "core[dev,render]" fastapi uvicorn`.
"""

import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), log_level="info")
