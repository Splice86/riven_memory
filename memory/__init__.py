"""Riven Memory API server package.

Run with:
    uvicorn memory:app --reload --port 8030
    python -m memory
"""

import os

from memory.api import app

# Config path resolution (works from project root when memory is a package)
_module_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_module_dir, "config.yaml")

__all__ = ["app"]
