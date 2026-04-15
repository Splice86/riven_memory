"""Riven Memory API - package root for uvicorn entrypoint.

Run with:
    uvicorn __init__:app --reload --port 8030
    python -m __main__
"""

import os
import sys

# Ensure the project root and src layout are importable (works both in-place and installed)
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)  # Root for database/ package
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(1, _src)  # 1 so root __init__ is still found first

from api import app

# Config path resolution (project root)
_root = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_root, "config.yaml")

__all__ = ["app"]
