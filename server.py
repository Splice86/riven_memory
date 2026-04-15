"""Entry point for `python server.py` or `uvicorn __init__:app`."""

import os
import sys

import uvicorn

# Ensure project layout is importable
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(1, _src)

from config import get
from __init__ import app


def run():
    host = get('api.host', '0.0.0.0')
    port = get('api.port', 8030)
    uvicorn.run("__init__:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
