"""Entry point for `python server.py` or `uvicorn __init__:app`."""

import uvicorn
import riven_memory_config as cfg

from __init__ import app


def run():
    host = cfg.get('api.host', '0.0.0.0')
    port = cfg.get('api.port', 8030)
    uvicorn.run(app, host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
