"""Entry point for `python -m memory`."""

import uvicorn

from riven_memory import app

uvicorn.run(app, host="0.0.0.0", port=8030, reload=True)
