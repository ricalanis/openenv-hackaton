"""
FastAPI application for the DataSage Answering Environment.

This module creates an HTTP server that exposes the AnsweringEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Import from local models.py (PYTHONPATH includes /app/env in Docker)
from models import AnsweringAction, AnsweringObservation

from .answering_environment import AnsweringEnvironment


# Create the app with web interface
app = create_app(
    AnsweringEnvironment,
    AnsweringAction,
    AnsweringObservation,
    env_name="datasage_answering",
    max_concurrent_envs=4,
)


from fastapi import Request


@app.post("/reset-with-seed")
async def reset_with_seed(request: Request):
    """Reset environment with a specific seed for reproducible state."""
    body = await request.json()
    seed = body.get("seed")
    domain = body.get("domain")
    env = AnsweringEnvironment()
    obs = env.reset(seed=seed, domain=domain)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
