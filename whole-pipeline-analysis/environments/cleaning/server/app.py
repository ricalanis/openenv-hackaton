"""
FastAPI application for the DataSage Cleaning Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from models import CleaningAction, CleaningObservation
from .cleaning_environment import CleaningEnvironment


app = create_app(
    CleaningEnvironment,
    CleaningAction,
    CleaningObservation,
    env_name="datasage_cleaning",
    max_concurrent_envs=4,
)


from fastapi import Request


@app.post("/reset-with-seed")
async def reset_with_seed(request: Request):
    """Reset environment with a specific seed for reproducible state."""
    body = await request.json()
    seed = body.get("seed")
    domain = body.get("domain")
    env = CleaningEnvironment()
    obs = env.reset(seed=seed, domain=domain)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
