"""FastAPI app for the FSDS cleaning environment."""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .cleaning_environment import FSDSCleaningEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.cleaning_environment import FSDSCleaningEnvironment

app = create_app(
    FSDSCleaningEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="fsds_cleaning_env",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
