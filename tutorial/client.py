"""Client for the Personal Assistant environment."""

from core.http_env_client import HTTPEnvClient
from my_env.models import MyAction, MyObservation, MyState


class MyEnv(HTTPEnvClient[MyAction, MyObservation, MyState]):
    """
    HTTP client for the personal assistant world modeling environment.

    Usage:
        # Connect to a running Space
        env = MyEnv(base_url="https://YOUR_USERNAME-my-env.hf.space")

        # Or load from Hub (pulls Docker container locally)
        env = MyEnv.from_hub("YOUR_USERNAME/my-env")

        # Synchronous usage
        with env.sync() as client:
            obs, reward = client.reset()
            result = client.step(MyAction(
                tool_name="check_calendar",
                tool_args={}
            ))
    """

    pass
