"""Client for the FSDS cleaning environment."""

from openenv.core.mcp_client import MCPToolClient


class FSDSCleaningEnv(MCPToolClient):
    """Client wrapper for the FSDS cleaning environment.

    Usage:
        from fsds_cleaning_env import FSDSCleaningEnv

        with FSDSCleaningEnv(base_url="https://<space>.hf.space").sync() as env:
            env.reset(task_id="ecommerce_mobile")
            tools = env.list_tools()
            result = env.call_tool("get_task_brief")
    """

    pass
