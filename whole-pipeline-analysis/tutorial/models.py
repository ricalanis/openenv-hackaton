"""
Models for the Personal Assistant World Modeling Environment.

This environment simulates a personal assistant handling scheduling
conflicts, email triage, and task delegation — matching the OpenEnv
Hackathon "World Modeling: Personalized Tasks" problem statement (3.2).
"""

from pydantic import BaseModel, Field


class MyAction(BaseModel):
    """Action the agent takes each step."""

    tool_name: str = Field(
        description="Name of the tool to use: "
        "check_calendar, send_email, send_message, "
        "reschedule_meeting, delegate_task, check_inbox"
    )
    tool_args: dict = Field(
        default_factory=dict,
        description="Arguments for the tool (varies by tool)",
    )


class MyObservation(BaseModel):
    """What the environment returns after each step."""

    result: str = Field(description="Text description of what happened")
    available_tools: list[str] = Field(
        default_factory=lambda: [
            "check_calendar",
            "send_email",
            "send_message",
            "reschedule_meeting",
            "delegate_task",
            "check_inbox",
        ]
    )
    task_completed: bool = Field(default=False)
    pending_conflicts: int = Field(default=0)


class MyState(BaseModel):
    """Internal episode state tracked by the server."""

    step_count: int = 0
    max_steps: int = 10
    task_description: str = ""
    history: list[dict] = Field(default_factory=list)
    conflicts_resolved: int = 0
    total_conflicts: int = 1
    emails_sent: int = 0
    score: float = 0.0
