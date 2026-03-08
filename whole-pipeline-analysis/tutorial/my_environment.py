"""
Personal Assistant World Modeling Environment — Server Logic.

Simulates realistic personal task handling: scheduling conflicts,
email triage, dinner/drive planning, and task delegation.

Hackathon track: Statement 3.2 (Personalized Tasks)
"""

import random
from core.environment import Environment

# Import models — adjust path if running inside the OpenEnv scaffold
try:
    from my_env.models import MyAction, MyObservation, MyState
except ImportError:
    from models import MyAction, MyObservation, MyState


# ── Scenario templates ──────────────────────────────────────────────
SCENARIOS = [
    {
        "description": (
            "You have a dinner reservation at 7:00 PM with your partner, but "
            "your boss just scheduled a mandatory team sync at 6:30 PM that "
            "could run until 7:30 PM. Handle the conflict professionally "
            "without upsetting either party."
        ),
        "calendar": [
            {"event": "Team standup", "time": "9:00 AM", "duration": "30m"},
            {"event": "Focus time", "time": "10:00 AM", "duration": "2h"},
            {"event": "Lunch", "time": "12:00 PM", "duration": "1h"},
            {"event": "Boss sync (MANDATORY)", "time": "6:30 PM", "duration": "1h"},
            {"event": "Dinner with partner", "time": "7:00 PM", "duration": "2h"},
        ],
        "inbox": [
            {"from": "boss@work.com", "subject": "Urgent: Team sync moved to 6:30 PM",
             "body": "Need everyone on this call, no exceptions."},
            {"from": "partner@home.com", "subject": "Re: Dinner tonight",
             "body": "Can't wait! I already made the reservation."},
        ],
        "resolution_keys": ["reschedule_meeting", "send_message_partner"],
        "total_conflicts": 1,
    },
    {
        "description": (
            "You received 5 emails this morning. Two are urgent (client "
            "escalation and a deadline reminder), two are informational, and "
            "one is spam. Triage them correctly and respond to the urgent ones."
        ),
        "calendar": [
            {"event": "Morning standup", "time": "9:30 AM", "duration": "15m"},
            {"event": "Client call", "time": "2:00 PM", "duration": "1h"},
        ],
        "inbox": [
            {"from": "client@acme.com", "subject": "URGENT: Production is down",
             "body": "Our dashboard has been unresponsive since 6 AM. Need help ASAP."},
            {"from": "pm@work.com", "subject": "Reminder: Q3 report due by EOD",
             "body": "Please submit your section of the Q3 report today."},
            {"from": "newsletter@tech.io", "subject": "This week in AI",
             "body": "Top stories from the AI world..."},
            {"from": "hr@work.com", "subject": "Benefits enrollment open",
             "body": "Open enrollment period is now through March 15."},
            {"from": "spam@deals.biz", "subject": "You won a free iPhone!!!",
             "body": "Click here to claim your prize..."},
        ],
        "resolution_keys": ["reply_client", "reply_pm"],
        "total_conflicts": 2,
    },
]

AVAILABLE_TOOLS = [
    "check_calendar",
    "send_email",
    "send_message",
    "reschedule_meeting",
    "delegate_task",
    "check_inbox",
]


class MyEnvironment(Environment):
    """
    Personal assistant environment that tests whether an LLM can:
    - Gather context (calendar, inbox) before acting
    - Resolve scheduling conflicts diplomatically
    - Triage and respond to emails appropriately
    - Delegate tasks when appropriate

    Reward signal:
      +0.1  for information gathering (check_calendar, check_inbox)
      +0.3  for appropriate communication
      +0.5  for partially resolving the scenario
      +1.0  for fully resolving the scenario
      -0.1  for irrelevant or harmful actions
      -0.2  for repeated identical actions
    """

    def __init__(self):
        super().__init__()
        self.state: MyState | None = None
        self.scenario: dict | None = None
        self._checked_calendar = False
        self._checked_inbox = False
        self._resolved = set()

    async def reset(self):
        self.scenario = random.choice(SCENARIOS)
        self.state = MyState(
            step_count=0,
            max_steps=10,
            task_description=self.scenario["description"],
            history=[],
            conflicts_resolved=0,
            total_conflicts=self.scenario["total_conflicts"],
        )
        self._checked_calendar = False
        self._checked_inbox = False
        self._resolved = set()

        obs = MyObservation(
            result=self.scenario["description"],
            available_tools=AVAILABLE_TOOLS,
            task_completed=False,
            pending_conflicts=self.scenario["total_conflicts"],
        )
        return obs, 0.0

    async def step(self, action: MyAction):
        self.state.step_count += 1
        self.state.history.append(action.model_dump())

        reward = 0.0
        done = False
        result = ""

        # Penalize repeated identical actions
        if len(self.state.history) >= 2:
            prev = self.state.history[-2]
            if prev == action.model_dump():
                reward -= 0.2
                result = "You already did that. Try something different."
                return self._make_result(result, reward, done)

        # ── Tool dispatch ────────────────────────────────────────
        tool = action.tool_name.lower().strip()

        if tool == "check_calendar":
            cal = self.scenario["calendar"]
            lines = [f"  {e['time']} — {e['event']} ({e['duration']})" for e in cal]
            result = "Your calendar today:\n" + "\n".join(lines)
            reward = 0.1 if not self._checked_calendar else 0.0
            self._checked_calendar = True

        elif tool == "check_inbox":
            inbox = self.scenario["inbox"]
            lines = [f"  From: {e['from']} | Subject: {e['subject']}" for e in inbox]
            result = "Inbox:\n" + "\n".join(lines)
            reward = 0.1 if not self._checked_inbox else 0.0
            self._checked_inbox = True

        elif tool == "send_email":
            to = action.tool_args.get("to", "").lower()
            body = action.tool_args.get("body", "").lower()
            subject = action.tool_args.get("subject", "")
            result = f"Email sent to {to}. Subject: {subject}"
            self.state.emails_sent += 1

            # Check if this resolves a key
            if "client" in to and any(w in body for w in ["looking into", "investigating", "on it", "fix"]):
                reward = 0.5
                self._resolved.add("reply_client")
            elif ("pm" in to or "manager" in to) and any(w in body for w in ["submit", "report", "send", "done"]):
                reward = 0.3
                self._resolved.add("reply_pm")
            elif "boss" in to and "reschedule" in body:
                reward = 0.5
                self._resolved.add("reschedule_meeting")
            else:
                reward = 0.1

        elif tool == "send_message":
            to = action.tool_args.get("to", "").lower()
            body = action.tool_args.get("body", "").lower()
            result = f"Message sent to {to}."

            if "partner" in to and any(w in body for w in ["late", "delay", "reschedule", "sorry"]):
                reward = 0.3
                self._resolved.add("send_message_partner")
            else:
                reward = 0.1

        elif tool == "reschedule_meeting":
            new_time = action.tool_args.get("new_time", "earlier")
            result = f"Meeting rescheduled to {new_time}. Calendar updated."
            reward = 0.5
            self._resolved.add("reschedule_meeting")

        elif tool == "delegate_task":
            to = action.tool_args.get("to", "teammate")
            task = action.tool_args.get("task", "unspecified")
            result = f"Task '{task}' delegated to {to}."
            reward = 0.2

        else:
            result = f"Unknown tool '{action.tool_name}'. Available: {', '.join(AVAILABLE_TOOLS)}"
            reward = -0.1

        # ── Check if scenario is fully resolved ──────────────────
        keys = set(self.scenario["resolution_keys"])
        self.state.conflicts_resolved = len(self._resolved & keys)
        if keys.issubset(self._resolved):
            done = True
            reward += 1.0  # bonus for full resolution
            result += "\n✅ All conflicts resolved! Great job."

        # End on max steps
        if self.state.step_count >= self.state.max_steps:
            done = True

        self.state.score += reward
        return self._make_result(result, reward, done)

    def _make_result(self, result, reward, done):
        pending = self.state.total_conflicts - self.state.conflicts_resolved
        obs = MyObservation(
            result=result,
            available_tools=AVAILABLE_TOOLS,
            task_completed=done,
            pending_conflicts=pending,
        )
        return obs, reward, done

    async def get_state(self):
        return self.state
