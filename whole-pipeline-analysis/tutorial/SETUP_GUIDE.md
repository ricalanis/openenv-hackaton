# OpenEnv Hackathon — Full Setup Guide

## Your Goal

Build a **World Modeling** environment (Statement 3) — either professional tasks
(tool/API interaction, enterprise workflows) or personalized tasks (personal
assistant, email triage, scheduling conflicts). Deploy it on HF Spaces using
OpenEnv 0.2.1, then show a minimal GRPO training script in Colab.

---

## Step 0 — Prerequisites

You already have a Hugging Face account. Make sure you can log in at
<https://huggingface.co>. You'll also need:

- **Python 3.10+** on your laptop (or use Colab for everything)
- **Docker Desktop** installed and running (for local testing)
- **Git** installed

---

## Step 1 — Create a Hugging Face Access Token

1. Go to <https://huggingface.co/settings/tokens>
2. Click **"Create new token"**
3. Name it `openenv-hackathon`, select **Write** permissions
4. Copy the token — you'll need it in Step 3

---

## Step 2 — Install OpenEnv & Scaffold Your Environment

```bash
# Install OpenEnv (includes the CLI)
pip install git+https://github.com/meta-pytorch/OpenEnv.git

# Log in to Hugging Face
huggingface-cli login
# Paste your token from Step 1 when prompted

# Scaffold a new environment
openenv init my_env

# You'll get this structure:
# my_env/
# ├── __init__.py
# ├── client.py          ← Client code (used by training script)
# ├── models.py          ← Action, Observation, State pydantic models
# ├── openenv.yaml       ← Environment metadata
# ├── pyproject.toml     ← Python dependencies
# ├── uv.lock
# ├── README.md
# └── server/
#     ├── __init__.py
#     ├── app.py          ← FastAPI entrypoint (usually untouched)
#     ├── my_environment.py  ← YOUR LOGIC GOES HERE
#     ├── requirements.txt
#     └── Dockerfile
```

---

## Step 3 — Build Your Environment Logic

The three files you'll edit most are:

### 3a. `models.py` — Define Actions, Observations, State

```python
from pydantic import BaseModel

class MyAction(BaseModel):
    """What the agent sends to the environment each step."""
    tool_name: str          # e.g. "send_email", "check_calendar"
    tool_args: dict = {}    # arguments for the tool

class MyObservation(BaseModel):
    """What the environment returns after each step."""
    result: str             # text description of what happened
    available_tools: list[str]
    task_completed: bool = False

class MyState(BaseModel):
    """Internal episode state."""
    step_count: int = 0
    task_description: str = ""
    history: list[dict] = []
```

### 3b. `server/my_environment.py` — Core Logic

```python
from core.environment import Environment
from my_env.models import MyAction, MyObservation, MyState

class MyEnvironment(Environment):
    """
    World Modeling environment: a personal assistant that must
    handle scheduling conflicts, email replies, and task delegation.
    """

    def __init__(self):
        super().__init__()
        self.state = None

    async def reset(self):
        """Initialize a new episode with a task scenario."""
        self.state = MyState(
            step_count=0,
            task_description=(
                "You have a dinner at 7pm but your boss just scheduled "
                "a mandatory meeting at 6:30pm. Handle the conflict."
            ),
            history=[],
        )
        return MyObservation(
            result=self.state.task_description,
            available_tools=["check_calendar", "send_email",
                             "reschedule_meeting", "send_message"],
            task_completed=False,
        ), 0.0  # initial reward

    async def step(self, action: MyAction):
        """Execute an action and return observation + reward."""
        self.state.step_count += 1
        self.state.history.append(action.model_dump())

        # ── YOUR REWARD LOGIC ──
        reward = 0.0
        done = False

        if action.tool_name == "check_calendar":
            result = "Calendar: Dinner 7pm, Meeting 6:30-7:30pm (mandatory)"
            reward = 0.1  # small reward for gathering info

        elif action.tool_name == "send_email":
            to = action.tool_args.get("to", "")
            body = action.tool_args.get("body", "")
            if "boss" in to.lower() and "reschedule" in body.lower():
                result = "Email sent to boss requesting meeting reschedule."
                reward = 0.5
            elif "dinner" in to.lower():
                result = "Message sent about running late to dinner."
                reward = 0.3
            else:
                result = f"Email sent to {to}."
                reward = 0.1

        elif action.tool_name == "reschedule_meeting":
            result = "Meeting rescheduled to 5:30pm. Conflict resolved!"
            reward = 1.0
            done = True

        elif action.tool_name == "send_message":
            result = f"Message sent: {action.tool_args.get('body', '')}"
            reward = 0.2

        else:
            result = f"Unknown tool: {action.tool_name}"
            reward = -0.1

        # End episode after 10 steps
        if self.state.step_count >= 10:
            done = True

        observation = MyObservation(
            result=result,
            available_tools=["check_calendar", "send_email",
                             "reschedule_meeting", "send_message"],
            task_completed=done,
        )
        return observation, reward, done

    async def state(self):
        return self.state
```

### 3c. `client.py` — Client the Training Script Uses

```python
from core.http_env_client import HTTPEnvClient
from my_env.models import MyAction, MyObservation, MyState

class MyEnv(HTTPEnvClient[MyAction, MyObservation, MyState]):
    """Client for the personal assistant environment."""
    pass
```

---

## Step 4 — Test Locally with Docker

```bash
cd my_env

# Build the Docker image
openenv build

# Run locally
docker run -d -p 8001:8000 openenv-my_env:latest

# Quick smoke test in Python
python -c "
from my_env.client import MyEnv
from my_env.models import MyAction

env = MyEnv(base_url='http://localhost:8001')
with env.sync() as client:
    obs, reward = client.reset()
    print('Initial:', obs.result)
    
    result = client.step(MyAction(
        tool_name='check_calendar', tool_args={}
    ))
    print('Step 1:', result.observation.result, 'reward:', result.reward)
    
    result = client.step(MyAction(
        tool_name='reschedule_meeting',
        tool_args={'new_time': '5:30pm'}
    ))
    print('Step 2:', result.observation.result, 'reward:', result.reward)
"
```

If you see observations and rewards printing, you're good!

---

## Step 5 — Deploy to Hugging Face Spaces

```bash
cd my_env

# This pushes your environment as a Docker Space
openenv push --repo-id YOUR_HF_USERNAME/my-env

# Or make it public (required for hackathon — repos must be public)
openenv push --repo-id YOUR_HF_USERNAME/my-env
```

After pushing, your environment will be live at:
```
https://YOUR_HF_USERNAME-my-env.hf.space
```

Verify it's running by visiting the URL — you should see the OpenEnv health
check page. Then test from Python:

```python
from my_env.client import MyEnv
env = MyEnv(base_url="https://YOUR_HF_USERNAME-my-env.hf.space")
```

---
    
## Step 6 — Minimal GRPO Training Script (Colab)

See the companion file `openenv_training_colab.py` — paste it cell-by-cell
into a Colab notebook. The key changes for YOUR environment:

1. Install your env client:
   ```
   !pip install git+https://huggingface.co/spaces/YOUR_USERNAME/my-env
   ```

2. Import your classes:
   ```python
   from my_env.client import MyEnv
   from my_env.models import MyAction
   env = MyEnv(base_url="https://YOUR_USERNAME-my-env.hf.space")
   ```

3. In the reward function, step through YOUR environment:
   ```python
   def env_reward_fn(completions, **kwargs):
       rewards = []
       for text in completions:
           env.reset()
           # Parse the model's output into your action format
           action = parse_model_output_to_action(text)
           result = env.step(action)
           rewards.append(float(result.reward))
       return rewards
   ```

---

## Step 7 — Submission Checklist

Before submitting, confirm:

- [ ] GitHub repo is **public**
- [ ] Environment uses **OpenEnv >= 0.2.1**
- [ ] Environment is deployed on **HF Spaces** and accessible
- [ ] Training script runs in **Colab** (Unsloth or HF TRL)
- [ ] Training shows **reward improvement** (screenshot reward curves)
- [ ] **1-minute demo video** uploaded to YouTube
- [ ] Submitted via the hackathon submission form

---

## Quick Reference

| What | Command / URL |
|------|---------------|
| Create HF token | https://huggingface.co/settings/tokens |
| Scaffold env | `openenv init my_env` |
| Build Docker | `openenv build` |
| Test locally | `docker run -d -p 8001:8000 openenv-my_env:latest` |
| Deploy to Spaces | `openenv push --repo-id USER/my-env` |
| Your Space URL | `https://USER-my-env.hf.space` |
| Unsloth notebooks | https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks |
| TRL + OpenEnv docs | https://huggingface.co/docs/trl/en/openenv |
| OpenEnv GitHub | https://github.com/meta-pytorch/OpenEnv |
| Hackathon Discord | https://discord.gg/VBcf6VtfY6 |
