"""Configuration for the DataSage demo."""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# --- Environment URLs (HuggingFace Spaces) ---
SPACE_URLS = {
    "cleaning": "https://ricalanis-datasage-cleaning.hf.space",
    "enrichment": "https://ricalanis-datasage-enrichment.hf.space",
    "answering": "https://ricalanis-datasage-answering.hf.space",
}

# --- Model Registry ---
MODELS = {
    "qwen3-8b": {
        "display_name": "Qwen3-8B",
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-8B",
        "hf_provider": "fireworks-ai",
        "description": "Qwen3-8B via HuggingFace Inference API (fireworks-ai)",
        "color": "#7C3AED",
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o-mini",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "description": "OpenAI GPT-4o-mini - strong general-purpose model",
        "color": "#10A37F",
    },
    "datasage-cleaning": {
        "display_name": "DataSage Cleaning",
        "provider": "huggingface-lora",
        "model_id": "ricalanis/cleaning-grpo",
        "description": "GRPO fine-tuned LoRA on Qwen2.5-3B for data cleaning (requires GPU)",
        "color": "#F59E0B",
    },
    "datasage-enrichment": {
        "display_name": "DataSage Enrichment",
        "provider": "huggingface-lora",
        "model_id": "ricalanis/enrichment-grpo",
        "description": "GRPO fine-tuned LoRA on Qwen2.5-3B for data enrichment (requires GPU)",
        "color": "#EF4444",
    },
    "datasage-answering": {
        "display_name": "DataSage Answering",
        "provider": "huggingface-lora",
        "model_id": "ricalanis/answering-grpo",
        "description": "GRPO fine-tuned LoRA on Qwen2.5-3B for persona-aware Q&A (requires GPU)",
        "color": "#3B82F6",
    },
}

# --- Model groupings for comparison ---
MODEL_GROUPS = {
    "cleaning": ["qwen3-8b", "gpt-4o-mini", "datasage-cleaning"],
    "enrichment": ["qwen3-8b", "gpt-4o-mini", "datasage-enrichment"],
    "answering": ["qwen3-8b", "gpt-4o-mini", "datasage-answering"],
}

# --- Domain configuration ---
DOMAINS = ["hr", "sales", "pm", "it_ops"]
DOMAIN_DISPLAY = {
    "hr": "HR & People",
    "sales": "Sales & Revenue",
    "pm": "Project Management",
    "it_ops": "IT Operations",
}

PERSONAS = ["executive", "manager", "ic"]
PERSONA_DISPLAY = {
    "executive": "Executive (C-Suite)",
    "manager": "Manager (Ops Lead)",
    "ic": "Individual Contributor",
}

# --- API Keys ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Benchmark settings ---
DEFAULT_EPISODES = 5
MAX_CLEANING_STEPS = 15
MAX_ENRICHMENT_STEPS = 12
