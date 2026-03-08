"""Model provider abstraction layer for multi-model comparison."""

import json
import os
import re
from abc import ABC, abstractmethod

from .config import MODELS, OPENAI_API_KEY, HF_TOKEN


class ModelProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_id: str, display_name: str):
        self.model_id = model_id
        self.display_name = display_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
        pass

    def generate_action(self, observation: dict, task: str) -> dict:
        system_prompt = self._get_system_prompt(task)
        user_prompt = self._format_observation(observation, task)
        response = self.generate(user_prompt, system_prompt, max_tokens=300)
        return self._parse_action(response, task)

    def _get_system_prompt(self, task: str) -> str:
        prompts = {
            "cleaning": (
                "You are a data quality agent. Given a data observation with quality issues, "
                "output ONLY a JSON action. No explanation. Format: "
                '{"operation": "<op>", "column": "<col>", "value": "<val>", "params": {}}. '
                "Operations: fill_null, fix_type, remove_duplicate, standardize, trim, correct_typo. "
                "For fill_null use value 'median' for numeric, 'mode' for categorical."
            ),
            "enrichment": (
                "You are a data enrichment agent. Output ONLY a JSON action. Format: "
                '{"operation": "<op>", "field_name": "<name>", "params": {}}. '
                "Operations: add_field, lookup, compute_derived, add_category. "
                "Pick from the possible_enrichments list. Use a different field each time."
            ),
            "answering": (
                "You are a data analyst answering questions for a specific persona. "
                "Adapt language: Executive=strategic/financial, Manager=operational/actionable, "
                "IC=plain/personal. Cite specific data columns. Be data-grounded."
            ),
        }
        return prompts.get(task, "You are a helpful data assistant.")

    def _format_observation(self, observation: dict, task: str) -> str:
        if task == "cleaning":
            return (
                f"Domain: {observation.get('domain', 'unknown')}\n"
                f"DQ Score: {observation.get('dq_score', 'N/A')}\n"
                f"DQ Report: {observation.get('dq_report', '')}\n"
                f"Columns:\n{observation.get('columns_info', '')}\n"
                f"Data Preview:\n{observation.get('data_preview', '')[:500]}\n"
                f"Step {observation.get('step_number', '?')}/{observation.get('max_steps', '?')}\n\n"
                "Output a single JSON action."
            )
        elif task == "enrichment":
            return (
                f"Domain: {observation.get('domain', 'unknown')}\n"
                f"Schema:\n{observation.get('schema_info', '')}\n"
                f"Coverage: {observation.get('enrichment_coverage', 0)}\n"
                f"Fields Added: {observation.get('fields_added', [])}\n"
                f"Possible Enrichments: {observation.get('possible_enrichments', [])}\n"
                f"Step {observation.get('step_number', '?')}/{observation.get('max_steps', '?')}\n\n"
                "Output a single JSON action. Choose a field NOT already added."
            )
        elif task == "answering":
            return (
                f"Domain: {observation.get('domain', 'unknown')}\n"
                f"Persona: {observation.get('persona', 'unknown')} - {observation.get('persona_description', '')}\n"
                f"Question: {observation.get('question', 'N/A')}\n"
                f"Available Columns: {observation.get('available_columns', [])}\n"
                f"Column Stats:\n{observation.get('column_stats', '')[:800]}\n"
                f"Dataset Summary:\n{observation.get('dataset_summary', '')[:400]}\n\n"
                "Answer the question in the persona's style."
            )
        return json.dumps(observation, indent=2)

    def _parse_action(self, response: str, task: str) -> dict:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        if task == "answering":
            available = []
            cited = [c for c in available if c.lower() in response.lower()]
            return {"answer": response, "cited_columns": cited, "reasoning": ""}

        defaults = {
            "cleaning": {"operation": "trim", "column": "Status", "params": {}},
            "enrichment": {"operation": "add_field", "field_name": "unknown", "params": {}},
        }
        return defaults.get(task, {"raw": response})


class OpenAIProvider(ModelProvider):
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.model_id, messages=messages,
                max_tokens=max_tokens, temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error: {e}]"


class HuggingFaceProvider(ModelProvider):
    def __init__(self, model_id: str, display_name: str, hf_provider: str = "fireworks-ai"):
        super().__init__(model_id, display_name)
        self.hf_provider = hf_provider

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                provider=self.hf_provider,
                token=HF_TOKEN or None,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat_completion(
                model=self.model_id, messages=messages,
                max_tokens=max_tokens, temperature=0.7,
            )
            content = response.choices[0].message.content
            # Strip Qwen3 thinking tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            return f"[HuggingFace Error: {e}]"


def get_provider(model_key: str) -> ModelProvider:
    config = MODELS[model_key]
    if config["provider"] == "openai":
        return OpenAIProvider(config["model_id"], config["display_name"])
    elif config["provider"] == "huggingface":
        return HuggingFaceProvider(
            config["model_id"], config["display_name"],
            hf_provider=config.get("hf_provider", "fireworks-ai"),
        )
    else:
        raise ValueError(f"Provider {config['provider']} for {model_key} requires GPU inference")


def is_model_available(model_key: str) -> bool:
    config = MODELS[model_key]
    if config["provider"] == "openai":
        return bool(OPENAI_API_KEY)
    if config["provider"] == "huggingface-lora":
        return False  # LoRA adapters need GPU
    return True
