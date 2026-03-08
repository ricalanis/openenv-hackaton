"""Training configuration constants."""

WANDB_PROJECT = "datasage"
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"

SPACE_URLS = {
    "cleaning": "https://ricalanis-datasage-cleaning.hf.space",
    "enrichment": "https://ricalanis-datasage-enrichment.hf.space",
    "answering": "https://ricalanis-datasage-answering.hf.space",
}

HF_MODEL_REPOS = {
    "cleaning": "ricalanis/datasage-qwen-cleaning",
    "enrichment": "ricalanis/datasage-qwen-enrichment",
    "answering": "ricalanis/datasage-qwen-answering",
}

# GRPO hyperparameters per stage
TRAINING_CONFIGS = {
    "cleaning": {
        "learning_rate": 5e-6,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "max_completion_length": 256,
        "max_prompt_length": 1024,
    },
    "enrichment": {
        "learning_rate": 5e-6,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "max_completion_length": 256,
        "max_prompt_length": 1024,
    },
    "answering": {
        "learning_rate": 3e-6,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "max_completion_length": 256,
        "max_prompt_length": 1024,
    },
}
