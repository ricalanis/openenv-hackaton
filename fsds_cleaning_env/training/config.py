"""Experiment configuration for the training harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    task_id: str
    n_episodes: int
    agent: str
    base_url: str
    max_steps_per_episode: int
    log_dir: str
    log_interval: int
    seed: Optional[int]
    output_dir: str
    # Curriculum settings (only used when curriculum=True)
    curriculum: bool = False
    curriculum_task_ids: List[str] = field(default_factory=list)
    curriculum_mode: str = "round_robin"  # "round_robin" | "random"
    curriculum_start_level: str = "easy"  # "easy" | "medium" | "hard"

    @classmethod
    def _parse(cls, d: dict) -> "ExperimentConfig":
        return cls(
            task_id=d.get("task_id", "ecommerce_mobile"),
            n_episodes=int(d.get("n_episodes", 50)),
            agent=d.get("agent", "heuristic"),
            base_url=d.get("base_url", "http://localhost:8000"),
            max_steps_per_episode=int(d.get("max_steps_per_episode", 18)),
            log_dir=d.get("log_dir", "logs"),
            log_interval=int(d.get("log_interval", 10)),
            seed=d.get("seed"),
            output_dir=d.get("output_dir", "training_output"),
            curriculum=bool(d.get("curriculum", False)),
            curriculum_task_ids=list(d.get("curriculum_task_ids", [])),
            curriculum_mode=d.get("curriculum_mode", "round_robin"),
            curriculum_start_level=d.get("curriculum_start_level", "easy"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls._parse(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls._parse(d)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        suf = p.suffix.lower()
        if suf == ".yaml" or suf == ".yml":
            return cls.from_yaml(p)
        if suf == ".json":
            return cls.from_json(p)
        raise ValueError(f"Unsupported config format: {suf}. Use .json or .yaml")
