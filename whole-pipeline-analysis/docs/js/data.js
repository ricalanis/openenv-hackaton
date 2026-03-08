// Auto-generated from demo/data/*.json — do not edit manually
const DATA = {
  "datasageVsBase": {
    "datasage": {
      "cleaning": 0.9618,
      "enrichment": 0.2,
      "answering": 0.6612
    },
    "base_qwen": {
      "cleaning": 0.9609,
      "enrichment": 0.2,
      "answering": 0.5734
    },
    "improvement": {
      "cleaning": 0.1,
      "enrichment": 0.0,
      "answering": 15.3
    }
  },
  "benchmarkComparison": {
    "gpt4o_mini": {
      "cleaning": 0.9606,
      "enrichment": 0.0167,
      "answering": 0.7116
    },
    "qwen3_8b": {
      "cleaning": 0.9631,
      "enrichment": 0.0167,
      "answering": 0.5151
    }
  },
  "perDomainAnswering": {
    "datasage": {
      "sales": 0.6389,
      "pm": 0.7593,
      "it_ops": 0.66
    },
    "base_qwen": {
      "hr": 0.6425,
      "sales": 0.4667,
      "pm": 0.5454,
      "it_ops": 0.5864
    },
    "gpt4o_mini": {
      "sales": 0.8181,
      "hr": 0.681,
      "pm": 0.7033,
      "it_ops": 0.8725
    },
    "qwen3_8b": {
      "pm": 0.538,
      "hr": 0.5659,
      "it_ops": 0.6041,
      "sales": 0.195
    }
  },
  "perPersona": {
    "datasage": {
      "Executive": 0.5527,
      "Individual Contributor": 0.7461,
      "Manager": 0.7933
    },
    "base_qwen": {
      "Individual Contributor": 0.6296,
      "Executive": 0.542,
      "Manager": 0.5506
    },
    "gpt4o_mini": {
      "Individual Contributor": 0.8648,
      "Executive": 0.4594,
      "Manager": 0.876
    },
    "qwen3_8b": {
      "Executive": 0.4673,
      "Manager": 0.8016,
      "Individual Contributor": 0.5
    }
  },
  "heatmap": {
    "cleaning": {
      "hr": 0.9606,
      "sales": 0.9618,
      "pm": 0.9629,
      "it_ops": 0.9619
    },
    "enrichment": {
      "hr": 0.2,
      "sales": 0.2,
      "pm": 0.2,
      "it_ops": 0.2
    },
    "answering": {
      "sales": 0.6389,
      "pm": 0.7593,
      "it_ops": 0.66
    }
  },
  "qaShowcase": [
    {
      "model": "qwen3-8b",
      "domain": "pm",
      "persona": "Executive",
      "question": "How is resource utilization across teams?",
      "answer": "<think>\nOkay, let me tackle this query. The user is an executive asking about resource utilization across teams. They care about costs, ROI, strategic risk, portfolio trends, and year-over-year comparisons. So I need to focus on high-level metrics and strategic implications.\n\nFirst, looking at the a...",
      "cited_columns": [
        "Status",
        "Priority",
        "EstimatedHours",
        "ActualHours",
        "CompletionPct"
      ],
      "reward": 0.73
    },
    {
      "model": "qwen3-8b",
      "domain": "it_ops",
      "persona": "Manager",
      "question": "How many tickets are breaching SLA?",
      "answer": "<think>\nOkay, the user is a manager in IT operations, so they need actionable insights. The question is about how many tickets are breaching SLA. Let me check the available data.\n\nFirst, the columns given include SLATarget and ResolvedDate. The SLATarget has a mean of 47.6 and a standard deviation o...",
      "cited_columns": [
        "Status",
        "CreatedDate",
        "ResolvedDate",
        "SLATarget"
      ],
      "reward": 0.7083
    }
  ],
  "trainingConfig": {
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "quantization": "4-bit (BnB)",
    "lora_r": 16,
    "lora_alpha": 16,
    "optimizer": "AdamW 8-bit",
    "epochs": 3,
    "beta": 0.001,
    "epsilon": 0.2,
    "loss_type": "BNPO",
    "max_steps_per_task": 192
  },
  "environments": {
    "cleaning": {
      "name": "Data Cleaning",
      "color": "#F59E0B",
      "description": "Detects and fixes data quality issues \u2014 missing values, duplicates, type errors",
      "reward_functions": [
        "cleaning_env_reward",
        "cleaning_json_format_reward"
      ],
      "metric": "Data Quality Score",
      "hf_space": "https://huggingface.co/spaces/ricalanis/datasage-cleaning",
      "lora_repo": "ricalanis/cleaning-grpo",
      "datasage_reward": 0.962,
      "base_reward": 0.961
    },
    "enrichment": {
      "name": "Data Enrichment",
      "color": "#EF4444",
      "description": "Adds computed columns and derived features from existing data",
      "reward_functions": [
        "enrichment_env_reward",
        "enrichment_json_format_reward",
        "source_relevance_reward"
      ],
      "metric": "Coverage (fields added)",
      "hf_space": "https://huggingface.co/spaces/ricalanis/datasage-enrichment",
      "lora_repo": "ricalanis/enrichment-grpo",
      "datasage_reward": 0.2,
      "base_reward": 0.2
    },
    "answering": {
      "name": "Data Answering",
      "color": "#3B82F6",
      "description": "Answers natural language questions grounded in the dataset",
      "reward_functions": [
        "answering_env_reward",
        "answering_json_format_reward",
        "patronus_reward_fn",
        "persona_match_reward"
      ],
      "metric": "Composite Reward",
      "hf_space": "https://huggingface.co/spaces/ricalanis/datasage-answering",
      "lora_repo": "ricalanis/answering-grpo",
      "datasage_reward": 0.661,
      "base_reward": 0.573
    }
  },
  "links": {
    "github": "https://github.com/ricalanis/openenv-hackaton",
    "lora_repos": {
      "cleaning": "ricalanis/cleaning-grpo",
      "enrichment": "ricalanis/enrichment-grpo",
      "answering": "ricalanis/answering-grpo"
    },
    "hf_spaces": {
      "cleaning": "https://huggingface.co/spaces/ricalanis/datasage-cleaning",
      "enrichment": "https://huggingface.co/spaces/ricalanis/datasage-enrichment",
      "answering": "https://huggingface.co/spaces/ricalanis/datasage-answering"
    }
  },
  "radar": {
    "datasage": [
      0.962,
      0.2,
      0.661
    ],
    "base_qwen": [
      0.961,
      0.2,
      0.573
    ],
    "gpt4o_mini": [
      0.961,
      0.017,
      0.712
    ],
    "qwen3_8b": [
      0.963,
      0.017,
      0.515
    ]
  }
};
