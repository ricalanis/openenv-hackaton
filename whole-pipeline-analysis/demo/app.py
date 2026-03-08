"""
DataSage Demo - Multi-Model Agentic Comparison Platform
========================================================
LangGraph-based agentic system comparing Qwen3-8B, GPT-4o-mini,
and GRPO fine-tuned DataSage models across data operations tasks.

Run: python app.py
"""

import json
import os
import sys

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from backend.config import MODELS, DOMAINS, DOMAIN_DISPLAY, PERSONAS, PERSONA_DISPLAY
from backend.standard_benchmarks import (
    STANDARD_BENCHMARKS, DOMAIN_BENCHMARKS,
    get_benchmark_summary, get_category_scores, get_all_benchmarks,
)

# ---------------------------------------------------------------------------
# Load real results
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "benchmark_results.json")
with open(DATA_PATH) as f:
    ALL_DATA = json.load(f)

REAL = ALL_DATA["real_results"]
PROJECTED = ALL_DATA["projected_results"]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COLORS = {
    "gpt-4o-mini": "#10A37F",
    "qwen3-8b": "#7C3AED",
    "datasage-cleaning": "#F59E0B",
    "datasage-enrichment": "#EF4444",
    "datasage-answering": "#3B82F6",
    "datasage-ensemble": "#EC4899",
}

DISPLAY_NAMES = {
    "gpt-4o-mini": "GPT-4o-mini",
    "qwen3-8b": "Qwen3-8B",
    "datasage-cleaning": "DataSage Cleaning (projected)",
    "datasage-enrichment": "DataSage Enrichment (projected)",
    "datasage-answering": "DataSage Answering (projected)",
    "datasage-ensemble": "DataSage Ensemble (projected)",
}


# ---------------------------------------------------------------------------
# Chart: Answering comparison (REAL DATA)
# ---------------------------------------------------------------------------

def build_answering_scatter():
    """Scatter plot of all real answering episodes."""
    rows = []
    for model_key in ["gpt-4o-mini", "qwen3-8b"]:
        data = REAL["answering"][model_key]
        for ep in data["per_episode"]:
            rows.append({
                "Model": DISPLAY_NAMES[model_key],
                "Domain": DOMAIN_DISPLAY.get(ep["domain"], ep["domain"]),
                "Persona": PERSONA_DISPLAY.get(ep["persona"], ep["persona"]),
                "Reward": ep["reward"],
                "model_key": model_key,
            })

    df = pd.DataFrame(rows)
    fig = px.strip(
        df, x="Model", y="Reward", color="Model",
        hover_data=["Domain", "Persona"],
        color_discrete_map={DISPLAY_NAMES[k]: v for k, v in COLORS.items()},
    )

    # Add mean lines
    for model_key in ["gpt-4o-mini", "qwen3-8b"]:
        mean_r = REAL["answering"][model_key]["reward_mean"]
        name = DISPLAY_NAMES[model_key]
        fig.add_hline(
            y=mean_r, line_dash="dash",
            line_color=COLORS[model_key],
            annotation_text=f"{name} mean: {mean_r:.3f}",
            annotation_position="top right",
        )

    fig.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Answering Task: Per-Episode Rewards (REAL DATA)",
        title_x=0.5,
        yaxis=dict(title="Environment Reward", range=[0, 1]),
        showlegend=False,
    )
    return fig


def build_answering_by_persona():
    """Answering results broken down by persona."""
    rows = []
    for model_key in ["gpt-4o-mini", "qwen3-8b"]:
        data = REAL["answering"][model_key]
        persona_rewards = {}
        for ep in data["per_episode"]:
            p = ep["persona"]
            persona_rewards.setdefault(p, []).append(ep["reward"])

        for persona, rewards in persona_rewards.items():
            rows.append({
                "Model": DISPLAY_NAMES[model_key],
                "Persona": PERSONA_DISPLAY.get(persona, persona),
                "Mean Reward": sum(rewards) / len(rewards),
                "N": len(rewards),
                "model_key": model_key,
            })

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Persona", y="Mean Reward", color="Model", barmode="group",
        text="Mean Reward",
        color_discrete_map={DISPLAY_NAMES[k]: v for k, v in COLORS.items()},
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Answering by Persona (REAL DATA)",
        title_x=0.5,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def build_answering_by_domain():
    """Answering results broken down by domain."""
    rows = []
    for model_key in ["gpt-4o-mini", "qwen3-8b"]:
        data = REAL["answering"][model_key]
        domain_rewards = {}
        for ep in data["per_episode"]:
            d = ep["domain"]
            domain_rewards.setdefault(d, []).append(ep["reward"])

        for domain, rewards in domain_rewards.items():
            rows.append({
                "Model": DISPLAY_NAMES[model_key],
                "Domain": DOMAIN_DISPLAY.get(domain, domain),
                "Mean Reward": sum(rewards) / len(rewards),
                "N": len(rewards),
                "model_key": model_key,
            })

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Domain", y="Mean Reward", color="Model", barmode="group",
        text="Mean Reward",
        color_discrete_map={DISPLAY_NAMES[k]: v for k, v in COLORS.items()},
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Answering by Domain (REAL DATA)",
        title_x=0.5,
        yaxis=dict(range=[0, 1]),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Full comparison (real + projected)
# ---------------------------------------------------------------------------

def build_full_comparison():
    """Bar chart comparing all models across all tasks."""
    rows = [
        {"Model": "GPT-4o-mini", "Task": "Cleaning (reward)", "Score": REAL["cleaning"]["gpt-4o-mini"]["avg_reward_mean"], "Type": "Real"},
        {"Model": "GPT-4o-mini", "Task": "Enrichment (coverage)", "Score": REAL["enrichment"]["gpt-4o-mini"]["final_coverage_mean"], "Type": "Real"},
        {"Model": "GPT-4o-mini", "Task": "Answering (reward)", "Score": REAL["answering"]["gpt-4o-mini"]["reward_mean"], "Type": "Real"},
        {"Model": "Qwen3-8B", "Task": "Cleaning (reward)", "Score": REAL["cleaning"]["qwen3-8b"]["avg_reward_mean"], "Type": "Real"},
        {"Model": "Qwen3-8B", "Task": "Enrichment (coverage)", "Score": REAL["enrichment"]["qwen3-8b"]["final_coverage_mean"], "Type": "Real"},
        {"Model": "Qwen3-8B", "Task": "Answering (reward)", "Score": REAL["answering"]["qwen3-8b"]["reward_mean"], "Type": "Real"},
        {"Model": "DataSage (projected)", "Task": "Cleaning (reward)", "Score": PROJECTED["datasage_cleaning"]["projected_avg_reward"], "Type": "Projected"},
        {"Model": "DataSage (projected)", "Task": "Enrichment (coverage)", "Score": PROJECTED["datasage_enrichment"]["projected_coverage"], "Type": "Projected"},
        {"Model": "DataSage (projected)", "Task": "Answering (reward)", "Score": PROJECTED["datasage_answering"]["projected_reward"], "Type": "Projected"},
    ]

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Task", y="Score", color="Model", barmode="group",
        pattern_shape="Type", pattern_shape_map={"Real": "", "Projected": "/"},
        text="Score",
        color_discrete_map={
            "GPT-4o-mini": COLORS["gpt-4o-mini"],
            "Qwen3-8B": COLORS["qwen3-8b"],
            "DataSage (projected)": COLORS["datasage-ensemble"],
        },
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Full Model Comparison (Real + Projected)",
        title_x=0.5,
        yaxis=dict(range=[0, 1.1], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def build_radar_comparison():
    """Radar chart comparing models."""
    categories = ["Cleaning", "Enrichment", "Answering"]

    fig = go.Figure()

    # GPT-4o-mini (real)
    gpt_values = [
        REAL["cleaning"]["gpt-4o-mini"]["avg_reward_mean"],
        REAL["enrichment"]["gpt-4o-mini"]["final_coverage_mean"],
        REAL["answering"]["gpt-4o-mini"]["reward_mean"],
    ]
    gpt_values.append(gpt_values[0])
    fig.add_trace(go.Scatterpolar(
        r=gpt_values, theta=categories + [categories[0]],
        fill='toself', name='GPT-4o-mini (real)',
        line_color=COLORS["gpt-4o-mini"], opacity=0.7,
    ))

    # Qwen3-8B (real)
    qwen_values = [
        REAL["cleaning"]["qwen3-8b"]["avg_reward_mean"],
        REAL["enrichment"]["qwen3-8b"]["final_coverage_mean"],
        REAL["answering"]["qwen3-8b"]["reward_mean"],
    ]
    qwen_values.append(qwen_values[0])
    fig.add_trace(go.Scatterpolar(
        r=qwen_values, theta=categories + [categories[0]],
        fill='toself', name='Qwen3-8B (real)',
        line_color=COLORS["qwen3-8b"], opacity=0.7,
    ))

    # DataSage (projected)
    ds_values = [
        PROJECTED["datasage_cleaning"]["projected_avg_reward"],
        PROJECTED["datasage_enrichment"]["projected_coverage"],
        PROJECTED["datasage_answering"]["projected_reward"],
    ]
    ds_values.append(ds_values[0])
    fig.add_trace(go.Scatterpolar(
        r=ds_values, theta=categories + [categories[0]],
        fill='toself', name='DataSage ensemble (projected)',
        line_color=COLORS["datasage-ensemble"], opacity=0.7,
        line_dash="dash",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
            bgcolor="#16213e",
            angularaxis=dict(gridcolor="#333"),
        ),
        height=450,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        title_text="Pipeline Coverage Radar",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Standard benchmarks
# ---------------------------------------------------------------------------

def build_standard_benchmarks_chart():
    bms = STANDARD_BENCHMARKS
    # Use qwen3-8b scores (same family, similar to 3B but larger)
    models_map = {"qwen3-8b": "qwen-base", "gpt-4o-mini": "gpt-4o-mini"}
    benchmark_names = list(bms.keys())

    fig = go.Figure()
    for display_key, data_key in [("gpt-4o-mini", "gpt-4o-mini"), ("qwen3-8b", "qwen-base")]:
        scores = [bms[b].get(data_key, 0) for b in benchmark_names]
        fig.add_trace(go.Bar(
            name=DISPLAY_NAMES.get(display_key, display_key),
            x=benchmark_names, y=scores,
            marker_color=COLORS.get(display_key, "#888"),
        ))

    fig.update_layout(
        barmode="group", height=450,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Standard NLP Benchmarks (Published Scores)",
        title_x=0.5,
        yaxis=dict(title="Score (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-30),
    )
    return fig


def build_domain_benchmarks_chart():
    bms = DOMAIN_BENCHMARKS
    models = [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("qwen3-8b", "qwen-base"),
        ("datasage-cleaning", "datasage-cleaning"),
        ("datasage-enrichment", "datasage-enrichment"),
        ("datasage-answering", "datasage-answering"),
    ]
    benchmark_names = list(bms.keys())

    fig = go.Figure()
    for display_key, data_key in models:
        scores = [bms[b].get(data_key, 0) for b in benchmark_names]
        fig.add_trace(go.Bar(
            name=DISPLAY_NAMES.get(display_key, display_key),
            x=benchmark_names, y=scores,
            marker_color=COLORS.get(display_key, "#888"),
        ))

    fig.update_layout(
        barmode="group", height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        title_text="Domain-Specific Benchmarks (DataSage Specialization, estimated)",
        title_x=0.5,
        yaxis=dict(title="Score (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-30),
    )
    return fig


def build_category_radar():
    cats = get_category_scores()
    models = [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("qwen3-8b", "qwen-base"),
        ("datasage-answering", "datasage-answering"),
    ]
    categories = list(cats.keys())

    fig = go.Figure()
    for display_key, data_key in models:
        values = [cats[c].get(data_key, 0) for c in categories]
        values.append(values[0])
        labels = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, fill='toself',
            name=DISPLAY_NAMES.get(display_key, display_key),
            line_color=COLORS.get(display_key, "#888"),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"),
            bgcolor="#16213e",
            angularaxis=dict(gridcolor="#333"),
        ),
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        title_text="Capability Profile by Category",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Live agent demo
# ---------------------------------------------------------------------------

def run_live_comparison(task, model_key, n_episodes):
    """Run live episodes using the LangGraph agent."""
    from backend.agent import run_episode

    results = []
    output_lines = [f"### Running {n_episodes} episodes of {task} with {DISPLAY_NAMES.get(model_key, model_key)}\n"]

    for ep in range(int(n_episodes)):
        result = run_episode(task, "hr", model_key, persona="executive", seed=42 + ep)
        results.append(result)
        metrics = result.get("metrics", {})
        output_lines.append(f"**Episode {ep+1}**: {json.dumps(metrics, indent=2)}")

    # Summary table
    table_rows = []
    for i, r in enumerate(results):
        row = {"Episode": i + 1}
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                row[k] = round(v, 4)
        table_rows.append(row)

    df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame()

    # Trace
    traces = []
    if results:
        trace = results[0].get("trace", [])
        for t in trace:
            traces.append(f"  `{t['node']}` step={t.get('step', '-')}")

    trace_text = "**Agent Trace (Episode 1):**\n" + "\n".join(traces) if traces else "No trace"

    return "\n\n".join(output_lines), df, trace_text


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #333; border-radius: 12px; padding: 20px;
    text-align: center; margin: 8px;
}
.metric-value { font-size: 2.2em; font-weight: bold; color: #EC4899; }
.metric-label { color: #9CA3AF; font-size: 0.9em; margin-top: 4px; }
.hero-section {
    text-align: center; padding: 30px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #0a2463 50%, #1a1a2e 100%);
    border-radius: 16px; margin-bottom: 20px; border: 1px solid #333;
}
.hero-title {
    font-size: 2.2em; font-weight: bold;
    background: linear-gradient(90deg, #EC4899, #8B5CF6, #3B82F6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #9CA3AF; font-size: 1.05em; }
.real-badge {
    display: inline-block; background: #10B981; color: white;
    padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold;
}
.projected-badge {
    display: inline-block; background: #F59E0B; color: black;
    padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold;
}
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def create_app():
    gpt_ans = REAL["answering"]["gpt-4o-mini"]["reward_mean"]
    qwen_ans = REAL["answering"]["qwen3-8b"]["reward_mean"]

    with gr.Blocks(title="DataSage - Multi-Model Agentic Demo") as app:

        # Hero
        gr.HTML(f"""
        <div class="hero-section">
            <div class="hero-title">DataSage Multi-Model Comparison</div>
            <div class="hero-subtitle">
                LangGraph Agentic System &bull; Real benchmarks against live HF Space environments
            </div>
            <div style="margin-top:16px; display:flex; justify-content:center; gap:30px; flex-wrap:wrap;">
                <div class="metric-card" style="display:inline-block; min-width:140px;">
                    <div class="metric-value" style="color:#10A37F;">{gpt_ans:.3f}</div>
                    <div class="metric-label">GPT-4o-mini Answering <span class="real-badge">REAL</span></div>
                </div>
                <div class="metric-card" style="display:inline-block; min-width:140px;">
                    <div class="metric-value" style="color:#7C3AED;">{qwen_ans:.3f}</div>
                    <div class="metric-label">Qwen3-8B Answering <span class="real-badge">REAL</span></div>
                </div>
                <div class="metric-card" style="display:inline-block; min-width:140px;">
                    <div class="metric-value" style="color:#10B981;">+{(gpt_ans - qwen_ans):.3f}</div>
                    <div class="metric-label">GPT advantage</div>
                </div>
                <div class="metric-card" style="display:inline-block; min-width:140px;">
                    <div class="metric-value" style="color:#EC4899;">0.85</div>
                    <div class="metric-label">DataSage target <span class="projected-badge">PROJECTED</span></div>
                </div>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ============================================================
            # TAB 1: Real Results
            # ============================================================
            with gr.Tab("Real Results (Live Environments)"):
                gr.Markdown("""
## Real Benchmark Results
<span class="real-badge">REAL</span> All results below were collected by running GPT-4o-mini and Qwen3-8B
against the **live HuggingFace Space environments** via the LangGraph agent on 2026-03-08.
                """)

                gr.Markdown("### Answering Task (Best Differentiation)")
                gr.Markdown(
                    "The answering task shows clear model differentiation. "
                    "Each episode randomly assigns a domain and persona. "
                    f"**GPT-4o-mini: {gpt_ans:.3f}** vs **Qwen3-8B: {qwen_ans:.3f}** mean reward."
                )

                with gr.Row():
                    gr.Plot(build_answering_scatter())

                with gr.Row():
                    with gr.Column():
                        gr.Plot(build_answering_by_persona())
                    with gr.Column():
                        gr.Plot(build_answering_by_domain())

                gr.Markdown("### Answering Episodes (Raw Data)")
                ans_rows = []
                for model_key in ["gpt-4o-mini", "qwen3-8b"]:
                    for ep in REAL["answering"][model_key]["per_episode"]:
                        ans_rows.append({
                            "Model": DISPLAY_NAMES[model_key],
                            "Domain": DOMAIN_DISPLAY.get(ep["domain"], ep["domain"]),
                            "Persona": PERSONA_DISPLAY.get(ep["persona"], ep["persona"]),
                            "Reward": ep["reward"],
                        })
                gr.Dataframe(pd.DataFrame(ans_rows), label="All Answering Episodes")

                gr.Markdown("---")
                gr.Markdown("### Cleaning & Enrichment Tasks")
                gr.Markdown("""
**Cleaning**: The environment starts with DQ score > 0.95 (the done threshold),
so episodes end after 1 step for both models. Both achieve ~0.96 reward.
This is the real environment behavior - the 15% corruption rate on 50 rows
doesn't drop quality below the threshold.

**Enrichment**: Both models achieve exactly 0.20 coverage (1 out of 5 possible enrichments)
across all 12 steps. After successfully adding the first enrichment field, both models
fail to add subsequent unique fields - they either repeat the same field name or
generate invalid operations.
                """)

                clean_enrich_rows = []
                for model_key in ["gpt-4o-mini", "qwen3-8b"]:
                    c = REAL["cleaning"][model_key]
                    e = REAL["enrichment"][model_key]
                    clean_enrich_rows.append({
                        "Model": DISPLAY_NAMES[model_key],
                        "Cleaning DQ": f"{c['final_dq_mean']:.4f}",
                        "Cleaning Reward": f"{c['avg_reward_mean']:.4f}",
                        "Cleaning Steps": c["steps_mean"],
                        "Enrichment Coverage": f"{e['final_coverage_mean']:.4f}",
                        "Enrichment Steps": e["steps_mean"],
                    })
                gr.Dataframe(pd.DataFrame(clean_enrich_rows), label="Cleaning & Enrichment Summary")

            # ============================================================
            # TAB 2: Full Comparison (Real + Projected)
            # ============================================================
            with gr.Tab("Full Comparison"):
                gr.Markdown("""
## Full Model Comparison
<span class="real-badge">REAL</span> GPT-4o-mini and Qwen3-8B scores from live environments.
<span class="projected-badge">PROJECTED</span> DataSage fine-tuned model scores based on GRPO training rewards.

DataSage models are LoRA adapters on Qwen2.5-3B-Instruct, trained with GRPO
(Group Relative Policy Optimization) using environment feedback rewards.
They require GPU inference via Unsloth and cannot be benchmarked via standard API.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Plot(build_full_comparison())
                    with gr.Column():
                        gr.Plot(build_radar_comparison())

                gr.Markdown("### Comparison Table")
                comp_rows = [
                    {
                        "Model": "GPT-4o-mini",
                        "Cleaning": f"{REAL['cleaning']['gpt-4o-mini']['avg_reward_mean']:.3f}",
                        "Enrichment": f"{REAL['enrichment']['gpt-4o-mini']['final_coverage_mean']:.3f}",
                        "Answering": f"{REAL['answering']['gpt-4o-mini']['reward_mean']:.3f}",
                        "Source": "REAL",
                    },
                    {
                        "Model": "Qwen3-8B",
                        "Cleaning": f"{REAL['cleaning']['qwen3-8b']['avg_reward_mean']:.3f}",
                        "Enrichment": f"{REAL['enrichment']['qwen3-8b']['final_coverage_mean']:.3f}",
                        "Answering": f"{REAL['answering']['qwen3-8b']['reward_mean']:.3f}",
                        "Source": "REAL",
                    },
                    {
                        "Model": "DataSage Ensemble",
                        "Cleaning": f"{PROJECTED['datasage_cleaning']['projected_avg_reward']:.3f}",
                        "Enrichment": f"{PROJECTED['datasage_enrichment']['projected_coverage']:.3f}",
                        "Answering": f"{PROJECTED['datasage_answering']['projected_reward']:.3f}",
                        "Source": "PROJECTED",
                    },
                ]
                gr.Dataframe(pd.DataFrame(comp_rows))

                gr.Markdown("""
### Key Findings
1. **Answering is the differentiating task**: GPT-4o-mini achieves 0.712 vs Qwen3-8B's 0.515 (+38%)
2. **Cleaning environment is too easy**: DQ starts above 0.95 threshold, limiting comparison
3. **Enrichment is equally hard for both**: Neither model can add more than 1/5 enrichments
4. **GRPO fine-tuning targets these gaps**: DataSage models are specifically trained to handle
   multi-step data operations that general-purpose models struggle with
                """)

            # ============================================================
            # TAB 3: Standard Benchmarks
            # ============================================================
            with gr.Tab("Standard Benchmarks"):
                gr.Markdown("""
## Standard NLP Benchmarks
Published scores for base models. DataSage domain-specific scores are estimated
based on fine-tuning impact analysis.
                """)
                gr.Plot(build_standard_benchmarks_chart())

                gr.Markdown("### Domain-Specific Benchmarks (estimated)")
                gr.Plot(build_domain_benchmarks_chart())
                gr.Plot(build_category_radar())

                # All benchmarks table
                all_bm = get_all_benchmarks()
                bm_rows = []
                for name, bm in all_bm.items():
                    row = {"Benchmark": name, "Category": bm["category"]}
                    for mk in ["gpt-4o-mini", "qwen-base", "datasage-cleaning",
                                "datasage-enrichment", "datasage-answering"]:
                        display = {"gpt-4o-mini": "GPT-4o-mini", "qwen-base": "Qwen Base"}.get(mk, mk)
                        row[display] = bm.get(mk, "-")
                    bm_rows.append(row)
                gr.Dataframe(pd.DataFrame(bm_rows), label="All Benchmark Scores")

            # ============================================================
            # TAB 4: Live Agent Demo
            # ============================================================
            with gr.Tab("Live Agent Demo"):
                gr.Markdown("""
## Interactive LangGraph Agent
Run episodes against live HF Space environments with different models.
Requires OpenAI API key for GPT and HF token for Qwen.
                """)

                with gr.Row():
                    task_select = gr.Dropdown(
                        choices=["cleaning", "enrichment", "answering"],
                        value="answering", label="Task",
                    )
                    model_select = gr.Dropdown(
                        choices=["gpt-4o-mini", "qwen3-8b"],
                        value="gpt-4o-mini", label="Model",
                    )
                    episodes_select = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1, label="Episodes",
                    )

                run_btn = gr.Button("Run Agent", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        results_md = gr.Markdown("*Click 'Run Agent' to start...*")
                    with gr.Column():
                        results_table = gr.Dataframe(label="Metrics")

                trace_output = gr.Markdown("*Agent trace will appear here...*")

                run_btn.click(
                    fn=run_live_comparison,
                    inputs=[task_select, model_select, episodes_select],
                    outputs=[results_md, results_table, trace_output],
                )

            # ============================================================
            # TAB 5: Architecture
            # ============================================================
            with gr.Tab("Architecture"):
                gr.Markdown("""
## System Architecture

### LangGraph Agent Flow
```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Initialize │────>│ Select Action  │────>│ Execute Action  │
│  (env reset)│     │ (model infer)  │     │  (env step)     │
└─────────────┘     └────────────────┘     └───────┬─────────┘
                           ^                        │
                           │    ┌──────────┐        │
                           │    │ Evaluate │<───────┤ done?
                           │    │ (metrics)│   yes  │
                           │    └──────────┘        │
                           └────────────────────────┘ no
```

### Multi-Model Switching
```
              ┌───────────────────────┐
              │    LangGraph Agent    │
              │   (model_key param)   │
              └───────┬───────────────┘
                      │
         ┌────────────┼────────────┐
         v            v            v
  ┌──────────┐ ┌──────────┐ ┌───────────┐
  │ Qwen3-8B │ │ GPT-4o   │ │ DataSage  │
  │  (HF/    │ │  mini    │ │(LoRA+GPU) │
  │ fireworks)│ │ (OpenAI) │ │           │
  └────┬─────┘ └────┬─────┘ └─────┬─────┘
       └─────────────┼─────────────┘
                     v
              ┌──────────────┐
              │ HF Space Env │
              │ (live API)   │
              └──────────────┘
```

### Training Pipeline (DataSage)
```
  Qwen 2.5-3B ──> LoRA Adapter ──> GRPO Trainer
                                        │
          ┌─────────────────────────────┤
          v              v              v
   ┌──────────┐   ┌───────────┐  ┌───────────┐
   │ Cleaning │   │ Enrichment│  │ Answering │
   │   Env    │   │    Env    │  │    Env    │
   └──────────┘   └───────────┘  └───────────┘
        │               │              │
        v               v              v
   cleaning-grpo  enrichment-grpo  answering-grpo
```

### HuggingFace Models
| Model | Type | HF Repo |
|-------|------|---------|
| Cleaning | LoRA adapter | [ricalanis/cleaning-grpo](https://huggingface.co/ricalanis/cleaning-grpo) |
| Enrichment | LoRA adapter | [ricalanis/enrichment-grpo](https://huggingface.co/ricalanis/enrichment-grpo) |
| Answering | LoRA adapter | [ricalanis/answering-grpo](https://huggingface.co/ricalanis/answering-grpo) |

### Environment Spaces
| Environment | Status | URL |
|-------------|--------|-----|
| Cleaning | Live | [ricalanis/datasage-cleaning](https://huggingface.co/spaces/ricalanis/datasage-cleaning) |
| Enrichment | Live | [ricalanis/datasage-enrichment](https://huggingface.co/spaces/ricalanis/datasage-enrichment) |
| Answering | Live | [ricalanis/datasage-answering](https://huggingface.co/spaces/ricalanis/datasage-answering) |
                """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
