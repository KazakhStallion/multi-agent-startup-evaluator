# Reads summary.csv and metrics.json from the eval run and generates 8 figures.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EVAL_DIR = Path(__file__).parents[2] / "data" / "finance" / "evaluation"
FIGURES_DIR = EVAL_DIR / "figures"

sns.set_theme(style="whitegrid")


def load_data():
    csv_path = EVAL_DIR / "summary.csv"
    metrics_path = EVAL_DIR / "metrics.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"summary.csv not found")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found")

    df = pd.read_csv(csv_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return df, metrics


def plot_score_comparison(df):
    # one grouped bar per pitch, three bars per group (one per model)
    pivot = df.groupby(["pitch_name", "model"])["score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Score Comparison Across Pitches and Models")
    ax.set_xlabel("Startup Pitch")
    ax.set_ylabel("Average Score (1-10)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Model")
    ax.set_ylim(0, 11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_comparison", dpi=150)
    plt.close(fig)
    print("saved score_comparison")


def plot_avg_score_per_model(df):
    stats = df.groupby("model")["score"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        stats["model"],
        stats["mean"],
        yerr=stats["std"].fillna(0),
        capsize=5,
        color=sns.color_palette("muted", len(stats)),
    )
    ax.set_title("Average Score per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score (1-10)")
    ax.set_ylim(0, 11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "avg_score_per_model", dpi=150)
    plt.close(fig)
    print("saved avg_score_per_model")


def plot_score_heatmap(df):
    pivot = df.groupby(["pitch_name", "model"])["score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, vmin=1, vmax=10)
    ax.set_title("Mean Score per Pitch per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pitch")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_heatmap", dpi=150)
    plt.close(fig)
    print("saved score_heatmap")



def plot_math_accuracy(metrics):
    math_data = metrics.get("avg_math_accuracy", {})
    if not math_data:
        print("no math accuracy data, skipping that figure")
        return

    models = list(math_data.keys())
    rates = [math_data[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, rates, color=sns.color_palette("muted", len(models)))
    ax.set_title("Average Math Accuracy per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("% of Correct Calculations")
    ax.set_ylim(0, 115)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{rate:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "math_accuracy", dpi=150)
    plt.close(fig)
    print("saved math_accuracy")


def plot_rubric_scores(metrics):
    rubric = metrics.get("avg_rubric_score", {})
    if not rubric:
        print("no rubric data, skipping that figure")
        return

    models = list(rubric.keys())
    scores = [rubric[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, scores, color=sns.color_palette("muted", len(models)))
    ax.set_title("Average Rubric Quality Score per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rubric Score (0-10)")
    ax.set_ylim(0, 11)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{s:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rubric_scores", dpi=150)
    plt.close(fig)
    print("saved rubric_scores")


# def plot_risk_flagging(df):
#     sector_risk = df.groupby(["model", "sector"])["risks_count"].mean().reset_index()
#     pivot = sector_risk.pivot(index="sector", columns="model", values="risks_count")

#     fig, ax = plt.subplots(figsize=(12, 6))
#     pivot.plot(kind="bar", ax=ax, width=0.75)
#     ax.set_title("Average Risks Flagged per Model by Sector")
#     ax.set_xlabel("Sector")
#     ax.set_ylabel("Avg Risks Flagged")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.legend(title="Model")
#     fig.tight_layout()
#     fig.savefig(FIGURES_DIR / "risk_flagging", dpi=150)
#     plt.close(fig)
#     print("saved risk_flagging")


def plot_model_agreement(metrics):
    agreement_data = metrics.get("inter_model_agreement", {})
    if not agreement_data:
        print("no agreement data, skipping that figure")
        return

    # build a sorted list of model names from the pairwise keys
    model_set = set()
    for key in agreement_data:
        for part in key.split(" vs "):
            model_set.add(part.strip())
    model_names = sorted(model_set)
    n = len(model_names)

    # diagonal is 100% (a model always agrees with itself)
    matrix = np.full((n, n), 100.0)
    for key, value in agreement_data.items():
        parts = [p.strip() for p in key.split(" vs ")]
        if len(parts) == 2:
            try:
                i = model_names.index(parts[0])
                j = model_names.index(parts[1])
                matrix[i][j] = value
                matrix[j][i] = value
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax,
        vmin=0,
        vmax=100,
        cmap="Blues",
    )
    ax.set_title("Inter-model Agreement Rate (%)\n% of pitches where two models scored within 2 point")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_agreement", dpi=150)
    plt.close(fig)
    print("saved model_agreement")


def plot_score_consistency(df):
    # box plot per model — tight box means consistent, wide means unreliable
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df, x="model", y="score", ax=ax, palette="muted")
    ax.set_title("Score Distribution per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (1-10)")
    ax.set_ylim(0, 11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_consistency", dpi=150)
    plt.close(fig)
    print("saved score_consistency")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, metrics = load_data()
    print(f"loaded {len(df)} rows, generating figures...")

    plot_score_comparison(df)
    plot_avg_score_per_model(df)
    plot_score_heatmap(df)
    plot_math_accuracy(metrics)
    plot_rubric_scores(metrics)
    plot_model_agreement(metrics)
    plot_score_consistency(df)

    print(f"done, all figures in {FIGURES_DIR}")
