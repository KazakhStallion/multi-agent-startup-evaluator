import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "tech_lead" / "evaluation"
FIGURES_DIR = EVAL_DIR / "figures"

if sns:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("ggplot")


def load_data():
    summary_path = EVAL_DIR / "summary.csv"
    metrics_path = EVAL_DIR / "metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    df = pd.read_csv(summary_path)
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return df, metrics


def plot_verdict_distribution(df):
    counts = df["verdict"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="muted")
    else:
        ax.bar(counts.index, counts.values, color=["#4C72B0", "#55A868", "#C44E52"][: len(counts)])
    ax.set_title("Technical Verdict Distribution")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.05, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "verdict_distribution", dpi=150)
    plt.close(fig)
    print("saved verdict_distribution")


def plot_confidence_distribution(df):
    order = ["Low", "Medium", "High"]
    counts = df["confidence"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="crest")
    else:
        ax.bar(counts.index, counts.values, color=["#8172B3", "#64B5CD", "#CCB974"])
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.05, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distribution", dpi=150)
    plt.close(fig)
    print("saved confidence_distribution")


def plot_verdict_by_sector(df):
    pivot = (
        df.groupby(["sector", "verdict"]).size().reset_index(name="count").pivot(
            index="sector", columns="verdict", values="count"
        ).fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Verdict Distribution by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Count")
    ax.legend(title="Verdict")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "verdict_by_sector", dpi=150)
    plt.close(fig)
    print("saved verdict_by_sector")


def plot_rubric_scores(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.histplot(df["rubric_score"], bins=[0, 1, 2, 3, 4, 5, 6], discrete=True, ax=ax, color="#4C72B0")
    else:
        ax.hist(df["rubric_score"], bins=[0, 1, 2, 3, 4, 5, 6], color="#4C72B0", edgecolor="white")
    ax.set_title("Rubric Score Distribution")
    ax.set_xlabel("Rubric Score (0-5)")
    ax.set_ylabel("Count")
    ax.set_xlim(-0.2, 5.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rubric_score_distribution", dpi=150)
    plt.close(fig)
    print("saved rubric_score_distribution")


def plot_summary_length(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns:
        sns.histplot(df["summary_words"], bins=12, ax=ax, color="#55A868")
    else:
        ax.hist(df["summary_words"], bins=12, color="#55A868", edgecolor="white")
    ax.set_title("Technical Summary Length Distribution")
    ax.set_xlabel("Words in Technical Summary")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "summary_word_count_distribution", dpi=150)
    plt.close(fig)
    print("saved summary_word_count_distribution")


def plot_completeness_heatmap(df):
    subset = df[["risks_count", "plan_items", "questions_count", "summary_words", "rubric_score"]]
    corr = subset.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=-1, vmax=1)
    else:
        im = ax.imshow(corr.values, cmap="YlGnBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Output Feature Correlation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "output_feature_correlation", dpi=150)
    plt.close(fig)
    print("saved output_feature_correlation")


def plot_overview_table(metrics):
    rows = [
        ("Total Files", metrics.get("total_files", 0)),
        ("Schema Pass Rate (%)", metrics.get("schema_pass_rate", 0.0)),
        ("Verdict Valid Rate (%)", metrics.get("verdict_valid_rate", 0.0)),
        ("Confidence Valid Rate (%)", metrics.get("confidence_valid_rate", 0.0)),
        ("Avg Risks Flagged", metrics.get("avg_risks_flagged", 0.0)),
        ("Avg Summary Words", metrics.get("avg_summary_words", 0.0)),
        ("Avg Rubric Score", metrics.get("avg_rubric_score", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Tech Lead Evaluation Overview", pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluation_overview", dpi=150)
    plt.close(fig)
    print("saved evaluation_overview")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, metrics = load_data()
    print(f"loaded {len(df)} rows, generating figures...")

    plot_verdict_distribution(df)
    plot_confidence_distribution(df)
    plot_verdict_by_sector(df)
    plot_rubric_scores(df)
    plot_summary_length(df)
    plot_completeness_heatmap(df)
    plot_overview_table(metrics)

    print(f"done, all figures in {FIGURES_DIR}")
