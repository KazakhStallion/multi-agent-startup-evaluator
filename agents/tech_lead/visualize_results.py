import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def is_model_eval(df):
    return "model" in df.columns and "verdict_score" in df.columns


def plot_old_verdict_distribution(df):
    counts = df["verdict"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="muted")
    else:
        ax.bar(counts.index, counts.values, color=["#4C72B0", "#55A868", "#C44E52"][: len(counts)])
    ax.set_title("Technical Verdict Distribution")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Count")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "verdict_distribution", dpi=150)
    plt.close(fig)


def plot_old_confidence_distribution(df):
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
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distribution", dpi=150)
    plt.close(fig)


def plot_old_verdict_by_sector(df):
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


def plot_old_rubric_scores(df):
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


def plot_old_summary_length(df):
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


def plot_old_feature_correlation(df):
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


def plot_old_overview(metrics):
    rows = [
        ("Total Files", metrics.get("total_files", 0)),
        ("Schema Pass Rate (%)", metrics.get("schema_pass_rate", 0.0)),
        ("Verdict Valid Rate (%)", metrics.get("verdict_valid_rate", 0.0)),
        ("Confidence Valid Rate (%)", metrics.get("confidence_valid_rate", 0.0)),
        ("Avg Risks Flagged", metrics.get("avg_risks_flagged", 0.0)),
        ("Avg Summary Words", metrics.get("avg_summary_words", 0.0)),
        ("Avg Rubric Score", metrics.get("avg_rubric_score", 0.0)),
    ]
    save_overview_table(rows, "Tech Lead Evaluation Overview", "evaluation_overview")


def plot_model_score_comparison(df):
    pivot = df.groupby(["pitch_name", "model"])["verdict_score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Verdict Score Comparison Across Pitches and Models")
    ax.set_xlabel("Startup Pitch")
    ax.set_ylabel("Average Verdict Score (1=No-Go, 2=Pivot, 3=Go)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Model")
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_comparison", dpi=150)
    plt.close(fig)


def plot_model_avg_score(df):
    stats = df.groupby("model")["verdict_score"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        stats["model"],
        stats["mean"],
        yerr=stats["std"].fillna(0),
        capsize=5,
        color=(sns.color_palette("muted", len(stats)) if sns else ["#4C72B0", "#55A868", "#C44E52"][: len(stats)]),
    )
    ax.set_title("Average Verdict Score per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Verdict Score")
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "avg_score_per_model", dpi=150)
    plt.close(fig)


def plot_model_score_heatmap(df):
    pivot = df.groupby(["pitch_name", "model"])["verdict_score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 10))
    if sns:
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, vmin=1, vmax=3)
    else:
        im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=1, vmax=3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Mean Verdict Score per Pitch per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pitch")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_heatmap", dpi=150)
    plt.close(fig)


def plot_model_rubric_scores(metrics):
    rubric = metrics.get("avg_rubric_score", {})
    if not rubric:
        return

    models = list(rubric.keys())
    scores = [rubric[model] for model in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, scores, color=(sns.color_palette("muted", len(models)) if sns else ["#4C72B0", "#55A868", "#C44E52"][: len(models)]))
    ax.set_title("Average Rubric Quality Score per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rubric Score (0-5)")
    ax.set_ylim(0, 5.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{score:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rubric_scores", dpi=150)
    plt.close(fig)


def plot_model_risk_flagging(df):
    sector_risk = df.groupby(["model", "sector"])["risks_count"].mean().reset_index()
    pivot = sector_risk.pivot(index="sector", columns="model", values="risks_count")

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Average Risks Flagged per Model by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Avg Risks Flagged")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_flagging", dpi=150)
    plt.close(fig)


def plot_model_agreement(metrics):
    agreement_data = metrics.get("inter_model_agreement", {})
    if not agreement_data:
        return

    model_set = set()
    for key in agreement_data:
        for part in key.split(" vs "):
            model_set.add(part.strip())
    model_names = sorted(model_set)
    size = len(model_names)
    matrix = np.full((size, size), 100.0)

    for key, value in agreement_data.items():
        parts = [part.strip() for part in key.split(" vs ")]
        if len(parts) != 2:
            continue
        i = model_names.index(parts[0])
        j = model_names.index(parts[1])
        matrix[i][j] = value
        matrix[j][i] = value

    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
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
    else:
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticklabels(model_names)
        for i in range(size):
            for j in range(size):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Inter-model Agreement Rate (%)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_agreement", dpi=150)
    plt.close(fig)


def plot_model_score_consistency(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    if sns:
        sns.boxplot(data=df, x="model", y="verdict_score", ax=ax, palette="muted")
    else:
        grouped = [df[df["model"] == model]["verdict_score"] for model in sorted(df["model"].unique())]
        ax.boxplot(grouped, labels=sorted(df["model"].unique()))
    ax.set_title("Verdict Score Distribution per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Verdict Score")
    ax.set_ylim(0.5, 3.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_consistency", dpi=150)
    plt.close(fig)


def plot_model_confidence_distribution(df):
    pivot = (
        df.groupby(["model", "confidence"]).size().reset_index(name="count").pivot(
            index="model", columns="confidence", values="count"
        ).fillna(0)
    )
    confidence_order = [column for column in ["Low", "Medium", "High"] if column in pivot.columns]
    if confidence_order:
        pivot = pivot[confidence_order]

    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Confidence Distribution by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.legend(title="Confidence")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_by_model", dpi=150)
    plt.close(fig)


def save_overview_table(rows, title, filename):
    fig, ax = plt.subplots(figsize=(8, max(3.6, 0.45 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_model_overview(metrics, df):
    rows = [
        ("Total Rows", len(df)),
        ("Models Present", ", ".join(sorted(df["model"].unique()))),
        ("Verdict Variance Keys", len(metrics.get("verdict_variance_per_model", {}))),
        ("Agreement Pairs", len(metrics.get("inter_model_agreement", {}))),
    ]

    for model, value in metrics.get("avg_risks_flagged", {}).items():
        rows.append((f"Avg Risks: {model}", value))
    for model, value in metrics.get("avg_rubric_score", {}).items():
        rows.append((f"Avg Rubric: {model}", value))

    save_overview_table(rows, "Tech Lead Model Evaluation Overview", "evaluation_overview")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, metrics = load_data()
    print(f"loaded {len(df)} rows, generating figures...")

    if is_model_eval(df):
        plot_model_score_comparison(df)
        plot_model_avg_score(df)
        plot_model_score_heatmap(df)
        plot_model_rubric_scores(metrics)
        plot_model_risk_flagging(df)
        plot_model_agreement(metrics)
        plot_model_score_consistency(df)
        plot_model_confidence_distribution(df)
        plot_model_overview(metrics, df)
    else:
        plot_old_verdict_distribution(df)
        plot_old_confidence_distribution(df)
        plot_old_verdict_by_sector(df)
        plot_old_rubric_scores(df)
        plot_old_summary_length(df)
        plot_old_feature_correlation(df)
        plot_old_overview(metrics)

    print(f"done, all figures in {FIGURES_DIR}")
