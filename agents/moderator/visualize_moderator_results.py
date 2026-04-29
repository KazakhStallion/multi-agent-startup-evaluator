import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


EVAL_DIR = Path(__file__).parents[2] / "data" / "moderator" / "evaluation"
FIGURES_DIR = EVAL_DIR / "figures"

sns.set_theme(style="whitegrid")


def load_data():
    summary_path = EVAL_DIR / "moderator_eval_summary.json"
    runs_path = EVAL_DIR / "moderator_eval_runs.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"run evaluate_moderator.py first, missing {summary_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"run evaluate_moderator.py first, missing {runs_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with runs_path.open("r", encoding="utf-8") as f:
        runs = json.load(f)

    return summary, runs


def plot_decision_distribution(summary):
    dist = summary.get("final_decision_distribution", {})
    labels = ["Go", "Pivot", "No-Go"]
    values = [dist.get(label, {}).get("count", 0) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=sns.color_palette("muted", len(labels)))
    ax.set_title("Moderator Final Decision Distribution")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Run Count")
    ymax = max(values) if values else 0
    ax.set_ylim(0, max(1, ymax) + 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decision_distribution.png", dpi=150)
    plt.close(fig)
    print("saved decision_distribution")


def plot_confidence_distribution(summary):
    dist = summary.get("confidence_distribution", {})
    labels = ["Low", "Medium", "High"]
    values = [dist.get(label, {}).get("count", 0) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=sns.color_palette("deep", len(labels)))
    ax.set_title("Moderator Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Run Count")
    ymax = max(values) if values else 0
    ax.set_ylim(0, max(1, ymax) + 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distribution.png", dpi=150)
    plt.close(fig)
    print("saved confidence_distribution")


def plot_disagreement_per_run(runs):
    labels = [row.get("startup", f"run_{i+1}") for i, row in enumerate(runs)]
    values = [float(row.get("disagreement_rate", 0.0)) * 100 for row in runs]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    bars = ax.bar(positions, values, color=sns.color_palette("crest", len(labels) if labels else 1))
    ax.set_title("Disagreement Rate per Run")
    ax.set_xlabel("Startup")
    ax.set_ylabel("Disagreement Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "disagreement_per_run.png", dpi=150)
    plt.close(fig)
    print("saved disagreement_per_run")


def plot_avg_disagreement(summary):
    avg_rate = float(summary.get("avg_disagreement_rate", 0.0)) * 100

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Average"], [avg_rate], color=sns.color_palette("rocket", 1))
    ax.set_title("Average Committee Disagreement")
    ax.set_ylabel("Disagreement Rate (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{avg_rate:.1f}%",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "avg_disagreement.png", dpi=150)
    plt.close(fig)
    print("saved avg_disagreement")


def plot_vote_entropy_per_run(runs):
    labels = [row.get("startup", f"run_{i+1}") for i, row in enumerate(runs)]
    values = [float(row.get("vote_entropy_normalized", 0.0)) * 100 for row in runs]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    bars = ax.bar(positions, values, color=sns.color_palette("magma", len(labels) if labels else 1))
    ax.set_title("Vote Polarization per Run (Entropy)")
    ax.set_xlabel("Startup")
    ax.set_ylabel("Normalized Entropy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vote_polarization_per_run.png", dpi=150)
    plt.close(fig)
    print("saved vote_polarization_per_run")


def plot_sector_disagreement(summary):
    by_sector = summary.get("by_sector", {})
    if not by_sector:
        print("no sector data, skipping sector_disagreement")
        return

    sectors = sorted(by_sector.keys())
    values = [float(by_sector[s].get("avg_disagreement_rate", 0.0)) * 100 for s in sectors]
    positions = list(range(len(sectors)))

    fig, ax = plt.subplots(figsize=(max(7, len(sectors) * 1.2), 4.8))
    bars = ax.bar(positions, values, color=sns.color_palette("viridis", len(sectors)))
    ax.set_title("Average Disagreement by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Disagreement Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(positions)
    ax.set_xticklabels(sectors, rotation=30, ha="right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sector_disagreement.png", dpi=150)
    plt.close(fig)
    print("saved sector_disagreement")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_data, run_rows = load_data()
    print(f"loaded {len(run_rows)} moderator runs, generating figures...")

    plot_decision_distribution(summary_data)
    plot_confidence_distribution(summary_data)
    plot_disagreement_per_run(run_rows)
    plot_avg_disagreement(summary_data)
    plot_vote_entropy_per_run(run_rows)
    plot_sector_disagreement(summary_data)

    print(f"done, all figures in {FIGURES_DIR}")

