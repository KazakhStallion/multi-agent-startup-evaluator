import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_GLOB = "data/committee_pipeline/*_committee_pipeline.json"
OUT_DIR = PROJECT_ROOT / "data" / "moderator" / "evaluation"

VALID_DECISIONS = {"Go", "Pivot", "No-Go"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}


def _safe_text(value, fallback):
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _safe_decision(value):
    decision = _safe_text(value, "Pivot")
    return decision if decision in VALID_DECISIONS else "Pivot"


def _safe_confidence(value):
    confidence = _safe_text(value, "Low").title()
    return confidence if confidence in VALID_CONFIDENCE else "Low"


def _load_payload(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decision_counts(agent_rows):
    counts = Counter()
    for row in agent_rows:
        counts[_safe_decision(row.get("decision"))] += 1
    return dict(counts)


def _vote_entropy(vote_counts, total_votes):
    if total_votes <= 0:
        return 0.0, 0.0

    probs = []
    for label in ["Go", "Pivot", "No-Go"]:
        count = float(vote_counts.get(label, 0))
        if count > 0:
            probs.append(count / total_votes)

    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(3)
    normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0
    return entropy, normalized


def _majority_margin(vote_counts, total_votes):
    if total_votes <= 0:
        return 0.0
    top = sorted([int(vote_counts.get("Go", 0)), int(vote_counts.get("Pivot", 0)), int(vote_counts.get("No-Go", 0))], reverse=True)
    if len(top) < 2:
        return 1.0
    return (top[0] - top[1]) / total_votes


def evaluate_one(path: Path):
    payload = _load_payload(path)
    moderator = payload.get("moderator_output", {}) if isinstance(payload, dict) else {}
    committee = payload.get("committee_inputs", []) if isinstance(payload, dict) else []
    startup = payload.get("startup", {}) if isinstance(payload, dict) else {}

    startup_name = _safe_text(startup.get("name"), path.stem)
    startup_sector = _safe_text(startup.get("sector"), "unknown").lower()
    final_decision = _safe_decision(moderator.get("final_decision"))
    confidence = _safe_confidence(moderator.get("confidence"))

    committee_rows = [x for x in committee if isinstance(x, dict)]
    committee_size = len(committee_rows)

    disagree_count = 0
    for row in committee_rows:
        if _safe_decision(row.get("decision")) != final_decision:
            disagree_count += 1

    disagreement_rate = (disagree_count / committee_size) if committee_size > 0 else 0.0
    vote_counts = _decision_counts(committee_rows)
    vote_entropy, vote_entropy_norm = _vote_entropy(vote_counts, committee_size)
    majority_margin = _majority_margin(vote_counts, committee_size)

    return {
        "startup": startup_name,
        "sector": startup_sector,
        "file": str(path),
        "committee_size": committee_size,
        "final_decision": final_decision,
        "confidence": confidence,
        "disagree_count": disagree_count,
        "disagreement_rate": round(disagreement_rate, 4),
        "vote_entropy": round(vote_entropy, 4),
        "vote_entropy_normalized": round(vote_entropy_norm, 4),
        "majority_margin": round(majority_margin, 4),
        "vote_counts": vote_counts,
    }


def _pct_counts(counter_obj, total):
    out = {}
    for key in ["Go", "Pivot", "No-Go"]:
        count = int(counter_obj.get(key, 0))
        out[key] = {
            "count": count,
            "pct": round((count / total) * 100, 2) if total > 0 else 0.0,
        }
    return out


def _pct_counts_conf(counter_obj, total):
    out = {}
    for key in ["Low", "Medium", "High"]:
        count = int(counter_obj.get(key, 0))
        out[key] = {
            "count": count,
            "pct": round((count / total) * 100, 2) if total > 0 else 0.0,
        }
    return out


def summarize(rows):
    total = len(rows)
    overall_decision_counter = Counter(row["final_decision"] for row in rows)
    overall_confidence_counter = Counter(row["confidence"] for row in rows)

    if total > 0:
        avg_disagreement = sum(row["disagreement_rate"] for row in rows) / total
        avg_vote_entropy = sum(row["vote_entropy"] for row in rows) / total
        avg_vote_entropy_normalized = sum(row["vote_entropy_normalized"] for row in rows) / total
        avg_majority_margin = sum(row["majority_margin"] for row in rows) / total
    else:
        avg_disagreement = 0.0
        avg_vote_entropy = 0.0
        avg_vote_entropy_normalized = 0.0
        avg_majority_margin = 0.0

    by_sector = {}
    sector_buckets = {}
    for row in rows:
        sector = row.get("sector", "unknown")
        sector_buckets.setdefault(sector, []).append(row)

    for sector, items in sector_buckets.items():
        n = len(items)
        sector_decision_counter = Counter(item["final_decision"] for item in items)
        sector_confidence_counter = Counter(item["confidence"] for item in items)
        mean_disagreement = sum(item["disagreement_rate"] for item in items) / n if n else 0.0
        mean_entropy_norm = sum(item["vote_entropy_normalized"] for item in items) / n if n else 0.0
        by_sector[sector] = {
            "num_runs": n,
            "avg_disagreement_rate": round(mean_disagreement, 4),
            "avg_vote_entropy_normalized": round(mean_entropy_norm, 4),
            "final_decision_distribution": _pct_counts(sector_decision_counter, n),
            "confidence_distribution": _pct_counts_conf(sector_confidence_counter, n),
        }

    return {
        "num_runs": total,
        "avg_disagreement_rate": round(avg_disagreement, 4),
        "avg_vote_entropy": round(avg_vote_entropy, 4),
        "avg_vote_entropy_normalized": round(avg_vote_entropy_normalized, 4),
        "avg_majority_margin": round(avg_majority_margin, 4),
        "final_decision_distribution": _pct_counts(overall_decision_counter, total),
        "confidence_distribution": _pct_counts_conf(overall_confidence_counter, total),
        "by_sector": by_sector,
    }


def _write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows):
    fieldnames = [
        "startup",
        "sector",
        "file",
        "committee_size",
        "final_decision",
        "confidence",
        "disagree_count",
        "disagreement_rate",
        "vote_entropy",
        "vote_entropy_normalized",
        "majority_margin",
        "go_votes",
        "pivot_votes",
        "no_go_votes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            votes = row.get("vote_counts", {})
            writer.writerow(
                {
                    "startup": row["startup"],
                    "sector": row["sector"],
                    "file": row["file"],
                    "committee_size": row["committee_size"],
                    "final_decision": row["final_decision"],
                    "confidence": row["confidence"],
                    "disagree_count": row["disagree_count"],
                    "disagreement_rate": row["disagreement_rate"],
                    "vote_entropy": row["vote_entropy"],
                    "vote_entropy_normalized": row["vote_entropy_normalized"],
                    "majority_margin": row["majority_margin"],
                    "go_votes": int(votes.get("Go", 0)),
                    "pivot_votes": int(votes.get("Pivot", 0)),
                    "no_go_votes": int(votes.get("No-Go", 0)),
                }
            )


def _collect_files(single_file, input_glob):
    if single_file:
        path = Path(single_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return [path]

    pattern_path = Path(input_glob)
    if pattern_path.is_absolute():
        root = pattern_path.anchor or "/"
        return sorted(Path(root).glob(str(pattern_path).lstrip("/")))

    return sorted(PROJECT_ROOT.glob(input_glob))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate moderator outputs across committee pipeline runs."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Single committee pipeline json to evaluate.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=DEFAULT_INPUT_GLOB,
        help="Glob for committee pipeline files when --file is not provided.",
    )
    args = parser.parse_args()

    files = _collect_files(args.file, args.input_glob)
    files = [p for p in files if p.exists() and p.suffix.lower() == ".json"]
    if not files:
        print("No input files found.")
        return

    rows = [evaluate_one(path) for path in files]
    summary = summarize(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_DIR / "moderator_eval_summary.json", summary)
    _write_json(OUT_DIR / "moderator_eval_runs.json", rows)
    _write_csv(OUT_DIR / "moderator_eval_runs.csv", rows)

    print(f"evaluated files: {len(rows)}")
    print(f"avg disagreement rate: {summary['avg_disagreement_rate']:.4f}")
    print(f"avg vote entropy (norm): {summary['avg_vote_entropy_normalized']:.4f}")
    print(f"avg majority margin: {summary['avg_majority_margin']:.4f}")
    print(f"decision distribution: {summary['final_decision_distribution']}")
    print(f"confidence distribution: {summary['confidence_distribution']}")
    print(f"saved summary: {OUT_DIR / 'moderator_eval_summary.json'}")
    print(f"saved runs: {OUT_DIR / 'moderator_eval_runs.json'}")
    print(f"saved csv: {OUT_DIR / 'moderator_eval_runs.csv'}")


if __name__ == "__main__":
    main()

