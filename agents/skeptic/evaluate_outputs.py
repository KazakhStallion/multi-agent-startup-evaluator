import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "skeptic" / "outputs"
EVAL_DIR = PROJECT_ROOT / "data" / "skeptic" / "evaluation"
SUMMARY_PATH = EVAL_DIR / "summary.csv"
METRICS_PATH = EVAL_DIR / "metrics.json"
ALL_RUNS_PATH = EVAL_DIR / "all_runs.json"

REQUIRED_KEYS = [
    "agent",
    "role",
    "summary",
    "decision",
    "confidence",
    "scorecard",
    "key_strengths",
    "key_risks",
    "key_questions",
    "next_steps",
    "debate",
]
VALID_VERDICTS = {"Go", "Pivot", "No-Go"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}
SCORECARD_KEYS = {
    "execution_feasibility",
    "scalability",
    "evidence_quality",
    "risk_level",
}


def _load_output_files() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*_skeptic_analysis.json"))


def _safe_list(value) -> list:
    return value if isinstance(value, list) else []


def _rubric_score(analysis: dict) -> int:
    score = 0
    summary_words = len(str(analysis.get("summary", "")).split())

    if analysis.get("decision") in VALID_VERDICTS:
        score += 1
    if analysis.get("confidence") in VALID_CONFIDENCE:
        score += 1
    if len(_safe_list(analysis.get("key_risks"))) >= 3:
        score += 1
    if len(_safe_list(analysis.get("key_questions"))) >= 3:
        score += 1
    if summary_words >= 20:
        score += 1
    if isinstance(analysis.get("scorecard"), dict) and SCORECARD_KEYS.issubset(analysis["scorecard"].keys()):
        score += 1
    return score


def _evaluate_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) if isinstance(payload, dict) else {}
    identity = payload.get("startup_identity", {}) if isinstance(payload, dict) else {}

    missing_keys = [key for key in REQUIRED_KEYS if key not in analysis]
    scorecard = analysis.get("scorecard", {}) if isinstance(analysis.get("scorecard"), dict) else {}
    missing_scorecard = sorted(SCORECARD_KEYS - set(scorecard.keys()))

    summary = str(analysis.get("summary", ""))
    summary_words = len(summary.split())
    verdict = analysis.get("decision", "Missing")
    confidence = analysis.get("confidence", "Missing")

    return {
        "file": path.name,
        "startup_name": identity.get("name", "Unknown"),
        "sector": identity.get("sector", "Unknown"),
        "verdict": verdict,
        "confidence": confidence,
        "risks_count": len(_safe_list(analysis.get("key_risks"))),
        "questions_count": len(_safe_list(analysis.get("key_questions"))),
        "next_steps_count": len(_safe_list(analysis.get("next_steps"))),
        "summary_words": summary_words,
        "missing_required_keys": len(missing_keys) + len(missing_scorecard),
        "has_schema_issues": bool(missing_keys or missing_scorecard),
        "verdict_valid": verdict in VALID_VERDICTS,
        "confidence_valid": confidence in VALID_CONFIDENCE,
        "rubric_score": _rubric_score(analysis),
    }


def _write_summary(rows: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "startup_name",
        "sector",
        "verdict",
        "confidence",
        "risks_count",
        "questions_count",
        "next_steps_count",
        "summary_words",
        "missing_required_keys",
        "has_schema_issues",
        "verdict_valid",
        "confidence_valid",
        "rubric_score",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_all_runs(rows: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with ALL_RUNS_PATH.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def _compute_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "total_files": 0,
            "schema_pass_rate": 0.0,
            "verdict_distribution": {},
            "confidence_distribution": {},
            "avg_risks_flagged": 0.0,
            "avg_summary_words": 0.0,
            "avg_rubric_score": 0.0,
        }

    verdict_counts = Counter(row["verdict"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    total = len(rows)
    schema_ok = sum(1 for row in rows if not row["has_schema_issues"])
    valid_verdict = sum(1 for row in rows if row["verdict_valid"])
    valid_conf = sum(1 for row in rows if row["confidence_valid"])

    return {
        "total_files": total,
        "schema_pass_rate": round(100 * schema_ok / total, 2),
        "verdict_valid_rate": round(100 * valid_verdict / total, 2),
        "confidence_valid_rate": round(100 * valid_conf / total, 2),
        "verdict_distribution": dict(verdict_counts),
        "confidence_distribution": dict(confidence_counts),
        "avg_risks_flagged": round(mean(row["risks_count"] for row in rows), 2),
        "avg_summary_words": round(mean(row["summary_words"] for row in rows), 2),
        "avg_rubric_score": round(mean(row["rubric_score"] for row in rows), 2),
    }


def _write_metrics(metrics: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    files = _load_output_files()
    rows = [_evaluate_file(path) for path in files]

    _write_summary(rows)
    _write_all_runs(rows)
    metrics = _compute_metrics(rows)
    _write_metrics(metrics)

    print(f"Evaluated files: {len(rows)}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"All runs: {ALL_RUNS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
