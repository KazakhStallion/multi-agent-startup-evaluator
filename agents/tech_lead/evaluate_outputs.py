import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "tech_lead" / "outputs"
EVAL_DIR = PROJECT_ROOT / "data" / "tech_lead" / "evaluation"
SUMMARY_PATH = EVAL_DIR / "summary.csv"
METRICS_PATH = EVAL_DIR / "metrics.json"
ALL_RUNS_PATH = EVAL_DIR / "all_runs.json"

REQUIRED_KEYS = [
    "technical_summary",
    "architecture_feasibility",
    "scalability_outlook",
    "security_and_reliability_risks",
    "build_plan_90_days",
    "tech_due_diligence_questions",
    "technical_verdict",
    "confidence",
]

VALID_VERDICTS = {"Go", "Pivot", "No-Go"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}


def _load_output_files() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*_technical_analysis.json"))


def _safe_list(value) -> list:
    return value if isinstance(value, list) else []


def _rubric_score(analysis: dict) -> int:
    score = 0
    risks = _safe_list(analysis.get("security_and_reliability_risks"))
    plan = _safe_list(analysis.get("build_plan_90_days"))
    questions = _safe_list(analysis.get("tech_due_diligence_questions"))
    summary_words = len(str(analysis.get("technical_summary", "")).split())

    if len(risks) >= 3:
        score += 1
    if len(plan) >= 3:
        score += 1
    if len(questions) >= 3:
        score += 1
    if analysis.get("technical_verdict") in VALID_VERDICTS:
        score += 1
    if summary_words >= 25:
        score += 1
    return score


def _evaluate_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) if isinstance(payload, dict) else {}
    identity = payload.get("startup_identity", {}) if isinstance(payload, dict) else {}

    missing_keys = [key for key in REQUIRED_KEYS if key not in analysis]
    risks = _safe_list(analysis.get("security_and_reliability_risks"))
    plan = _safe_list(analysis.get("build_plan_90_days"))
    questions = _safe_list(analysis.get("tech_due_diligence_questions"))

    verdict = analysis.get("technical_verdict", "Missing")
    confidence = analysis.get("confidence", "Missing")
    summary = str(analysis.get("technical_summary", ""))
    summary_words = len(summary.split())

    return {
        "file": path.name,
        "startup_name": identity.get("name", "Unknown"),
        "sector": identity.get("sector", "Unknown"),
        "verdict": verdict,
        "confidence": confidence,
        "risks_count": len(risks),
        "plan_items": len(plan),
        "questions_count": len(questions),
        "summary_words": summary_words,
        "missing_required_keys": len(missing_keys),
        "has_schema_issues": bool(missing_keys),
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
        "plan_items",
        "questions_count",
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
