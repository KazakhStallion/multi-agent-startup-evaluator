"""
Run the Technical Lead evaluation across the same three models used by market_analyst:
- gpt-oss-120b
- Kimi-K2.5
- llama3.1:8b

Outputs:
- data/tech_lead/evaluation/all_runs.json
- data/tech_lead/evaluation/summary.csv
- data/tech_lead/evaluation/metrics.json
"""

import csv
import json
import os
import time
from itertools import combinations
from pathlib import Path
from statistics import variance

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tech_lead"
EVAL_DIR = DATA_DIR / "evaluation"
DATASET_PATH = DATA_DIR / "tech_cases_clean.json"

RUNS_PER_MODEL = 3
TEST_CASE_LIMIT = 12


def _load_local_env_file():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if load_dotenv:
    load_dotenv()
else:
    _load_local_env_file()


api_key = os.getenv("OPENAI_API_KEY")
vt_client = OpenAI(api_key=api_key, base_url="https://llm-api.arc.vt.edu/api/v1/") if api_key else None
ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")

MODELS = [
    {"name": "gpt-oss-120b", "client": vt_client},
    {"name": "Kimi-K2.5", "client": vt_client},
    {"name": "llama3.1:8b", "client": ollama_client},
]

VERDICT_TO_SCORE = {"No-Go": 1, "Pivot": 2, "Go": 3}


def load_test_cases(limit=TEST_CASE_LIMIT):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("tech_cases_clean.json must be a JSON list")
    return cases[:limit]


def build_prompt(startup):
    identity = startup.get("identity", {})
    business = startup.get("business", {})
    finances = startup.get("finances", {})

    startup_data = f"""
Name: {identity.get('name', 'Unknown')}
Sector: {identity.get('sector', 'Unknown')}
Location: {identity.get('location', 'Unknown')}
Business Model: {business.get('model', 'Unknown')}
Description: {business.get('description', 'Unknown')}
Team Size: {finances.get('employee_count', 'Unknown')}
Runway: {finances.get('runway', 'Unknown')}
"""

    return f"""You are the Technical Lead agent in an AI investment committee.
Evaluate this startup from a technology strategy perspective.
Use only the startup data below. If data is missing, say Unknown.

STARTUP DATA:
{startup_data}

Return ONLY valid JSON with exactly these keys:
{{
  "technical_summary": "string",
  "architecture_feasibility": {{
    "assessment": "Low/Medium/High",
    "reasoning": "string"
  }},
  "scalability_outlook": {{
    "assessment": "Low/Medium/High",
    "reasoning": "string"
  }},
  "security_and_reliability_risks": ["risk 1", "risk 2", "risk 3"],
  "build_plan_90_days": ["milestone 1", "milestone 2", "milestone 3"],
  "tech_due_diligence_questions": ["question 1", "question 2", "question 3"],
  "technical_verdict": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High"
}}""".strip()


def validate_result(parsed):
    if not isinstance(parsed, dict):
        return {}

    parsed.setdefault("technical_summary", "No summary provided.")
    parsed.setdefault("architecture_feasibility", {"assessment": "Medium", "reasoning": "Not provided"})
    parsed.setdefault("scalability_outlook", {"assessment": "Medium", "reasoning": "Not provided"})
    parsed.setdefault("security_and_reliability_risks", [])
    parsed.setdefault("build_plan_90_days", [])
    parsed.setdefault("tech_due_diligence_questions", [])
    parsed.setdefault("technical_verdict", "Pivot")
    parsed.setdefault("confidence", "Low")
    return parsed


def run_single(startup, model_name, model_client):
    if model_client is None:
        return None

    prompt = build_prompt(startup)
    try:
        response = model_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return validate_result(parsed)
    except Exception as exc:
        print(f"    call failed: {exc}")
        return None


def rubric_score(result):
    score = 0
    if len(result.get("security_and_reliability_risks", [])) >= 3:
        score += 1
    if len(result.get("build_plan_90_days", [])) >= 3:
        score += 1
    if len(result.get("tech_due_diligence_questions", [])) >= 3:
        score += 1
    if result.get("technical_verdict") in {"Go", "Pivot", "No-Go"}:
        score += 1
    if len(str(result.get("technical_summary", "")).split()) >= 25:
        score += 1
    return score


def extract_metrics(result, startup, model, run_num):
    identity = startup.get("identity", {})
    verdict = result.get("technical_verdict", "Pivot")
    return {
        "pitch_name": identity.get("name", "Unknown"),
        "sector": identity.get("sector", "Unknown"),
        "model": model,
        "run": run_num,
        "verdict": verdict,
        "verdict_score": VERDICT_TO_SCORE.get(verdict, 2),
        "confidence": result.get("confidence", "Low"),
        "risks_count": len(result.get("security_and_reliability_risks", [])),
        "plan_items": len(result.get("build_plan_90_days", [])),
        "questions_count": len(result.get("tech_due_diligence_questions", [])),
        "summary_words": len(str(result.get("technical_summary", "")).split()),
        "rubric_score": rubric_score(result),
    }


def safe_variance(values):
    if len(values) < 2:
        return 0.0
    try:
        return variance(values)
    except Exception:
        return 0.0


def compute_ml_metrics(rows):
    if not rows:
        return {}

    models = sorted({row["model"] for row in rows})
    pitches = sorted({row["pitch_name"] for row in rows})

    verdict_variance_per_model = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        per_pitch = [
            safe_variance([r["verdict_score"] for r in model_rows if r["pitch_name"] == pitch])
            for pitch in pitches
        ]
        verdict_variance_per_model[model] = round(sum(per_pitch) / len(per_pitch), 4) if per_pitch else 0.0

    inter_model_agreement = {}
    for m1, m2 in combinations(models, 2):
        agree, total = 0, 0
        for pitch in pitches:
            s1 = [r["verdict_score"] for r in rows if r["model"] == m1 and r["pitch_name"] == pitch]
            s2 = [r["verdict_score"] for r in rows if r["model"] == m2 and r["pitch_name"] == pitch]
            if s1 and s2:
                if abs(sum(s1) / len(s1) - sum(s2) / len(s2)) <= 1:
                    agree += 1
                total += 1
        inter_model_agreement[f"{m1} vs {m2}"] = round(agree / total * 100, 1) if total else 0.0

    avg_risks_flagged = {
        model: round(
            sum(r["risks_count"] for r in rows if r["model"] == model)
            / max(1, sum(1 for r in rows if r["model"] == model)),
            2,
        )
        for model in models
    }

    avg_rubric_score = {
        model: round(
            sum(r["rubric_score"] for r in rows if r["model"] == model)
            / max(1, sum(1 for r in rows if r["model"] == model)),
            2,
        )
        for model in models
    }

    return {
        "verdict_variance_per_model": verdict_variance_per_model,
        "inter_model_agreement": inter_model_agreement,
        "avg_risks_flagged": avg_risks_flagged,
        "avg_rubric_score": avg_rubric_score,
    }


if __name__ == "__main__":
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    test_cases = load_test_cases()
    total_calls = len(test_cases) * len(MODELS) * RUNS_PER_MODEL
    done = 0
    all_rows = []
    all_results = []

    print(
        f"starting technical evaluation: {len(test_cases)} pitches, "
        f"{len(MODELS)} models, {RUNS_PER_MODEL} runs each"
    )
    print(f"total calls: {total_calls}, saving to {EVAL_DIR}")
    print()

    for startup in test_cases:
        identity = startup.get("identity", {})
        pitch_name = identity.get("name", "Unknown")
        sector = identity.get("sector", "Unknown")
        print(f"pitch: {pitch_name} ({sector})")

        for model in MODELS:
            for run in range(1, RUNS_PER_MODEL + 1):
                done += 1
                print(f"  [{done}/{total_calls}] {model['name']} run {run}...", end=" ", flush=True)
                result = run_single(startup, model["name"], model["client"])
                time.sleep(1)

                if result is None:
                    print("skipped")
                    continue

                row = extract_metrics(result, startup, model["name"], run)
                all_rows.append(row)
                all_results.append(
                    {
                        "pitch": pitch_name,
                        "sector": sector,
                        "model": model["name"],
                        "run": run,
                        "result": result,
                    }
                )
                print(
                    f"verdict={row['verdict']}, risks={row['risks_count']}, "
                    f"rubric={row['rubric_score']}/5"
                )
        print()

    print(f"all calls done. {len(all_rows)} successful out of {total_calls}")
    print("computing metrics and saving...")

    metrics = compute_ml_metrics(all_rows)

    with (EVAL_DIR / "all_runs.json").open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)
    print(f"saved all_runs.json ({len(all_results)} entries)")

    if all_rows:
        with (EVAL_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"saved summary.csv ({len(all_rows)} rows)")

    with (EVAL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("saved metrics.json")
