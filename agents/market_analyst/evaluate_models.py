"""
I use this to run the market analyst agent across 3 models on 12 test pitches,
3 runs each, so I can compare how consistent and reliable each model is.
Total is 108 API calls, takes a few minutes to finish.
"""

import csv
import json
import os
import sys
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
DATA_DIR = PROJECT_ROOT / "data" / "market_analyst"
EVAL_DIR = DATA_DIR / "evaluation"


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


# need the sibling agent module for evidence building and result validation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_analyst_agent import get_sector_evidence, _validate_result, cb_data, fail_data


api_key = os.getenv("OPENAI_API_KEY")
vt_client = OpenAI(api_key=api_key, base_url="https://llm-api.arc.vt.edu/api/v1/") if api_key else None
ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")

MODELS = [
    {"name": "gpt-oss-120b", "client": vt_client},
    {"name": "Kimi-K2.5", "client": vt_client},
    {"name": "llama3.1:8b", "client": ollama_client},
]

RUNS_PER_MODEL = 3

# 12 pitches across all 7 sectors. mix of strong, weak, and ambiguous on purpose
# so the models have to actually differentiate and not just give everything a 3
# All these pitches I generated using Claude, which are not real startups.
TEST_PITCHES = [
    {
        "name": "LedgerFlow",
        "sector": "fintech",
        # solid B2B fintech, real customers and MRR - should score higher
        "description": (
            "Automated accounts payable platform for mid-size construction companies. "
            "Integrates with QuickBooks and major ERPs to process invoices without manual data entry. "
            "Currently processing $12M in invoices per month across 8 paying customers, "
            "averaging $4,200 MRR each."
        ),
    },
    {
        "name": "CryptoSave",
        "sector": "fintech",
        # no customers, no model, just vibes - should score low
        "description": (
            "App that rounds up everyday purchases and converts spare change into cryptocurrency. "
            "We think crypto is popular with younger users so the market is big. "
            "No paying customers yet, still deciding which coins to support."
        ),
    },
    {
        "name": "CareCoord",
        "sector": "healthtech",
        # clinical pilot with real outcome data, strong pitch
        "description": (
            "Post-discharge care coordination platform for hospital systems. Automates patient "
            "follow-up and books PCP appointments within 72 hours of discharge. "
            "Piloting with 3 hospital networks and showing a 15% reduction in 30-day readmission rates."
        ),
    },
    {
        "name": "WellnessAI",
        "sector": "healthtech",
        # zero validation, typical AI health chatbot idea - weak
        "description": (
            "AI chatbot that gives wellness advice based on symptoms you type in. "
            "Goal is to be the go-to personal health assistant for millennials. "
            "No clinical validation, no hospital partnerships, no paying users."
        ),
    },
    {
        "name": "ReturnKit",
        "sector": "ecommerce",
        # paying customers and solid MRR growth, decent pitch
        "description": (
            "Returns management platform for DTC brands on Shopify. Cuts processing time by 60% "
            "and converts 30% of returns into exchanges instead. "
            "45 paying merchants at $399/month, $18K MRR, growing 20% month over month."
        ),
    },
    {
        "name": "NicheShop",
        "sector": "ecommerce",
        # landing page with zero transactions - clearly weak
        "description": (
            "Marketplace for rare collectibles and limited edition items. "
            "We believe collectors everywhere are underserved by existing platforms. "
            "Have a landing page with 20 listings and zero completed transactions so far."
        ),
    },
    {
        "name": "ProposalDesk",
        "sector": "saas",
        # real product, paying users, but 8% churn is a flag
        "description": (
            "Proposal and contract management tool built for digital marketing agencies. "
            "Auto-generates proposals from templates, tracks opens, and collects e-signatures. "
            "120 active subscribers at $49/month with 8% monthly churn."
        ),
    },
    {
        "name": "TeamSync",
        "sector": "saas",
        # no product, no customer definition, classic pre-idea stage
        "description": (
            "Platform to help distributed teams collaborate better using AI features. "
            "Still exploring exact product direction but know remote work is a pain point. "
            "No shipped product, no revenue, no clear customer definition yet."
        ),
    },
    {
        "name": "LocalPulse",
        "sector": "media",
        # small but real, 12 cities and some ad revenue
        "description": (
            "Hyperlocal news app for neighborhoods and small towns funded through local business ads. "
            "12 cities live, 40K monthly active readers, $8K MRR. "
            "Low CAC because community contributors write most of the content."
        ),
    },
    {
        "name": "SiteSense",
        "sector": "hardware",
        # IoT with actual pilots and decent hardware margins
        "description": (
            "IoT sensor kit for construction job sites that monitors worker location, equipment usage, "
            "and safety compliance in real time. Piloting with 4 general contractors. "
            "Hardware gross margin at 38% with a $299/month software subscription per kit."
        ),
    },
    {
        "name": "MealMap",
        "sector": "food",
        # marketplace model with real GMV growth
        "description": (
            "Platform connecting corporate offices with local caterers for recurring weekly lunch orders. "
            "12% commission per order. 22 office accounts active, "
            "GMV growing 15% month over month for the last 5 months."
        ),
    },
    {
        "name": "HealthyBite",
        "sector": "food",
        # pre-product, made food for friends, classic underprepared pitch
        "description": (
            "Subscription meal delivery service for busy professionals who want to eat healthy. "
            "The market for healthy food is massive and growing. Made meals for friends who liked them. "
            "Need funding to rent a commercial kitchen and hire a head chef."
        ),
    },
]

  
def build_prompt(name, sector, description, evidence):
    return f"""You are the Market Analyst agent in an AI investment committee.
Evaluate the market opportunity for this startup using only the
evidence provided. Separate evidence-based claims from assumptions.
Do not fabricate numbers not present in the evidence.

STARTUP:
Name: {name}
Sector: {sector}
Description: {description}

MARKET EVIDENCE FROM REAL DATASETS:
{evidence}

Return a JSON object with exactly these keys:
{{
  "agent": "Market Analyst",
  "market_sizing": {{
    "tam_estimate": "string",
    "sam_estimate": "string",
    "evidence": "string — cite specific numbers from the evidence above"
  }},
  "competitive_landscape": "string",
  "demand_validation": "string",
  "market_timing": "string",
  "key_claims": ["claim 1", "claim 2", "claim 3"],
  "risks": [
    {{"description": "string", "severity": "high/medium/low", "evidence": "string"}}
  ],
  "score": 3,
  "recommendation": "string"
}}""".strip()


def run_single(pitch, model_name, model_client):
    if model_client is None:
        return None

    evidence = get_sector_evidence(pitch["sector"], cb_data, fail_data)
    prompt = build_prompt(pitch["name"], pitch["sector"], pitch["description"], evidence)

    try:
        response = model_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return _validate_result(parsed)
    except Exception as e:
        print(f"    call failed: {e}")
        return None


def is_evidence_cited(result):
    # just check if there's a number in the evidence field
    # if the model used our dataset it would have cited company counts or funding amounts
    ms = result.get("market_sizing", {})
    evidence_text = ms.get("evidence", "") if isinstance(ms, dict) else ""
    return any(c.isdigit() for c in str(evidence_text))


def score_rubric(result):
    # 5-point automatic quality check, one point each
    score = 0
    risks = result.get("risks", [])

    if len(risks) >= 2:
        score += 1
    if risks and all(str(r.get("evidence", "")).strip() for r in risks):
        score += 1
    if len(result.get("key_claims", [])) >= 2:
        score += 1
    raw_score = result.get("score", 0)
    if isinstance(raw_score, (int, float)) and 1 <= int(raw_score) <= 5:
        score += 1
    if len(str(result.get("recommendation", "")).split()) > 10:
        score += 1

    return score


def extract_metrics(result, pitch, model, run_num):
    risks = result.get("risks", [])
    return {
        "pitch_name": pitch["name"],
        "sector": pitch["sector"],
        "model": model,
        "run": run_num,
        "score": result.get("score", 0),
        "risks_count": len(risks),
        "claims_count": len(result.get("key_claims", [])),
        "evidence_cited": is_evidence_cited(result),
        "risk_severity_high": sum(1 for r in risks if r.get("severity") == "high"),
        "recommendation_length": len(str(result.get("recommendation", "")).split()),
        "rubric_score": score_rubric(result),
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

    models = sorted({r["model"] for r in rows})
    pitches = sorted({r["pitch_name"] for r in rows})

    # how much does each model vary across its 3 runs on the same pitch
    score_variance_per_model = {}
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        per_pitch = [
            safe_variance([r["score"] for r in model_rows if r["pitch_name"] == p])
            for p in pitches
        ]
        score_variance_per_model[model] = round(sum(per_pitch) / len(per_pitch), 4) if per_pitch else 0.0

    # for each pair of models, how often do they agree on a pitch (within 1 score point)
    inter_model_agreement = {}
    for m1, m2 in combinations(models, 2):
        agree, total = 0, 0
        for pitch in pitches:
            s1 = [r["score"] for r in rows if r["model"] == m1 and r["pitch_name"] == pitch]
            s2 = [r["score"] for r in rows if r["model"] == m2 and r["pitch_name"] == pitch]
            if s1 and s2:
                if abs(sum(s1) / len(s1) - sum(s2) / len(s2)) <= 1:
                    agree += 1
                total += 1
        inter_model_agreement[f"{m1} vs {m2}"] = round(agree / total * 100, 1) if total else 0.0

    # did the model actually use numbers from our dataset or just make things up
    evidence_grounding_rate = {}
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        cited = sum(1 for r in model_rows if r["evidence_cited"])
        evidence_grounding_rate[model] = round(cited / len(model_rows) * 100, 1) if model_rows else 0.0

    avg_risks_flagged = {
        model: round(
            sum(r["risks_count"] for r in rows if r["model"] == model) /
            max(1, sum(1 for r in rows if r["model"] == model)),
            2,
        )
        for model in models
    }

    avg_rubric_score = {
        model: round(
            sum(r["rubric_score"] for r in rows if r["model"] == model) /
            max(1, sum(1 for r in rows if r["model"] == model)),
            2,
        )
        for model in models
    }

    return {
        "score_variance_per_model": score_variance_per_model,
        "inter_model_agreement": inter_model_agreement,
        "evidence_grounding_rate": evidence_grounding_rate,
        "avg_risks_flagged": avg_risks_flagged,
        "avg_rubric_score": avg_rubric_score,
    }


if __name__ == "__main__":
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_results = []
    total_calls = len(TEST_PITCHES) * len(MODELS) * RUNS_PER_MODEL
    done = 0

    print(f"starting evaluation: {len(TEST_PITCHES)} pitches, {len(MODELS)} models, {RUNS_PER_MODEL} runs each")
    print(f"total calls: {total_calls}, saving to {EVAL_DIR}")
    print()

    for pitch in TEST_PITCHES:
        print(f"pitch: {pitch['name']} ({pitch['sector']})")
        for model in MODELS:
            for run in range(1, RUNS_PER_MODEL + 1):
                done += 1
                print(f"  [{done}/{total_calls}] {model['name']} run {run}...", end=" ", flush=True)

                result = run_single(pitch, model["name"], model["client"])
                time.sleep(1)

                if result is None:
                    print("skipped")
                    continue

                row = extract_metrics(result, pitch, model["name"], run)
                all_rows.append(row)
                all_results.append({
                    "pitch": pitch["name"],
                    "sector": pitch["sector"],
                    "model": model["name"],
                    "run": run,
                    "result": result,
                })
                print(f"score={row['score']}, risks={row['risks_count']}, rubric={row['rubric_score']}/5")
        print()

    print(f"all calls done. {len(all_rows)} successful out of {total_calls}")
    print("computing metrics and saving...")

    metrics = compute_ml_metrics(all_rows)

    with (EVAL_DIR / "all_runs.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved all_runs.json ({len(all_results)} entries)")

    if all_rows:
        with (EVAL_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"saved summary.csv ({len(all_rows)} rows)")

    with (EVAL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("saved metrics.json")
