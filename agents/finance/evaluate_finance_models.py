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
DATA_DIR = PROJECT_ROOT / "data" / "finance"
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
from finance_agent import FinanceAgent


api_key = os.getenv("OPENAI_API_KEY")
vt_client = OpenAI(api_key=api_key, base_url="https://llm-api.arc.vt.edu/api/v1/") if api_key else None
ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")

MODELS = [
    {"name": "gpt-oss-120b", "client": vt_client},
    {"name": "Kimi-K2.5", "client": vt_client},
    {"name": "llama3.1:8b-instruct-q3_K_M", "client": ollama_client},
]

RUNS_PER_MODEL = 3

# 12 pitches across all 7 sectors. mix of strong, weak, and ambiguous on purpose
# so the models have to actually differentiate and not just give everything a 3
# All these pitches I generated using Claude, which are not real startups.
# ADDED FINANCE INFO WITH GEMINI
TEST_PITCHES = [
    {
        "name": "LedgerFlow",
        "sector": "fintech",
        "business": {"description": "Automated accounts payable platform for mid-size construction companies. Integrates with QuickBooks and major ERPs."},
        "finances": {
            "revenue": 403200,    # 8 customers * $4,200 * 12 months
            "burn_rate": 45000,   # Moderate burn for B2B SaaS
            "funding": 1200000    # Seed round
        }
    },
    {
        "name": "CryptoSave",
        "sector": "fintech",
        "business": {"description": "App that rounds up everyday purchases and converts spare change into cryptocurrency."},
        "finances": {
            "revenue": 0,         # "No paying customers yet"
            "burn_rate": 12000,   # Small team/founder burn
            "funding": 50000      # Small pre-seed/angel
        }
    },
    {
        "name": "CareCoord",
        "sector": "healthtech",
        "business": {"description": "Post-discharge care coordination platform for hospital systems. Piloting with 3 hospital networks."},
        "finances": {
            "revenue": 75000,     # Small pilot fees
            "burn_rate": 60000,   # High burn due to clinical/sales complexity
            "funding": 2500000    # Heavy R&D funding
        }
    },
    {
        "name": "WellnessAI",
        "sector": "healthtech",
        "business": {"description": "AI chatbot that gives wellness advice based on symptoms. No clinical validation or hospital partnerships."},
        "finances": {
            "revenue": 0,         # "No paying users"
            "burn_rate": 25000,   # Infrastructure and marketing burn
            "funding": 150000     # Early angel check
        }
    },
    {
        "name": "ReturnKit",
        "sector": "ecommerce",
        "business": {"description": "Returns management platform for DTC brands on Shopify. 45 paying merchants."},
        "finances": {
            "revenue": 215460,    # 45 merchants * $399 * 12 months
            "burn_rate": 30000,   # Lean operations
            "funding": 750000     # Seed stage
        }
    },
    {
        "name": "NicheShop",
        "sector": "ecommerce",
        "business": {"description": "Marketplace for rare collectibles. Landing page with zero completed transactions."},
        "finances": {
            "revenue": 0, 
            "burn_rate": 5000, 
            "funding": 10000
        }
    },
    {
        "name": "ProposalDesk",
        "sector": "saas",
        "business": {"description": "Proposal tool for marketing agencies. 120 subscribers at $49/month with 8% monthly churn."},
        "finances": {
            "revenue": 70560,     # 120 * $49 * 12
            "burn_rate": 15000, 
            "funding": 200000     # Troubled startup with high churn
        }
    },
    {
        "name": "TeamSync",
        "sector": "saas",
        "business": {"description": "Platform for distributed teams. Still exploring product direction; no shipped product."},
        "finances": {
            "revenue": 0, 
            "burn_rate": 20000, 
            "funding": 400000
        }
    },
    {
        "name": "LocalPulse",
        "sector": "media",
        "business": {"description": "Hyperlocal news app funded through local business ads. 12 cities live."},
        "finances": {
            "revenue": 96000,     # $8k MRR * 12
            "burn_rate": 10000,   # Low burn, community-driven
            "funding": 150000
        }
    },
    {
        "name": "SiteSense",
        "sector": "hardware",
        "business": {"description": "IoT sensor kit for construction sites. Hardware gross margin at 38%."},
        "finances": {
            "revenue": 125000,    # Mix of device sales + pilot subs
            "burn_rate": 80000,   # High hardware R&D/manufacturing burn
            "funding": 3000000    # Significant VC funding
        }
    },
    {
        "name": "MealMap",
        "sector": "food",
        "business": {"description": "Platform connecting corporate offices with local caterers for recurring weekly lunch orders. 12% commission per order."},
        "finances": {
            "revenue": 144000,    # Commission based on recurring lunch orders
            "burn_rate": 35000, 
            "funding": 1100000
        }
    },
    {
        "name": "HealthyBite",
        "sector": "food",
        "business": {"description": "Subscription meal delivery service. Pre-product, made food for friends."},
        "finances": {
            "revenue": 0, 
            "burn_rate": 2000,    # Minimal/personal burn
            "funding": 0
        }
    },
]




def run_single(pitch, model_name, model_client):
    if model_client is None:
        return None

    finance_agent = FinanceAgent(client=model_client, model=model_name)

    try:
        response = finance_agent.analyze(pitch)
        return response
    except Exception as e:
        print(f"    call failed: {e}")
        return None




def verify_finance_math(result, pitch):
    """Checks if the LLM's runway calculation matches the actual data."""
    accuracy = {}
    fin = pitch['finances']
    # Ground Truth Calculations
    true_cash = float(fin.get('funding', 0))
    true_burn = float(fin.get('burn_rate', 0))
    true_annual_rev = float(fin.get('revenue', 0))
    true_monthly_rev = true_annual_rev / 12

    def get_num(val):
        """Safely extracts a float from a string, or returns 0.0."""
        if val is None or isinstance(val, (list, dict)):
            return 0.0
        try:
            # This regex-style strip handles "$1,200.50 MRR" correctly
            cleaned = ''.join(c for c in str(val) if c.isdigit() or c == '.')
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0

    # 1. Runway Check (Cash / Burn)
    try:
        expected_runway = true_cash / true_burn if true_burn > 0 else 0
        llm_runway = get_num(result.get("runway", {}).get("months", 0))
        accuracy['runway'] = abs(llm_runway - expected_runway) < (expected_runway * 0.05)
    except: accuracy['runway'] = False

    try:
        expected_eff = true_burn / true_monthly_rev if true_monthly_rev > 0 else 0
        llm_eff = get_num(result.get("burn_efficiency", {}).get("result", 0))
        accuracy['burn_efficiency'] = abs(llm_eff - expected_eff) < (expected_eff * 0.1)
    except: accuracy['burn_efficiency'] = False

    return accuracy


def safe_int(val, default=0):
    """Safely converts a value to an integer, returning a default if it fails."""
    try:
        # Strip any extra text the LLM might have added (e.g., "Rating: 8")
        if isinstance(val, str):
            val = ''.join(c for c in val if c.isdigit() or c == '.')
        return int(float(val))
    except (ValueError, TypeError):
        return default

def score_rubric(result, math_accuracy_pct):
    score = 0
    
    # ... previous rubric checks ...

    # --- THE SAFE FIX ---
    risk_level = str(result.get("capital_risk", {}).get("level", "")).lower()
    
    # Safely convert to int using our helper
    total_rating = safe_int(result.get("total_rating", 0)) 
    
    # Now this comparison is guaranteed not to crash
    if not (risk_level == "high" and total_rating > 8):
        score += 1

    if result.get("final_decision") in ["Go", "No-Go", "Pivot"]:
        score += 1

    return score

# Score result based on number of inputs, math, etc.
def score_rubric(result, math_accuracy_pct):
    score = 0
    
    if math_accuracy_pct >= 95:
        score += 2
    elif math_accuracy_pct >= 70:
        score += 1

    required_keys = [
        "burn_efficiency", "capital_risk", "burn_multiple", "runway", 
        "unit_economics", "ten_x_goal", "capital_intensity", 
        "founder_inquiry", "total_rating", "final_decision"
    ]
    if all(k in result for k in required_keys):
        score += 1

    # SAFE FIXES USING THE HELPER
    if len(str(safe_nested_get(result, "runway", "calculation"))) > 5:
        score += 1

    reasoning_text = str(safe_nested_get(result, "capital_risk", "reasoning"))
    if len(reasoning_text.split()) > 15:
        score += 1

    if safe_nested_get(result, "ten_x_goal", "target_revenue", "Unknown") != "Unknown":
        score += 1

    if len(str(safe_nested_get(result, "unit_economics", "sustainability"))) > 10:
        score += 1

    risk_level = str(safe_nested_get(result, "capital_risk", "level")).lower()
    total_rating = safe_int(result.get("total_rating", 0))
    if not (risk_level == "high" and total_rating > 8):
        score += 1

    if "?" in str(result.get("founder_inquiry", "")) and len(str(result.get("founder_inquiry", ""))) > 10:
        score += 1

    if result.get("final_decision") in ["Go", "No-Go", "Pivot"]:
        score += 1

    return score

def safe_nested_get(data, top_key, sub_key, default=""):
    """Safely gets a nested key, ensuring the top level is actually a dictionary."""
    top_level = data.get(top_key, {})
    if isinstance(top_level, dict):
        return top_level.get(sub_key, default)
    return default


def extract_metrics(result, pitch, model, run_num):
    math_checks = verify_finance_math(result, pitch)
    
    num_checks = len(math_checks)
    num_correct = sum(1 for v in math_checks.values() if v is True)
    math_accuracy_pct = (num_correct / num_checks) * 100 if num_checks > 0 else 0

    return {
        "pitch_name": pitch["name"],
        "sector": pitch["sector"],
        "model": model,
        "run": run_num,
        "score": safe_int(result.get("total_rating", 0)),
        "math_accuracy": math_accuracy_pct,
        # SAFE FIXES USING THE HELPER
        "risk_level": safe_nested_get(result, "capital_risk", "level", "No Level Returned"),
        "capital_intensity": safe_nested_get(result, "capital_intensity", "rating", "No Rating Returned"),
        "recommendation_length": len(str(safe_nested_get(result, "final_decision", "recommendation")).split()),
        "rubric_score": score_rubric(result, math_accuracy_pct),
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
    avg_math_accuracy = {
        model: round(
            sum(r["math_accuracy"] for r in rows if r["model"] == model) /
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
        "avg_math_accuracy": avg_math_accuracy,
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

                raw_result = run_single(pitch, model["name"], model["client"])
                time.sleep(1)
                
                if raw_result is None:
                    print("skipped (API failure)")
                    continue

                if isinstance(raw_result, str):
                    try:
                        # Sometimes models wrap JSON in markdown blocks like ```json ... ```
                        clean_str = raw_result.strip().removeprefix("```json").removesuffix("```").strip()
                        result = json.loads(clean_str)
                    except Exception as e:
                        print(f"skipped (JSON Parse Error: {e})")
                        continue
                else:
                    result = raw_result

                row = extract_metrics(result, pitch, model["name"], run)
                all_rows.append(row)
                all_results.append({
                    "pitch": pitch["name"],
                    "sector": pitch["sector"],
                    "model": model["name"],
                    "run": run,
                    "result": result,
                })
                print(f"score={row['score']}, math={row['math_accuracy']}, rubric={row['rubric_score']}/10")
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
