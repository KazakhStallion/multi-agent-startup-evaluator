import json
import os
from copy import deepcopy

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
vt_client = (
    OpenAI(
        api_key=api_key,
        base_url="https://llm-api.arc.vt.edu/api/v1/",
    )
    if api_key
    else None
)

DEFAULT_RESULT = {
    "agent": "Finance",
    "burn_efficiency": {
        "calculation": "monthly_burn / monthly_revenue",
        "result": 0.0,
        "reasoning": "Not enough data.",
    },
    "capital_risk": {
        "level": "high",
        "reasoning": "Not enough data.",
    },
    "burn_multiple": {
        "calculation": "monthly_burn / monthly_revenue",
        "result": 0.0,
        "reasoning": "Not enough data.",
    },
    "runway": {
        "calculation": "cash / monthly_burn",
        "months": 0.0,
        "reasoning": "Not enough data.",
    },
    "ten_x_goal": {
        "target_revenue_usd": 0.0,
        "calculation": "funding * 10",
    },
    "capital_intensity": {
        "rating": "high",
        "reasoning": "Not enough data.",
    },
    "founder_inquiry": "What concrete metric will prove this business is durable over the next 12 months?",
    "total_rating": 3,
    "final_decision": {
        "decision": "Pivot",
        "recommendation": "Need clearer financial signals.",
    },
}


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(n, d):
    if d == 0:
        return 0.0
    return n / d


def _deterministic_snapshot(startup):
    finance = startup.get("finances", {}) if isinstance(startup, dict) else {}

    annual_rev = _to_float(finance.get("revenue", 0.0))
    monthly_rev = annual_rev / 12.0
    monthly_burn = _to_float(finance.get("burn_rate", 0.0))
    cash = _to_float(finance.get("funding", 0.0))

    burn_efficiency = _safe_div(monthly_burn, monthly_rev)
    runway_months = _safe_div(cash, monthly_burn)

    return {
        "annual_revenue": annual_rev,
        "monthly_revenue": monthly_rev,
        "monthly_burn": monthly_burn,
        "cash": cash,
        "burn_efficiency": burn_efficiency,
        "burn_multiple": burn_efficiency,
        "runway_months": runway_months,
    }


def _fallback_result(startup):
    snap = _deterministic_snapshot(startup)

    risk = "high"
    decision = "Pivot"
    rating = 3

    if snap["monthly_revenue"] > 0 and snap["burn_efficiency"] <= 1.5 and snap["runway_months"] >= 18:
        risk = "low"
        decision = "Go"
        rating = 8
    elif snap["monthly_revenue"] > 0 and snap["runway_months"] >= 9:
        risk = "medium"
        decision = "Go"
        rating = 6

    capital_intensity = "high"
    if snap["monthly_burn"] < 20000:
        capital_intensity = "low"
    elif snap["monthly_burn"] < 60000:
        capital_intensity = "medium"

    result = deepcopy(DEFAULT_RESULT)
    result["burn_efficiency"] = {
        "calculation": f"{snap['monthly_burn']:.2f} / {snap['monthly_revenue']:.2f}",
        "result": round(snap["burn_efficiency"], 4),
        "reasoning": "Computed directly from provided startup finances.",
    }
    result["burn_multiple"] = {
        "calculation": f"{snap['monthly_burn']:.2f} / {snap['monthly_revenue']:.2f}",
        "result": round(snap["burn_multiple"], 4),
        "reasoning": "Using monthly revenue as net-new revenue proxy.",
    }
    result["runway"] = {
        "calculation": f"{snap['cash']:.2f} / {snap['monthly_burn']:.2f}",
        "months": round(snap["runway_months"], 2),
        "reasoning": "Straight cash runway with zero-growth assumption.",
    }
    result["capital_risk"] = {
        "level": risk,
        "reasoning": "Risk is based on runway and burn efficiency.",
    }
    result["capital_intensity"] = {
        "rating": capital_intensity,
        "reasoning": "Rated from monthly burn level.",
    }
    result["ten_x_goal"] = {
        "target_revenue_usd": round(snap["annual_revenue"] * 10, 2),
        "calculation": f"{snap['annual_revenue']:.2f} * 10",
    }
    result["total_rating"] = rating
    result["final_decision"] = {
        "decision": decision,
        "recommendation": "Decision comes from runway, burn efficiency, and current revenue profile.",
    }
    result["founder_inquiry"] = (
        "What concrete plan gets runway above 18 months without breaking growth momentum?"
    )
    return result


def _extract_json_block(text):
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _validate_result(result):
    out = deepcopy(DEFAULT_RESULT)
    if not isinstance(result, dict):
        return out

    out.update(result)
    out["agent"] = "Finance"

    for key in ["burn_efficiency", "capital_risk", "burn_multiple", "runway", "ten_x_goal", "capital_intensity", "final_decision"]:
        if not isinstance(out.get(key), dict):
            out[key] = deepcopy(DEFAULT_RESULT[key])

    if not isinstance(out.get("founder_inquiry"), str) or not out["founder_inquiry"].strip():
        out["founder_inquiry"] = DEFAULT_RESULT["founder_inquiry"]

    # Normalize decision values
    decision = str(out.get("final_decision", {}).get("decision", "Pivot")).strip().lower()
    decision_map = {
        "go": "Go",
        "no-go": "No-Go",
        "no go": "No-Go",
        "pivot": "Pivot",
    }
    out["final_decision"]["decision"] = decision_map.get(decision, "Pivot")

    # Normalize risk/intensity levels
    risk = str(out.get("capital_risk", {}).get("level", "high")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "high"
    out["capital_risk"]["level"] = risk

    intensity = str(out.get("capital_intensity", {}).get("rating", "high")).strip().lower()
    if intensity not in {"low", "medium", "high"}:
        intensity = "high"
    out["capital_intensity"]["rating"] = intensity

    # Numeric field cleanup
    out["burn_efficiency"]["result"] = round(_to_float(out.get("burn_efficiency", {}).get("result", 0.0)), 4)
    out["burn_multiple"]["result"] = round(_to_float(out.get("burn_multiple", {}).get("result", 0.0)), 4)
    out["runway"]["months"] = round(_to_float(out.get("runway", {}).get("months", 0.0)), 2)
    out["ten_x_goal"]["target_revenue_usd"] = round(_to_float(out.get("ten_x_goal", {}).get("target_revenue_usd", 0.0)), 2)

    rating = int(_to_float(out.get("total_rating", 3), 3))
    out["total_rating"] = max(1, min(10, rating))

    if not isinstance(out["final_decision"].get("recommendation"), str) or not out["final_decision"]["recommendation"].strip():
        out["final_decision"]["recommendation"] = DEFAULT_RESULT["final_decision"]["recommendation"]

    return out


class FinanceAgent:
    def __init__(self, client=vt_client, model="gpt-oss-120b"):
        self.client = client
        self.model = model

    def _build_prompt(self, startup, snap):
        startup_data = f"""
Company Name: {startup.get('identity', {}).get('name')}
Annual Revenue: {snap['annual_revenue']}
Monthly Revenue: {snap['monthly_revenue']:.2f}
Monthly Burn: {snap['monthly_burn']}
Cash: {snap['cash']}

Deterministic checks (must stay consistent):
- burn_efficiency = {snap['burn_efficiency']:.6f}
- burn_multiple = {snap['burn_multiple']:.6f}
- runway_months = {snap['runway_months']:.6f}
""".strip()

        prompt = """
AUDIT TASK: Analyze this startup from a finance investor lens.
Use the numbers exactly as given. Do not invent new numeric inputs.

COMPANY DATA
{startup_data}

Return ONLY valid JSON with exactly these keys:
{{
  "agent": "Finance",
  "burn_efficiency": {{
    "calculation": "string showing the math",
    "result": number,
    "reasoning": "string"
  }},
  "capital_risk": {{
    "level": "low/medium/high",
    "reasoning": "string"
  }},
  "burn_multiple": {{
    "calculation": "string showing the math",
    "result": number,
    "reasoning": "string"
  }},
  "runway": {{
    "calculation": "cash / monthly_burn",
    "months": number,
    "reasoning": "string"
  }},
  "ten_x_goal": {{
    "target_revenue_usd": number,
    "calculation": "string"
  }},
  "capital_intensity": {{
    "rating": "low/medium/high",
    "reasoning": "string"
  }},
  "founder_inquiry": "string",
  "total_rating": 5,
  "final_decision": {{
    "decision": "Go/No-Go/Pivot",
    "recommendation": "string"
  }}
}}
""".strip()

        return prompt.format(startup_data=startup_data)

    def analyze(self, startup):
        fallback = _fallback_result(startup)

        if self.client is None:
            return json.dumps(fallback)

        try:
            snap = _deterministic_snapshot(startup)
            final_prompt = self._build_prompt(startup, snap)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            parsed = json.loads(_extract_json_block(raw_text))
            validated = _validate_result(parsed)
            return json.dumps(validated)
        except Exception as e:
            fail = _validate_result(fallback)
            fail["final_decision"]["recommendation"] = (
                f"Fallback used because model call failed: {e}"
            )
            return json.dumps(fail)
