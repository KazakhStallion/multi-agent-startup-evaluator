import json
from pathlib import Path

from agents.committee_adapters import (
    finance_to_committee_output,
    legal_to_committee_output,
    market_to_committee_output,
    product_to_committee_output,
)
from agents.finance.finance_agent import FinanceAgent
from agents.legal.legal_agent import run_legal_agent
from agents.market_analyst.market_analyst_agent import run_market_analyst
from agents.committee_second_round import run_committee_second_round
from agents.moderator.moderator_agent import ModeratorAgent
from agents.product_lead.product_lead_agent import run_product_lead
from agents.skeptic.skeptic_agent import SkepticAgent
from agents.tech_lead.technical_lead_agent import TechnicalLeadAgent


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "data" / "committee_pipeline"


def _build_default_startup():
    return {
        "name": "PayFlow",
        "sector": "fintech",
        "description": "B2B payment automation for small businesses",
        "identity": {
            "name": "PayFlow",
            "sector": "fintech",
            "location": "US",
        },
        "business": {
            "description": "B2B payment automation for small businesses",
            "model": "SaaS",
            "problem": "Manual AP workflows waste time and cause payment errors for SMBs.",
            "solution": "Automated invoice intake, approval routing, and payment scheduling.",
            "target_customer": "US small businesses with 10-200 employees",
            "pricing": "$299/month + usage tiers",
            "traction": "Pilots with 8 paying SMB customers",
        },
        "team": {
            "founders": "2 founders, one ex-fintech engineer and one SMB finance operator",
        },
        "finances": {
            "revenue": 1200000,
            "burn_rate": 85000,
            "funding": 2000000,
            "runway": "24 months",
            "employee_count": 12,
        },
    }


def run_committee_pipeline(startup: dict | None = None, *, second_round: bool = True):
    startup = startup or _build_default_startup()

    # six specialists (native JSON each)
    market_native = run_market_analyst(
        name=startup.get("name", ""),
        sector=startup.get("sector", ""),
        description=startup.get("description", startup.get("business", {}).get("description", "")),
    )
    finance_native_raw = FinanceAgent().analyze(startup)
    finance_native = (
        json.loads(finance_native_raw) if isinstance(finance_native_raw, str) else finance_native_raw
    )
    tech_native = TechnicalLeadAgent().analyze_structured(startup)
    skeptic_native = SkepticAgent().analyze_structured(startup)
    legal_native = run_legal_agent(
        name=startup.get("name", ""),
        sector=startup.get("sector", ""),
        description=startup.get("description", startup.get("business", {}).get("description", "")),
    )
    product_native = run_product_lead(
        name=startup.get("name", ""),
        sector=startup.get("sector", ""),
        description=startup.get("description", startup.get("business", {}).get("description", "")),
    )

    # adapters -> committee rows
    committee_inputs_initial = [
        market_to_committee_output(market_native),
        finance_to_committee_output(finance_native),
        tech_native,
        skeptic_native,
        legal_to_committee_output(legal_native),
        product_to_committee_output(product_native),
    ]

    committee_inputs = (
        run_committee_second_round(startup, committee_inputs_initial)
        if second_round
        else committee_inputs_initial
    )

    moderator = ModeratorAgent()
    moderator_output = moderator.synthesize(startup, committee_inputs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{startup['name'].lower()}_committee_pipeline.json"
    payload = {
        "startup": startup,
        "native_outputs": {
            "market_analyst": market_native,
            "finance": finance_native,
            "tech_lead": tech_native,
            "skeptic": skeptic_native,
            "legal": legal_native,
            "product_lead": product_native,
        },
        "committee_inputs": committee_inputs,
        "moderator_output": moderator_output,
        "pipeline_meta": {"second_round": bool(second_round)},
    }
    if second_round:
        payload["committee_inputs_initial"] = committee_inputs_initial
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {"output_path": str(out_path), "decision": moderator_output.get("final_decision")}


if __name__ == "__main__":
    result = run_committee_pipeline()
    print(f"saved: {result['output_path']}")
    print(f"final moderator decision: {result['decision']}")
