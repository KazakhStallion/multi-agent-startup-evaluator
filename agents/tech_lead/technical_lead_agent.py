import json
from pathlib import Path

from agents.committee_utils import (
    build_client,
    default_model_for,
    normalize_startup,
    slugify,
    startup_to_text,
    validate_committee_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tech_lead"
OUTPUT_DIR = DATA_DIR / "outputs"

FALLBACK_SUMMARY = "Not enough evidence to complete the technical review."
FALLBACK_THESIS = "The startup may be interesting, but the technical execution case is still under-specified."


def _build_prompt(startup: dict) -> str:
    startup_data = startup_to_text(startup)
    return f"""
You are the Technical Lead agent in an AI investment committee.
Evaluate the startup only from a technical execution, architecture, scalability, and reliability perspective.
Use only the startup data below. If data is missing, say Unknown and explain what should be validated.

STARTUP DATA:
{startup_data}

Return ONLY valid JSON with exactly these keys:
{{
  "summary": "2-4 sentence technical interpretation",
  "decision": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High",
  "scorecard": {{
    "execution_feasibility": {{
      "assessment": "Low/Medium/High",
      "reasoning": "string"
    }},
    "scalability": {{
      "assessment": "Low/Medium/High",
      "reasoning": "string"
    }},
    "evidence_quality": {{
      "assessment": "Low/Medium/High",
      "reasoning": "string"
    }},
    "risk_level": {{
      "assessment": "Low/Medium/High",
      "reasoning": "string"
    }}
  }},
  "key_strengths": [
    "strength 1",
    "strength 2",
    "strength 3"
  ],
  "key_risks": [
    "risk 1",
    "risk 2",
    "risk 3"
  ],
  "key_questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "next_steps": [
    "next step 1",
    "next step 2",
    "next step 3"
  ],
  "debate": {{
    "core_thesis": "string",
    "challenge_for_committee": "string",
    "what_would_change_my_mind": "string"
  }}
}}
""".strip()


def _with_legacy_tech_fields(result: dict) -> dict:
    enriched = dict(result)
    enriched["technical_summary"] = result["summary"]
    enriched["architecture_feasibility"] = result["scorecard"]["execution_feasibility"]
    enriched["scalability_outlook"] = result["scorecard"]["scalability"]
    enriched["security_and_reliability_risks"] = list(result["key_risks"])
    enriched["build_plan_90_days"] = list(result["next_steps"])
    enriched["tech_due_diligence_questions"] = list(result["key_questions"])
    enriched["technical_verdict"] = result["decision"]
    return enriched


def _validate_result(result: dict) -> dict:
    validated = validate_committee_output(
        result,
        agent="Technical Lead",
        role="technical",
        fallback_summary=FALLBACK_SUMMARY,
        fallback_thesis=FALLBACK_THESIS,
    )
    return _with_legacy_tech_fields(validated)


class TechnicalLeadAgent:
    def __init__(self, use_local: bool = False, model: str = "auto"):
        self.use_local = use_local
        self.client, self.provider = build_client()
        self.model = model if model != "auto" else default_model_for(self.provider)

    def analyze_structured(self, startup: dict) -> dict:
        normalized_startup = normalize_startup(startup)

        if self.use_local:
            return self._local_analysis(normalized_startup)

        if self.client is None:
            return self._build_fallback_result(
                normalized_startup,
                "No LLM client configured. Falling back to local technical heuristic.",
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": _build_prompt(normalized_startup)}],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            return _validate_result(parsed)
        except Exception as exc:
            return self._build_fallback_result(
                normalized_startup,
                f"LLM analysis failed: {exc}",
            )

    def analyze(self, startup: dict) -> str:
        return json.dumps(self.analyze_structured(startup), indent=2)

    def analyze_and_save(self, startup: dict) -> dict:
        normalized_startup = normalize_startup(startup)
        parsed_result = self.analyze_structured(normalized_startup)

        startup_name = normalized_startup["identity"]["name"]
        startup_slug = slugify(startup_name)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{startup_slug}_technical_analysis.json"

        payload = {
            "agent": "Technical Lead",
            "startup_identity": normalized_startup["identity"],
            "analysis": parsed_result,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return {"output_path": str(output_path), "analysis": parsed_result}

    def _build_fallback_result(self, startup: dict, reason: str) -> dict:
        fallback = self._local_analysis(startup)
        risks = list(fallback["key_risks"])
        risks.insert(0, reason)
        fallback["key_risks"] = risks[:4]
        fallback["security_and_reliability_risks"] = list(fallback["key_risks"])
        fallback["confidence"] = "Low"
        fallback["scorecard"]["evidence_quality"] = {
            "assessment": "Low",
            "reasoning": reason,
        }
        return _validate_result(fallback)

    def _local_analysis(self, startup: dict) -> dict:
        identity = startup["identity"]
        business = startup["business"]
        finances = startup["finances"]

        description = business["description"].lower()
        sector = identity["sector"].lower()
        traction = business["traction"].lower()
        runway = finances["runway"].lower()

        execution_assessment = "Medium"
        scalability_assessment = "Medium"
        evidence_assessment = "Low"
        risk_assessment = "Medium"
        decision = "Pivot"

        if any(keyword in description for keyword in ["api", "platform", "automation", "workflow", "dashboard"]):
            scalability_assessment = "High"
        if sector in {"hardware", "biotech", "aerospace", "deeptech"}:
            execution_assessment = "Low"
            risk_assessment = "High"
            decision = "No-Go"
        elif scalability_assessment == "High":
            decision = "Go"

        if traction not in {"unknown", "n/a"}:
            evidence_assessment = "Medium"
        if "pilot" in traction or "contract" in traction or "paying" in traction:
            evidence_assessment = "High"

        if runway not in {"unknown", "n/a"} and any(
            token in runway for token in ["short", "<", "1 month", "2 month", "3 month"]
        ):
            risk_assessment = "High"
            decision = "Pivot"

        result = {
            "summary": (
                "Local heuristic analysis based on the startup description, sector, traction, and runway fields."
            ),
            "decision": decision,
            "confidence": "Low",
            "scorecard": {
                "execution_feasibility": {
                    "assessment": execution_assessment,
                    "reasoning": "Estimated from sector complexity and the limited implementation detail in the pitch.",
                },
                "scalability": {
                    "assessment": scalability_assessment,
                    "reasoning": "Estimated from product keywords and likely software leverage.",
                },
                "evidence_quality": {
                    "assessment": evidence_assessment,
                    "reasoning": "Based on how much traction and operating detail the pitch actually provides.",
                },
                "risk_level": {
                    "assessment": risk_assessment,
                    "reasoning": "Based on sector difficulty, missing architecture detail, and any runway warning signs.",
                },
            },
            "key_strengths": [
                "The pitch describes a concrete workflow problem rather than a vague idea.",
                "The product sounds software-driven, which can support leverage if execution is solid.",
                "There is at least a plausible initial wedge for a focused MVP.",
            ],
            "key_risks": [
                "No explicit architecture diagram or deployment model is provided.",
                "No concrete security posture, compliance boundary, or threat model is described.",
                "No reliability targets, observability plan, or incident response process is described.",
            ],
            "key_questions": [
                "What does the real system architecture and data flow look like?",
                "How will the team handle security, privacy, and compliance from day one?",
                "What are the most likely scaling bottlenecks and how will they be mitigated?",
            ],
            "next_steps": [
                "Draft the reference architecture and core non-functional requirements.",
                "Define the MVP delivery plan with instrumentation, auth, and CI/CD controls.",
                "Run security and load-test planning before scaling customer commitments.",
            ],
            "debate": {
                "core_thesis": (
                    "The startup is most investable if the team can show a credible architecture and disciplined execution plan."
                ),
                "challenge_for_committee": (
                    "Are we underwriting a real technical moat, or just assuming the architecture is straightforward?"
                ),
                "what_would_change_my_mind": (
                    "A concrete architecture review, customer-backed requirements, and proof that the founding team can ship reliably."
                ),
            },
        }
        return _validate_result(result)


def run_technical_lead(
    name: str,
    sector: str,
    description: str,
    *,
    location: str = "Unknown",
    business_model: str = "Unknown",
    problem: str = "Unknown",
    solution: str = "Unknown",
    target_customer: str = "Unknown",
    pricing: str = "Unknown",
    traction: str = "Unknown",
    founders: str = "Unknown",
    revenue: str = "Unknown",
    burn_rate: str = "Unknown",
    funding: str = "Unknown",
    runway: str = "Unknown",
    team_size: str = "Unknown",
    use_local: bool = False,
    model: str = "auto",
) -> dict:
    startup = {
        "identity": {
            "name": name,
            "sector": sector,
            "location": location,
        },
        "business": {
            "description": description,
            "model": business_model,
            "problem": problem,
            "solution": solution,
            "target_customer": target_customer,
            "pricing": pricing,
            "traction": traction,
        },
        "team": {
            "founders": founders,
        },
        "finances": {
            "revenue": revenue,
            "burn_rate": burn_rate,
            "funding": funding,
            "runway": runway,
            "employee_count": team_size,
        },
    }
    agent = TechnicalLeadAgent(use_local=use_local, model=model)
    return agent.analyze_structured(startup)
