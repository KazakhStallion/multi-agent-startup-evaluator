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
DATA_DIR = PROJECT_ROOT / "data" / "skeptic"
OUTPUT_DIR = DATA_DIR / "outputs"

FALLBACK_SUMMARY = "The investment case is still too brittle to accept at face value."
FALLBACK_THESIS = "The committee should assume the pitch is weaker than it sounds until the missing evidence is proven."


def _build_prompt(startup: dict) -> str:
    startup_data = startup_to_text(startup)
    return f"""
You are the Skeptic agent in an AI investment committee.
Your role is to pressure-test the startup, identify hidden assumptions, and surface the strongest reasons this company could fail.
Be rigorous, skeptical, and specific. Use only the startup data below. If data is missing, say so directly.

STARTUP DATA:
{startup_data}

Return ONLY valid JSON with exactly these keys:
{{
  "summary": "2-4 sentence skeptical interpretation",
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


def _validate_result(result: dict) -> dict:
    return validate_committee_output(
        result,
        agent="Skeptic",
        role="skeptical",
        fallback_summary=FALLBACK_SUMMARY,
        fallback_thesis=FALLBACK_THESIS,
    )


class SkepticAgent:
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
                "No LLM client configured. Falling back to local skeptical heuristic.",
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
        output_path = OUTPUT_DIR / f"{startup_slug}_skeptic_analysis.json"

        payload = {
            "agent": "Skeptic",
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
        fallback["confidence"] = "Low"
        fallback["scorecard"]["evidence_quality"] = {
            "assessment": "Low",
            "reasoning": reason,
        }
        return _validate_result(fallback)

    def _local_analysis(self, startup: dict) -> dict:
        identity = startup["identity"]
        business = startup["business"]
        team = startup["team"]
        finances = startup["finances"]

        sector = identity["sector"].lower()
        description = business["description"].lower()
        traction = business["traction"].lower()
        founders = team["founders"].lower()
        runway = finances["runway"].lower()

        execution_assessment = "Medium"
        scalability_assessment = "Medium"
        evidence_assessment = "Low"
        risk_assessment = "High"
        decision = "No-Go"
        confidence = "Medium"

        if traction not in {"unknown", "n/a"} or founders not in {"unknown", "n/a"}:
            evidence_assessment = "Medium"
            decision = "Pivot"
        if "contract" in traction or "paying" in traction or "pilot" in traction:
            evidence_assessment = "High"
            decision = "Pivot"
        if any(keyword in description for keyword in ["api", "platform", "workflow", "automation"]):
            scalability_assessment = "High"
        if sector in {"saas", "fintech"} and evidence_assessment == "High":
            execution_assessment = "Medium"
        if runway not in {"unknown", "n/a"} and any(
            token in runway for token in ["12", "18", "24", "long"]
        ):
            risk_assessment = "Medium"
        if evidence_assessment == "High" and risk_assessment == "Medium":
            decision = "Pivot"
            confidence = "Low"

        result = {
            "summary": (
                "Local skeptical heuristic focused on missing evidence, execution gaps, and reasons the startup could fail."
            ),
            "decision": decision,
            "confidence": confidence,
            "scorecard": {
                "execution_feasibility": {
                    "assessment": execution_assessment,
                    "reasoning": "The pitch may describe a plausible product, but execution claims are still only lightly substantiated.",
                },
                "scalability": {
                    "assessment": scalability_assessment,
                    "reasoning": "Scalability depends on whether the delivery model is truly software-leveraged or still services-heavy.",
                },
                "evidence_quality": {
                    "assessment": evidence_assessment,
                    "reasoning": "Most founder pitches leave critical diligence questions unresolved unless traction is explicit.",
                },
                "risk_level": {
                    "assessment": risk_assessment,
                    "reasoning": "Missing details should be treated as risk until the startup proves otherwise.",
                },
            },
            "key_strengths": [
                "The pitch is at least aimed at a specific customer and workflow.",
                "There is a concrete enough product story to interrogate rather than pure vision.",
                "If the traction claims are real, there may be the beginnings of a wedge.",
            ],
            "key_risks": [
                "The startup may be overstating product readiness relative to the evidence shown.",
                "There may be hidden distribution, compliance, or onboarding friction not visible in the pitch.",
                "The team has not yet proven that customer enthusiasm translates into repeatable execution.",
            ],
            "key_questions": [
                "What hard evidence proves customers will keep paying after the pilot or founder-led sales motion ends?",
                "Which assumption in the pitch would most quickly break the business if it is wrong?",
                "What has the team already learned that contradicts the original plan?",
            ],
            "next_steps": [
                "Pressure-test the traction claims with customer references and retention evidence.",
                "Ask for the ugliest known operational bottleneck rather than the happiest-case roadmap.",
                "Require a short list of falsifiable milestones before treating this as a go decision.",
            ],
            "debate": {
                "core_thesis": (
                    "The pitch should be treated as unproven until the team demonstrates real evidence instead of relying on narrative quality."
                ),
                "challenge_for_committee": (
                    "Which assumption are we most tempted to gloss over because the story sounds coherent?"
                ),
                "what_would_change_my_mind": (
                    "Verified traction, referenceable customers, and evidence that the ugliest risks are already understood by the founders."
                ),
            },
        }
        return _validate_result(result)


def run_skeptic_agent(
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
    agent = SkepticAgent(use_local=use_local, model=model)
    return agent.analyze_structured(startup)
