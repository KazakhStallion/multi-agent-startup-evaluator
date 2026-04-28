import json
import os
import re
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from groq import Groq
except ImportError:
    Groq = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tech_lead"
OUTPUT_DIR = DATA_DIR / "outputs"
VT_BASE_URL = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_OPENAI_MODEL = "gpt-oss-120b"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
VALID_ASSESSMENTS = {"Low", "Medium", "High"}
VALID_VERDICTS = {"Go", "Pivot", "No-Go"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}

DEFAULT_RESULT = {
    "agent": "Technical Lead",
    "technical_summary": "Not enough evidence to complete the technical review.",
    "architecture_feasibility": {
        "assessment": "Medium",
        "reasoning": "No validated architecture details were available.",
    },
    "scalability_outlook": {
        "assessment": "Medium",
        "reasoning": "No validated scaling plan was available.",
    },
    "security_and_reliability_risks": ["Needs manual technical due diligence review."],
    "build_plan_90_days": ["Clarify the technical plan before making a final call."],
    "tech_due_diligence_questions": ["What is the actual system architecture and operating plan?"],
    "technical_verdict": "Pivot",
    "confidence": "Low",
}


def _load_local_env_file() -> None:
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


def _build_client():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key, base_url=VT_BASE_URL), "openai"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and Groq is not None:
        return Groq(api_key=groq_key), "groq"

    return None, "none"


def _default_model_for(provider: str) -> str:
    if provider == "groq":
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_MODEL


def _build_startup_data(startup: dict) -> str:
    identity = startup.get("identity", {})
    business = startup.get("business", {})
    finances = startup.get("finances", {})

    return f"""
Company Name: {identity.get('name', 'Unknown')}
Sector: {identity.get('sector', 'Unknown')}
Location: {identity.get('location', 'Unknown')}
Business Model: {business.get('model', 'Unknown')}
Description: {business.get('description', 'Unknown')}
Team Size (if provided): {finances.get('employee_count', 'Unknown')}
Runway (if provided): {finances.get('runway', 'Unknown')}
""".strip()


def _build_prompt(startup: dict) -> str:
    startup_data = _build_startup_data(startup)
    return f"""
You are the Technical Lead agent in an AI investment committee.
Evaluate this startup from a technology strategy perspective.
Use only the startup data below. If data is missing, say Unknown and explain what should be validated.

STARTUP DATA:
{startup_data}

Return ONLY valid JSON with exactly these keys:
{{
  "technical_summary": "2-4 sentence technical interpretation",
  "architecture_feasibility": {{
    "assessment": "Low/Medium/High",
    "reasoning": "string"
  }},
  "scalability_outlook": {{
    "assessment": "Low/Medium/High",
    "reasoning": "string"
  }},
  "security_and_reliability_risks": [
    "risk 1",
    "risk 2",
    "risk 3"
  ],
  "build_plan_90_days": [
    "milestone 1",
    "milestone 2",
    "milestone 3"
  ],
  "tech_due_diligence_questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "technical_verdict": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High"
}}
""".strip()


def _deep_copy_defaults() -> dict:
    return json.loads(json.dumps(DEFAULT_RESULT))


def _normalize_assessment_block(value: object, fallback: dict) -> dict:
    if not isinstance(value, dict):
        return dict(fallback)

    assessment = str(value.get("assessment", fallback["assessment"])).strip().title()
    if assessment not in VALID_ASSESSMENTS:
        assessment = fallback["assessment"]

    reasoning = str(value.get("reasoning", fallback["reasoning"])).strip() or fallback["reasoning"]
    return {"assessment": assessment, "reasoning": reasoning}


def _normalize_list(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)

    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized or list(fallback)


def _validate_result(result: dict) -> dict:
    validated = _deep_copy_defaults()
    if isinstance(result, dict):
        validated.update(result)

    validated["technical_summary"] = (
        str(validated.get("technical_summary", DEFAULT_RESULT["technical_summary"])).strip()
        or DEFAULT_RESULT["technical_summary"]
    )
    validated["architecture_feasibility"] = _normalize_assessment_block(
        validated.get("architecture_feasibility"),
        DEFAULT_RESULT["architecture_feasibility"],
    )
    validated["scalability_outlook"] = _normalize_assessment_block(
        validated.get("scalability_outlook"),
        DEFAULT_RESULT["scalability_outlook"],
    )
    validated["security_and_reliability_risks"] = _normalize_list(
        validated.get("security_and_reliability_risks"),
        DEFAULT_RESULT["security_and_reliability_risks"],
    )
    validated["build_plan_90_days"] = _normalize_list(
        validated.get("build_plan_90_days"),
        DEFAULT_RESULT["build_plan_90_days"],
    )
    validated["tech_due_diligence_questions"] = _normalize_list(
        validated.get("tech_due_diligence_questions"),
        DEFAULT_RESULT["tech_due_diligence_questions"],
    )

    verdict = str(validated.get("technical_verdict", DEFAULT_RESULT["technical_verdict"])).strip()
    if verdict not in VALID_VERDICTS:
        verdict = DEFAULT_RESULT["technical_verdict"]
    validated["technical_verdict"] = verdict

    confidence = str(validated.get("confidence", DEFAULT_RESULT["confidence"])).strip().title()
    if confidence not in VALID_CONFIDENCE:
        confidence = DEFAULT_RESULT["confidence"]
    validated["confidence"] = confidence
    validated["agent"] = "Technical Lead"

    return validated


class TechnicalLeadAgent:
    def __init__(self, use_local: bool = False, model: str = "auto"):
        self.use_local = use_local
        self.client, self.provider = _build_client()
        self.model = model if model != "auto" else _default_model_for(self.provider)

    def analyze_structured(self, startup: dict) -> dict:
        if self.use_local:
            return self._local_analysis(startup)

        if self.client is None:
            return self._build_fallback_result(
                startup,
                "No LLM client configured. Falling back to local technical heuristic.",
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": _build_prompt(startup)}],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            return _validate_result(parsed)
        except Exception as exc:
            return self._build_fallback_result(
                startup,
                f"LLM analysis failed: {exc}",
            )

    def analyze(self, startup: dict) -> str:
        return json.dumps(self.analyze_structured(startup), indent=2)

    def analyze_and_save(self, startup: dict) -> dict:
        parsed_result = self.analyze_structured(startup)

        identity = startup.get("identity", {})
        startup_name = identity.get("name", "unknown_startup")
        startup_slug = self._slugify(startup_name)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{startup_slug}_technical_analysis.json"

        payload = {
            "agent": "Technical Lead",
            "startup_identity": identity,
            "analysis": parsed_result,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return {"output_path": str(output_path), "analysis": parsed_result}

    def _build_fallback_result(self, startup: dict, reason: str) -> dict:
        fallback = self._local_analysis(startup)
        risks = list(fallback["security_and_reliability_risks"])
        risks.insert(0, reason)
        fallback["security_and_reliability_risks"] = risks[:4]
        fallback["confidence"] = "Low"
        return _validate_result(fallback)

    def _local_analysis(self, startup: dict) -> dict:
        identity = startup.get("identity", {})
        business = startup.get("business", {})
        description = str(business.get("description", "Unknown")).lower()
        sector = str(identity.get("sector", "Unknown")).lower()
        runway = str(startup.get("finances", {}).get("runway", "Unknown")).lower()

        scalability = "Medium"
        feasibility = "Medium"
        verdict = "Pivot"

        if any(keyword in description for keyword in ["ai", "ml", "platform", "api", "automation"]):
            scalability = "High"
        if sector in {"biotech", "hardware", "aerospace", "deeptech"}:
            feasibility = "Low"
            verdict = "No-Go"
        elif scalability == "High":
            verdict = "Go"

        if runway not in {"unknown", "n/a"} and any(
            token in runway for token in ["short", "<", "1 month", "2 month", "3 month"]
        ):
            verdict = "Pivot"

        result = {
            "technical_summary": (
                "Local heuristic analysis based on sector, product description, and any runway data provided."
            ),
            "architecture_feasibility": {
                "assessment": feasibility,
                "reasoning": "Estimated from sector complexity and available description only.",
            },
            "scalability_outlook": {
                "assessment": scalability,
                "reasoning": "Estimated from product keywords and likely software leverage.",
            },
            "security_and_reliability_risks": [
                "No explicit architecture diagram or deployment model provided.",
                "No explicit security posture, compliance scope, or threat model provided.",
                "No reliability SLO/SLA, incident, or observability plan described.",
            ],
            "build_plan_90_days": [
                "Define system architecture and non-functional requirements.",
                "Ship MVP with telemetry, authentication, and CI/CD guardrails.",
                "Run load and security tests, then prioritize remediation backlog.",
            ],
            "tech_due_diligence_questions": [
                "What does your reference architecture and data flow look like?",
                "How will you handle security, privacy, and compliance from day one?",
                "What are your scalability bottlenecks and mitigation plan?",
            ],
            "technical_verdict": verdict,
            "confidence": "Low",
        }
        return _validate_result(result)

    @staticmethod
    def _slugify(value: str) -> str:
        lowered = str(value).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        return slug or "unknown_startup"


def run_technical_lead(
    name: str,
    sector: str,
    description: str,
    *,
    location: str = "Unknown",
    business_model: str = "Unknown",
    team_size: str = "Unknown",
    runway: str = "Unknown",
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
        },
        "finances": {
            "employee_count": team_size,
            "runway": runway,
        },
    }
    agent = TechnicalLeadAgent(use_local=use_local, model=model)
    return agent.analyze_structured(startup)

