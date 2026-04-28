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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VT_BASE_URL = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_OPENAI_MODEL = "gpt-oss-120b"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
VALID_ASSESSMENTS = {"Low", "Medium", "High"}
VALID_VERDICTS = {"Go", "Pivot", "No-Go"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}
SCORECARD_KEYS = [
    "execution_feasibility",
    "scalability",
    "evidence_quality",
    "risk_level",
]

DEFAULT_STARTUP = {
    "metadata": {
        "source_file": "Unknown",
        "source_dataset": "Unknown",
        "label": "Unknown",
    },
    "identity": {
        "name": "Unknown",
        "sector": "Unknown",
        "location": "Unknown",
    },
    "business": {
        "description": "Unknown",
        "model": "Unknown",
        "problem": "Unknown",
        "solution": "Unknown",
        "target_customer": "Unknown",
        "pricing": "Unknown",
        "traction": "Unknown",
    },
    "team": {
        "founders": "Unknown",
    },
    "finances": {
        "revenue": "Unknown",
        "burn_rate": "Unknown",
        "funding": "Unknown",
        "runway": "Unknown",
        "employee_count": "Unknown",
    },
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


def build_client():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key, base_url=VT_BASE_URL), "openai"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and Groq is not None:
        return Groq(api_key=groq_key), "groq"

    return None, "none"


def default_model_for(provider: str) -> str:
    if provider == "groq":
        return DEFAULT_GROQ_MODEL
    return DEFAULT_OPENAI_MODEL


def deep_copy(value):
    return json.loads(json.dumps(value))


def slugify(value: str) -> str:
    lowered = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_startup"


def _normalize_text(value, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def normalize_startup(startup: dict) -> dict:
    normalized = deep_copy(DEFAULT_STARTUP)
    payload = startup if isinstance(startup, dict) else {}

    metadata = payload.get("metadata", {})
    identity = payload.get("identity", {})
    business = payload.get("business", {})
    team = payload.get("team", {})
    finances = payload.get("finances", {})

    normalized["metadata"]["source_file"] = _normalize_text(
        metadata.get("source_file") or metadata.get("source")
    )
    normalized["metadata"]["source_dataset"] = _normalize_text(metadata.get("source_dataset"))
    normalized["metadata"]["label"] = _normalize_text(metadata.get("label"))

    normalized["identity"]["name"] = _normalize_text(identity.get("name"))
    normalized["identity"]["sector"] = _normalize_text(identity.get("sector"))
    normalized["identity"]["location"] = _normalize_text(identity.get("location"))

    normalized["business"]["description"] = _normalize_text(business.get("description"))
    normalized["business"]["model"] = _normalize_text(
        business.get("model") or business.get("business_model")
    )
    normalized["business"]["problem"] = _normalize_text(business.get("problem"))
    normalized["business"]["solution"] = _normalize_text(business.get("solution"))
    normalized["business"]["target_customer"] = _normalize_text(
        business.get("target_customer")
    )
    normalized["business"]["pricing"] = _normalize_text(business.get("pricing"))
    normalized["business"]["traction"] = _normalize_text(business.get("traction"))

    normalized["team"]["founders"] = _normalize_text(
        team.get("founders") or team.get("team") or payload.get("team")
    )

    normalized["finances"]["revenue"] = _normalize_text(finances.get("revenue"))
    normalized["finances"]["burn_rate"] = _normalize_text(finances.get("burn_rate"))
    normalized["finances"]["funding"] = _normalize_text(finances.get("funding"))
    normalized["finances"]["runway"] = _normalize_text(finances.get("runway"))
    normalized["finances"]["employee_count"] = _normalize_text(finances.get("employee_count"))

    return normalized


def startup_to_text(startup: dict) -> str:
    normalized = normalize_startup(startup)
    metadata = normalized["metadata"]
    identity = normalized["identity"]
    business = normalized["business"]
    team = normalized["team"]
    finances = normalized["finances"]

    return f"""
Company Name: {identity['name']}
Sector: {identity['sector']}
Location: {identity['location']}
Source Label: {metadata['label']}
Description: {business['description']}
Business Model: {business['model']}
Problem: {business['problem']}
Solution: {business['solution']}
Target Customer: {business['target_customer']}
Pricing: {business['pricing']}
Traction: {business['traction']}
Founding Team: {team['founders']}
Revenue: {finances['revenue']}
Burn Rate: {finances['burn_rate']}
Funding: {finances['funding']}
Runway: {finances['runway']}
Employee Count: {finances['employee_count']}
""".strip()


def normalize_assessment(value, fallback: dict) -> dict:
    if not isinstance(value, dict):
        return dict(fallback)

    assessment = _normalize_text(value.get("assessment"), fallback["assessment"]).title()
    if assessment not in VALID_ASSESSMENTS:
        assessment = fallback["assessment"]

    reasoning = _normalize_text(value.get("reasoning"), fallback["reasoning"])
    return {"assessment": assessment, "reasoning": reasoning}


def normalize_list(value, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)

    normalized = [_normalize_text(item, "").strip() for item in value]
    cleaned = [item for item in normalized if item]
    return cleaned or list(fallback)


def committee_output_template(agent: str, role: str, summary: str, thesis: str) -> dict:
    return {
        "agent": agent,
        "role": role,
        "summary": summary,
        "decision": "Pivot",
        "confidence": "Low",
        "scorecard": {
            "execution_feasibility": {
                "assessment": "Medium",
                "reasoning": "Needs deeper validation.",
            },
            "scalability": {
                "assessment": "Medium",
                "reasoning": "Scalability claims need more evidence.",
            },
            "evidence_quality": {
                "assessment": "Medium",
                "reasoning": "Pitch-level evidence is incomplete.",
            },
            "risk_level": {
                "assessment": "Medium",
                "reasoning": "There are unresolved execution and diligence risks.",
            },
        },
        "key_strengths": ["No validated strengths were provided."],
        "key_risks": ["Needs manual committee review."],
        "key_questions": ["What missing evidence would change the decision?"],
        "next_steps": ["Collect the missing diligence inputs before investing."],
        "debate": {
            "core_thesis": thesis,
            "challenge_for_committee": "What is the strongest reason this startup could still fail?",
            "what_would_change_my_mind": "Stronger customer, product, and operating evidence.",
        },
    }


def validate_committee_output(
    result: dict,
    *,
    agent: str,
    role: str,
    fallback_summary: str,
    fallback_thesis: str,
) -> dict:
    template = committee_output_template(agent, role, fallback_summary, fallback_thesis)
    validated = deep_copy(template)
    if isinstance(result, dict):
        validated.update(result)

    validated["agent"] = agent
    validated["role"] = role
    validated["summary"] = _normalize_text(validated.get("summary"), fallback_summary)

    decision = _normalize_text(validated.get("decision"), "Pivot")
    if decision not in VALID_VERDICTS:
        decision = "Pivot"
    validated["decision"] = decision

    confidence = _normalize_text(validated.get("confidence"), "Low").title()
    if confidence not in VALID_CONFIDENCE:
        confidence = "Low"
    validated["confidence"] = confidence

    scorecard = validated.get("scorecard")
    if not isinstance(scorecard, dict):
        scorecard = {}
    validated["scorecard"] = {
        key: normalize_assessment(scorecard.get(key), template["scorecard"][key])
        for key in SCORECARD_KEYS
    }

    validated["key_strengths"] = normalize_list(
        validated.get("key_strengths"),
        template["key_strengths"],
    )
    validated["key_risks"] = normalize_list(
        validated.get("key_risks"),
        template["key_risks"],
    )
    validated["key_questions"] = normalize_list(
        validated.get("key_questions"),
        template["key_questions"],
    )
    validated["next_steps"] = normalize_list(
        validated.get("next_steps"),
        template["next_steps"],
    )

    debate = validated.get("debate")
    if not isinstance(debate, dict):
        debate = {}
    default_debate = template["debate"]
    validated["debate"] = {
        "core_thesis": _normalize_text(debate.get("core_thesis"), default_debate["core_thesis"]),
        "challenge_for_committee": _normalize_text(
            debate.get("challenge_for_committee"),
            default_debate["challenge_for_committee"],
        ),
        "what_would_change_my_mind": _normalize_text(
            debate.get("what_would_change_my_mind"),
            default_debate["what_would_change_my_mind"],
        ),
    }

    return validated
