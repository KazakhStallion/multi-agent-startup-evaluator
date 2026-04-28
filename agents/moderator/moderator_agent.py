import json
from pathlib import Path

from agents.committee_utils import (
    VALID_CONFIDENCE,
    VALID_VERDICTS,
    build_client,
    default_model_for,
    normalize_list,
    normalize_startup,
    slugify,
    startup_to_text,
)
from agents.skeptic.skeptic_agent import SkepticAgent
from agents.tech_lead.technical_lead_agent import TechnicalLeadAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "moderator"
OUTPUT_DIR = DATA_DIR / "outputs"

DECISION_ORDER = {"No-Go": 1, "Pivot": 2, "Go": 3}
ORDER_TO_DECISION = {value: key for key, value in DECISION_ORDER.items()}

DEFAULT_RESULT = {
    "agent": "Moderator",
    "committee_size": 2,
    "final_decision": "Pivot",
    "confidence": "Low",
    "decision_summary": "The committee needs more evidence before reaching a stronger decision.",
    "consensus_points": ["The committee agrees that more diligence is required."],
    "disagreements": ["The current agent set does not fully agree on the investment case."],
    "top_risks": ["The startup still has unresolved execution and evidence gaps."],
    "required_follow_ups": ["Collect the missing diligence inputs before investing."],
    "agent_positions": [],
}


def _build_prompt(startup: dict, agent_outputs: list[dict]) -> str:
    startup_text = startup_to_text(startup)
    outputs_text = json.dumps(agent_outputs, indent=2)
    return f"""
You are the Moderator agent in an AI investment committee.
Your job is to synthesize the current committee debate into a final investment decision.
Use only the startup data and the agent outputs below.

STARTUP DATA:
{startup_text}

AGENT OUTPUTS:
{outputs_text}

Return ONLY valid JSON with exactly these keys:
{{
  "final_decision": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High",
  "decision_summary": "2-4 sentence synthesis",
  "consensus_points": [
    "point 1",
    "point 2",
    "point 3"
  ],
  "disagreements": [
    "disagreement 1",
    "disagreement 2"
  ],
  "top_risks": [
    "risk 1",
    "risk 2",
    "risk 3"
  ],
  "required_follow_ups": [
    "follow-up 1",
    "follow-up 2",
    "follow-up 3"
  ],
  "agent_positions": [
    {{
      "agent": "Technical Lead",
      "decision": "Go/Pivot/No-Go",
      "confidence": "Low/Medium/High",
      "core_thesis": "string"
    }}
  ]
}}
""".strip()


def _normalize_positions(value) -> list[dict]:
    if not isinstance(value, list):
        return []

    positions = []
    for item in value:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision", "Pivot")).strip()
        if decision not in VALID_VERDICTS:
            decision = "Pivot"
        confidence = str(item.get("confidence", "Low")).strip().title()
        if confidence not in VALID_CONFIDENCE:
            confidence = "Low"
        positions.append(
            {
                "agent": str(item.get("agent", "Unknown")).strip() or "Unknown",
                "decision": decision,
                "confidence": confidence,
                "core_thesis": str(item.get("core_thesis", "No thesis provided.")).strip()
                or "No thesis provided.",
            }
        )
    return positions


def _validate_result(result: dict, committee_size: int, default_positions: list[dict]) -> dict:
    validated = json.loads(json.dumps(DEFAULT_RESULT))
    if isinstance(result, dict):
        validated.update(result)

    validated["agent"] = "Moderator"
    validated["committee_size"] = committee_size

    decision = str(validated.get("final_decision", "Pivot")).strip()
    if decision not in VALID_VERDICTS:
        decision = "Pivot"
    validated["final_decision"] = decision

    confidence = str(validated.get("confidence", "Low")).strip().title()
    if confidence not in VALID_CONFIDENCE:
        confidence = "Low"
    validated["confidence"] = confidence

    validated["decision_summary"] = (
        str(validated.get("decision_summary", DEFAULT_RESULT["decision_summary"])).strip()
        or DEFAULT_RESULT["decision_summary"]
    )
    validated["consensus_points"] = normalize_list(
        validated.get("consensus_points"),
        DEFAULT_RESULT["consensus_points"],
    )
    validated["disagreements"] = normalize_list(
        validated.get("disagreements"),
        DEFAULT_RESULT["disagreements"],
    )
    validated["top_risks"] = normalize_list(
        validated.get("top_risks"),
        DEFAULT_RESULT["top_risks"],
    )
    validated["required_follow_ups"] = normalize_list(
        validated.get("required_follow_ups"),
        DEFAULT_RESULT["required_follow_ups"],
    )

    positions = _normalize_positions(validated.get("agent_positions"))
    validated["agent_positions"] = positions or default_positions

    return validated


class ModeratorAgent:
    def __init__(self, use_local: bool = False, model: str = "auto"):
        self.use_local = use_local
        self.client, self.provider = build_client()
        self.model = model if model != "auto" else default_model_for(self.provider)

    def synthesize(self, startup: dict, agent_outputs: list[dict]) -> dict:
        normalized_startup = normalize_startup(startup)
        normalized_outputs = self._normalize_agent_outputs(agent_outputs)

        if self.use_local:
            return self._local_synthesis(normalized_outputs)

        if self.client is None:
            return self._local_synthesis(
                normalized_outputs,
                extra_risk="No LLM client configured. Falling back to local moderator heuristic.",
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": _build_prompt(normalized_startup, normalized_outputs)}],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            return _validate_result(parsed, len(normalized_outputs), self._default_positions(normalized_outputs))
        except Exception as exc:
            return self._local_synthesis(
                normalized_outputs,
                extra_risk=f"LLM synthesis failed: {exc}",
            )

    def synthesize_and_save(self, startup: dict, agent_outputs: list[dict]) -> dict:
        normalized_startup = normalize_startup(startup)
        final_result = self.synthesize(normalized_startup, agent_outputs)

        startup_slug = slugify(normalized_startup["identity"]["name"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{startup_slug}_committee_decision.json"

        payload = {
            "agent": "Moderator",
            "startup_identity": normalized_startup["identity"],
            "inputs": agent_outputs,
            "decision": final_result,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return {"output_path": str(output_path), "decision": final_result}

    def _normalize_agent_outputs(self, agent_outputs: list[dict]) -> list[dict]:
        normalized = []
        for item in agent_outputs:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "agent": str(item.get("agent", "Unknown")).strip() or "Unknown",
                    "role": str(item.get("role", "unknown")).strip() or "unknown",
                    "summary": str(item.get("summary", "No summary provided.")).strip()
                    or "No summary provided.",
                    "decision": str(item.get("decision", "Pivot")).strip()
                    if str(item.get("decision", "Pivot")).strip() in VALID_VERDICTS
                    else "Pivot",
                    "confidence": str(item.get("confidence", "Low")).strip().title()
                    if str(item.get("confidence", "Low")).strip().title() in VALID_CONFIDENCE
                    else "Low",
                    "key_strengths": normalize_list(item.get("key_strengths"), ["No strengths provided."]),
                    "key_risks": normalize_list(item.get("key_risks"), ["No risks provided."]),
                    "key_questions": normalize_list(item.get("key_questions"), ["No questions provided."]),
                    "next_steps": normalize_list(item.get("next_steps"), ["No next steps provided."]),
                    "debate": {
                        "core_thesis": str(
                            item.get("debate", {}).get("core_thesis", "No thesis provided.")
                        ).strip()
                        or "No thesis provided."
                    },
                }
            )
        return normalized

    def _default_positions(self, agent_outputs: list[dict]) -> list[dict]:
        return [
            {
                "agent": item["agent"],
                "decision": item["decision"],
                "confidence": item["confidence"],
                "core_thesis": item["debate"]["core_thesis"],
            }
            for item in agent_outputs
        ]

    def _local_synthesis(self, agent_outputs: list[dict], extra_risk: str | None = None) -> dict:
        positions = self._default_positions(agent_outputs)
        decisions = [DECISION_ORDER[item["decision"]] for item in positions] or [DECISION_ORDER["Pivot"]]
        average = sum(decisions) / len(decisions)

        if any(item["decision"] == "No-Go" for item in positions) and any(
            item["decision"] == "Go" for item in positions
        ):
            final_decision = "Pivot"
        else:
            final_decision = ORDER_TO_DECISION[round(average)]

        risk_pool = []
        follow_up_pool = []
        for item in agent_outputs:
            risk_pool.extend(item["key_risks"])
            follow_up_pool.extend(item["key_questions"][:2])
            follow_up_pool.extend(item["next_steps"][:1])
        if extra_risk:
            risk_pool.insert(0, extra_risk)

        summary = (
            "The moderator combined the current technical and skeptical views into a provisional committee decision."
        )
        if final_decision == "Go":
            summary = "The current two-agent committee leans positive, but the go decision still depends on validating the flagged risks."
        elif final_decision == "No-Go":
            summary = "The current two-agent committee sees too much unresolved risk to support an investment decision."

        result = {
            "final_decision": final_decision,
            "confidence": "Low" if any(item["confidence"] == "Low" for item in positions) else "Medium",
            "decision_summary": summary,
            "consensus_points": [
                "Both agents focused on execution quality rather than market storytelling alone.",
                "Both agents surfaced diligence gaps that still need explicit validation.",
                "The current decision should be treated as provisional until more committee roles are added.",
            ],
            "disagreements": [
                f"{item['agent']} position: {item['decision']} because {item['core_thesis']}"
                for item in positions
                if len({pos['decision'] for pos in positions}) > 1
            ]
            or ["The current two agents are directionally aligned."],
            "top_risks": risk_pool[:5] or DEFAULT_RESULT["top_risks"],
            "required_follow_ups": follow_up_pool[:5] or DEFAULT_RESULT["required_follow_ups"],
            "agent_positions": positions,
        }
        return _validate_result(result, len(agent_outputs), positions)


def run_two_agent_committee(
    startup: dict,
    *,
    tech_use_local: bool = False,
    skeptic_use_local: bool = False,
    moderator_use_local: bool = False,
    tech_model: str = "auto",
    skeptic_model: str = "auto",
    moderator_model: str = "auto",
    save: bool = False,
) -> dict:
    normalized_startup = normalize_startup(startup)

    tech_agent = TechnicalLeadAgent(use_local=tech_use_local, model=tech_model)
    skeptic_agent = SkepticAgent(use_local=skeptic_use_local, model=skeptic_model)
    moderator_agent = ModeratorAgent(use_local=moderator_use_local, model=moderator_model)

    tech_output = tech_agent.analyze_structured(normalized_startup)
    skeptic_output = skeptic_agent.analyze_structured(normalized_startup)
    agent_outputs = [tech_output, skeptic_output]

    if save:
        moderator_result = moderator_agent.synthesize_and_save(normalized_startup, agent_outputs)
        final_decision = moderator_result["decision"]
        output_path = moderator_result["output_path"]
    else:
        final_decision = moderator_agent.synthesize(normalized_startup, agent_outputs)
        output_path = None

    return {
        "startup": normalized_startup,
        "agent_outputs": agent_outputs,
        "moderator_output": final_decision,
        "output_path": output_path,
    }
