"""Peer pass on committee rows after adapters (does not re-run each native agent module)."""
import copy
import json

from agents.committee_utils import (
    build_client,
    default_model_for,
    normalize_startup,
    startup_to_text,
    validate_committee_output,
)


def _digest_peers(rows: list[dict], skip_agent: str) -> str:
    lines = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("agent") == skip_agent:
            continue
        name = r.get("agent", "?")
        dec = r.get("decision", "?")
        conf = r.get("confidence", "?")
        deb = r.get("debate") if isinstance(r.get("debate"), dict) else {}
        thesis = (deb.get("core_thesis") or "").strip()
        if len(thesis) > 300:
            thesis = thesis[:300] + "..."
        lines.append(f"- {name}: {dec} ({conf}). {thesis}")
    return "\n".join(lines) if lines else "(no peer rows)"


def _trim_row_for_prompt(row: dict) -> dict:
    deb = row.get("debate") if isinstance(row.get("debate"), dict) else {}
    return {
        "decision": row.get("decision"),
        "confidence": row.get("confidence"),
        "summary": row.get("summary"),
        "key_risks": (row.get("key_risks") or [])[:5],
        "debate": {
            "core_thesis": deb.get("core_thesis"),
            "challenge_for_committee": deb.get("challenge_for_committee"),
        },
    }


def _build_prompt(startup_norm: dict, row: dict, round1_rows: list[dict]) -> str:
    agent = row.get("agent", "Committee member")
    role = row.get("role", "general")
    peers = _digest_peers(round1_rows, agent)
    startup_block = startup_to_text(startup_norm)
    prior = json.dumps(_trim_row_for_prompt(row), indent=2)
    return f"""
You are the {agent} on a seed-stage investment committee (focus: {role}).

Round 1 is finished. Here is what the OTHER members concluded:
{peers}

The startup pitch (unchanged):
{startup_block}

Your own round-1 assessment (excerpt):
{prior}

Write your ROUND-2 assessment. You may keep or change Go/Pivot/No-Go based on peer arguments.
Return ONLY valid JSON with these keys:
{{
  "summary": "2-4 sentences",
  "decision": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High",
  "scorecard": {{
    "execution_feasibility": {{"assessment": "Low/Medium/High", "reasoning": "..."}},
    "scalability": {{"assessment": "Low/Medium/High", "reasoning": "..."}},
    "evidence_quality": {{"assessment": "Low/Medium/High", "reasoning": "..."}},
    "risk_level": {{"assessment": "Low/Medium/High", "reasoning": "..."}}
  }},
  "key_strengths": ["...", "...", "..."],
  "key_risks": ["...", "...", "..."],
  "key_questions": ["...", "...", "..."],
  "next_steps": ["...", "...", "..."],
  "debate": {{
    "core_thesis": "...",
    "challenge_for_committee": "...",
    "what_would_change_my_mind": "..."
  }}
}}
""".strip()


def _validate_row(parsed: dict, row: dict) -> dict:
    agent = str(row.get("agent", "Unknown")).strip() or "Unknown"
    role = str(row.get("role", "unknown")).strip() or "unknown"
    fb_sum = str(row.get("summary") or "Round 2 summary unavailable.")
    deb = row.get("debate") if isinstance(row.get("debate"), dict) else {}
    fb_thesis = str(deb.get("core_thesis") or "Thesis unavailable.")
    return validate_committee_output(
        parsed if isinstance(parsed, dict) else {},
        agent=agent,
        role=role,
        fallback_summary=fb_sum,
        fallback_thesis=fb_thesis,
    )


def run_committee_second_round(
    startup: dict,
    committee_round1: list[dict],
    *,
    model: str = "auto",
) -> list[dict]:
    startup_norm = normalize_startup(startup)
    client, provider = build_client()
    mname = model if model != "auto" else default_model_for(provider)

    if client is None:
        return copy.deepcopy(committee_round1)

    out: list[dict] = []
    for i, row in enumerate(committee_round1):
        if not isinstance(row, dict):
            out.append(row)
            continue
        prompt = _build_prompt(startup_norm, row, committee_round1)
        try:
            resp = client.chat.completions.create(
                model=mname,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            out.append(_validate_row(parsed, row))
        except Exception:
            out.append(copy.deepcopy(row))

    return out
