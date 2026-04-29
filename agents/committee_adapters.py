from agents.committee_utils import validate_committee_output


def _safe_text(value, fallback):
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _tight_text(value, fallback, max_len=220):
    text = _safe_text(value, fallback)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text

    split_points = [". ", "; ", " - "]
    cut = len(text)
    for marker in split_points:
        idx = text.find(marker)
        if 40 <= idx <= max_len:
            cut = min(cut, idx + 1)
    if cut == len(text):
        cut = max_len

    trimmed = text[:cut].rstrip(" ,;:-")
    if not trimmed.endswith("."):
        trimmed += "."
    return trimmed


def _join_text_list(value, fallback):
    if isinstance(value, list):
        parts = [_tight_text(x, "", max_len=180) for x in value if str(x).strip()]
        parts = [p for p in parts if p]
        if parts:
            return "; ".join(parts)
    return _safe_text(value, fallback)


def _is_bucket_label(text):
    low_signal = {"low", "medium", "high", "modest", "moderate", "strong", "weak"}
    return str(text).strip().lower() in low_signal


def _scorecard_from_score(score_1_to_10):
    # simple mapping to keep scores roughly comparable across agents
    if score_1_to_10 >= 8:
        level = "High"
    elif score_1_to_10 >= 5:
        level = "Medium"
    else:
        level = "Low"

    risk_level = "Low" if score_1_to_10 >= 8 else "Medium" if score_1_to_10 >= 5 else "High"
    return {
        "execution_feasibility": {
            "assessment": level,
            "reasoning": f"Derived from score {score_1_to_10}/10 and consistency of returned evidence.",
        },
        "scalability": {
            "assessment": level,
            "reasoning": f"Scalability confidence tracks score bucket at {score_1_to_10}/10.",
        },
        "evidence_quality": {
            "assessment": level,
            "reasoning": "Evidence quality follows how specific and testable the supporting points are.",
        },
        "risk_level": {
            "assessment": risk_level,
            "reasoning": f"Risk is inversely mapped from score bucket ({score_1_to_10}/10).",
        },
    }


def market_to_committee_output(result):
    score_1_to_5 = int(result.get("score", 3))
    score_1_to_10 = max(1, min(10, score_1_to_5 * 2))

    decision = "Pivot"
    if score_1_to_5 >= 4:
        decision = "Go"
    elif score_1_to_5 <= 2:
        decision = "No-Go"

    market_sizing = result.get("market_sizing", {}) if isinstance(result.get("market_sizing"), dict) else {}
    confidence = "Medium" if score_1_to_5 >= 3 else "Low"

    adapted = {
        "summary": _tight_text(result.get("competitive_landscape"), "No market summary provided."),
        "decision": decision,
        "confidence": confidence,
        "scorecard": _scorecard_from_score(score_1_to_10),
        "key_strengths": [
            _tight_text(market_sizing.get("tam_estimate"), "TAM estimate missing."),
            _tight_text(market_sizing.get("sam_estimate"), "SAM estimate missing."),
            _tight_text(result.get("market_timing"), "Market timing notes missing."),
        ],
        "key_risks": [
            _tight_text(r.get("description"), "Risk description missing.", max_len=180)
            for r in result.get("risks", [])
            if isinstance(r, dict)
        ][:5]
        or [
            _tight_text(
                result.get("demand_validation"),
                "No explicit market risks returned.",
            )
        ],
        "key_questions": [
            "What proof exists for repeatable customer demand in the exact target niche?",
            "What is the startup's most defensible market wedge against incumbents?",
            "Which assumption in TAM/SAM has the highest downside if wrong?",
        ],
        "next_steps": [
            "Get customer interviews and conversion proof in the target segment.",
            "List the alternatives buyers are using right now.",
            "Test pricing and retention assumptions with real prospects.",
        ],
        "debate": {
            "core_thesis": _safe_text(result.get("recommendation"), "No market thesis provided."),
            "challenge_for_committee": "Are we overestimating demand from broad market stats?",
            "what_would_change_my_mind": _tight_text(
                market_sizing.get("evidence"),
                "Need clearer evidence tied to this startup's segment.",
            ),
        },
    }
    return validate_committee_output(
        adapted,
        agent="Market Analyst",
        role="market",
        fallback_summary="Market output was incomplete.",
        fallback_thesis="Need better market evidence before a stronger verdict.",
    )


def finance_to_committee_output(result):
    total_rating = int(result.get("total_rating", 5))
    raw_decision = str(result.get("final_decision", {}).get("decision", "Pivot")).strip().lower()

    decision = "Pivot"
    if raw_decision in {"go"}:
        decision = "Go"
    elif raw_decision in {"no-go", "no go"}:
        decision = "No-Go"

    confidence = "High" if total_rating >= 8 else "Medium" if total_rating >= 5 else "Low"
    burn_eff = result.get("burn_efficiency", {}) if isinstance(result.get("burn_efficiency"), dict) else {}
    runway = result.get("runway", {}) if isinstance(result.get("runway"), dict) else {}
    cap_risk = result.get("capital_risk", {}) if isinstance(result.get("capital_risk"), dict) else {}
    cap_intensity = result.get("capital_intensity", {}) if isinstance(result.get("capital_intensity"), dict) else {}

    adapted = {
        "summary": _safe_text(
            result.get("final_decision", {}).get("recommendation"),
            "No finance summary provided.",
        ),
        "decision": decision,
        "confidence": confidence,
        "scorecard": {
            "execution_feasibility": {
                "assessment": "High" if total_rating >= 7 else "Medium" if total_rating >= 4 else "Low",
                "reasoning": f"Mapped from finance rating {total_rating}/10 and stability of cash profile.",
            },
            "scalability": {
                "assessment": "High" if _safe_text(cap_intensity.get("rating"), "high").lower() == "low" else "Medium",
                "reasoning": _tight_text(cap_intensity.get("reasoning"), "Based on burn profile."),
            },
            "evidence_quality": {
                "assessment": "High",
                "reasoning": "Finance output has explicit calculations.",
            },
            "risk_level": {
                "assessment": _safe_text(cap_risk.get("level"), "medium").title(),
                "reasoning": _safe_text(cap_risk.get("reasoning"), "Risk reasoning not provided."),
            },
        },
        "key_strengths": [
            f"Burn efficiency: {_safe_text(burn_eff.get('result'), 'n/a')}",
            f"Runway months: {_safe_text(runway.get('months'), 'n/a')}",
            _tight_text(
                runway.get("reasoning"),
                "Runway gives enough time to execute core milestones.",
                max_len=170,
            ),
        ],
        "key_risks": [
            f"Capital risk level: {_safe_text(cap_risk.get('level'), 'unknown')}",
            _tight_text(cap_risk.get("reasoning"), "No explicit capital risk reasoning provided."),
            f"Capital intensity: {_safe_text(cap_intensity.get('rating'), 'unknown')}",
            "Burn is still close to revenue, so margin error is thin if growth slows.",
        ],
        "key_questions": [
            _safe_text(result.get("founder_inquiry"), "What financial milestone must be hit before the next raise?"),
        ],
        "next_steps": [
            "Track burn efficiency month to month.",
            "Set runway guardrails and fundraise trigger points.",
            "Validate unit economics with cohort data.",
        ],
        "debate": {
            "core_thesis": _safe_text(
                result.get("final_decision", {}).get("recommendation"),
                "Finance thesis missing.",
            ),
            "challenge_for_committee": "Is the current runway sufficient if growth slips below plan?",
            "what_would_change_my_mind": _safe_text(
                cap_risk.get("reasoning"),
                "Cleaner runway and burn profile with verified retention.",
            ),
        },
    }
    return validate_committee_output(
        adapted,
        agent="Finance",
        role="finance",
        fallback_summary="Finance output was incomplete.",
        fallback_thesis="Need a clearer financial risk profile before a stronger verdict.",
    )


def legal_to_committee_output(result):
    score = int(result.get("score", 5))
    verdict_raw = str(result.get("legal_verdict", "Hold")).strip().lower()

    decision = "Pivot"
    if verdict_raw == "go":
        decision = "Go"
    elif verdict_raw in {"no-go", "no go"}:
        decision = "No-Go"

    confidence = "High" if score >= 8 else "Medium" if score >= 5 else "Low"

    adapted = {
        "summary": _safe_text(
            result.get("ip_defensibility"),
            "No legal summary provided.",
        ),
        "decision": decision,
        "confidence": confidence,
        "scorecard": _scorecard_from_score(score),
        "key_strengths": [
            f"Regulatory burden: {_safe_text(result.get('regulatory_burden'), 'Unknown')}",
            _tight_text(result.get("ip_defensibility"), "IP defensibility not provided."),
        ],
        "key_risks": [
            _tight_text(item, "Missing legal risk detail.", max_len=180)
            for item in result.get("critical_red_flags", [])
        ][:5]
        or ["No explicit legal red flags returned."],
        "key_questions": [
            "Which compliance obligations must be met before scaling?",
            "What legal blocker can halt go-to-market if not handled now?",
            "What is the highest legal risk concentration by geography?",
        ],
        "next_steps": [
            _tight_text(item, "Legal mitigation step missing.", max_len=180)
            for item in result.get("mitigation_requirements", [])
        ][:5]
        or ["Create a legal mitigation plan before committing capital."],
        "debate": {
            "core_thesis": _tight_text(
                result.get("ip_defensibility"),
                "Legal thesis missing.",
            ),
            "challenge_for_committee": "Are we underestimating regulatory and liability exposure?",
            "what_would_change_my_mind": "Proof of compliance readiness and lower legal exposure.",
        },
    }
    return validate_committee_output(
        adapted,
        agent="Legal",
        role="legal",
        fallback_summary="Legal output was incomplete.",
        fallback_thesis="Need better legal/compliance evidence before a stronger verdict.",
    )


def product_to_committee_output(result):
    score = int(result.get("score", 5))
    verdict_raw = str(result.get("product_verdict", "Hold")).strip().lower()

    decision = "Pivot"
    if verdict_raw == "go":
        decision = "Go"
    elif verdict_raw in {"no-go", "no go"}:
        decision = "No-Go"

    confidence = "High" if score >= 8 else "Medium" if score >= 5 else "Low"

    product_diff = _safe_text(result.get("product_differentiation"), "")
    feature_priority = _safe_text(result.get("feature_priority"), "")
    scalability_assessment = _safe_text(result.get("scalability_assessment"), "")
    summary_text = product_diff
    if _is_bucket_label(summary_text) or not summary_text:
        summary_text = feature_priority or scalability_assessment or "No product summary provided."

    first_strength = product_diff
    if _is_bucket_label(first_strength) or not first_strength:
        first_strength = feature_priority or "Product differentiation not provided."

    adapted = {
        "summary": _tight_text(summary_text, "No product summary provided."),
        "decision": decision,
        "confidence": confidence,
        "scorecard": _scorecard_from_score(score),
        "key_strengths": [
            _tight_text(first_strength, "Product differentiation not provided."),
            _tight_text(result.get("scalability_assessment"), "Scalability assessment not provided."),
        ],
        "key_risks": [
            _tight_text(item, "Missing product risk detail.", max_len=180)
            for item in result.get("execution_risks", [])
        ][:5]
        or ["No explicit product execution risks returned."],
        "key_questions": [
            "What product signal most strongly indicates PMF progress?",
            "Which user segment has the clearest willingness to pay?",
            "What feature decision carries the highest execution risk?",
        ],
        "next_steps": [
            "Prioritize the roadmap around the highest-confidence user pain.",
            "Validate feature adoption using cohort-level usage data.",
            "Reduce execution risk by sequencing delivery in milestones.",
        ],
        "debate": {
            "core_thesis": _safe_text(
                _join_text_list(
                    result.get("product_market_fit_signals"),
                    "Product thesis missing.",
                ),
                "Product thesis missing.",
            ),
            "challenge_for_committee": "Are we overestimating differentiation versus incumbents?",
            "what_would_change_my_mind": "Stronger PMF evidence and lower execution risk.",
        },
    }
    return validate_committee_output(
        adapted,
        agent="Product Lead",
        role="product",
        fallback_summary="Product output was incomplete.",
        fallback_thesis="Need better PMF and execution evidence before a stronger verdict.",
    )

