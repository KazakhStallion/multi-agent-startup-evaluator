from agents.committee_utils import validate_committee_output


def _safe_text(value, fallback):
    text = str(value).strip() if value is not None else ""
    return text or fallback


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
            "reasoning": "Mapped from overall score.",
        },
        "scalability": {
            "assessment": level,
            "reasoning": "Mapped from overall score.",
        },
        "evidence_quality": {
            "assessment": level,
            "reasoning": "Based on score and returned evidence.",
        },
        "risk_level": {
            "assessment": risk_level,
            "reasoning": "Inverse of score bucket.",
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
        "summary": _safe_text(result.get("competitive_landscape"), "No market summary provided."),
        "decision": decision,
        "confidence": confidence,
        "scorecard": _scorecard_from_score(score_1_to_10),
        "key_strengths": [
            _safe_text(market_sizing.get("tam_estimate"), "TAM estimate missing."),
            _safe_text(market_sizing.get("sam_estimate"), "SAM estimate missing."),
            _safe_text(result.get("demand_validation"), "Demand validation missing."),
        ],
        "key_risks": [
            _safe_text(r.get("description"), "Risk description missing.")
            for r in result.get("risks", [])
            if isinstance(r, dict)
        ][:5]
        or ["No explicit market risks returned."],
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
            "what_would_change_my_mind": _safe_text(
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
                "reasoning": "Mapped from finance rating.",
            },
            "scalability": {
                "assessment": "High" if _safe_text(cap_intensity.get("rating"), "high").lower() == "low" else "Medium",
                "reasoning": _safe_text(cap_intensity.get("reasoning"), "Based on burn profile."),
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
            f"Capital intensity: {_safe_text(cap_intensity.get('rating'), 'n/a')}",
        ],
        "key_risks": [
            _safe_text(cap_risk.get("reasoning"), "No explicit capital risk reasoning provided."),
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
            _safe_text(result.get("ip_defensibility"), "IP defensibility not provided."),
        ],
        "key_risks": [
            _safe_text(item, "Missing legal risk detail.")
            for item in result.get("critical_red_flags", [])
        ][:5]
        or ["No explicit legal red flags returned."],
        "key_questions": [
            "Which compliance obligations must be met before scaling?",
            "What legal blocker can halt go-to-market if not handled now?",
            "What is the highest legal risk concentration by geography?",
        ],
        "next_steps": [
            _safe_text(item, "Legal mitigation step missing.")
            for item in result.get("mitigation_requirements", [])
        ][:5]
        or ["Create a legal mitigation plan before committing capital."],
        "debate": {
            "core_thesis": _safe_text(
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

    adapted = {
        "summary": _safe_text(
            result.get("product_differentiation"),
            "No product summary provided.",
        ),
        "decision": decision,
        "confidence": confidence,
        "scorecard": _scorecard_from_score(score),
        "key_strengths": [
            _safe_text(result.get("product_differentiation"), "Product differentiation not provided."),
            _safe_text(result.get("scalability_assessment"), "Scalability assessment not provided."),
        ],
        "key_risks": [
            _safe_text(item, "Missing product risk detail.")
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
                result.get("product_market_fit_signals"),
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

