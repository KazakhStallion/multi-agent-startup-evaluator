# multi-agent-startup-evaluator
Multi-Agent Debate & Decision System for Early-Stage Startup Evaluation

Early-stage startup evaluation is uncertain, multi-dimensional, and adversarial. Investors have to weigh technical feasibility, market demand, financial viability, product execution, regulatory exposure, and downside risk with incomplete information. This repo is structured around a committee model where specialized agents evaluate the same startup pitch from different perspectives, debate their conclusions, and feed a moderator that produces a final decision.

## Current status
The long-term target is a 6-agent committee:
- Market Analyst
- Technical Lead
- Finance / VC
- Product Lead
- Risk / Legal
- Skeptic

## Current flow
1. Load one startup pitch JSON using the shared startup schema below.
2. Pass that same pitch to All Agents.
4. Both agents return the same core output schema.
5. Pass those two outputs into `ModeratorAgent`.
6. The moderator returns the current committee decision: `Go`, `Pivot`, or `No-Go`.

Later, the same pattern can be extended so the moderator ingests all 6 agent outputs instead of just 2.

## Shared input schema
All committee agents should receive a startup pitch JSON shaped like this:

```json
{
  "metadata": {
    "source_file": "all_synthetic_pitches.json",
    "source_dataset": "tech_lead_synthetic",
    "label": "Synthetic"
  },
  "identity": {
    "name": "ShiftPay",
    "sector": "fintech",
    "location": "United States"
  },
  "business": {
    "description": "ShiftPay is a cloud-native payroll platform built for multi-unit restaurant groups.",
    "model": "B2B SaaS",
    "problem": "Restaurant groups still compile hours from multiple systems and run payroll manually.",
    "solution": "ShiftPay syncs operational data, automates payroll, and flags compliance exceptions.",
    "target_customer": "Restaurant groups with 10-200 locations",
    "pricing": "$45 per location per month plus $2 per active employee",
    "traction": "Three pilots, 85 locations, 4,200 employees, two signed contracts"
  },
  "team": {
    "founders": "Founder has payroll automation experience; CTO previously built restaurant POS pipelines."
  },
  "finances": {
    "revenue": "Unknown",
    "burn_rate": "Unknown",
    "funding": "Unknown",
    "runway": "12 months",
    "employee_count": "8"
  }
}
```

### Required vs optional fields
The absolute minimum useful fields are:
- `identity.name`
- `identity.sector`
- `business.description`

Strongly recommended fields:
- `business.problem`
- `business.solution`
- `business.target_customer`
- `business.traction`
- `team.founders`
- `finances.runway`

If fields are unknown, use the string `"Unknown"` rather than omitting them.

## Shared agent output schema
`Technical Lead` and `Skeptic` now return the same core schema:

```json
{
  "agent": "Technical Lead",
  "role": "technical",
  "summary": "2-4 sentence interpretation",
  "decision": "Go",
  "confidence": "Medium",
  "scorecard": {
    "execution_feasibility": {
      "assessment": "High",
      "reasoning": "The product scope looks achievable for the team and category."
    },
    "scalability": {
      "assessment": "High",
      "reasoning": "The product appears software-leveraged rather than services-heavy."
    },
    "evidence_quality": {
      "assessment": "Medium",
      "reasoning": "There is some traction but still missing diligence detail."
    },
    "risk_level": {
      "assessment": "Medium",
      "reasoning": "There are still unresolved implementation and operating risks."
    }
  },
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
  "debate": {
    "core_thesis": "The main case this agent is making",
    "challenge_for_committee": "The hardest thing the committee should confront",
    "what_would_change_my_mind": "What evidence would materially shift the decision"
  }
}
```

### Notes
- `decision` must be one of `Go`, `Pivot`, or `No-Go`.
- `confidence` must be one of `Low`, `Medium`, or `High`.
- Every `scorecard` field uses `Low`, `Medium`, or `High`.
- `Technical Lead` currently also emits legacy alias fields so its existing tech-only tooling stays usable.

## Moderator output schema
The moderator currently consumes the 2 agent outputs above and returns:

```json
{
  "agent": "Moderator",
  "committee_size": 2,
  "final_decision": "Pivot",
  "confidence": "Low",
  "decision_summary": "Short synthesis of the current committee view",
  "consensus_points": [
    "point 1",
    "point 2"
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
    {
      "agent": "Technical Lead",
      "decision": "Go",
      "confidence": "Medium",
      "core_thesis": "The strongest technical case for the company"
    },
    {
      "agent": "Skeptic",
      "decision": "Pivot",
      "confidence": "Medium",
      "core_thesis": "The strongest argument against taking the pitch at face value"
    }
  ]
}
```

## Code entry points
Current aligned modules:
- `agents/tech_lead/technical_lead_agent.py`
- `agents/skeptic/skeptic_agent.py`
- `agents/moderator/moderator_agent.py`
- `agents/committee_utils.py`

Useful functions:
- `run_technical_lead(...)`
- `run_skeptic_agent(...)`
- `run_two_agent_committee(startup, ...)`

## Example usage
Run the current 2-agent committee in Python:

```python
from agents.moderator.moderator_agent import run_two_agent_committee

startup = {
    "identity": {
        "name": "ShiftPay",
        "sector": "fintech",
        "location": "United States",
    },
    "business": {
        "description": "Cloud-native payroll platform for multi-unit restaurant groups.",
        "model": "B2B SaaS",
        "problem": "Restaurant payroll is manual and compliance-heavy.",
        "solution": "Automated payroll, tax, and exception handling.",
        "target_customer": "Restaurant groups with 10-200 locations",
        "pricing": "$45/location/month plus usage fees",
        "traction": "Three pilots and two signed contracts",
    },
    "team": {
        "founders": "Payroll automation operator plus restaurant POS infrastructure lead"
    },
    "finances": {
        "runway": "12 months",
        "employee_count": "8"
    }
}

result = run_two_agent_committee(
    startup,
    tech_use_local=True,
    skeptic_use_local=True,
    moderator_use_local=True,
    save=False,
)
```

## Output locations
When using the save-capable methods:
- Tech Lead outputs go to `data/tech_lead/outputs/`
- Skeptic outputs go to `data/skeptic/outputs/`
- Moderator outputs go to `data/moderator/outputs/`

## Next planned expansion
Once the other teammates finish their agents, the intended next step is:
1. normalize them to the same shared input schema
2. normalize them to the same shared output schema
3. pass all 6 outputs into the moderator
4. keep the moderator schema stable while expanding `committee_size`
