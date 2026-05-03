# multi-agent-startup-evaluator

Multi-Agent Debate and Decision System for early-stage startup evaluation.

This project runs a committee-style flow: six role-specific agents read the same startup payload, outputs are normalized into one schema, an optional peer revision pass lets roles update once after seeing others, then a moderator produces the final Go/Pivot/No-Go memo with disagreement, risks, and follow-ups.

## What is implemented now

Current committee roles:

- Market Analyst
- Finance
- Technical Lead
- Legal
- Product Lead
- Skeptic

Core pipeline is live in `run_committee_pipeline.py` and writes one trace per startup to `data/committee_pipeline/*_committee_pipeline.json`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Environment keys:

- `OPENAI_API_KEY` (used against `https://llm-api.arc.vt.edu/api/v1/`)
- `GROQ_API_KEY` (fallback provider if OpenAI key is absent)

If neither key is present, several components fall back to local heuristics so the pipeline still runs, but output quality is lower.

## Shared startup input schema

All roles expect a startup dictionary shaped roughly like:

```json
{
  "metadata": {
    "source_file": "optional.json",
    "source_dataset": "optional_tag",
    "label": "optional"
  },
  "identity": {
    "name": "ShiftPay",
    "sector": "fintech",
    "location": "United States"
  },
  "business": {
    "description": "Cloud payroll + AP automation for multi-unit operators",
    "model": "B2B SaaS",
    "problem": "Manual AP and payroll workflows",
    "solution": "Automated intake, approvals, and payment routing",
    "target_customer": "Restaurant groups with 10-200 locations",
    "pricing": "$45/location/month + usage",
    "traction": "Three pilots and two signed contracts"
  },
  "team": {
    "founders": "Domain operator + engineering lead"
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

Minimum useful fields:

- `identity.name`
- `identity.sector`
- `business.description`

`normalize_startup` in `agents/committee_utils.py` fills missing text as `"Unknown"` so fields stay explicit.

## Committee output schema (normalized)

Adapters and validators produce one shared shape per role:

```json
{
  "agent": "Technical Lead",
  "role": "technical",
  "summary": "2-4 sentence assessment",
  "decision": "Go/Pivot/No-Go",
  "confidence": "Low/Medium/High",
  "scorecard": {
    "execution_feasibility": {
      "assessment": "Low/Medium/High",
      "reasoning": "..."
    },
    "scalability": { "assessment": "Low/Medium/High", "reasoning": "..." },
    "evidence_quality": { "assessment": "Low/Medium/High", "reasoning": "..." },
    "risk_level": { "assessment": "Low/Medium/High", "reasoning": "..." }
  },
  "key_strengths": ["..."],
  "key_risks": ["..."],
  "key_questions": ["..."],
  "next_steps": ["..."],
  "debate": {
    "core_thesis": "...",
    "challenge_for_committee": "...",
    "what_would_change_my_mind": "..."
  }
}
```

Rules enforced in code:

- `decision` in `{Go, Pivot, No-Go}`
- `confidence` in `{Low, Medium, High}`
- scorecard assessments in `{Low, Medium, High}`

## Pipeline flow (current)

1. Run six specialist modules and store raw role outputs in `native_outputs`.
2. Map role outputs into `committee_inputs_initial` shape (via adapters + validator).
3. Optional peer revision pass (`agents/committee_second_round.py`) rewrites each row once with peer context.
4. Moderator runs structured `debate_round` generation and final synthesis.
5. Save one JSON trace with startup, native outputs, committee rows, moderator output, and `pipeline_meta`.

`second_round` defaults to `True` in `run_committee_pipeline(...)`.

## Running the project

### Single run

```bash
python run_committee_pipeline.py
```

Programmatic call:

```python
from run_committee_pipeline import run_committee_pipeline

result = run_committee_pipeline(second_round=True)   # default
print(result["output_path"], result["decision"])
```

Disable peer revision:

```python
run_committee_pipeline(second_round=False)
```

### Batch run

```bash
python run_committee_batch.py
python run_committee_batch.py --no-second-round
```

## Moderator outputs and evaluation

Moderator output includes:

- `final_decision`
- `confidence`
- `decision_summary`
- `consensus_points`
- `disagreements`
- `top_risks`
- `required_follow_ups`
- `agent_positions`
- `debate_round` (rebuttals + key shifts)

Evaluation script:

```bash
python agents/moderator/evaluate_moderator.py
```

This writes summary/runs CSV/JSON and figures under `data/moderator/evaluation/`.

## Streamlit dashboard

```bash
streamlit run ui/debate_dashboard.py
```

The UI can:

- view saved committee runs
- show final positions and moderator decision
- show round-one snapshot when peer revision is enabled
- run a new startup from sidebar form

## Main files

- `run_committee_pipeline.py` - orchestrator
- `run_committee_batch.py` - batch execution
- `agents/committee_adapters.py` - role adapters to shared schema
- `agents/committee_second_round.py` - peer revision pass
- `agents/committee_utils.py` - normalization/validation/client routing
- `agents/moderator/moderator_agent.py` - rebuttals + synthesis
- `agents/moderator/evaluate_moderator.py` - metrics
- `ui/debate_dashboard.py` - Streamlit app

## Example notebook

`notebooks/example_committee_run.ipynb` shows a minimal smoke test from Python and how to inspect a saved trace.
