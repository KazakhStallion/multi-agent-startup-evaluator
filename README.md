# multi-agent-startup-evaluator

Capstone project: six specialist-style agents read the same startup JSON, we squash their native outputs into one committee schema, optionally let each role revise once after seeing peers, then a moderator model writes Go / Pivot / No-Go plus risks and follow-ups. Everything worth keeping lands in `data/committee_pipeline/` as JSON. There is no single fine-tuned model checkpoint; calls go out to the VT OpenAI-compatible endpoint and/or Groq using keys in your environment (see below).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env` from the sample if you have one, or export:

- `OPENAI_API_KEY` — used with base URL `https://llm-api.arc.vt.edu/api/v1/` (Virginia Tech ARC) for `gpt-oss-120b` by default in `agents/committee_utils.py`.
- `GROQ_API_KEY` — optional fallback; default model there is `llama-3.3-70b-versatile`.

If neither key is set, several agents and the moderator fall back to small local heuristics so imports still work; quality is not the same as hitting the API.

## Run the full committee pipeline

From repo root:

```bash
python run_committee_pipeline.py
```

That runs the default PayFlow-shaped fixture, writes `data/committee_pipeline/payflow_committee_pipeline.json`, and prints the path plus moderator label.

In code, `second_round` defaults to **on** (six extra LLM calls: each role refines its committee row using other members’ round-one stances). To skip that for a faster or cheaper run:

```python
from run_committee_pipeline import run_committee_pipeline
run_committee_pipeline(second_round=False)
```

Batch over several startups (list lives in `run_committee_batch.py`):

```bash
python run_committee_batch.py
# skip peer revision for every startup in the batch:
python run_committee_batch.py --no-second-round
```

## Streamlit UI

```bash
streamlit run ui/debate_dashboard.py
```

Pick a saved `*_committee_pipeline.json`, or use the sidebar form to run a new startup. If `committee_inputs_initial` is in the file, the UI can show round-one vs after peer round.

## Moderator evaluation (figures + summary JSON)

After you have one or more committee JSON files under `data/committee_pipeline/`:

```bash
python agents/moderator/evaluate_moderator.py
```

Outputs go to `data/moderator/evaluation/` (summary JSON, CSV, and matplotlib figures). Your `.gitignore` may ignore `*.png`; that is why plots sometimes do not show up in git unless you adjust ignore rules.

## What’s in a committee trace

Rough shape of `data/committee_pipeline/<name>_committee_pipeline.json`:

- `startup` — input pitch
- `native_outputs` — raw JSON per agent before adapters
- `committee_inputs` — normalized rows the moderator actually saw (after peer round when that ran)
- `committee_inputs_initial` — only when peer revision ran; snapshot before that pass
- `moderator_output` — final decision, confidence, summary lists, `debate_round` (moderator-generated rebuttal block), etc.
- `pipeline_meta` — e.g. `"second_round": true/false`

## Repo layout (high level)

| Path | What it is |
|------|------------|
| `run_committee_pipeline.py` | Orchestrates six agents → adapters → optional `committee_second_round` → moderator |
| `agents/committee_adapters.py` | Maps heterogeneous agent JSON into the shared committee shape |
| `agents/committee_second_round.py` | Peer revision pass on committee rows |
| `agents/moderator/moderator_agent.py` | Debate prompt + final synthesis |
| `agents/moderator/evaluate_moderator.py` | Metrics over saved runs |
| `ui/debate_dashboard.py` | Streamlit front end |

Older helpers like `run_two_agent_committee` in `moderator_agent.py` still exist for debugging but are not the main demo anymore.

## Example notebook

See `notebooks/example_committee_run.ipynb` — load a saved trace or call the pipeline from Python.

## Data / preprocessing / checkpoints

- **Data:** Pitches are JSON you provide or fixtures under `data/`. Some agents (market, legal, product) read static JSON corpora under their own `data/...` folders; paths are in the agent modules. There is no separate mandatory preprocessing pipeline for the committee JSON itself beyond `normalize_startup` in `committee_utils.py`.
- **Model weights:** Not shipped. Inference goes through the APIs above; install deps from `requirements.txt` and set keys.

## Submission checklist (course-style)

| Requirement | Where |
|-------------|--------|
| README | This file |
| Dependencies | `requirements.txt` |
| Example notebook | `notebooks/example_committee_run.ipynb` |
| Preprocessing | Inline normalization + agent-specific readers (no standalone ETL script required for the committee flow) |
| Checkpoints | N/A — use API keys per above |

## License / course

Virginia Tech capstone — update if you need a formal license block.
