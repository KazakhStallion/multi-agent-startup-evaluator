# tech_lead data

Dedicated dataset and artifacts for the Technical Lead agent.

## Layout
- `tech_cases_clean.json`: normalized input cases for technical review.
- `outputs/`: per-startup technical analysis JSON files.
- `evaluation/`: future evaluation metrics and reports.
- `synthetic_cases/`: future synthetic technology-focused pitch cases.

## Build dataset
Run:

```bash
python agents/tech_lead/data_loader.py
```

## Run Technical Batch
Run a local heuristic smoke batch:

```bash
python agents/tech_lead/run_batch.py --use-local --limit 5
```

Run against LLM:

```bash
python agents/tech_lead/run_batch.py --limit 20
```

By default the tech lead agent now follows the same client setup as the other agents:
- prefers `OPENAI_API_KEY` with the ARC VT base URL and auto-selects `gpt-oss-120b`
- falls back to `GROQ_API_KEY` if available and auto-selects `llama-3.3-70b-versatile`
- falls back to the local heuristic if no LLM client is configured

Useful flags:
- `--start 20` starts at record index 20.
- `--limit 50` processes only 50 cases.
- `--dataset path/to/file.json` overrides the input dataset file.
- `--model auto` uses the provider-specific default model.
