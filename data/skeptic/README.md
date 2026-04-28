# skeptic data

Dedicated dataset and artifacts for the Skeptic agent.

## Layout
- `skeptic_cases_clean.json`: normalized input cases for skeptical review.
- `outputs/`: per-startup skeptic analysis JSON files.
- `evaluation/`: summary reports and metrics for skeptic outputs.
- `synthetic_cases/`: optional local skeptic pitch source.

## Build dataset
Run:

```bash
python agents/skeptic/data_loader.py
```

If `data/processed/` exists, the loader will normalize those records first.
Otherwise it falls back to synthetic pitches, defaulting to `data/tech_lead/synthetic_cases/all_synthetic_pitches.json` if no skeptic-specific synthetic file exists yet.

## Run Skeptic Batch
Run a local heuristic smoke batch:

```bash
python agents/skeptic/run_batch.py --use-local --limit 5
```

Run against LLM:

```bash
python agents/skeptic/run_batch.py --limit 20
```

Useful flags:
- `--start 20` starts at record index 20.
- `--limit 50` processes only 50 cases.
- `--dataset path/to/file.json` overrides the input dataset file.
- `--model auto` uses the provider-specific default model.

## Evaluate saved outputs
Run:

```bash
python agents/skeptic/evaluate_outputs.py
```
