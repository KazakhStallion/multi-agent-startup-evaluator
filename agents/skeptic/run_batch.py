import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.skeptic.skeptic_agent import SkepticAgent

DATASET_PATH = PROJECT_ROOT / "data" / "skeptic" / "skeptic_cases_clean.json"


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON list of startup objects.")
    return payload


def _select_cases(cases: list[dict], start: int, limit: int | None) -> list[dict]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    sliced = cases[start:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0 when provided")
        sliced = sliced[:limit]
    return sliced


def run_batch(dataset_path: Path, start: int, limit: int | None, use_local: bool, model: str) -> dict:
    all_cases = _load_cases(dataset_path)
    selected = _select_cases(all_cases, start=start, limit=limit)
    if not selected:
        return {
            "total_available": len(all_cases),
            "processed": 0,
            "failed": 0,
            "elapsed_seconds": 0.0,
            "failures": [],
        }

    agent = SkepticAgent(use_local=use_local, model=model)
    failures: list[dict] = []
    begin = perf_counter()

    print(f"Loaded {len(all_cases)} cases from {dataset_path}")
    print(f"Processing {len(selected)} case(s) starting at index {start}")
    if use_local:
        print("Mode: local-heuristic")
    else:
        print(f"Mode: llm ({agent.provider}, model={agent.model})")

    for idx, startup in enumerate(selected, start=start):
        name = startup.get("identity", {}).get("name", f"case_{idx}")
        try:
            result = agent.analyze_and_save(startup)
            print(f"[OK] #{idx} {name} -> {result['output_path']}")
        except Exception as exc:
            failures.append({"index": idx, "name": name, "error": str(exc)})
            print(f"[FAIL] #{idx} {name}: {exc}")

    elapsed = perf_counter() - begin
    return {
        "total_available": len(all_cases),
        "processed": len(selected) - len(failures),
        "failed": len(failures),
        "elapsed_seconds": round(elapsed, 2),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Skeptic agent over skeptic_cases_clean.json")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to skeptic dataset JSON",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of cases to process (default: all remaining)",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local heuristic mode instead of LLM API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Model name when running with LLM mode (default: provider-specific auto selection)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_summary = run_batch(
        dataset_path=args.dataset,
        start=args.start,
        limit=args.limit,
        use_local=args.use_local,
        model=args.model,
    )

    print("\nBatch summary:")
    print(json.dumps(batch_summary, indent=2))
