import argparse
import json
import time
from pathlib import Path

from run_committee_pipeline import OUT_DIR, run_committee_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
PROGRESS_PATH = PROJECT_ROOT / "data" / "committee_pipeline" / "batch_progress.json"


STARTUPS = [
    {
        "name": "LedgerFlow",
        "sector": "fintech",
        "description": "Automated accounts payable platform for mid-size construction companies.",
        "business": {"description": "Automated accounts payable platform for mid-size construction companies."},
        "finances": {"revenue": 403200, "burn_rate": 45000, "funding": 1200000, "employee_count": 10},
    },
    {
        "name": "CryptoSave",
        "sector": "fintech",
        "description": "Rounds up purchases and converts spare change into crypto.",
        "business": {"description": "Rounds up purchases and converts spare change into crypto."},
        "finances": {"revenue": 0, "burn_rate": 12000, "funding": 50000, "employee_count": 5},
    },
    {
        "name": "CareCoord",
        "sector": "healthtech",
        "description": "Post-discharge care coordination platform for hospital systems.",
        "business": {"description": "Post-discharge care coordination platform for hospital systems."},
        "finances": {"revenue": 75000, "burn_rate": 60000, "funding": 2500000, "employee_count": 18},
    },
    {
        "name": "WellnessAI",
        "sector": "healthtech",
        "description": "AI chatbot for wellness advice without clinical validation yet.",
        "business": {"description": "AI chatbot for wellness advice without clinical validation yet."},
        "finances": {"revenue": 0, "burn_rate": 25000, "funding": 150000, "employee_count": 7},
    },
    {
        "name": "ReturnKit",
        "sector": "ecommerce",
        "description": "Returns management platform for DTC brands.",
        "business": {"description": "Returns management platform for DTC brands."},
        "finances": {"revenue": 215460, "burn_rate": 30000, "funding": 750000, "employee_count": 11},
    },
    {
        "name": "NicheShop",
        "sector": "ecommerce",
        "description": "Marketplace for rare collectibles with near-zero traction.",
        "business": {"description": "Marketplace for rare collectibles with near-zero traction."},
        "finances": {"revenue": 0, "burn_rate": 5000, "funding": 10000, "employee_count": 3},
    },
    {
        "name": "ProposalDesk",
        "sector": "saas",
        "description": "Proposal tool for marketing agencies with high churn.",
        "business": {"description": "Proposal tool for marketing agencies with high churn."},
        "finances": {"revenue": 70560, "burn_rate": 15000, "funding": 200000, "employee_count": 6},
    },
    {
        "name": "TeamSync",
        "sector": "saas",
        "description": "Collaboration platform still pre-product.",
        "business": {"description": "Collaboration platform still pre-product."},
        "finances": {"revenue": 0, "burn_rate": 20000, "funding": 400000, "employee_count": 8},
    },
    {
        "name": "LocalPulse",
        "sector": "media",
        "description": "Hyperlocal news app monetized through local ads.",
        "business": {"description": "Hyperlocal news app monetized through local ads."},
        "finances": {"revenue": 96000, "burn_rate": 10000, "funding": 150000, "employee_count": 9},
    },
    {
        "name": "SiteSense",
        "sector": "hardware",
        "description": "IoT sensor kit for construction sites.",
        "business": {"description": "IoT sensor kit for construction sites."},
        "finances": {"revenue": 125000, "burn_rate": 80000, "funding": 3000000, "employee_count": 20},
    },
    {
        "name": "MealMap",
        "sector": "food",
        "description": "Recurring office catering marketplace.",
        "business": {"description": "Recurring office catering marketplace."},
        "finances": {"revenue": 144000, "burn_rate": 35000, "funding": 1100000, "employee_count": 14},
    },
    {
        "name": "HealthyBite",
        "sector": "food",
        "description": "Pre-product subscription meal delivery startup.",
        "business": {"description": "Pre-product subscription meal delivery startup."},
        "finances": {"revenue": 0, "burn_rate": 2000, "funding": 0, "employee_count": 2},
    },
]


def _slug(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch in {"_", "-"}).replace("-", "")


def _output_path_for(startup_name: str) -> Path:
    return OUT_DIR / f"{startup_name.lower()}_committee_pipeline.json"


def _load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"runs": []}
    try:
        with PROGRESS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data
    except Exception:
        pass
    return {"runs": []}


def _save_progress(progress: dict):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def _already_succeeded(progress: dict, startup_name: str) -> bool:
    key = _slug(startup_name)
    for item in progress.get("runs", []):
        if not isinstance(item, dict):
            continue
        if _slug(str(item.get("startup", ""))) == key and item.get("status") == "ok":
            return True
    return False


def _record(progress: dict, startup_name: str, status: str, message: str, elapsed_sec: float, attempt: int):
    progress.setdefault("runs", [])
    progress["runs"].append(
        {
            "startup": startup_name,
            "status": status,
            "message": message,
            "elapsed_sec": round(elapsed_sec, 2),
            "attempt": attempt,
            "timestamp": int(time.time()),
        }
    )
    _save_progress(progress)


def run_batch(max_startups: int | None, retries: int, force: bool):
    progress = _load_progress()
    completed = 0

    for startup in STARTUPS:
        name = startup["name"]
        out_path = _output_path_for(name)

        if not force:
            if out_path.exists():
                print(f"skip (file exists): {name}")
                continue
            if _already_succeeded(progress, name):
                print(f"skip (already successful): {name}")
                continue

        success = False
        last_error = ""
        for attempt in range(1, retries + 2):
            t0 = time.time()
            try:
                result = run_committee_pipeline(startup)
                elapsed = time.time() - t0
                msg = f"decision={result.get('decision', 'unknown')}"
                print(f"ok: {name} -> {msg} ({elapsed:.1f}s)")
                _record(progress, name, "ok", msg, elapsed, attempt)
                success = True
                completed += 1
                break
            except Exception as exc:
                elapsed = time.time() - t0
                last_error = str(exc)
                print(f"fail: {name} (attempt {attempt}) -> {last_error}")
                _record(progress, name, "fail", last_error, elapsed, attempt)
                if attempt <= retries:
                    time.sleep(2)

        if not success:
            print(f"gave up: {name} -> {last_error}")

        if max_startups is not None and completed >= max_startups:
            print(f"stopping early after {completed} successful runs (--max-startups)")
            break

    print(f"progress file: {PROGRESS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run committee pipeline over a startup batch.")
    parser.add_argument(
        "--max-startups",
        type=int,
        default=None,
        help="Stop after N successful runs in this invocation.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per startup after initial failure.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if output file already exists.",
    )
    args = parser.parse_args()
    run_batch(max_startups=args.max_startups, retries=args.retries, force=args.force)

