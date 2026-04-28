import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.committee_utils import normalize_startup
SOURCE_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data" / "skeptic"
OUTPUT_PATH = DATA_DIR / "skeptic_cases_clean.json"
PRIMARY_SYNTHETIC_PATH = DATA_DIR / "synthetic_cases" / "all_synthetic_pitches.json"
FALLBACK_SYNTHETIC_PATH = PROJECT_ROOT / "data" / "tech_lead" / "synthetic_cases" / "all_synthetic_pitches.json"


def _synthetic_row_to_startup(payload: dict) -> dict:
    return normalize_startup(
        {
            "metadata": {
                "source_file": "all_synthetic_pitches.json",
                "source_dataset": "skeptic_synthetic",
                "label": "Synthetic",
            },
            "identity": {
                "name": payload.get("name", "Unknown"),
                "sector": payload.get("sector", "Unknown"),
                "location": payload.get("location", "Unknown"),
            },
            "business": {
                "description": payload.get("description", "Unknown"),
                "model": payload.get("business_model", "Unknown"),
                "problem": payload.get("problem", "Unknown"),
                "solution": payload.get("solution", "Unknown"),
                "target_customer": payload.get("target_customer", "Unknown"),
                "pricing": payload.get("pricing", "Unknown"),
                "traction": payload.get("traction", "Unknown"),
            },
            "team": {
                "founders": payload.get("team", "Unknown"),
            },
            "finances": {
                "employee_count": payload.get("team_size", "Unknown"),
                "runway": payload.get("runway", "Unknown"),
            },
        }
    )


def _load_synthetic_rows() -> list[dict]:
    synthetic_path = PRIMARY_SYNTHETIC_PATH if PRIMARY_SYNTHETIC_PATH.exists() else FALLBACK_SYNTHETIC_PATH
    if not synthetic_path.exists():
        return []

    with synthetic_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return [_synthetic_row_to_startup(row) for row in rows]


def build_skeptic_dataset() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "evaluation").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "synthetic_cases").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    if SOURCE_DIR.exists():
        for file_path in sorted(SOURCE_DIR.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            records.append(normalize_startup(payload))

    if not records:
        records = _load_synthetic_rows()

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    return records


if __name__ == "__main__":
    rows = build_skeptic_dataset()
    print("Built skeptic dataset")
    print(f"Rows: {len(rows)}")
    print(f"Output: {OUTPUT_PATH}")
