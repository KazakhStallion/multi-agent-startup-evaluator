import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data" / "tech_lead"
OUTPUT_PATH = DATA_DIR / "tech_cases_clean.json"
SYNTHETIC_PATH = DATA_DIR / "synthetic_cases" / "all_synthetic_pitches.json"


def _normalize_case(payload: dict, source_file: str) -> dict:
    identity = payload.get("identity", {})
    business = payload.get("business", {})
    finances = payload.get("finances", {})
    metadata = payload.get("metadata", {})

    return {
        "metadata": {
            "source_file": source_file,
            "source_dataset": metadata.get("source", "Unknown"),
            "label": metadata.get("label", "Unknown"),
        },
        "identity": {
            "name": identity.get("name", "Unknown"),
            "sector": identity.get("sector", "Unknown"),
            "location": identity.get("location", "Unknown"),
        },
        "business": {
            "description": business.get("description", "Unknown"),
            "model": business.get("model", "Unknown"),
        },
        "finances": {
            "employee_count": finances.get("employee_count", "Unknown"),
            "runway": finances.get("runway", "Unknown"),
        },
    }


def _normalize_synthetic_case(payload: dict) -> dict:
    return {
        "metadata": {
            "source_file": "all_synthetic_pitches.json",
            "source_dataset": "tech_lead_synthetic",
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
        },
        "finances": {
            "employee_count": payload.get("team_size", "Unknown"),
            "runway": "Unknown",
        },
    }


def build_tech_lead_dataset() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "evaluation").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "synthetic_cases").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    if SOURCE_DIR.exists():
        for file_path in sorted(SOURCE_DIR.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            records.append(_normalize_case(payload, file_path.name))

    if not records and SYNTHETIC_PATH.exists():
        with SYNTHETIC_PATH.open("r", encoding="utf-8") as file:
            synthetic_rows = json.load(file)
        records = [_normalize_synthetic_case(row) for row in synthetic_rows]

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    return records


if __name__ == "__main__":
    rows = build_tech_lead_dataset()
    print("Built tech lead dataset")
    print(f"Rows: {len(rows)}")
    print(f"Output: {OUTPUT_PATH}")
