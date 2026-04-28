# This script is used to clean the raw market datasets once and save them as JSON,
# so the market analyst agent can load them fast without redoing the CSV.

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "legal"
CRUNCHBASE_OUTPUT = OUTPUT_DIR / "crunchbase_clean.json"
FAILURES_OUTPUT = OUTPUT_DIR / "failures_clean.json"

CRUNCHBASE_KEEP_COLUMNS = [
    "name",
    "market",
    "category_list",
    "status",
    "country_code",
    "region",
    "city",
    "funding_total_usd",
    "funding_rounds",
    "founded_year",
    "first_funding_at",
    "last_funding_at",
    "seed",
    "venture",
    "round_A",
    "round_B",
    "round_C",
    "round_D",
]

FAILURES_KEEP_COLUMNS = [
    "Name",
    "Sector",
    "sector_tag",
    "Years of Operation",
    "What They Did",
    "How Much They Raised",
    "Why They Failed",
    "Takeaway",
    "Giants",
    "No Budget",
    "Competition",
    "Poor Market Fit",
    "Acquisition Stagnation",
    "Platform Dependency",
    "Monetization Failure",
    "Niche Limits",
    "Execution Flaws",
    "Trend Shifts",
    "Toxicity/Trust Issues",
    "Regulatory Pressure",
    "Overhype",
    "High Operational Costs",
    "source_file",
]

ALL_BINARY_COLS = [
    "Giants",
    "No Budget",
    "Competition",
    "Poor Market Fit",
    "Acquisition Stagnation",
    "Platform Dependency",
    "Monetization Failure",
    "Niche Limits",
    "Execution Flaws",
    "Trend Shifts",
    "Toxicity/Trust Issues",
    "Regulatory Pressure",
    "Overhype",
    "High Operational Costs",
]


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_records(df: pd.DataFrame) -> list[dict]:
    serializable = df.astype(object).where(pd.notna(df), None)
    return serializable.to_dict(orient="records")


def _save_json(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def _print_null_counts(df: pd.DataFrame, columns: list[str]) -> None:
    print("Null counts:")
    for column in columns:
        if column in df.columns:
            print(f"  {column}: {int(df[column].isna().sum())}")


def clean_crunchbase() -> list[dict]:
    _ensure_output_dir()

    crunchbase_path = RAW_DIR / "investments_VC.csv"
    df = pd.read_csv(crunchbase_path, encoding="latin-1")
    df.columns = df.columns.str.strip()

    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.strip()
        df.loc[df["market"].isin(["", "nan", "None"]), "market"] = pd.NA

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
        df.loc[df["name"].isin(["", "nan", "None"]), "name"] = pd.NA

    if "funding_total_usd" in df.columns:
        df["funding_total_usd"] = (
            df["funding_total_usd"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )
        df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    if "founded_year" in df.columns:
        df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce")

    keep_columns = [column for column in CRUNCHBASE_KEEP_COLUMNS if column in df.columns]
    df = df[keep_columns]

    df = df.dropna(subset=["name", "market"], how="all")

    records = _to_records(df)
    _save_json(records, CRUNCHBASE_OUTPUT)

    market_counts = (
        df["market"]
        .dropna()
        .value_counts()
        .head(15)
        .to_dict()
        if "market" in df.columns
        else {}
    )

    print("Saved cleaned Crunchbase data")
    print(f"Rows: {len(df)}")
    print(f"Output: {CRUNCHBASE_OUTPUT}")
    print("Top markets:")
    for market, count in market_counts.items():
        print(f"  {market}: {count}")
    _print_null_counts(df, ["name", "market", "funding_total_usd", "status"])

    return records


def _read_failure_csv(file_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding="latin-1")


def _extract_sector_tag(filename: str) -> str:
    return filename.replace(".csv", "").split("(")[-1].replace(")", "").strip()


def clean_failures() -> list[dict]:
    _ensure_output_dir()

    frames: list[pd.DataFrame] = []

    for file_path in sorted(RAW_DIR.iterdir()):
        filename = file_path.name

        if filename == "Startup Failures.csv":
            continue

        if file_path.suffix.lower() != ".csv" or "failure" not in filename.lower():
            continue

        df = _read_failure_csv(file_path)
        df.columns = df.columns.str.strip()

        # I keep a sector tag from the filename because the raw files are not perfectly consistent.
        sector_tag = _extract_sector_tag(filename)
        df["sector_tag"] = sector_tag
        df["source_file"] = filename
        frames.append(df)

        print(f"Loaded {filename}")
        print(f"  rows: {len(df)}")
        print(f"  sector_tag: {sector_tag}")

    combined = pd.concat(frames, ignore_index=True)

    for column in ALL_BINARY_COLS:
        if column not in combined.columns:
            combined[column] = 0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0).astype(int)

    keep_columns = [column for column in FAILURES_KEEP_COLUMNS if column in combined.columns]
    combined = combined[keep_columns]
    combined = combined.dropna(subset=["Name"])

    records = _to_records(combined)
    _save_json(records, FAILURES_OUTPUT)

    print("Saved cleaned failure data")
    print(f"Rows: {len(combined)}")
    print(f"Output: {FAILURES_OUTPUT}")

    unique_sector_tags = sorted(tag for tag in combined["sector_tag"].dropna().unique().tolist())
    print("Sector tags:")
    for tag in unique_sector_tags:
        print(f"  {tag}")

    print("Failure reason totals:")
    for column in ALL_BINARY_COLS:
        if column in combined.columns:
            print(f"  {column}: {int(combined[column].sum())}")

    return records


if __name__ == "__main__":
    crunchbase_records = clean_crunchbase()
    failure_records = clean_failures()

    print("Finished building the legal agent data files")
    print(f"Crunchbase rows saved: {len(crunchbase_records)}")
    print(f"Failure rows saved: {len(failure_records)}")
