import json
import os
from pathlib import Path
from statistics import median

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "product_lead"
OUTPUT_DIR = DATA_DIR / "outputs"
CRUNCHBASE_PATH = DATA_DIR / "crunchbase_clean.json"
FAILURES_PATH = DATA_DIR / "failures_clean.json"


def _load_local_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    # I keep this small fallback so the script still works even if dotenv is missing.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if load_dotenv:
    load_dotenv()
else:
    _load_local_env_file()


def _load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


cb_data = _load_json_records(CRUNCHBASE_PATH)
fail_data = _load_json_records(FAILURES_PATH)

api_key = os.getenv("OPENAI_API_KEY")
client = (
    OpenAI(
        api_key=api_key,
        base_url="https://llm-api.arc.vt.edu/api/v1/",
    )
    if api_key
    else None
)

SECTOR_MAP = {
    "fintech": {
        "crunchbase": [
            "Finance",
            "Payments",
            "Financial Services",
            "Banking",
            "Lending",
            "Insurance",
            "Cryptocurrency",
            "Bitcoin",
        ],
        "failures": ["Finance and Insurance"],
    },
    "healthtech": {
        "crunchbase": [
            "Health Care",
            "Biotechnology",
            "Medical",
            "Health and Wellness",
            "Pharmaceuticals",
            "Fitness",
        ],
        "failures": ["Health Care"],
    },
    "ecommerce": {
        "crunchbase": [
            "E-Commerce",
            "Retail",
            "Marketplaces",
            "Shopping",
            "Fashion",
            "Luxury Goods",
        ],
        "failures": ["Retail Trade"],
    },
    "saas": {
        "crunchbase": [
            "Software",
            "Enterprise Software",
            "SaaS",
            "Cloud Computing",
            "Developer Tools",
            "Productivity Tools",
        ],
        "failures": ["Information"],
    },
    "media": {
        "crunchbase": [
            "News",
            "Publishing",
            "Media and Entertainment",
            "Social Media",
            "Content",
        ],
        "failures": ["Information"],
    },
    "hardware": {
        "crunchbase": [
            "Hardware",
            "Robotics",
            "Manufacturing",
            "Electronics",
            "Drones",
            "Wearables",
        ],
        "failures": ["Manufacturing"],
    },
    "food": {
        "crunchbase": [
            "Food and Beverages",
            "Restaurants",
            "Food Delivery",
            "Agriculture",
        ],
        "failures": ["Accommodation and Food Services"],
    },
}

BINARY_REASON_COLUMNS = [
    "Giants",
    "Competition",
    "Poor Market Fit",
    "Monetization Failure",
    "No Budget",
    "Execution Flaws",
    "Trend Shifts",
    "Regulatory Pressure",
    "Niche Limits",
    "Platform Dependency",
    "Overhype",
    "High Operational Costs",
]

DEFAULT_RESULT = {
    "agent": "Product_Lead",
    "product_differentiation": "Unknown",
    "feature_priority": "Not enough evidence to assess roadmap",
    "scalability_assessment": "Unknown",
    "ip_defensibility": "Not enough evidence",
    "execution_risks": ["Analysis failed or incomplete"],
    "product_market_fit_signals": ["Needs manual customer discovery"],
    "score": 5,
    "product_verdict": "Hold"
}


def _format_currency(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


def _normalize_status(status: str | None) -> str:
    if not status:
        return "unknown"
    cleaned = str(status).strip().lower()
    if cleaned in {"operating", "acquired", "closed"}:
        return cleaned
    return "other"


def _matches_any_market(record: dict, targets: list[str]) -> bool:
    market_value = str(record.get("market") or "").strip().lower()
    category_value = str(record.get("category_list") or "").lower()
    for target in targets:
        target_lower = target.lower()
        if market_value == target_lower or target_lower in category_value:
            return True
    return False


def _resolve_sector_matches(
    sector_input: str, cb_data: list[dict], fail_data: list[dict]
) -> tuple[list[dict], list[dict]]:
    sector_key = sector_input.strip().lower()

    if sector_key in SECTOR_MAP:
        cb_markets = SECTOR_MAP[sector_key]["crunchbase"]
        fail_sectors = SECTOR_MAP[sector_key]["failures"]

        cb_matches = [record for record in cb_data if _matches_any_market(record, cb_markets)]
        fail_matches = [
            record
            for record in fail_data
            if record.get("sector_tag") in fail_sectors or record.get("Sector") in fail_sectors
        ]
    else:
        cb_matches = [
            record
            for record in cb_data
            if sector_key in str(record.get("market", "")).lower()
            or sector_key in str(record.get("category_list", "")).lower()
        ]
        fail_matches = [
            record
            for record in fail_data
            if sector_key in str(record.get("Sector", "")).lower()
            or sector_key in str(record.get("sector_tag", "")).lower()
        ]

    return cb_matches, fail_matches


def _build_sector_snapshot(cb_matches: list[dict], fail_matches: list[dict]) -> dict:
    fundings = [
        record["funding_total_usd"]
        for record in cb_matches
        if record.get("funding_total_usd") is not None
    ]
    median_funding = median(fundings) if fundings else None

    top_funded = sorted(
        cb_matches,
        key=lambda record: record.get("funding_total_usd") or 0,
        reverse=True,
    )[:5]

    status_counts = {"operating": 0, "acquired": 0, "closed": 0, "other": 0, "unknown": 0}
    for record in cb_matches:
        normalized = _normalize_status(record.get("status"))
        status_counts[normalized] = status_counts.get(normalized, 0) + 1

    years = [record.get("founded_year") for record in cb_matches if record.get("founded_year") is not None]
    year_range = "N/A"
    if years:
        year_range = f"{int(min(years))} to {int(max(years))}"

    total_failures = len(fail_matches)
    reason_totals = {
        column: sum(int(record.get(column) or 0) for record in fail_matches)
        for column in BINARY_REASON_COLUMNS
    }
    top_reasons = sorted(reason_totals.items(), key=lambda item: item[1], reverse=True)[:3]

    examples = fail_matches[:4]
    return {
        "total_companies": len(cb_matches),
        "median_funding": median_funding,
        "top_funded": top_funded,
        "status_counts": status_counts,
        "year_range": year_range,
        "total_failures": total_failures,
        "top_reasons": top_reasons,
        "examples": examples,
    }


def _format_top_companies(records: list[dict]) -> str:
    if not records:
        return "None found"

    items = []
    for record in records:
        name = record.get("name", "Unknown")
        funding = _format_currency(record.get("funding_total_usd"))
        items.append(f"{name} ({funding})")
    return ", ".join(items)


def _format_failure_examples(records: list[dict]) -> str:
    if not records:
        return "No failure examples found"

    lines = []
    for record in records:
        name = record.get("Name", "Unknown")
        what_they_did = record.get("What They Did", "Unknown")
        why_they_failed = record.get("Why They Failed", "Unknown")
        lines.append(f"- {name}: {what_they_did} | Failed because: {why_they_failed}")
    return "\n".join(lines)


def get_sector_evidence(sector_input: str, cb_data: list[dict], fail_data: list[dict]) -> str:
    cb_matches, fail_matches = _resolve_sector_matches(sector_input, cb_data, fail_data)
    snapshot = _build_sector_snapshot(cb_matches, fail_matches)

    top_reasons_text = (
        ", ".join(
            f"{reason} ({count})" for reason, count in snapshot["top_reasons"] if count > 0
        )
        or "None found"
    )
    reason_totals = {
        column: sum(int(record.get(column) or 0) for record in fail_matches)
        for column in BINARY_REASON_COLUMNS
    }
    exec_count = reason_totals.get("Execution Flaws", 0)
    fit_count = reason_totals.get("Poor Market Fit", 0)


    lines = [
        "Crunchbase evidence",
        f"Sector: {sector_input}",
        f"Companies found: {snapshot['total_companies']}",
        f"Median funding: {_format_currency(snapshot['median_funding'])}",
        f"Top funded: {_format_top_companies(snapshot['top_funded'])}",
        (
            "Status: "
            f"{snapshot['status_counts']['operating']} operating, "
            f"{snapshot['status_counts']['acquired']} acquired, "
            f"{snapshot['status_counts']['closed']} closed"
        ),
        f"Funding years: {snapshot['year_range']}",
        "",
        "Failure evidence",
        f"Total failures in sector: {snapshot['total_failures']}",
        f"Top failure reasons: {top_reasons_text}",
        f"Execution Flaw Failures: {exec_count}",     # Updated for Product Lead
        f"Poor Market Fit Failures: {fit_count}",   # Updated for Product Lead
        "Examples:",
        _format_failure_examples(snapshot["examples"]),
    ]

    return "\n".join(lines)


def _deep_copy_defaults() -> dict:
    return json.loads(json.dumps(DEFAULT_RESULT))


def _validate_result(result: dict) -> dict:
    validated = _deep_copy_defaults()
    validated.update(result)

    for list_key in ["execution_risks", "product_market_fit_signals"]:
        if not isinstance(validated.get(list_key), list):
            validated[list_key] = []


    raw_score = validated.get("score", 5)
    try:
        # Strip common formatting if the LLM gets fancy like '3/10'
        if isinstance(raw_score, str):
            clean_score = raw_score.split('/')[0].strip()
            score = int(float(clean_score))
        else:
            score = int(raw_score)
    except (ValueError, TypeError):
        score = 5


    validated["score"] = min(max(score, 1), 10)
    validated["agent"] = "Product_Lead"

    return validated


def _build_fallback_result(
    sector: str,
    evidence: str,
    cb_matches: list[dict],
    fail_matches: list[dict],
) -> dict:
    fallback = _deep_copy_defaults()
    fallback["execution_risks"] = [f"API Failed. Needs manual review for {sector}."]
    return fallback


def run_product_lead(name: str, sector: str, description: str) -> dict:
    cb_matches, fail_matches = _resolve_sector_matches(sector, cb_data, fail_data)
    evidence = get_sector_evidence(sector, cb_data, fail_data)



    prompt = f"""
        You are the Product Lead (CPO) on an AI investment committee.
        Your job is to evaluate if the startup has built a defensible product that solves a high-priority problem.

        Focus on:
        1. Problem-Solution Fit: Does the 'Solution' actually kill the 'Problem' described? 
        2. Differentiation: How does this stand out against the 'Top Funded' incumbents in the evidence? 
        3. Execution Risk: Use the Historical Evidence to see if 'Execution Flaws' or 'Platform Dependency' have killed similar startups.
        4. Scalability: Is the product model designed for low-friction growth, or is it too manual?

        STARTUP CONTEXT:
        Name: {name}
        Sector: {sector}
        Description: {description}

        HISTORICAL MARKET EVIDENCE:
        {evidence}

        Return ONLY a valid JSON object with exactly these keys:
        {{
        "agent": "Product_Lead",
        "product_differentiation": "High/Medium/Low - how they win vs incumbents",
        "feature_priority": "Assessment of if they are building the right features first",
        "scalability_assessment": "Analysis of the tech/product growth potential",
        "ip_defensibility": "Assessment of moats (patents, network effects, or data moats)",
        "execution_risks": ["Specific product-related risks e.g., Platform Dependency, Overhype"],
        "product_market_fit_signals": ["Early evidence or historical warnings from the sector"],
        "score": integer 1-10,
        "product_verdict": "Go/Hold/No-Go"
        }}

        SCORE GUIDELINES (1-10):
        - 10: Painkiller solution, clear moats, and high technical defensibility.
        - 5: Standard product; "me-too" solution with a slight execution edge.
        - 1: High 'Execution Flaws' history in sector, no differentiation, or extreme 'Platform Dependency'.
        """.strip()


    if client is not None:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
    else:
        parsed = _build_fallback_result(sector, evidence, cb_matches, fail_matches)

    return _validate_result(parsed)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # I use a few different sectors here so I can sanity check the matching quickly.
    test_startups = [
        {
            "name": "PayFlow",
            "sector": "fintech",
            "description": "B2B payment automation for small businesses",
        },
        {
            "name": "MediTrack",
            "sector": "healthtech",
            "description": "Patient data management platform for clinics",
        },
        {
            "name": "ShopLocal",
            "sector": "ecommerce",
            "description": "Marketplace connecting local boutique stores to shoppers",
        },
    ]

    for startup in test_startups:
        result = run_product_lead(
            name=startup["name"],
            sector=startup["sector"],
            description=startup["description"],
        )
        print(json.dumps(result, indent=2))

        output_path = OUTPUT_DIR / f"{startup['name'].lower()}_analysis.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
