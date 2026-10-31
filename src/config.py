"""Configuration and extensible endpoint registry for the OpenFDA pipeline."""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_FILE = RAW_DATA_DIR / "_checkpoint.json"

WINDOW_MONTHS = 36


def rolling_window(months: int = WINDOW_MONTHS, today: date | None = None) -> tuple[str, str]:
    """Returns (start, end) as YYYYMMDD: the first day of the month `months` back, through today (UTC)."""
    today = today or datetime.now(timezone.utc).date()
    month_index = today.year * 12 + (today.month - 1) - (months - 1)
    start = date(month_index // 12, month_index % 12 + 1, 1)
    return start.strftime("%Y%m%d"), today.strftime("%Y%m%d")


WINDOW_START, WINDOW_END = rolling_window()


@dataclass(frozen=True)
class EndpointConfig:
    domain: str
    enforcement_url: str
    adverse_event_url: str
    date_field: str = "report_date"
    start_date: str = WINDOW_START  # Rolling 36-month window ending today
    end_date: str = WINDOW_END
    page_limit: int = 1000
    max_pages: int = 10           # Up to 10,000 records per run


# Extensible registry: onboarding Medical Devices or Food requires only adding a key
ENDPOINT_REGISTRY: Dict[str, EndpointConfig] = {
    "drug": EndpointConfig(
        domain="drug",
        enforcement_url="https://api.fda.gov/drug/enforcement.json",
        adverse_event_url="https://api.fda.gov/drug/event.json",
    ),
    # Ready for future expansion:
    # "device": EndpointConfig(
    #     domain="device",
    #     enforcement_url="https://api.fda.gov/device/enforcement.json",
    #     adverse_event_url="https://api.fda.gov/device/event.json",
    # ),
}

VALID_CLASSIFICATIONS: List[str] = ["Class I", "Class II", "Class III"]
VALID_STATUSES: List[str] = ["Ongoing", "Completed", "Terminated", "Pending"]

SEVERITY_WEIGHTS: Dict[str, int] = {
    "Class I": 10,
    "Class II": 5,
    "Class III": 1,
}

EXPECTED_REQUIRED_FIELDS: List[str] = [
    "recall_number",
    "event_id",
    "classification",
    "status",
    "recalling_firm",
    "report_date",
    "recall_initiation_date",
    "reason_for_recall",
    "voluntary_mandated",
    "distribution_pattern",
]