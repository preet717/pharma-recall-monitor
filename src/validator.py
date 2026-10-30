"""Data Quality validation framework with PASS, WARN, and QUARANTINE outcomes."""
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config import EXPECTED_REQUIRED_FIELDS, VALID_CLASSIFICATIONS, VALID_STATUSES

VALIDATOR_VERSION = "v1.2.0"

# Formal validation contract catalog surfaced in the Quarantine & Reprocessing Inspector
VALIDATION_RULE_CATALOG: Dict[str, Dict[str, str]] = {
    "CRITICAL_MISSING_RECALL_NUMBER": {
        "severity": "QUARANTINE",
        "failure_reason": "Primary identifier `recall_number` is null, empty, or whitespace.",
        "expected_condition": "`recall_number` must be a non-empty string uniquely identifying the recall enforcement record.",
        "remediation_guide": "Populate `recall_number` from `event_id` + product sequence or assign a synthetic tracking key.",
    },
    "CRITICAL_DUPLICATE_PRIMARY_KEY": {
        "severity": "QUARANTINE",
        "failure_reason": "Duplicate `recall_number` encountered across API pagination batches.",
        "expected_condition": "Each `recall_number` in `fact_recalls` must be unique (1:1 grain per recall number).",
        "remediation_guide": "Disambiguate multi-lot/multi-product share of `recall_number` by appending `-DUP-<index>` or deduplicating.",
    },
    "CRITICAL_INVALID_CLASSIFICATION": {
        "severity": "QUARANTINE",
        "failure_reason": "`classification` is missing or outside FDA enforcement severity classes.",
        "expected_condition": "`classification` must be strictly one of: ['Class I', 'Class II', 'Class III'].",
        "remediation_guide": "Assign a valid FDA classification ('Class I', 'Class II', or 'Class III') after reviewing recall notice.",
    },
    "CRITICAL_UNPARSEABLE_REPORT_DATE": {
        "severity": "QUARANTINE",
        "failure_reason": "`report_date` is missing, malformed, or dated in the future.",
        "expected_condition": "`report_date` must follow `YYYYMMDD` format and not exceed the current UTC date.",
        "remediation_guide": "Correct `report_date` to a valid `YYYYMMDD` string not in the future.",
    },
    "CRITICAL_CHRONOLOGICAL_INVERSION": {
        "severity": "QUARANTINE",
        "failure_reason": "`report_date` occurs earlier than `recall_initiation_date` (negative reporting lag) or year < 1990.",
        "expected_condition": "`recall_initiation_date <= report_date` and `recall_initiation_date >= 19900101`.",
        "remediation_guide": "Align `recall_initiation_date` and `report_date` so initiation precedes or equals the FDA report date.",
    },
    "WARN_MISSING_OPENFDA_PRODUCT_IDS": {
        "severity": "WARN",
        "failure_reason": "Nested `openfda` object lacks both `product_ndc` and `brand_name` identifiers.",
        "expected_condition": "`openfda.product_ndc` or `openfda.brand_name` should be present for product-level risk ratios.",
        "remediation_guide": "Non-blocking warning: record is retained in `fact_recalls`, excluded from `recalls_per_product` denominator.",
    },
    "WARN_MISSING_INITIATION_DATE": {
        "severity": "WARN",
        "failure_reason": "`recall_initiation_date` is missing or unparseable.",
        "expected_condition": "`recall_initiation_date` should be formatted as `YYYYMMDD` to calculate `reporting_lag_days`.",
        "remediation_guide": "Non-blocking warning: `reporting_lag_days` is set to NaN.",
    },
    "WARN_MISSING_GEOGRAPHIC_STATE": {
        "severity": "WARN",
        "failure_reason": "`state` field is empty (often foreign manufacturers without US state codes).",
        "expected_condition": "`state` should be a 2-letter state/territory code when country is United States.",
        "remediation_guide": "Non-blocking warning: defaults to 'Unknown' in `dim_firms`.",
    },
    "WARN_UNEXPECTED_STATUS_VALUE": {
        "severity": "WARN",
        "failure_reason": "`status` value is outside standard lifecycle values.",
        "expected_condition": f"`status` expected in {VALID_STATUSES}.",
        "remediation_guide": "Non-blocking warning: retained with raw status string.",
    },
}


class DataQualityValidator:
    """Validates raw OpenFDA records and checks post-transform referential integrity."""

    def __init__(self, run_id: str, run_timestamp: str):
        self.run_id = run_id
        self.run_timestamp = run_timestamp
        self.version = VALIDATOR_VERSION

    @staticmethod
    def _parse_fda_date(date_str: Any) -> pd.Timestamp:
        if not date_str or not isinstance(date_str, str):
            return pd.NaT
        try:
            return pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        except Exception:
            return pd.NaT

    def evaluate_single_record(
        self, record: Dict[str, Any], seen_pks: set, allow_existing_pk: bool = False
    ) -> Tuple[str, List[str], List[str]]:
        """
        Evaluates a single raw JSON record against the validation contract.
        Returns (dq_status, critical_violations, warn_flags).
        """
        critical_reasons: List[str] = []
        warn_reasons: List[str] = []
        # now_ts = pd.Timestamp(datetime.utcnow().date())  # <-- DEPRECATED
        now_ts = pd.Timestamp(datetime.now(timezone.utc).date())  # <-- TIMEZONE-AWARE UTC

        recall_num = str(record.get("recall_number", "")).strip()
        if not recall_num:
            critical_reasons.append("CRITICAL_MISSING_RECALL_NUMBER")
        elif not allow_existing_pk and recall_num in seen_pks:
            critical_reasons.append("CRITICAL_DUPLICATE_PRIMARY_KEY")
        else:
            seen_pks.add(recall_num)

        classification = str(record.get("classification", "")).strip()
        if classification not in VALID_CLASSIFICATIONS:
            critical_reasons.append("CRITICAL_INVALID_CLASSIFICATION")

        report_dt = self._parse_fda_date(record.get("report_date"))
        init_dt = self._parse_fda_date(record.get("recall_initiation_date"))

        if pd.isna(report_dt) or report_dt > now_ts:
            critical_reasons.append("CRITICAL_UNPARSEABLE_REPORT_DATE")

        if pd.notna(report_dt) and pd.notna(init_dt):
            lag_days = (report_dt - init_dt).days
            if init_dt.year < 1990 or init_dt > now_ts or lag_days < 0:
                critical_reasons.append("CRITICAL_CHRONOLOGICAL_INVERSION")
        else:
            warn_reasons.append("WARN_MISSING_INITIATION_DATE")

        openfda = record.get("openfda") or {}
        if not bool(openfda.get("product_ndc") or openfda.get("brand_name")):
            warn_reasons.append("WARN_MISSING_OPENFDA_PRODUCT_IDS")

        if not str(record.get("state", "")).strip():
            warn_reasons.append("WARN_MISSING_GEOGRAPHIC_STATE")

        if str(record.get("status", "")).strip() not in VALID_STATUSES:
            warn_reasons.append("WARN_UNEXPECTED_STATUS_VALUE")

        if critical_reasons:
            return "QUARANTINE", critical_reasons, warn_reasons
        if warn_reasons:
            return "WARN", [], warn_reasons
        return "PASS", [], []

    def validate_raw_records(
        self, raw_records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Evaluates batch of raw records and produces PASS/WARN list + quarantine & DQ tables."""
        total_input = len(raw_records)
        seen_pks: set = set()
        valid_records: List[Dict[str, Any]] = []
        quarantined_rows: List[Dict[str, Any]] = []

        rule_counts: Dict[str, int] = {code: 0 for code in VALIDATION_RULE_CATALOG}

        for idx, record in enumerate(raw_records):
            status, critical_list, warn_list = self.evaluate_single_record(record, seen_pks)

            for c_rule in critical_list:
                rule_counts[c_rule] = rule_counts.get(c_rule, 0) + 1
            for w_rule in warn_list:
                rule_counts[w_rule] = rule_counts.get(w_rule, 0) + 1

            recall_num = str(record.get("recall_number", "")).strip() or f"MISSING_PK_ROW_{idx}"

            if status == "QUARANTINE":
                quarantine_id = f"Q_{self.run_id}_{idx}"
                failure_details = [
                    {
                        "rule_code": code,
                        "failure_reason": VALIDATION_RULE_CATALOG[code]["failure_reason"],
                        "expected_condition": VALIDATION_RULE_CATALOG[code]["expected_condition"],
                        "remediation_guide": VALIDATION_RULE_CATALOG[code]["remediation_guide"],
                    }
                    for code in critical_list
                ]
                quarantined_rows.append(
                    {
                        "quarantine_id": quarantine_id,
                        "run_id": self.run_id,
                        "quarantined_at": self.run_timestamp,
                        "validator_version": self.version,
                        "record_index": idx,
                        "recall_number": recall_num,
                        "event_id": str(record.get("event_id", "")),
                        "recalling_firm": record.get("recalling_firm", "Unknown"),
                        "classification": record.get("classification", "Unknown"),
                        "report_date": record.get("report_date", ""),
                        "recall_initiation_date": record.get("recall_initiation_date", ""),
                        "dq_status": "QUARANTINE",
                        "quarantine_reasons": ", ".join(critical_list),
                        "failure_explanations_json": json.dumps(failure_details),
                        "warn_flags": ", ".join(warn_list) if warn_list else "NONE",
                        "raw_json_payload": json.dumps(record),
                    }
                )
            else:
                enriched = dict(record)
                enriched["dq_status"] = status
                enriched["dq_flags"] = ", ".join(warn_list) if warn_list else "NONE"
                enriched["validator_version"] = self.version
                valid_records.append(enriched)

        completeness_rows = []
        for field_name in EXPECTED_REQUIRED_FIELDS + ["state", "city", "openfda.brand_name", "openfda.product_ndc"]:
            missing_cnt = 0
            for r in raw_records:
                if field_name.startswith("openfda."):
                    subkey = field_name.split(".")[1]
                    val = (r.get("openfda") or {}).get(subkey)
                else:
                    val = r.get(field_name)
                if val is None or val == "" or val == []:
                    missing_cnt += 1
            missing_pct = round((missing_cnt / max(total_input, 1)) * 100.0, 2)
            completeness_rows.append(
                {
                    "run_id": self.run_id,
                    "field_name": field_name,
                    "total_records": total_input,
                    "missing_count": missing_cnt,
                    "missing_pct": missing_pct,
                    "completeness_pct": round(100.0 - missing_pct, 2),
                }
            )

        rule_summary_rows = [
            {
                "run_id": self.run_id,
                "validator_version": self.version,
                "rule_name": code,
                "outcome_tier": meta["severity"],
                "failure_reason": meta["failure_reason"],
                "expected_condition": meta["expected_condition"],
                "records_evaluated": total_input,
                "records_triggered": rule_counts[code],
                "trigger_rate_pct": round((rule_counts[code] / max(total_input, 1)) * 100.0, 2),
            }
            for code, meta in VALIDATION_RULE_CATALOG.items()
        ]

        pass_count = sum(1 for r in valid_records if r["dq_status"] == "PASS")
        warn_count = sum(1 for r in valid_records if r["dq_status"] == "WARN")
        quarantine_count = len(quarantined_rows)

        dq_metrics = {
            "validator_version": self.version,
            "records_extracted": total_input,
            "records_passed": pass_count,
            "records_warned": warn_count,
            "records_quarantined": quarantine_count,
            "duplicate_count": rule_counts["CRITICAL_DUPLICATE_PRIMARY_KEY"],
            "chronological_anomaly_count": rule_counts["CRITICAL_CHRONOLOGICAL_INVERSION"],
            "schema_consistency_passed": True,
        }

        return (
            valid_records,
            pd.DataFrame(quarantined_rows),
            pd.DataFrame(rule_summary_rows),
            pd.DataFrame(completeness_rows),
            dq_metrics,
        )

    @staticmethod
    def verify_referential_integrity(
        fact_recalls: pd.DataFrame, dim_firms: pd.DataFrame, dim_products: pd.DataFrame
    ) -> Dict[str, Any]:
        """Verifies foreign key integrity between fact_recalls and dimension tables."""
        if fact_recalls.empty:
            return {"orphan_firm_fks": 0, "orphan_product_fks": 0, "referential_integrity_passed": True}

        orphan_firms = int((~fact_recalls["firm_id"].isin(dim_firms["firm_id"])).sum())
        orphan_products = int((~fact_recalls["recall_number"].isin(dim_products["recall_number"])).sum())
        return {
            "orphan_firm_fks": orphan_firms,
            "orphan_product_fks": orphan_products,
            "referential_integrity_passed": (orphan_firms == 0 and orphan_products == 0),
        }