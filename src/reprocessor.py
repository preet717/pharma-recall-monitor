"""
Offline Quarantine Reprocessing Engine with Immutable Audit Logging.

Operates strictly on preserved `raw_json_payload` from `quarantined_records.parquet`
without making external API calls. Never overwrites original quarantine records.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import pandas as pd

from src.classifier import KeywordRuleClassifier
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.transforms import RecallTransformer
from src.validator import DataQualityValidator, VALIDATOR_VERSION


class QuarantineReprocessor:
    """Reprocesses quarantined records from preserved raw JSON and logs an immutable audit trail."""

    def __init__(self, processed_dir: Path = PROCESSED_DATA_DIR, raw_dir: Path = RAW_DATA_DIR):
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.audit_log_path = self.processed_dir / "reprocessing_audit_log.parquet"
        self.quarantine_path = self.processed_dir / "quarantined_records.parquet"
        self.fact_path = self.processed_dir / "fact_recalls.parquet"
        self.firms_path = self.processed_dir / "dim_firms.parquet"
        self.products_path = self.processed_dir / "dim_products.parquet"

    def get_audit_log(self) -> pd.DataFrame:
        """Returns the immutable reprocessing audit trail."""
        if self.audit_log_path.exists():
            return pd.read_parquet(self.audit_log_path)
        return pd.DataFrame(
            columns=[
                "attempt_id",
                "quarantine_id",
                "original_recall_number",
                "effective_recall_number",
                "reprocessed_at",
                "remediation_strategy",
                "operator_notes",
                "previous_status",
                "reprocessing_outcome",
                "violated_rules_before",
                "remaining_violations",
                "warn_flags",
                "validator_version",
                "classifier_version",
                "root_cause_assigned",
                "confidence_tier",
                "promoted_to_analytical_layer",
                "patched_json_payload",
            ]
        )

    def reprocess_record(
        self,
        quarantine_id: str,
        patched_payload: Optional[Dict[str, Any]] = None,
        remediation_strategy: str = "re_evaluate_as_is",
        operator_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Re-evaluates a quarantined record using its preserved raw JSON (or operator-patched JSON)
        without making any API calls. Appends an immutable audit record and promotes to Star Schema
        if the outcome is PASS or WARN.
        """
        if not self.quarantine_path.exists():
            raise FileNotFoundError("Quarantine table not found.")

        q_df = pd.read_parquet(self.quarantine_path)
        matched = q_df[q_df["quarantine_id"] == quarantine_id]
        if matched.empty:
            raise ValueError(f"Quarantine ID '{quarantine_id}' not found.")

        q_row = matched.iloc[0]
        original_payload = json.loads(q_row["raw_json_payload"])
        candidate_payload = dict(patched_payload) if patched_payload is not None else dict(original_payload)

        # Apply built-in automated remediation helper if requested by operator
        if remediation_strategy == "auto_remediate_known_anomalies":
            # 1. Disambiguate duplicate or missing recall_number using event_id + record_index
            rec_num = str(candidate_payload.get("recall_number", "")).strip()
            if not rec_num or "CRITICAL_DUPLICATE_PRIMARY_KEY" in str(q_row["quarantine_reasons"]):
                base_key = rec_num or f"EVT-{candidate_payload.get('event_id', 'UNK')}"
                candidate_payload["recall_number"] = f"{base_key}-R{q_row['record_index']}"

            # 2. Align inverted recall_initiation_date with report_date
            if "CRITICAL_CHRONOLOGICAL_INVERSION" in str(q_row["quarantine_reasons"]):
                rep_dt = candidate_payload.get("report_date")
                if rep_dt:
                    candidate_payload["recall_initiation_date"] = rep_dt

            # 3. Default missing/invalid classification to Class II (standard moderate hazard)
            if "CRITICAL_INVALID_CLASSIFICATION" in str(q_row["quarantine_reasons"]):
                candidate_payload["classification"] = "Class II"

        # Load existing primary keys in fact_recalls to preserve uniqueness check
        existing_pks: set = set()
        if self.fact_path.exists():
            fact_df = pd.read_parquet(self.fact_path)
            if not fact_df.empty and "recall_number" in fact_df.columns:
                existing_pks = set(fact_df["recall_number"].astype(str).tolist())
        else:
            fact_df = pd.DataFrame()

        now_iso = datetime.now(timezone.utc).isoformat()
        attempt_id = f"REPROC_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"

        validator = DataQualityValidator(run_id=attempt_id, run_timestamp=now_iso)
        new_status, critical_violations, warn_flags = validator.evaluate_single_record(
            candidate_payload, seen_pks=existing_pks, allow_existing_pk=False
        )

        classifier = KeywordRuleClassifier()
        transformer = RecallTransformer(classifier=classifier)
        promoted = False
        root_cause_assigned = "N/A (Still Quarantined)"
        confidence_tier = "N/A"

        if new_status in ("PASS", "WARN"):
            candidate_payload["dq_status"] = new_status
            candidate_payload["dq_flags"] = ", ".join(warn_flags) if warn_flags else "NONE"
            candidate_payload["validator_version"] = VALIDATOR_VERSION

            fact_row, product_row = transformer.transform_single_record(candidate_payload)
            root_cause_assigned = fact_row["root_cause_category"]
            confidence_tier = fact_row["confidence_tier"]

            # Promote to fact_recalls & dim_products idempotently
            new_fact_df = (
                pd.DataFrame([fact_row])
                if fact_df.empty
                else pd.concat([fact_df, pd.DataFrame([fact_row])], ignore_index=True)
            )
            new_fact_df = new_fact_df.drop_duplicates(subset=["recall_number"], keep="last")
            new_fact_df.to_parquet(self.fact_path, index=False)

            if self.products_path.exists():
                prod_df = pd.read_parquet(self.products_path)
                new_prod_df = (
                    pd.DataFrame([product_row])
                    if prod_df.empty
                    else pd.concat([prod_df, pd.DataFrame([product_row])], ignore_index=True)
                )
                new_prod_df = new_prod_df.drop_duplicates(subset=["recall_number"], keep="last")
            else:
                new_prod_df = pd.DataFrame([product_row])
            new_prod_df.to_parquet(self.products_path, index=False)

            # Recompute dim_firms using cached adverse events (zero API calls)
            ae_cache = self.raw_dir / "drug_adverse_events_counts.json"
            ae_counts = (
                json.loads(ae_cache.read_text(encoding="utf-8")).get("results", [])
                if ae_cache.exists()
                else []
            )
            new_firms_df = transformer.recompute_dim_firms(new_fact_df, ae_counts)
            new_firms_df.to_parquet(self.firms_path, index=False)
            promoted = True

        audit_entry = {
            "attempt_id": attempt_id,
            "quarantine_id": quarantine_id,
            "original_recall_number": str(q_row["recall_number"]),
            "effective_recall_number": str(candidate_payload.get("recall_number", "")),
            "reprocessed_at": now_iso,
            "remediation_strategy": remediation_strategy,
            "operator_notes": operator_notes or "Interactive reprocessing via Inspector",
            "previous_status": "QUARANTINE",
            "reprocessing_outcome": new_status,
            "violated_rules_before": str(q_row["quarantine_reasons"]),
            "remaining_violations": ", ".join(critical_violations) if critical_violations else "NONE",
            "warn_flags": ", ".join(warn_flags) if warn_flags else "NONE",
            "validator_version": VALIDATOR_VERSION,
            "classifier_version": classifier.version,
            "root_cause_assigned": root_cause_assigned,
            "confidence_tier": confidence_tier,
            "promoted_to_analytical_layer": promoted,
            "patched_json_payload": json.dumps(candidate_payload),
        }

        existing_audit = self.get_audit_log()
        updated_audit = (
            pd.DataFrame([audit_entry])
            if existing_audit.empty
            else pd.concat([existing_audit, pd.DataFrame([audit_entry])], ignore_index=True)
        )
        updated_audit.to_parquet(self.audit_log_path, index=False)

        return audit_entry