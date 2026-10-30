"""Unit test suite for classifier, validator, star-schema transforms, and offline reprocessor."""
import json
from pathlib import Path
import pandas as pd
import pytest

from src.classifier import KeywordRuleClassifier
from src.reprocessor import QuarantineReprocessor
from src.transforms import RecallTransformer, normalize_firm_name
from src.validator import DataQualityValidator


def test_classifier_unclassified_when_no_keywords_match():
    clf = KeywordRuleClassifier()
    res = clf.classify("Administrative withdrawal by distributor without specified defect.")
    assert res.root_cause_category == "Unclassified / Manual Review"
    assert res.confidence_tier == "UNCLASSIFIED"
    assert res.heuristic_score == 0.0
    assert res.competing_category_scores == "{}"


def test_classifier_multi_defect_and_low_confidence_ambiguity():
    clf = KeywordRuleClassifier()
    # Overlapping defects with identical weights: non-sterile (0.55) vs particulate matter (0.55)
    reason = "CGMP deviation: presence of particulate matter and non-sterile manufacturing environment."
    res = clf.classify(reason)
    assert res.is_multi_defect is True
    assert res.secondary_category != "NONE"
    assert res.confidence_tier == "LOW_CONFIDENCE"
    scores = json.loads(res.competing_category_scores)
    assert len(scores) >= 2


def test_classifier_evaluation_harness():
    clf = KeywordRuleClassifier()
    gold_df = pd.DataFrame(
        [
            {"reason_for_recall": "Presence of NDMA nitrosamine impurity above limit", "true_category": "Chemical Impurities & Nitrosamines"},
            {"reason_for_recall": "Subpotent assay results during stability testing", "true_category": "Potency, Assay & Dissolution Failures"},
        ]
    )
    eval_metrics = clf.evaluate_against_labeled_dataset(gold_df)
    assert eval_metrics["sample_size"] == 2
    assert eval_metrics["exact_match_accuracy"] == 1.0


def test_validator_pass_warn_quarantine_outcomes():
    validator = DataQualityValidator(run_id="TEST_RUN", run_timestamp="2026-10-30T00:00:00Z")
    raw_batch = [
        # 1. Clean PASS record
        {
            "recall_number": "D-0001-2026",
            "event_id": "90001",
            "classification": "Class I",
            "status": "Ongoing",
            "recalling_firm": "Acme Pharma Inc.",
            "state": "NJ",
            "report_date": "20260510",
            "recall_initiation_date": "20260501",
            "reason_for_recall": "Presence of glass particulate matter in vials.",
            "openfda": {"product_ndc": ["12345-678"], "brand_name": ["ACMECILLIN"]},
        },
        # 2. WARN record (missing openfda product IDs)
        {
            "recall_number": "D-0002-2026",
            "event_id": "90002",
            "classification": "Class II",
            "status": "Ongoing",
            "recalling_firm": "Acme Pharma Inc.",
            "state": "NJ",
            "report_date": "20260512",
            "recall_initiation_date": "20260505",
            "reason_for_recall": "Subpotent assay results.",
            "openfda": {},
        },
        # 3. QUARANTINE record (duplicate primary key + chronological inversion)
        {
            "recall_number": "D-0001-2026",
            "event_id": "90003",
            "classification": "Class II",
            "status": "Ongoing",
            "recalling_firm": "Beta Labs LLC",
            "state": "CA",
            "report_date": "20260401",
            "recall_initiation_date": "20260501",
            "reason_for_recall": "Mislabeled carton.",
            "openfda": {},
        },
    ]

    valid, q_df, rules_df, comp_df, metrics = validator.validate_raw_records(raw_batch)
    assert metrics["records_passed"] == 1
    assert metrics["records_warned"] == 1
    assert metrics["records_quarantined"] == 1
    assert "CRITICAL_DUPLICATE_PRIMARY_KEY" in q_df.iloc[0]["quarantine_reasons"]
    assert "CRITICAL_CHRONOLOGICAL_INVERSION" in q_df.iloc[0]["quarantine_reasons"]


def test_offline_reprocessor_preserves_quarantine_and_appends_immutable_audit(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    proc_dir.mkdir()
    raw_dir.mkdir()

    validator = DataQualityValidator(run_id="RUN_1", run_timestamp="2026-10-30T00:00:00Z")
    raw_records = [
        {
            "recall_number": "D-1000-2026",
            "event_id": "100",
            "classification": "Class I",
            "status": "Ongoing",
            "recalling_firm": "Apex Bio LLC",
            "state": "NY",
            "report_date": "20260301",
            "recall_initiation_date": "20260401",  # Inverted date -> QUARANTINE
            "reason_for_recall": "Non-sterile microbial contamination.",
            "openfda": {"product_ndc": ["99999-111"]},
        }
    ]
    valid, q_df, _, _, _ = validator.validate_raw_records(raw_records)
    assert len(valid) == 0
    assert len(q_df) == 1

    transformer = RecallTransformer()
    fact_df, firms_df, prod_df = transformer.build_star_schema(valid, [])

    fact_df.to_parquet(proc_dir / "fact_recalls.parquet", index=False)
    firms_df.to_parquet(proc_dir / "dim_firms.parquet", index=False)
    prod_df.to_parquet(proc_dir / "dim_products.parquet", index=False)
    q_df.to_parquet(proc_dir / "quarantined_records.parquet", index=False)

    reprocessor = QuarantineReprocessor(processed_dir=proc_dir, raw_dir=raw_dir)
    qid = q_df.iloc[0]["quarantine_id"]

    # Reprocess offline using automated remediation
    audit_entry = reprocessor.reprocess_record(
        quarantine_id=qid,
        remediation_strategy="auto_remediate_known_anomalies",
        operator_notes="Unit test offline remediation",
    )

    assert audit_entry["reprocessing_outcome"] == "PASS"
    assert audit_entry["promoted_to_analytical_layer"] is True

    # Verify original quarantine table is untouched (immutable)
    q_after = pd.read_parquet(proc_dir / "quarantined_records.parquet")
    assert len(q_after) == 1

    # Verify audit log recorded the attempt and fact_recalls received the promoted record
    audit_log = reprocessor.get_audit_log()
    assert len(audit_log) == 1
    promoted_fact = pd.read_parquet(proc_dir / "fact_recalls.parquet")
    assert len(promoted_fact) == 1
    assert promoted_fact.iloc[0]["root_cause_category"] == "Sterility & Microbial Contamination"