"""Unit test suite for classifier, validator, star-schema transforms, offline reprocessor, and analytics."""
from datetime import date
import json
from pathlib import Path
import pandas as pd

from src.analytics import build_insights, firm_scorecard
from src.classifier import KeywordRuleClassifier
from src.config import rolling_window
from src.extractor import merge_incremental_records
from src.reprocessor import QuarantineReprocessor, replay_remediations
from src.transforms import RecallTransformer
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

    # The inverted initiation date is cleared, not invented, so the record is promoted with a WARN
    assert audit_entry["reprocessing_outcome"] == "WARN"
    assert "WARN_MISSING_INITIATION_DATE" in audit_entry["warn_flags"]
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
    assert bool(promoted_fact.iloc[0]["is_remediated"]) is True
    assert pd.isna(promoted_fact.iloc[0]["reporting_lag_days"])

    # A full ETL rebuild re-quarantines the same raw record; the stored remediation must be replayed
    rebuild_validator = DataQualityValidator(run_id="RUN_2", run_timestamp="2026-10-31T00:00:00Z")
    valid_2, q_df_2, _, _, _ = rebuild_validator.validate_raw_records(raw_records)
    valid_2, q_df_2, replayed = replay_remediations(
        valid_2, q_df_2, reprocessor.load_promoted_remediations(), rebuild_validator
    )
    assert replayed == 1
    assert len(valid_2) == 1 and valid_2[0]["is_remediated"] is True
    assert bool(q_df_2.iloc[0]["remediated"]) is True


def test_auto_remediation_never_invents_classification(tmp_path: Path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    validator = DataQualityValidator(run_id="RUN_1", run_timestamp="2026-10-30T00:00:00Z")
    record = dict(_base_record(), classification="")
    _, q_df, _, _, _ = validator.validate_raw_records([record])
    q_df.to_parquet(proc_dir / "quarantined_records.parquet", index=False)

    reprocessor = QuarantineReprocessor(processed_dir=proc_dir, raw_dir=tmp_path)
    entry = reprocessor.reprocess_record(
        quarantine_id=q_df.iloc[0]["quarantine_id"], remediation_strategy="auto_remediate_known_anomalies"
    )
    assert entry["reprocessing_outcome"] == "QUARANTINE"
    assert "CRITICAL_INVALID_CLASSIFICATION" in entry["remaining_violations"]
    assert entry["promoted_to_analytical_layer"] is False


def _base_record(**overrides):
    record = {
        "recall_number": "D-2000-2026",
        "event_id": "200",
        "classification": "Class II",
        "status": "Ongoing",
        "recalling_firm": "Gamma Pharmaceuticals Inc.",
        "state": "NJ",
        "report_date": "20260310",
        "recall_initiation_date": "20260301",
        "reason_for_recall": "Subpotent assay results.",
        "voluntary_mandated": "Voluntary: Firm initiated",
        "distribution_pattern": "Nationwide",
        "openfda": {"product_ndc": ["11111-222"]},
    }
    record.update(overrides)
    return record


def test_validator_treats_null_recall_number_as_missing():
    validator = DataQualityValidator(run_id="T", run_timestamp="2026-10-30T00:00:00Z")
    status, critical, _ = validator.evaluate_single_record(_base_record(recall_number=None), set())
    assert status == "QUARANTINE"
    assert "CRITICAL_MISSING_RECALL_NUMBER" in critical


def test_missing_initiation_warning_only_fires_for_initiation_date():
    validator = DataQualityValidator(run_id="T", run_timestamp="2026-10-30T00:00:00Z")
    _, critical, warns = validator.evaluate_single_record(_base_record(report_date=None), set())
    assert "CRITICAL_UNPARSEABLE_REPORT_DATE" in critical
    assert "WARN_MISSING_INITIATION_DATE" not in warns


def test_schema_consistency_detects_missing_and_mistyped_fields():
    validator = DataQualityValidator(run_id="T", run_timestamp="2026-10-30T00:00:00Z")
    clean = _base_record()
    _, _, _, _, metrics = validator.validate_raw_records([clean])
    assert metrics["schema_consistency_passed"] is True

    drifted = _base_record(recall_number="D-2001-2026", event_id=201)  # int instead of str
    del drifted["distribution_pattern"]
    _, _, _, _, metrics = validator.validate_raw_records([clean, drifted])
    assert metrics["schema_consistency_passed"] is False
    assert json.loads(metrics["schema_missing_key_fields"]) == {"distribution_pattern": 1}
    assert json.loads(metrics["schema_wrong_type_fields"]) == {"event_id": 1}


def test_referential_integrity_flags_duplicate_fact_keys():
    transformer = RecallTransformer()
    fact_df, firms_df, prod_df = transformer.build_star_schema([_base_record(), _base_record()], [])
    result = DataQualityValidator.verify_referential_integrity(fact_df, firms_df, prod_df)
    assert result["duplicate_fact_pks"] == 1
    assert result["referential_integrity_passed"] is False


def test_incremental_merge_deduplicates_boundary_records():
    existing = [_base_record(status="Ongoing"), _base_record(recall_number="D-2001-2026")]
    fresh = [_base_record(status="Terminated"), _base_record(recall_number="D-2002-2026")]
    merged = merge_incremental_records(existing, fresh)
    by_pk = {r["recall_number"]: r for r in merged}
    assert len(merged) == 3
    assert by_pk["D-2000-2026"]["status"] == "Terminated"  # newest copy wins


def test_classifier_ignores_lead_as_a_verb():
    clf = KeywordRuleClassifier()
    verb = clf.classify("Defective container: non-sealed pouches which can lead to product leakage.")
    assert verb.root_cause_category != "Chemical Impurities & Nitrosamines"
    element = clf.classify("Chemical contamination; presence of lead above specification.")
    assert element.root_cause_category == "Chemical Impurities & Nitrosamines"


def test_rolling_window_spans_36_calendar_months():
    assert rolling_window(today=date(2026, 10, 31)) == ("20231101", "20261031")
    assert rolling_window(today=date(2026, 1, 5)) == ("20230201", "20260105")


def test_firm_scorecard_and_insights_follow_filtered_data():
    records = [
        _base_record(recall_number=f"D-30{i:02d}-2026", recalling_firm=f"Firm {i % 12} Inc.",
                     classification="Class I" if i % 4 == 0 else "Class II")
        for i in range(24)
    ]
    fact_df, firms_df, _ = RecallTransformer().build_star_schema(records, [])
    scorecard = firm_scorecard(fact_df[fact_df["classification"] == "Class I"], firms_df)
    assert scorecard["class_ii_recalls"].sum() == 0
    assert (scorecard["avg_severity_per_recall"] == 10).all()

    insights = build_insights(fact_df)
    assert any(text.startswith("**Severity:**") for text in insights)
    assert any(text.startswith("**Concentration:**") for text in insights)
    assert build_insights(fact_df.iloc[0:0]) == ["No recalls match the current filters."]