"""Main ETL entry point for the OpenFDA Pharma Recall & Compliance Monitor."""
import argparse
from datetime import datetime, timezone
import logging
import uuid
import pandas as pd

from src.config import ENDPOINT_REGISTRY, PROCESSED_DATA_DIR
from src.extractor import OpenFDAExtractor
from src.transforms import RecallTransformer
from src.validator import DataQualityValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("etl")


def run_pipeline(domain: str = "drug", incremental: bool = False) -> None:
    """Executes the end-to-end extraction, validation, transformation, and lineage logging."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"RUN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
    run_ts = datetime.now(timezone.utc).isoformat()

    config = ENDPOINT_REGISTRY[domain]
    logger.info("Starting ETL run %s for domain='%s' (window %s to %s)", run_id, domain, config.start_date, config.end_date)

    # 1. Extract
    extractor = OpenFDAExtractor(config)
    raw_recalls, telemetry = extractor.extract_enforcement_recalls(incremental=incremental)
    ae_counts = extractor.extract_contextual_adverse_events()

    # 2. Validate (PASS / WARN / QUARANTINE)
    validator = DataQualityValidator(run_id=run_id, run_timestamp=run_ts)
    valid_records, quarantined_df, dq_rules_df, dq_completeness_df, dq_metrics = validator.validate_raw_records(raw_recalls)

    # 3. Transform into Star Schema
    transformer = RecallTransformer()
    fact_recalls_df, dim_firms_df, dim_products_df = transformer.build_star_schema(valid_records, ae_counts)

    # 4. Verify Referential Integrity
    ref_integrity = validator.verify_referential_integrity(fact_recalls_df, dim_firms_df, dim_products_df)

    # 5. Persist Analytical & Observability Outputs
    fact_recalls_df.to_parquet(PROCESSED_DATA_DIR / "fact_recalls.parquet", index=False)
    dim_firms_df.to_parquet(PROCESSED_DATA_DIR / "dim_firms.parquet", index=False)
    dim_products_df.to_parquet(PROCESSED_DATA_DIR / "dim_products.parquet", index=False)
    quarantined_df.to_parquet(PROCESSED_DATA_DIR / "quarantined_records.parquet", index=False)
    dq_rules_df.to_parquet(PROCESSED_DATA_DIR / "dq_rule_summary.parquet", index=False)
    dq_completeness_df.to_parquet(PROCESSED_DATA_DIR / "dq_field_completeness.parquet", index=False)

    # 6. Append Lineage & Telemetry Record
    lineage_record = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "extraction_timestamp": run_ts,
                "domain": domain,
                "source_endpoint": config.enforcement_url,
                "source_mode": telemetry["source_mode"],
                "api_meta_last_updated": telemetry["api_meta_last_updated"],
                "api_calls_made": telemetry["api_calls_made"],
                "api_latency_ms": round(telemetry["api_latency_ms_total"], 2),
                "retry_count": telemetry["retry_count"],
                **dq_metrics,
                **ref_integrity,
            }
        ]
    )
    lineage_path = PROCESSED_DATA_DIR / "pipeline_lineage.parquet"
    if lineage_path.exists():
        existing_lineage = pd.read_parquet(lineage_path)
        lineage_record = pd.concat([existing_lineage, lineage_record], ignore_index=True)
    lineage_record.to_parquet(lineage_path, index=False)

    logger.info(
        "ETL Complete | Extracted=%d | Passed=%d | Warned=%d | Quarantined=%d | Duplicates=%d | RefIntegrity=%s",
        dq_metrics["records_extracted"],
        dq_metrics["records_passed"],
        dq_metrics["records_warned"],
        dq_metrics["records_quarantined"],
        dq_metrics["duplicate_count"],
        ref_integrity["referential_integrity_passed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenFDA Regulatory Intelligence ETL Pipeline")
    parser.add_argument("--domain", default="drug", choices=list(ENDPOINT_REGISTRY.keys()))
    parser.add_argument("--incremental", action="store_true", help="Extract incrementally from checkpoint watermark")
    args = parser.parse_args()
    run_pipeline(domain=args.domain, incremental=args.incremental)