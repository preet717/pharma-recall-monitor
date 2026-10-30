"""Star schema transformation and manufacturer compliance risk scoring."""
import hashlib
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.classifier import BaseRootCauseClassifier, KeywordRuleClassifier
from src.config import SEVERITY_WEIGHTS
from src.validator import VALIDATOR_VERSION


# def normalize_firm_name(name: Any) -> str:
#     """Standardizes manufacturer names for deterministic entity resolution."""
#     if not name or not str(name).strip():
#         return "UNKNOWN MANUFACTURER"
#     cleaned = str(name).upper().strip()
#     cleaned = re.sub(r"[,\.\(\)]", " ", cleaned)
#     cleaned = re.sub(
#         r"\b(INC|INCORPORATED|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|PHARMACEUTICALS|PHARMA|LABS|LABORATORIES)\b",
#         "",
#         cleaned,
#     )
#     cleaned = re.sub(r"\s+", " ", cleaned).strip()
#     return cleaned if cleaned else str(name).upper().strip()

def normalize_firm_name(name: Any) -> str:
    """Standardizes manufacturer names across enforcement and FAERS endpoints."""
    if not name or not str(name).strip():
        return "UNKNOWN MANUFACTURER"
    cleaned = str(name).upper().strip()
    # Strip DBA clauses and parenthetical suffixes
    cleaned = re.sub(r"\bDBA\b.*$", "", cleaned)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[,\.\-/]", " ", cleaned)
    # Harmonize HEALTHCARE -> HEALTH (e.g., Cardinal Healthcare -> Cardinal Health)
    cleaned = re.sub(r"\bHEALTHCARE\b", "HEALTH", cleaned)
    cleaned = re.sub(
        r"\b(INC|INCORPORATED|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|"
        r"PHARMACEUTICALS|PHARMACEUTICAL|PHARMA|LABS|LABORATORIES|LABORATORY|"
        r"USA|US|INDIA|GLOBAL|HOLDINGS|ENTERPRISES|SERVICES|PACKAGING|INDUSTRIES)\b",
        "",
        cleaned,
    )
    # Strip trailing plant/subsidiary numbers (e.g., CARDINAL HEALTH 107 -> CARDINAL HEALTH)
    cleaned = re.sub(r"\s+\d+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else str(name).upper().strip()

def generate_firm_id(normalized_name: str) -> str:
    """Generates a deterministic 12-character primary key for a normalized firm."""
    return "FIRM_" + hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:10].upper()


class RecallTransformer:
    """Transforms validated OpenFDA records into Star Schema analytical tables."""

    def __init__(self, classifier: BaseRootCauseClassifier | None = None):
        self.classifier = classifier or KeywordRuleClassifier()

    def transform_single_record(self, r: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transforms a single validated record into fact_recalls and dim_products row dicts."""
        recall_number = str(r.get("recall_number")).strip()
        raw_firm = r.get("recalling_firm", "Unknown Firm")
        firm_clean = normalize_firm_name(raw_firm)
        firm_id = generate_firm_id(firm_clean)

        report_dt = pd.to_datetime(r.get("report_date"), format="%Y%m%d", errors="coerce")
        init_dt = pd.to_datetime(r.get("recall_initiation_date"), format="%Y%m%d", errors="coerce")

        lag_days = (
            int((report_dt - init_dt).days)
            if pd.notna(report_dt) and pd.notna(init_dt) and (report_dt - init_dt).days >= 0
            else np.nan
        )

        classification = str(r.get("classification", "")).strip()
        severity_weight = SEVERITY_WEIGHTS.get(classification, 1)

        cls_result = self.classifier.classify(r.get("reason_for_recall", ""))

        dist_pattern = str(r.get("distribution_pattern", ""))
        is_nationwide = bool(re.search(r"nationwide|all 50 states|united states|us and puerto rico", dist_pattern, re.I))

        openfda = r.get("openfda") or {}
        ndc_list = openfda.get("product_ndc") or []
        brand_list = openfda.get("brand_name") or []
        generic_list = openfda.get("generic_name") or []
        route_list = openfda.get("route") or []
        ptype_list = openfda.get("product_type") or []
        mfr_list = openfda.get("manufacturer_name") or []
        openfda_mfr_clean = normalize_firm_name(mfr_list[0]) if mfr_list else "UNMAPPED"

        primary_ndc = ndc_list[0] if ndc_list else None
        primary_brand = brand_list[0].upper() if brand_list else None
        has_reliable_product_id = bool(primary_ndc or primary_brand)
        reliable_product_key = primary_ndc or primary_brand

        fact_row = {
            "recall_number": recall_number,
            "event_id": str(r.get("event_id", "")),
            "firm_id": firm_id,
            "recalling_firm_raw": raw_firm,
            "firm_name_clean": firm_clean,
            "city": r.get("city") or "Unknown",
            "state": r.get("state") or "Unknown",
            "country": r.get("country") or "United States",
            "status": r.get("status", "Unknown"),
            "classification": classification,
            "severity_weight": severity_weight,
            "voluntary_mandated": r.get("voluntary_mandated", "Unknown"),
            "recall_initiation_date": init_dt,
            "report_date": report_dt,
            "report_month": report_dt.strftime("%Y-%m") if pd.notna(report_dt) else "Unknown",
            "reporting_lag_days": lag_days,
            "is_nationwide": is_nationwide,
            "reason_for_recall": r.get("reason_for_recall", ""),
            "root_cause_category": cls_result.root_cause_category,
            "secondary_category": cls_result.secondary_category,
            "is_multi_defect": cls_result.is_multi_defect,
            "classification_method": cls_result.classification_method,
            "heuristic_score": cls_result.heuristic_score,
            "confidence_tier": cls_result.confidence_tier,
            "matched_keywords": cls_result.matched_keywords,
            "competing_category_scores": cls_result.competing_category_scores,
            "classifier_version": cls_result.classifier_version,
            "validator_version": r.get("validator_version", VALIDATOR_VERSION),
            "has_reliable_product_id": has_reliable_product_id,
            "reliable_product_key": reliable_product_key,
            "dq_status": r.get("dq_status", "PASS"),
            "dq_flags": r.get("dq_flags", "NONE"),
            "openfda_mfr_clean": openfda_mfr_clean,
        }

        product_row = {
            "recall_number": recall_number,
            "firm_id": firm_id,
            "product_ndc": primary_ndc or "UNMAPPED_NDC",
            "brand_name": primary_brand or "UNMAPPED_BRAND",
            "generic_name": generic_list[0].upper() if generic_list else "UNSPECIFIED_GENERIC",
            "route": route_list[0].upper() if route_list else "UNSPECIFIED_ROUTE",
            "product_type": ptype_list[0].upper() if ptype_list else "UNSPECIFIED_TYPE",
            "has_reliable_product_id": has_reliable_product_id,
            "product_description_short": str(r.get("product_description", ""))[:160],
        }
        return fact_row, product_row

    def recompute_dim_firms(
        self, fact_df: pd.DataFrame, adverse_event_counts: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Aggregates manufacturer dimension with enhanced risk metrics."""
        ae_map: Dict[str, int] = {}
        for item in adverse_event_counts:
            norm_term = normalize_firm_name(item.get("term", ""))
            ae_map[norm_term] = ae_map.get(norm_term, 0) + int(item.get("count", 0))

        if fact_df.empty:
            return pd.DataFrame()

        firm_groups = []
        for firm_id, grp in fact_df.groupby("firm_id", as_index=False):
            total_recalls = len(grp)
            c1_count = int((grp["classification"] == "Class I").sum())
            c2_count = int((grp["classification"] == "Class II").sum())
            c3_count = int((grp["classification"] == "Class III").sum())

            severity_score = (c1_count * 10) + (c2_count * 5) + (c3_count * 1)
            c1_pct = round((c1_count / total_recalls) * 100.0, 2)

            valid_products = grp.loc[grp["has_reliable_product_id"], "reliable_product_key"].dropna().unique()
            distinct_products = len(valid_products)
            recalls_per_product = (
                round(int(grp["has_reliable_product_id"].sum()) / distinct_products, 2)
                if distinct_products > 0
                else np.nan
            )

            clean_name = grp["firm_name_clean"].iloc[0]
            # Check primary firm name first, then fallback to mapped openfda.manufacturer_name
            ae_total = ae_map.get(clean_name, 0)
            if ae_total == 0 and "openfda_mfr_clean" in grp.columns:
                for alt_name in grp["openfda_mfr_clean"].dropna().unique():
                    if alt_name != "UNMAPPED" and alt_name in ae_map:
                        ae_total += ae_map[alt_name]

            firm_groups.append(
                {
                    "firm_id": firm_id,
                    "firm_name_clean": clean_name,
                    "primary_state": grp["state"].mode().iloc[0] if not grp["state"].mode().empty else "Unknown",
                    "primary_country": grp["country"].mode().iloc[0] if not grp["country"].mode().empty else "US",
                    "total_recalls": total_recalls,
                    "class_i_recalls": c1_count,
                    "class_ii_recalls": c2_count,
                    "class_iii_recalls": c3_count,
                    "class_i_pct": c1_pct,
                    "severity_weighted_score": severity_score,
                    "distinct_identified_products": distinct_products,
                    "recalls_per_product": recalls_per_product,
                    "median_reporting_lag_days": float(grp["reporting_lag_days"].median())
                    if grp["reporting_lag_days"].notna().any()
                    else np.nan,
                    # "contextual_adverse_event_count": ae_map.get(clean_name, 0),
                    "contextual_adverse_event_count": ae_total,
                }
            )
        return pd.DataFrame(firm_groups).sort_values("severity_weighted_score", ascending=False)

    def build_star_schema(
        self,
        valid_records: List[Dict[str, Any]],
        adverse_event_counts: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Produces fact_recalls, dim_firms, and dim_products DataFrames."""
        fact_rows: List[Dict[str, Any]] = []
        product_rows: List[Dict[str, Any]] = []

        for r in valid_records:
            f_row, p_row = self.transform_single_record(r)
            fact_rows.append(f_row)
            product_rows.append(p_row)

        fact_df = pd.DataFrame(fact_rows)
        dim_products_df = pd.DataFrame(product_rows)
        dim_firms_df = self.recompute_dim_firms(fact_df, adverse_event_counts)
        return fact_df, dim_firms_df, dim_products_df