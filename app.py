# """Operational regulatory and data-quality Streamlit dashboard."""

# import json
# from pathlib import Path

# import pandas as pd
# import streamlit as st

# from src.config import PROCESSED_DIR

# st.set_page_config(page_title="Pharma Recall Monitor", layout="wide")
# st.title("Pharma Recall Monitor")
# st.caption("Regulatory recall operations | Local processed data")


# def read_jsonl(path: Path) -> pd.DataFrame:
#     if not path.exists() or path.stat().st_size == 0:
#         return pd.DataFrame()
#     rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
#     return pd.DataFrame(rows)


# records = read_jsonl(PROCESSED_DIR / "star_schema.jsonl")
# quarantine = read_jsonl(PROCESSED_DIR / "quarantine.jsonl")
# audit = read_jsonl(PROCESSED_DIR / "reprocessing_audit.jsonl")

# metric_columns = st.columns(3)
# metric_columns[0].metric("Accepted recalls", len(records))
# metric_columns[1].metric("Quarantined records", len(quarantine))
# metric_columns[2].metric("Reprocessing runs", len(audit))

# recall_tab, quality_tab, audit_tab = st.tabs(["Recall register", "Data quality", "Reprocessing audit"])
# with recall_tab:
#     if records.empty:
#         st.info("No processed recall records found. Run the ETL reprocess command after placing raw extracts in data/raw.")
#     else:
#         if "defect_categories" in records:
#             options = sorted({category for categories in records["defect_categories"] if isinstance(categories, list) for category in categories})
#             selected = st.multiselect("Defect category", options)
#             if selected:
#                 records = records[records["defect_categories"].apply(lambda values: bool(set(values or []) & set(selected)))]
#         st.dataframe(records, use_container_width=True, hide_index=True)
# with quality_tab:
#     st.subheader("Quarantine")
#     st.dataframe(quarantine, use_container_width=True, hide_index=True)
# with audit_tab:
#     st.dataframe(audit, use_container_width=True, hide_index=True)



"""
Streamlit Operational Application: OpenFDA Pharma Recall & Compliance Monitor.
Run via: streamlit run app.py
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DATA_DIR
from src.reprocessor import QuarantineReprocessor
from src.validator import VALIDATION_RULE_CATALOG

st.set_page_config(
    page_title="OpenFDA Pharma Recall & Compliance Monitor",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(ttl=15)
def load_parquet_tables():
    """Loads processed Star Schema and observability Parquet tables."""
    tables = {}
    expected_files = [
        "fact_recalls",
        "dim_firms",
        "dim_products",
        "quarantined_records",
        "dq_rule_summary",
        "dq_field_completeness",
        "pipeline_lineage",
    ]
    for name in expected_files:
        path = PROCESSED_DATA_DIR / f"{name}.parquet"
        tables[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return tables


def main():
    st.title("🛡️ OpenFDA Drug Recall & Regulatory Compliance Intelligence")
    st.caption(
        "End-to-End Star Schema Analytics, Explainable Heuristic Defect Classification, "
        "and Offline Quarantine Reprocessing Platform (36-Month Enforcement Window)"
    )

    tables = load_parquet_tables()
    fact_df = tables["fact_recalls"]
    firms_df = tables["dim_firms"]
    quarantine_df = tables["quarantined_records"]
    dq_rules_df = tables["dq_rule_summary"]
    dq_comp_df = tables["dq_field_completeness"]
    lineage_df = tables["pipeline_lineage"]

    if fact_df.empty:
        st.error(
            "Processed analytical tables not found in `/data/processed/`. "
            "Please run `python etl.py` first to extract and model OpenFDA data."
        )
        st.stop()

    # Global Sidebar Filters
    st.sidebar.header("🔎 Global Analytical Filters")
    selected_classes = st.sidebar.multiselect(
        "Hazard Classification",
        options=["Class I", "Class II", "Class III"],
        default=["Class I", "Class II", "Class III"],
    )
    selected_statuses = st.sidebar.multiselect(
        "Enforcement Status",
        options=sorted(fact_df["status"].dropna().unique().tolist()),
        default=sorted(fact_df["status"].dropna().unique().tolist()),
    )
    selected_dq_tiers = st.sidebar.multiselect(
        "DQ Validation Status",
        options=["PASS", "WARN"],
        default=["PASS", "WARN"],
    )
    confidence_filter = st.sidebar.multiselect(
        "Heuristic Confidence Tier",
        options=["HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", "LOW_CONFIDENCE", "UNCLASSIFIED"],
        default=["HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", "LOW_CONFIDENCE", "UNCLASSIFIED"],
    )

    filtered_fact = fact_df[
        (fact_df["classification"].isin(selected_classes))
        & (fact_df["status"].isin(selected_statuses))
        & (fact_df["dq_status"].isin(selected_dq_tiers))
        & (fact_df["confidence_tier"].isin(confidence_filter))
    ].copy()

    tabs = st.tabs(
        [
            "📊 1. Executive Risk Overview",
            "🧠 2. Explainable Root-Cause Intelligence",
            "🏭 3. Manufacturer Risk Scorecard",
            "🛠️ 4. Quarantine & Reprocessing Inspector",
            "🔍 5. Data Quality & Pipeline Lineage",
        ]
    )

    # =========================================================================
    # TAB 1: EXECUTIVE RISK OVERVIEW
    # =========================================================================
    with tabs[0]:
        total_recalls = len(filtered_fact)
        c1_count = int((filtered_fact["classification"] == "Class I").sum())
        c1_pct = round((c1_count / max(total_recalls, 1)) * 100, 1)
        median_lag = (
            int(filtered_fact["reporting_lag_days"].median())
            if filtered_fact["reporting_lag_days"].notna().any()
            else 0
        )
        multi_defect_pct = round((filtered_fact["is_multi_defect"].mean() * 100), 1) if total_recalls else 0.0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Active Analytical Recalls", f"{total_recalls:,}")
        k2.metric("Class I (Life-Threatening)", f"{c1_count:,}", f"{c1_pct}% of total")
        k3.metric("Median Reporting Lag", f"{median_lag} days")
        k4.metric("Multi-Defect Recalls", f"{multi_defect_pct}%")
        k5.metric("Quarantined Records", f"{len(quarantine_df):,}")

        col_left, col_right = st.columns(2)
        with col_left:
            monthly_trend = (
                filtered_fact.groupby(["report_month", "classification"], as_index=False)
                .size()
                .rename(columns={"size": "recall_count"})
                .sort_values("report_month")
            )
            fig_trend = px.bar(
                monthly_trend,
                x="report_month",
                y="recall_count",
                color="classification",
                title="Monthly Recall Enforcement Volume by Hazard Class",
                color_discrete_map={"Class I": "#d62728", "Class II": "#ff7f0e", "Class III": "#1f77b4"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_right:
            rc_summary = (
                filtered_fact.groupby(["root_cause_category", "classification"], as_index=False)
                .size()
                .rename(columns={"size": "recall_count"})
            )
            fig_rc = px.bar(
                rc_summary,
                y="root_cause_category",
                x="recall_count",
                color="classification",
                orientation="h",
                title="Root-Cause Defect Taxonomy by Hazard Class",
                color_discrete_map={"Class I": "#d62728", "Class II": "#ff7f0e", "Class III": "#1f77b4"},
            )
            st.plotly_chart(fig_rc, use_container_width=True)

    # =========================================================================
    # TAB 2: EXPLAINABLE ROOT-CAUSE INTELLIGENCE
    # =========================================================================
    with tabs[1]:
        st.subheader("Explainable Heuristic Root-Cause Classifier")
        st.info(
            "**Methodology Note:** `heuristic_score` is an uncalibrated deterministic keyword-weight indicator "
            "(not a calibrated statistical probability). Records with overlapping defect categories or narrow "
            "margins (<0.15) are routed to `LOW_CONFIDENCE`, and records without keyword matches are assigned "
            "`Unclassified / Manual Review`."
        )

        c_left, c_right = st.columns(2)
        with c_left:
            tier_counts = (
                filtered_fact.groupby(["confidence_tier", "root_cause_category"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            fig_conf = px.bar(
                tier_counts,
                x="confidence_tier",
                y="count",
                color="root_cause_category",
                title="Distribution of Heuristic Confidence Tiers Across Categories",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        with c_right:
            fig_hist = px.histogram(
                filtered_fact,
                x="heuristic_score",
                color="confidence_tier",
                nbins=20,
                title="Uncalibrated Heuristic Score Distribution by Confidence Tier",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🔬 Audit Individual Classification Explanations & Competing Category Scores")
        tier_focus = st.selectbox(
            "Filter Inspector by Confidence Tier",
            options=["ALL", "LOW_CONFIDENCE", "UNCLASSIFIED", "MODERATE_CONFIDENCE", "HIGH_CONFIDENCE"],
            index=1,
        )
        inspect_df = (
            filtered_fact
            if tier_focus == "ALL"
            else filtered_fact[filtered_fact["confidence_tier"] == tier_focus]
        )
        st.dataframe(
            inspect_df[
                [
                    "recall_number",
                    "classification",
                    "root_cause_category",
                    "secondary_category",
                    "is_multi_defect",
                    "confidence_tier",
                    "heuristic_score",
                    "matched_keywords",
                    "competing_category_scores",
                    "reason_for_recall",
                    "classifier_version",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================================
    # TAB 3: MANUFACTURER RISK SCORECARD
    # =========================================================================
    with tabs[2]:
        st.subheader("Manufacturer Regulatory Risk Scorecard")
        st.caption(
            "**Risk Scoring Formula:** `Severity Weighted Score = (Class I × 10) + (Class II × 5) + (Class III × 1)`. "
            "`Recalls per Product` is computed strictly where reliable NDC/brand identifiers exist. "
            "`Contextual Adverse Events` are displayed separately as market-volume context and are NOT evidence of recall causation."
        )

        min_recalls = st.slider("Minimum Total Recalls per Firm", min_value=1, max_value=20, value=2)
        scored_firms = firms_df[firms_df["total_recalls"] >= min_recalls].copy()

        col_scatter, col_top = st.columns(2)
        with col_scatter:
            fig_firms = px.scatter(
                scored_firms.head(60),
                x="total_recalls",
                y="severity_weighted_score",
                size="class_i_pct",
                color="primary_state",
                hover_name="firm_name_clean",
                hover_data=["class_i_recalls", "recalls_per_product", "contextual_adverse_event_count"],
                title="Top Manufacturers: Total Recalls vs. Severity-Weighted Score (Size = Class I %)",
            )
            st.plotly_chart(fig_firms, use_container_width=True)

        with col_top:
            top15 = scored_firms.head(15).sort_values("severity_weighted_score", ascending=True)
            fig_bar = px.bar(
                top15,
                x="severity_weighted_score",
                y="firm_name_clean",
                color="class_i_pct",
                orientation="h",
                title="Top 15 Firms by Severity-Weighted Recall Score",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            scored_firms[
                [
                    "firm_name_clean",
                    "primary_state",
                    "primary_country",
                    "severity_weighted_score",
                    "total_recalls",
                    "class_i_recalls",
                    "class_i_pct",
                    "distinct_identified_products",
                    "recalls_per_product",
                    "median_reporting_lag_days",
                    "contextual_adverse_event_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================================
    # TAB 4: QUARANTINE & REPROCESSING INSPECTOR
    # =========================================================================
    with tabs[3]:
        st.subheader("🛠️ Interactive Quarantine & Offline Reprocessing Inspector")
        st.markdown(
            "Inspect records that failed critical validation checks, review violated rules and expected contracts, "
            "and trigger **offline reprocessing on the preserved `raw_json_payload` (zero API calls)** with an "
            "**immutable audit trail**."
        )

        reprocessor = QuarantineReprocessor()
        audit_df = reprocessor.get_audit_log()

        if quarantine_df.empty:
            st.success("No records are currently in `quarantined_records.parquet`.")
        else:
            # Filter quarantined records by rule violation
            all_violated_rules = sorted(
                {
                    r.strip()
                    for reasons in quarantine_df["quarantine_reasons"].dropna()
                    for r in reasons.split(",")
                    if r.strip()
                }
            )
            selected_rule_filter = st.multiselect(
                "Filter Quarantined Records by Violated Rule",
                options=all_violated_rules,
                default=all_violated_rules,
            )

            filtered_q = quarantine_df[
                quarantine_df["quarantine_reasons"].apply(
                    lambda x: any(rule in str(x) for rule in selected_rule_filter)
                )
            ].copy()

            # Annotate whether a quarantine_id has already been reprocessed
            reprocessed_ids = set(audit_df["quarantine_id"].unique()) if not audit_df.empty else set()
            filtered_q["reprocessed_in_audit_log"] = filtered_q["quarantine_id"].isin(reprocessed_ids)

            st.markdown(f"**Quarantined Records Matching Filter ({len(filtered_q)} rows):**")
            st.dataframe(
                filtered_q[
                    [
                        "quarantine_id",
                        "recall_number",
                        "recalling_firm",
                        "classification",
                        "report_date",
                        "recall_initiation_date",
                        "quarantine_reasons",
                        "warn_flags",
                        "reprocessed_in_audit_log",
                        "quarantined_at",
                        "validator_version",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.markdown("#### 🔍 Select a Quarantined Record by Recall Number / Quarantine ID")
            options = [
                f"{row['recall_number']} | {row['quarantine_id']} ({row['quarantine_reasons']})"
                for _, row in filtered_q.iterrows()
            ]
            if options:
                selected_option = st.selectbox("Select Record to Inspect & Reprocess", options=options)
                selected_qid = selected_option.split(" | ")[1].split(" (")[0].strip()
                q_record = filtered_q[filtered_q["quarantine_id"] == selected_qid].iloc[0]

                col_rules, col_json = st.columns(2)
                with col_rules:
                    st.markdown("##### ❌ Violated Validation Rules & Expected Conditions")
                    explanations = json.loads(q_record["failure_explanations_json"])
                    for exp in explanations:
                        with st.expander(f"Rule: {exp['rule_code']}", expanded=True):
                            st.markdown(f"**Failure Reason:** {exp['failure_reason']}")
                            st.markdown(f"**Expected Validation Condition:** {exp['expected_condition']}")
                            st.markdown(f"**Remediation Guidance:** {exp['remediation_guide']}")

                    st.markdown("##### ⚙️ Offline Reprocessing Controls (Zero API Calls)")
                    strategy = st.radio(
                        "Remediation Mode",
                        options=[
                            "auto_remediate_known_anomalies",
                            "manual_json_patch",
                        ],
                        format_func=lambda x: (
                            "Automated Remediation (Disambiguate duplicate PK / align inverted dates)"
                            if x == "auto_remediate_known_anomalies"
                            else "Apply Custom Operator Edits from JSON Editor"
                        ),
                    )
                    op_notes = st.text_input(
                        "Operator Audit Note",
                        value="Verified enforcement notice; remediated payload for analytical promotion.",
                    )

                with col_json:
                    st.markdown("##### 📄 Preserved Raw JSON Payload (Editable for Manual Remediation)")
                    pretty_raw = json.dumps(json.loads(q_record["raw_json_payload"]), indent=2)
                    edited_json_str = st.text_area(
                        "Raw JSON Payload (`raw_json_payload`)",
                        value=pretty_raw,
                        height=340,
                    )

                if st.button("🚀 Trigger Offline Reprocessing", type="primary"):
                    try:
                        patched_dict = json.loads(edited_json_str)
                        result = reprocessor.reprocess_record(
                            quarantine_id=selected_qid,
                            patched_payload=patched_dict,
                            remediation_strategy=strategy,
                            operator_notes=op_notes,
                        )
                        st.cache_data.clear()
                        if result["reprocessing_outcome"] in ("PASS", "WARN"):
                            st.success(
                                f"Reprocessing Succeeded! Outcome: `{result['reprocessing_outcome']}` | "
                                f"Effective Recall Number: `{result['effective_recall_number']}` | "
                                f"Promoted to Star Schema: `{result['promoted_to_analytical_layer']}` | "
                                f"Validator: `{result['validator_version']}` | Classifier: `{result['classifier_version']}`"
                            )
                        else:
                            st.warning(
                                f"Record remains quarantined (`{result['reprocessing_outcome']}`). "
                                f"Remaining violations: `{result['remaining_violations']}`"
                            )
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid JSON in payload editor: {exc}")
                    except Exception as exc:
                        st.error(f"Reprocessing error: {exc}")

        st.divider()
        st.markdown("#### 📜 Immutable Reprocessing Audit Trail (`reprocessing_audit_log.parquet`)")
        st.caption(
            "Original records in `quarantined_records.parquet` are never overwritten. Every reprocessing "
            "attempt is appended below with timestamp, operator notes, and validator/classifier versions."
        )
        latest_audit = reprocessor.get_audit_log()
        if latest_audit.empty:
            st.info("No reprocessing attempts logged yet. Select a quarantined record above to test reprocessing.")
        else:
            st.dataframe(latest_audit.sort_values("reprocessed_at", ascending=False), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5: DATA QUALITY & PIPELINE LINEAGE
    # =========================================================================
    with tabs[4]:
        st.subheader("Pipeline Observability, Data Quality & Lineage")
        if not lineage_df.empty:
            latest_run = lineage_df.iloc[-1]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Latest Run ID", str(latest_run["run_id"])[:16])
            m2.metric("Source Mode", str(latest_run["source_mode"]))
            m3.metric("API Latency (Total)", f"{latest_run['api_latency_ms']:.0f} ms")
            m4.metric("API Retries", int(latest_run["retry_count"]))
            m5.metric("Referential Integrity", "PASS" if latest_run["referential_integrity_passed"] else "FAIL")
            m6.metric("Duplicates Caught", int(latest_run["duplicate_count"]))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Validation Rule Outcomes (`dq_rule_summary.parquet`)")
            st.dataframe(dq_rules_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("##### Field-Level Completeness (`dq_field_completeness.parquet`)")
            fig_comp = px.bar(
                dq_comp_df.sort_values("completeness_pct"),
                x="completeness_pct",
                y="field_name",
                orientation="h",
                title="Field Completeness (%) Across Extracted Raw Records",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("##### Execution Lineage History (`pipeline_lineage.parquet`)")
        st.dataframe(lineage_df.sort_values("extraction_timestamp", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()