"""
Streamlit Operational Application: OpenFDA Pharma Recall & Compliance Monitor.
Run via: streamlit run app.py
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import build_insights, firm_scorecard
from src.config import PROCESSED_DATA_DIR
from src.reprocessor import QuarantineReprocessor

st.set_page_config(
    page_title="OpenFDA Pharma Recall & Compliance Monitor",
    page_icon="🛡️",
    layout="wide",
)

CONFIDENCE_LABELS = {
    "HIGH_CONFIDENCE": "High confidence",
    "MODERATE_CONFIDENCE": "Moderate confidence",
    "LOW_CONFIDENCE": "Low confidence (review)",
    "UNCLASSIFIED": "Unclassified (no keyword match)",
}


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
        "Which drug recalls are happening, why, and which manufacturers carry the most regulatory risk: "
        "FDA enforcement reports over a rolling 36-month window, with data-quality checks you can inspect."
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
    st.sidebar.header("🔎 Filters")
    min_date = fact_df["report_date"].min().date()
    max_date = fact_df["report_date"].max().date()
    date_range = st.sidebar.date_input(
        "FDA Report Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # While the user is mid-selection Streamlit returns a single date
    start_date, end_date = date_range if len(date_range) == 2 else (date_range[0], max_date)
    firm_query = st.sidebar.text_input("Search Recalling Firm", placeholder="e.g. Cardinal, Sun Pharma")
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
        "Root-Cause Confidence",
        options=list(CONFIDENCE_LABELS),
        default=list(CONFIDENCE_LABELS),
        format_func=CONFIDENCE_LABELS.get,
    )

    filtered_fact = fact_df[
        (fact_df["report_date"].dt.date.between(start_date, end_date))
        & (fact_df["classification"].isin(selected_classes))
        & (fact_df["status"].isin(selected_statuses))
        & (fact_df["dq_status"].isin(selected_dq_tiers))
        & (fact_df["confidence_tier"].isin(confidence_filter))
    ].copy()
    if firm_query.strip():
        filtered_fact = filtered_fact[
            filtered_fact["recalling_firm_raw"].str.contains(firm_query.strip(), case=False, regex=False, na=False)
        ]

    st.sidebar.caption(f"{len(filtered_fact):,} of {len(fact_df):,} recalls match the filters.")
    st.sidebar.download_button(
        "⬇️ Download Filtered Recalls (CSV)",
        data=filtered_fact.drop(columns=["competing_category_scores"]).to_csv(index=False).encode("utf-8"),
        file_name="filtered_recalls.csv",
        mime="text/csv",
    )

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
        k1.metric("Recalls in Scope", f"{total_recalls:,}")
        k2.metric("Class I (Most Serious)", f"{c1_count:,}", f"{c1_pct}% of total", delta_color="off")
        k3.metric("Median Initiation → FDA Report", f"{median_lag} days")
        k4.metric("Multi-Defect Recalls", f"{multi_defect_pct}%")
        k5.metric("Quarantined Records", f"{len(quarantine_df):,}")

        st.markdown("#### 💡 Key Findings")
        for insight in build_insights(filtered_fact):
            st.markdown(f"- {insight}")

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
            st.plotly_chart(fig_trend, width="stretch")

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
            st.plotly_chart(fig_rc, width="stretch")

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
            st.plotly_chart(fig_conf, width="stretch")

        with c_right:
            fig_hist = px.histogram(
                filtered_fact,
                x="heuristic_score",
                color="confidence_tier",
                nbins=20,
                title="Uncalibrated Heuristic Score Distribution by Confidence Tier",
            )
            st.plotly_chart(fig_hist, width="stretch")

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
            width="stretch",
            hide_index=True,
        )

    # =========================================================================
    # TAB 3: MANUFACTURER RISK SCORECARD
    # =========================================================================
    with tabs[2]:
        st.subheader("Manufacturer Regulatory Risk Scorecard")
        st.caption(
            "**Exposure** = (Class I × 10) + (Class II × 5) + (Class III × 1); it grows with recall volume. "
            "**Intensity** = exposure ÷ total recalls (1–10); it shows how serious a firm's recalls typically are, "
            "regardless of size. Scores follow the sidebar filters. Recalls per product only counts recalls with an "
            "NDC or brand identifier. Adverse-event counts are whole-window market context, not evidence of causation."
        )

        sc1, sc2 = st.columns(2)
        min_recalls = sc1.slider("Minimum Total Recalls per Firm", min_value=1, max_value=20, value=2)
        rank_by = sc2.radio(
            "Rank Firms By",
            options=["severity_weighted_score", "avg_severity_per_recall"],
            format_func=lambda x: "Exposure (total severity)" if x == "severity_weighted_score" else "Intensity (avg severity per recall)",
            horizontal=True,
        )
        scorecard = firm_scorecard(filtered_fact, firms_df)
        if scorecard.empty:
            st.info("No recalls match the current filters.")
        else:
            scored_firms = (
                scorecard[scorecard["total_recalls"] >= min_recalls]
                .sort_values([rank_by, "total_recalls"], ascending=False)
            )

            col_scatter, col_top = st.columns(2)
            with col_scatter:
                fig_firms = px.scatter(
                    scored_firms.head(60),
                    x="total_recalls",
                    y="avg_severity_per_recall",
                    size="severity_weighted_score",
                    color="class_i_pct",
                    color_continuous_scale="Reds",
                    hover_name="firm_display_name",
                    hover_data=["class_i_recalls", "recalls_per_product", "contextual_adverse_event_count"],
                    labels={
                        "total_recalls": "Total recalls",
                        "avg_severity_per_recall": "Intensity (avg severity)",
                        "class_i_pct": "Class I %",
                    },
                    title="Volume vs. Intensity (bubble size = exposure; top-right = highest concern)",
                )
                st.plotly_chart(fig_firms, width="stretch")

            with col_top:
                top15 = scored_firms.head(15).sort_values(rank_by, ascending=True)
                fig_bar = px.bar(
                    top15,
                    x=rank_by,
                    y="firm_display_name",
                    color="class_i_pct",
                    orientation="h",
                    labels={rank_by: "Score", "firm_display_name": "", "class_i_pct": "Class I %"},
                    title="Top 15 Firms by Selected Ranking",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_bar, width="stretch")

            st.dataframe(
                scored_firms[
                    [
                        "firm_display_name",
                        "primary_state",
                        "primary_country",
                        "severity_weighted_score",
                        "avg_severity_per_recall",
                        "total_recalls",
                        "class_i_recalls",
                        "class_i_pct",
                        "distinct_identified_products",
                        "recalls_per_product",
                        "median_reporting_lag_days",
                        "contextual_adverse_event_count",
                    ]
                ].rename(
                    columns={
                        "firm_display_name": "Firm",
                        "primary_state": "State",
                        "primary_country": "Country",
                        "severity_weighted_score": "Exposure",
                        "avg_severity_per_recall": "Intensity",
                        "total_recalls": "Recalls",
                        "class_i_recalls": "Class I",
                        "class_i_pct": "Class I %",
                        "distinct_identified_products": "Identified Products",
                        "recalls_per_product": "Recalls / Product",
                        "median_reporting_lag_days": "Median Lag (days)",
                        "contextual_adverse_event_count": "Adverse-Event Reports (context)",
                    }
                ),
                width="stretch",
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

            # Annotate whether a record has already been reprocessed. Quarantine IDs change on every ETL run,
            # so match on the stable raw-record fingerprint where both sides have it.
            reprocessed = filtered_q["quarantine_id"].isin(set(audit_df["quarantine_id"])) if not audit_df.empty else False
            if not audit_df.empty and "record_fingerprint" in audit_df and "record_fingerprint" in filtered_q:
                reprocessed = reprocessed | filtered_q["record_fingerprint"].isin(set(audit_df["record_fingerprint"].dropna()))
            filtered_q["reprocessed_in_audit_log"] = reprocessed

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
                width="stretch",
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
                            "Automated Remediation (synthetic key for missing/duplicate PK; clear inverted initiation date)"
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
            st.dataframe(latest_audit.sort_values("reprocessed_at", ascending=False), width="stretch", hide_index=True)

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
            st.dataframe(dq_rules_df, width="stretch", hide_index=True)
        with c2:
            st.markdown("##### Field-Level Completeness (`dq_field_completeness.parquet`)")
            fig_comp = px.bar(
                dq_comp_df.sort_values("completeness_pct"),
                x="completeness_pct",
                y="field_name",
                orientation="h",
                title="Field Completeness (%) Across Extracted Raw Records",
            )
            st.plotly_chart(fig_comp, width="stretch")

        st.markdown("##### Execution Lineage History (`pipeline_lineage.parquet`)")
        st.dataframe(lineage_df.sort_values("extraction_timestamp", ascending=False), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()