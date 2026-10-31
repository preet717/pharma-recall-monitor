"""Dashboard analytics: filter-aware firm scorecard and plain-language insight generation."""
from typing import List
import pandas as pd

from src.transforms import RecallTransformer

UNCLASSIFIED_CATEGORY = "Unclassified / Manual Review"


def firm_scorecard(fact_df: pd.DataFrame, dim_firms: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes firm risk metrics for the currently filtered recalls so the scorecard honours the
    sidebar filters. Adverse-event context is a whole-window figure, so it is carried over from
    `dim_firms` rather than recomputed.
    """
    if fact_df.empty:
        return pd.DataFrame()
    scorecard = RecallTransformer().recompute_dim_firms(fact_df, [])
    if not dim_firms.empty and "contextual_adverse_event_count" in dim_firms.columns:
        ae_by_firm = dim_firms.set_index("firm_id")["contextual_adverse_event_count"]
        scorecard["contextual_adverse_event_count"] = (
            scorecard["firm_id"].map(ae_by_firm).fillna(0).astype(int)
        )
    return scorecard


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def build_insights(fact_df: pd.DataFrame) -> List[str]:
    """Returns short, plain-English findings for the filtered recall population."""
    total = len(fact_df)
    if total == 0:
        return ["No recalls match the current filters."]

    insights: List[str] = []

    # 1. Trend: latest 6 months vs the 6 months before
    dated = fact_df.dropna(subset=["report_date"])
    if not dated.empty:
        latest = dated["report_date"].max()
        recent = dated[dated["report_date"] > latest - pd.DateOffset(months=6)]
        prior = dated[
            (dated["report_date"] <= latest - pd.DateOffset(months=6))
            & (dated["report_date"] > latest - pd.DateOffset(months=12))
        ]
        if len(prior) > 0:
            change = _pct(len(recent) - len(prior), len(prior))
            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            insights.append(
                f"**Trend:** {len(recent):,} recalls in the latest 6 months vs {len(prior):,} in the "
                f"6 months before ({direction} {abs(change)}%)."
            )

    # 2. Severity mix and what drives the most dangerous recalls
    class_i = fact_df[fact_df["classification"] == "Class I"]
    if len(class_i):
        classified_i = class_i[class_i["root_cause_category"] != UNCLASSIFIED_CATEGORY]
        driver = ""
        if len(classified_i):
            top_cause = classified_i["root_cause_category"].value_counts()
            driver = (
                f" The leading identified cause is **{top_cause.index[0]}** "
                f"({_pct(top_cause.iloc[0], len(class_i))}% of Class I)."
            )
        insights.append(
            f"**Severity:** {len(class_i):,} recalls ({_pct(len(class_i), total)}%) are Class I, "
            f"where use may cause serious harm or death.{driver}"
        )

    # 3. Overall root cause
    classified = fact_df[fact_df["root_cause_category"] != UNCLASSIFIED_CATEGORY]
    if len(classified):
        causes = classified["root_cause_category"].value_counts()
        insights.append(
            f"**Root cause:** {causes.index[0]} is the most common defect category "
            f"({_pct(causes.iloc[0], total)}% of recalls), followed by {causes.index[1] if len(causes) > 1 else 'none'}."
        )

    # 4. Concentration among manufacturers
    firm_counts = fact_df["firm_id"].value_counts()
    if len(firm_counts) >= 10:
        insights.append(
            f"**Concentration:** the top 10 of {len(firm_counts):,} recalling firms account for "
            f"{_pct(firm_counts.head(10).sum(), total)}% of recalls."
        )

    # 5. Speed and reach
    if fact_df["reporting_lag_days"].notna().any():
        insights.append(
            f"**Speed & reach:** median {int(fact_df['reporting_lag_days'].median())} days from recall "
            f"initiation to FDA report; {_pct(fact_df['is_nationwide'].sum(), total)}% were distributed nationwide."
        )

    # 6. Honest caveat about the heuristic classifier
    uncertain = fact_df["confidence_tier"].isin(["LOW_CONFIDENCE", "UNCLASSIFIED"]).sum()
    insights.append(
        f"**Caveat:** {_pct(uncertain, total)}% of recalls have a low-confidence or no root-cause match; "
        "categories are keyword-based decision support, not regulatory determinations."
    )
    return insights
