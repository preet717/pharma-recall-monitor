"""
Explainable root-cause classification engine for FDA recall reasons.

NOTE ON HEURISTIC SCORING:
`heuristic_score` is a deterministic, uncalibrated keyword-weight indicator in [0.0, 1.0],
NOT a statistically calibrated probability. Ambiguous or closely competing matches are
explicitly routed to `LOW_CONFIDENCE` for human review.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Tuple
import pandas as pd


@dataclass(frozen=True)
class ClassificationResult:
    root_cause_category: str
    secondary_category: str
    is_multi_defect: bool
    classification_method: str
    heuristic_score: float
    confidence_tier: str
    matched_keywords: str
    competing_category_scores: str
    classifier_version: str


class BaseRootCauseClassifier(ABC):
    """Extensible interface supporting rule-based, ML, or LLM classifiers."""

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def classify(self, reason_text: str) -> ClassificationResult:
        pass

    def evaluate_against_labeled_dataset(
        self, labeled_df: pd.DataFrame, text_col: str = "reason_for_recall", label_col: str = "true_category"
    ) -> Dict[str, Any]:
        """
        Benchmarks the classifier against a manually labeled evaluation dataset.
        Enables side-by-side comparison between deterministic rules and future ML/LLM models.
        """
        if labeled_df.empty:
            return {"sample_size": 0, "exact_match_accuracy": 0.0, "low_confidence_rate": 0.0}

        predictions = [self.classify(str(text)) for text in labeled_df[text_col]]
        pred_categories = [p.root_cause_category for p in predictions]
        true_categories = labeled_df[label_col].astype(str).tolist()

        matches = sum(1 for p, t in zip(pred_categories, true_categories) if p == t)
        low_conf = sum(1 for p in predictions if p.confidence_tier in ("LOW_CONFIDENCE", "UNCLASSIFIED"))
        total = len(true_categories)

        return {
            "classifier_version": self.version,
            "sample_size": total,
            "exact_match_accuracy": round(matches / total, 4),
            "low_confidence_or_unclassified_rate": round(low_conf / total, 4),
        }


class KeywordRuleClassifier(BaseRootCauseClassifier):
    """
    Deterministic weighted keyword classifier with competing-category tracking,
    multi-defect detection, and explicit heuristic confidence tiers.
    """

    VERSION = "v1.2.0-heuristic"
    METHOD = "deterministic_weighted_keyword_heuristic"

    # Minimum heuristic thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.45
    AMBIGUITY_MARGIN_THRESHOLD = 0.15  # Minimum gap between #1 and #2 category to avoid LOW_CONFIDENCE
    MULTI_DEFECT_SECOND_SCORE_MIN = 0.35

    TAXONOMY: Dict[str, List[Tuple[str, float]]] = {
        "Sterility & Microbial Contamination": [
            (r"\bnon-sterile\b|\bnonsterile\b", 0.55),
            (r"\bsterility\b", 0.50),
            (r"\bmicrobial\b", 0.50),
            (r"\bburkholderia\b|\bpseudomonas\b|\bstaphylococcus\b", 0.65),
            (r"\bendotoxin\b", 0.55),
            (r"\bfungal\b|\bmold\b|\byeast\b", 0.45),
            (r"\bbacterial\b", 0.45),
        ],
        "Chemical Impurities & Nitrosamines": [
            (r"\bnitrosamine\b|\bndma\b|\bnnds\b|\bnitroso\b", 0.65),
            (r"\bimpurity\b|\bimpurities\b", 0.45),
            (r"\bdegradation\b", 0.40),
            (r"\bbenzene\b", 0.65),
            (r"\bcross-contamination\b|\bcross contamination\b", 0.50),
            (r"\bheavy metal\b|\blead\b|\barsenic\b", 0.50),
        ],
        "Particulate Matter & Foreign Material": [
            (r"\bparticulate matter\b|\bparticulates\b|\bparticulate\b", 0.55),
            (r"\bglass\b", 0.45),
            (r"\bforeign tablet\b|\bforeign matter\b|\bforeign material\b|\bforeign substance\b", 0.55),
            (r"\bmetal\b|\bfiber\b|\bhair\b|\bsilicone\b|\binsect\b", 0.45),
            (r"\bprecipitate\b|\bcrystallization\b", 0.45),
        ],
        "Potency, Assay & Dissolution Failures": [
            (r"\bsubpotent\b|\bsuperpotent\b", 0.60),
            (r"\bdissolution\b", 0.55),
            (r"\bassay\b", 0.45),
            (r"\bout of specification\b|\boos\b", 0.40),
            (r"\bstability\b", 0.35),
            (r"\bpotency\b|\bstrength\b", 0.40),
        ],
        "Packaging, Labeling & Mix-Up Defects": [
            (r"\bmislabeled\b|\bmislabeling\b|\blabel mix-up\b|\bmix-up\b", 0.60),
            (r"\bwrong label\b|\bincorrect label\b|\blabeling error\b", 0.55),
            (r"\bundeclared\b", 0.55),
            (r"\bcontainer closure\b|\bleaking\b|\bdefective seal\b|\bcracked vial\b", 0.45),
            (r"\bpackage\b|\bpackaging\b|\bblister\b|\bcarton\b|\binsert\b", 0.30),
        ],
        "CGMP & Temperature/Storage Deviations": [
            (r"\bcgmp\b|\bgood manufacturing\b", 0.45),
            (r"\btemperature excursion\b|\bstorage condition\b|\bfreezing\b", 0.55),
            (r"\bunapproved new drug\b|\bwithout an approved\b", 0.55),
            (r"\black of assurance\b", 0.30),
        ],
    }

    @property
    def version(self) -> str:
        return self.VERSION

    def classify(self, reason_text: str) -> ClassificationResult:
        if not reason_text or not str(reason_text).strip():
            return ClassificationResult(
                root_cause_category="Unclassified / Manual Review",
                secondary_category="NONE",
                is_multi_defect=False,
                classification_method=self.METHOD,
                heuristic_score=0.0,
                confidence_tier="UNCLASSIFIED",
                matched_keywords="NONE",
                competing_category_scores="{}",
                classifier_version=self.VERSION,
            )

        text_lower = str(reason_text).lower()
        category_scores: Dict[str, float] = {}
        category_keywords: Dict[str, List[str]] = {}

        for category, patterns in self.TAXONOMY.items():
            raw_score = 0.0
            matched_terms: List[str] = []
            for regex_pattern, weight in patterns:
                found = re.findall(regex_pattern, text_lower)
                if found:
                    raw_score += weight
                    matched_terms.extend(set(found))
            if raw_score > 0.0:
                # Cap raw heuristic accumulation at 1.0
                category_scores[category] = round(min(1.0, raw_score), 2)
                category_keywords[category] = sorted(set(matched_terms))

        if not category_scores:
            return ClassificationResult(
                root_cause_category="Unclassified / Manual Review",
                secondary_category="NONE",
                is_multi_defect=False,
                classification_method=self.METHOD,
                heuristic_score=0.0,
                confidence_tier="UNCLASSIFIED",
                matched_keywords="NONE",
                competing_category_scores="{}",
                classifier_version=self.VERSION,
            )

        # Rank categories by descending heuristic score
        ranked = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
        top_category, top_score = ranked[0]

        secondary_category = "NONE"
        second_score = 0.0
        is_multi_defect = False

        if len(ranked) > 1:
            secondary_category, second_score = ranked[1]
            if second_score >= self.MULTI_DEFECT_SECOND_SCORE_MIN:
                is_multi_defect = True

        # Collect all matched keywords across top/competing categories for explainability
        all_matched: List[str] = []
        for cat, _ in ranked:
            prefix_terms = [f"{kw} ({cat.split()[0]})" for kw in category_keywords[cat]]
            all_matched.extend(prefix_terms)

        score_margin = round(top_score - second_score, 2)

        # Determine heuristic confidence tier
        if top_score < self.LOW_CONFIDENCE_THRESHOLD or (len(ranked) > 1 and score_margin < self.AMBIGUITY_MARGIN_THRESHOLD):
            confidence_tier = "LOW_CONFIDENCE"
        elif top_score >= self.HIGH_CONFIDENCE_THRESHOLD and score_margin >= 0.25:
            confidence_tier = "HIGH_CONFIDENCE"
        else:
            confidence_tier = "MODERATE_CONFIDENCE"

        return ClassificationResult(
            root_cause_category=top_category,
            secondary_category=secondary_category if is_multi_defect else "NONE",
            is_multi_defect=is_multi_defect,
            classification_method=self.METHOD,
            heuristic_score=top_score,
            confidence_tier=confidence_tier,
            matched_keywords=", ".join(all_matched),
            competing_category_scores=json.dumps(dict(ranked)),
            classifier_version=self.VERSION,
        )