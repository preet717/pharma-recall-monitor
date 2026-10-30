# OpenFDA Pharma Recall & Regulatory Compliance Intelligence Platform

## 1. Project Overview & Selected API
This repository implements an end-to-end regulatory intelligence and data quality platform built with **Python**, **Streamlit**, and the keyless **OpenFDA Public API** (`drug/enforcement.json` and `drug/event.json`).

### Business Problem
Pharmaceutical Quality Assurance (QA), Supply Chain Risk, and Regulatory Affairs teams need continuous visibility into FDA drug enforcement recalls, defect root causes, and manufacturer compliance trajectories. Raw OpenFDA enforcement feeds contain unstructured defect narratives, inconsistent manufacturer naming, missing product identifiers, duplicate recall numbers across multi-lot distributions, and occasional chronological anomalies.

This platform solves those challenges by:
1. **Resiliently extracting** 36 months of FDA Drug Enforcement recalls (November 2023 – October 2026) and contextual manufacturer adverse event counts with pagination, exponential backoff, incremental checkpointing, and offline raw caching.
2. **Validating data quality** via an explicit **`PASS` / `WARN` / `QUARANTINE`** contract that retains raw JSON payloads for quarantined records and enables zero-API-call offline reprocessing with an immutable audit trail.
3. **Classifying unstructured recall reasons** into regulatory root-cause categories with an explainable, versioned heuristic classifier that tracks competing category scores, multi-defect occurrences, and low-confidence ambiguity.
4. **Modeling a Star Schema** (`fact_recalls`, `dim_firms`, `dim_products`) in local Parquet storage and surfacing insights in a 5-tab operational Streamlit application.

### Selected API (Keyless)
- **Primary Endpoint**: `https://api.fda.gov/drug/enforcement.json` (No API key required).
- **Contextual Endpoint**: `https://api.fda.gov/drug/event.json` (Server-side aggregation by manufacturer name; no API key required).

---

## 2. How to Run the Project Locally

The project runs locally using the standard command flow below with zero external databases, Docker containers, or API keys required:

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python etl.py
streamlit run app.py