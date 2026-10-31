# OpenFDA Pharma Recall & Regulatory Compliance Monitor

An end-to-end data product that answers three questions for pharmaceutical Quality Assurance, Supply-Chain Risk and Regulatory Affairs teams:

1. **What is being recalled, and is it getting worse?** Volume and severity trends over a rolling 36-month window.
2. **Why are drugs being recalled?** Free-text recall reasons are classified into root-cause categories, each with a visible explanation and a confidence tier.
3. **Which manufacturers carry the most regulatory risk?** A firm scorecard that separates *exposure* (how much risk in total) from *intensity* (how severe a typical recall is).

Data quality is a first-class part of the product. Every record is validated, failures are quarantined with their raw JSON, and an operator can fix and reprocess them offline. Each fix is written to an append-only audit trail.

---

## 1. Selected API

**[OpenFDA](https://open.fda.gov/apis/): keyless and free.** No API key, account or configuration is needed.

| Endpoint | Used for |
|---|---|
| `https://api.fda.gov/drug/enforcement.json` | Primary data: every FDA drug recall enforcement report in the window (about 2,200 records) |
| `https://api.fda.gov/drug/event.json` | Context only: adverse-event report counts per manufacturer (FAERS), aggregated on the server with `count=` |

Without a key, OpenFDA allows about 1,000 requests per day per IP. A full ETL run uses roughly 10 requests.

**Why this API:** the data is real and has real quality problems. Nested `openfda` objects are empty about half the time, firm names are inconsistent, key fields are sometimes missing and the recall reasons are unstructured text. That gives the validation and modelling layers genuine work to do, and the business use case is concrete.

---

## 2. How to Run Locally

```bash
git clone https://github.com/preet717/pharma-recall-monitor.git
cd pharma-recall-monitor
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python etl.py
streamlit run app.py
```

**Notes**
- `python etl.py` needs internet access to `api.fda.gov` and takes about 30–60 seconds. It writes raw extracts to `data/raw/` and analytical tables to `data/processed/`. No manual data preparation is needed.
- If the API is unreachable on a later run, the ETL falls back to the last raw extract cached in `data/raw/` and records `source_mode = cached_raw_fallback` in the lineage table.
- `python etl.py --incremental` fetches only records from the last high-water date onwards and merges them into the cache.
- Tests: `pytest` (14 tests, no network access).
- Tested with Python 3.13. Dependencies use compatible version ranges (see `requirements.txt`).

---

## 3. Architecture

```
 OpenFDA drug/enforcement ──┐      ┌── OpenFDA drug/event (count by manufacturer)
                            ▼      ▼
  src/extractor.py    Pagination · retry with exponential backoff and jitter (429/5xx)
                      · 404 → empty page · raw JSON cached with SHA-256 · checkpoint/high-water mark
                      · deduplicated incremental merge · cached-raw fallback
                            │   data/raw/*.json
                            ▼
  src/validator.py    9-rule contract → PASS / WARN / QUARANTINE · schema-drift check
                      · field completeness · quarantine keeps raw JSON + stable fingerprint
                            │
  src/reprocessor.py  Replays operator fixes from the audit log (so a rebuild keeps them)
                            │   PASS + WARN records
                            ▼
  src/transforms.py   Firm-name normalisation → deterministic firm_id · reporting lag
  src/classifier.py   · nationwide flag · root-cause classifier · firm aggregates + adverse-event join
                            │
  src/validator.py    Referential integrity + primary-key uniqueness
                            │   data/processed/*.parquet  (+ pipeline_lineage appended per run)
                            ▼
  app.py              Streamlit (5 tabs) · src/analytics.py builds the filter-aware scorecard and findings
```

| Path | Responsibility |
|---|---|
| `etl.py` | Entry point that runs extract → validate → replay → transform → integrity check → persist → lineage |
| `src/config.py` | Paths, rolling window, endpoint registry, valid values, severity weights |
| `src/extractor.py` | API access, retries, checkpointing, raw cache, incremental merge |
| `src/validator.py` | Rule catalog, record validation, schema and completeness checks, integrity checks |
| `src/classifier.py` | Versioned keyword root-cause classifier behind an interface that an ML/LLM model could later implement |
| `src/transforms.py` | Star-schema construction and firm risk metrics |
| `src/reprocessor.py` | Offline quarantine reprocessing, audit log, replay of fixes during ETL |
| `src/analytics.py` | Dashboard analytics: firm scorecard for the filtered data, plain-English findings |
| `tests/test_pipeline.py` | Unit tests for every layer above |

**Design choices**
- **Local Parquet files, no database.** This keeps the brief's "no external databases" rule while still giving typed columnar tables.
- **Raw data is never modified.** Derived tables can always be rebuilt from `data/raw/`.
- **Versioned logic.** Every fact row carries `validator_version` and `classifier_version`, so a change in a metric can be traced to a change in the rules.
- **The dashboard makes no network calls.** It reads only from `data/processed/`.

---

## 4. Data Model

A star schema at **one row per FDA recall number** (each recall number identifies one recalled product).

| Table | Grain | Key columns |
|---|---|---|
| `fact_recalls` | One recall (`recall_number` PK) | `firm_id` (FK), `classification`, `severity_weight`, `report_date`, `recall_initiation_date`, `reporting_lag_days`, `is_nationwide`, `root_cause_category`, `secondary_category`, `confidence_tier`, `matched_keywords`, `dq_status`, `dq_flags`, `is_remediated` |
| `dim_firms` | One normalised manufacturer (`firm_id` PK) | `firm_display_name`, `total_recalls`, `class_i_pct`, `severity_weighted_score` (exposure), `avg_severity_per_recall` (intensity), `recalls_per_product`, `median_reporting_lag_days`, `contextual_adverse_event_count` |
| `dim_products` | One recall's product attributes (`recall_number` PK/FK) | `product_ndc`, `brand_name`, `generic_name`, `route`, `product_type` |

`dim_products` shares the fact table's grain: it is a product-attribute extension, not a deduplicated product dimension. OpenFDA product identifiers are missing for about 51% of recalls, so a true product dimension would leave most facts unlinked.

**Observability tables:** `quarantined_records`, `dq_rule_summary`, `dq_field_completeness`, `pipeline_lineage` (appended each run) and `reprocessing_audit_log` (append-only).

### Transformation logic

- **Firm entity resolution.** Names are upper-cased. DBA clauses, punctuation, legal suffixes (INC, LLC…) and generic words (PHARMACEUTICALS, USA…) are removed, so "Sun Pharmaceutical Industries, Inc." and "Sun Pharma USA" resolve to the same firm. `firm_id` is a SHA-256 hash of the normalised name, so it's stable across runs. The dashboard shows the firm's most common original name.
- **Reporting lag** = `report_date − recall_initiation_date`, in days (median 29). It's left empty when the initiation date is missing.
- **Severity weight:** Class I = 10, Class II = 5, Class III = 1. **Exposure** is the sum per firm and grows with volume. **Intensity** is exposure divided by recalls (1–10), so small firms with consistently severe recalls stand out.
- **Root-cause classification.** Weighted regex keywords across six categories. The classifier records competing category scores, a secondary category when several defects are present, and the matched keywords. Confidence tiers:
  - *High* or *Moderate*, depending on the score and the margin over the next category.
  - *Low*, when the top two categories are within 0.15 of each other.
  - *Unclassified*, when no keywords match.

  The score is an uncalibrated heuristic, not a probability.
- **Adverse-event context.** The top 999 manufacturers by FAERS report count, plus targeted batched queries for recalling manufacturers outside that top 999. These are joined on the normalised name, falling back to `openfda.manufacturer_name`, and 184 of 399 firms match. The counts are shown as market-volume context only.

---

## 5. Dashboard Overview

**Sidebar filters** apply across tabs: report-date range, firm search, hazard class, recall status, DQ status and classifier confidence. A CSV download of the filtered recalls is also available.

| Tab | Audience and purpose |
|---|---|
| **1. Executive Risk Overview** | KPIs, a **Key Findings** panel of plain-English insights generated from the filtered data (trend vs. the previous 6 months, Class I share and its main cause, top defect category, firm concentration, speed and reach, classifier caveat), a monthly trend by hazard class and root cause by class |
| **2. Root-Cause Intelligence** | Confidence-tier and score distributions, plus a table showing *why* each recall was classified (matched keywords, competing scores) |
| **3. Manufacturer Risk Scorecard** | A volume vs. intensity bubble chart, top 15 firms by exposure or intensity (selectable) and the full scorecard. Recomputed from the filtered recalls |
| **4. Quarantine & Reprocessing Inspector** | Violated rules with the reason, expected condition and remediation guidance; the editable raw JSON; offline reprocessing; the audit trail |
| **5. Data Quality & Lineage** | Latest-run health (source mode, latency, retries, integrity), rule trigger rates, field completeness and run history |

---

## 6. Data Quality Checks & Validation Logic

### Record-level rule contract (`src/validator.py`)

| Rule | Tier | Check | Latest run |
|---|---|---|---|
| `CRITICAL_MISSING_RECALL_NUMBER` | QUARANTINE | Primary key is null, empty or whitespace (JSON `null` included) | 1 |
| `CRITICAL_DUPLICATE_PRIMARY_KEY` | QUARANTINE | `recall_number` already seen in this run | 0 |
| `CRITICAL_INVALID_CLASSIFICATION` | QUARANTINE | Not one of Class I / II / III | 2 |
| `CRITICAL_UNPARSEABLE_REPORT_DATE` | QUARANTINE | Missing, not `YYYYMMDD`, or in the future | 0 |
| `CRITICAL_CHRONOLOGICAL_INVERSION` | QUARANTINE | Initiation date after the report date, in the future, or before 1990 | 0 |
| `WARN_MISSING_OPENFDA_PRODUCT_IDS` | WARN | No NDC or brand name, so the record is excluded from the recalls-per-product denominator | 1,116 (51%) |
| `WARN_MISSING_INITIATION_DATE` | WARN | Initiation date missing or unparseable, so the lag is left empty | 0 |
| `WARN_MISSING_GEOGRAPHIC_STATE` | WARN | `state` is empty (usually foreign firms) | 131 (6%) |
| `WARN_UNEXPECTED_STATUS_VALUE` | WARN | Status outside Ongoing / Completed / Terminated / Pending | 0 |

WARN records are kept in the analytical tables with their flags. QUARANTINE records are held back, with their full raw JSON and a stable fingerprint.

### Dataset-level checks
- **Schema drift:** every required field must be present and of type string (or null). Missing or mistyped fields are counted per field, logged as a warning and recorded in lineage.
- **Field completeness:** percentage missing for every required field plus `openfda.brand_name` and `openfda.product_ndc`.
- **Referential integrity:** no orphan `firm_id` or `recall_number` foreign keys, and **primary-key uniqueness** in `fact_recalls` and `dim_firms`. The uniqueness checks protect against bugs in incremental merges and reprocessing promotions.

### Operational reliability
- Retries with exponential backoff and jitter on 429/5xx and network errors. A 404 from OpenFDA means an empty page, not a failure.
- Falls back to the cached raw data when the live API fails, and records the source mode in lineage.
- Incremental runs restart at the high-water date inclusively, then **deduplicate on `recall_number` (newest copy wins)**, so boundary records and FDA status updates don't create false duplicates.
- Each run logs its run ID, latency, retry count, API calls, DQ counts and integrity results to `pipeline_lineage.parquet`.

### Quarantine remediation
- **Automatic fixes are limited to safe cases.** A missing or duplicate key gets a synthetic key from `event_id` plus the record index. An inverted initiation date is *cleared*, so the record continues with a WARN and no lag. **Hazard classification is never filled in automatically**, because it's a regulatory determination. The operator must patch it from the FDA notice.
- Every attempt is appended to `reprocessing_audit_log.parquet` with the operator note, the before/after rules, the patched payload and the validator/classifier versions. The original quarantine table is never overwritten.
- **Fixes survive a rebuild.** On each run, `etl.py` re-applies promoted fixes to records quarantined again (matched by raw-record fingerprint) and re-validates them against the current rules. A record the FDA has since corrected upstream isn't double-counted.

---

## 7. Assumptions

- `recall_number` uniquely identifies a recalled product. One recall *event* (`event_id`) can cover several products, which is why there are 2,193 recalls across 980 events.
- `report_date` (when the FDA published the recall) defines the analysis window. The window covers the 36 calendar months ending today and is recalculated on every run.
- A missing `country` is treated as the United States (OpenFDA omits it mostly for domestic firms).
- The "nationwide" flag comes from the free-text `distribution_pattern` (keywords such as *nationwide*, *all 50 states*).
- Severity weights of 10 / 5 / 1 reflect FDA's class definitions (serious harm → unlikely harm). They are a judgment call, kept in `src/config.py`.

## 8. Known Limitations

- **The classifier is keyword-based.** About 11% of recalls are unclassified and 9% are low-confidence. Its accuracy-check helper exists, but there is no labelled sample of real recalls to measure accuracy yet.
- **Firm-name normalisation is aggressive.** Removing generic words can merge distinct firms with similar names. Matching to FAERS manufacturer names is exact after normalisation (184 of 399 firms match).
- **Adverse-event counts are context, not causation.** They count all FAERS reports naming the manufacturer, not reports linked to a specific recall.
- **No full history of changes.** Each record is stored as last reported; status changes over time aren't tracked.
- **Data is capped at 10,000 records per run** (10 pages × 1,000). That's far above the current ~2,200 but would need raising for longer windows.
- Reprocessing from the dashboard writes to local Parquet files. It's designed for a single operator; there's no locking.
- A fresh clone needs internet access for the first `python etl.py`. The cached-raw fallback only works after one successful run.

---

## 9. How AI Was Used

The full transcript is in [`ai_transcript/ai_transcript.pdf`](ai_transcript/ai_transcript.pdf). A short summary of one debugging session is in [`ai_transcript/reprocessor-empty-dataframe.txt`](ai_transcript/reprocessor-empty-dataframe.txt).

**Tools:** Google Gemini for planning, design and the initial code; GitHub Copilot in the editor (guided by `.github/copilot-instructions.md`); and Claude Code for a final review and hardening pass against the evaluation criteria.

**How the work progressed**
1. **Breaking down the assignment and choosing the API.** I asked for keyless API options and weighed Open-Meteo, OpenFDA, USGS and World Bank/FX. I chose OpenFDA because its messy nested JSON supports meaningful DQ and modelling work, and it has a clear stakeholder (pharma QA / regulatory).
2. **Data model and ETL design.** I reviewed the proposed star schema, the PASS/WARN/QUARANTINE contract and the root-cause taxonomy before any UI was generated.
3. **Challenging the AI's suggestions** (examples from the transcript):
   - *Classifier confidence.* The first draft presented a weighted keyword score as a "confidence score". I rejected that framing and asked for an explicitly uncalibrated heuristic score, a LOW_CONFIDENCE tier for close competing categories, an "Unclassified / Manual Review" category instead of a forced label, stored matched keywords and competing scores, and an interface so an ML/LLM classifier could later be compared against a labelled set.
   - *Quarantine inspector.* I specified the requirements: filter by rule, inspect the raw JSON, see the violated rules and expected conditions, reprocess offline without API calls, and keep an immutable audit trail that never overwrites the original quarantine records.
4. **Debugging and validation with real output**:
   - The FAERS endpoint returned **403 Forbidden** for `limit=1000`. Diagnosis: the count endpoint caps at 999.
   - I questioned whether the global top-999 manufacturer list silently drops recalling firms. It did, so I added targeted batched queries for missing manufacturers and matching on both `recalling_firm` and `openfda.manufacturer_name`. Firms with adverse-event context rose from 97 to 184.
   - I fixed a `KeyError` when reprocessing against an empty fact table, and replaced deprecated `datetime.utcnow()` calls.
5. **Final review pass.** An AI review against the rubric found real defects, which were then fixed and covered with new tests (5 → 14):
   - incremental runs created false duplicates (5 per run)
   - dashboard remediations were lost on the next ETL run
   - the schema check was hard-coded to pass
   - JSON `null` recall numbers passed validation
   - auto-remediation invented a hazard class
   - `lead` (the verb) was matched as the contaminant

   That review also added the date and firm filters, the Key Findings panel and the exposure vs. intensity ranking.

Throughout, AI output was checked by running the pipeline and the tests and by reading the produced data, not accepted as written.

---

## 10. What I Would Improve With More Time

- **Measure the classifier:** hand-label about 200 recall reasons, report precision and recall per category with the existing accuracy-check helper, then compare against an LLM classifier behind the same interface.
- **Track recall history:** store a dated snapshot per run (type-2 history) to see how long recalls stay "Ongoing" and how often they are reclassified.
- **Better entity resolution:** fuzzy matching and the FDA establishment registry (FEI numbers) instead of rule-based name cleaning.
- **More domains:** add device and food recalls through the existing endpoint registry, which is already set up for it.
- **Normalise for market size:** recalls per approved product (from the Drugs@FDA or NDC endpoints) instead of raw counts.
- **Scheduling and alerting:** a nightly incremental run with alerts on new Class I recalls, and DQ thresholds that fail the run.
