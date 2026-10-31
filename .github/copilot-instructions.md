# Project Instructions

- This is a Python ETL and Streamlit project for public pharmaceutical recall data.
- Keep extraction explicit and opt-in; dashboard startup and offline processing must not make network requests.
- Preserve raw extracts. Reprocessing may replace derived Parquet tables in `data/processed/`, but must append audit entries rather than rewriting the audit log.
- Keep classifier and validation behavior transparent, versioned, and covered by focused pytest tests.
- Use the paths and module boundaries already established in `src/config.py`.
- Treat heuristic results as decision support, not regulatory determinations.
