"""Resilient API extractor with pagination, backoff retries, and checkpointing."""
import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests

from src.config import CHECKPOINT_FILE, EndpointConfig, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class OpenFDAExtractor:
    """Extracts enforcement and contextual adverse event data from OpenFDA."""

    def __init__(self, config: EndpointConfig, max_retries: int = 3, timeout: int = 20):
        self.config = config
        self.max_retries = max_retries
        self.timeout = timeout
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.telemetry: Dict[str, Any] = {
            "retry_count": 0,
            "api_latency_ms_total": 0.0,
            "api_calls_made": 0,
            "api_meta_last_updated": None,
            "source_mode": "live_api",
        }

    def _load_checkpoint(self) -> Dict[str, Any]:
        if CHECKPOINT_FILE.exists():
            try:
                return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint file found; starting fresh.")
        return {}

    def _save_checkpoint(self, domain: str, high_water_date: str, record_count: int, payload_hash: str) -> None:
        state = self._load_checkpoint()
        state[domain] = {
            "last_extracted_at": datetime.now(timezone.utc).isoformat(),
            "high_water_report_date": high_water_date,
            "last_record_count": record_count,
            "payload_sha256": payload_hash,
            "window_start": self.config.start_date,
            "window_end": self.config.end_date,
        }
        CHECKPOINT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _request_with_backoff(self, url: str) -> Dict[str, Any]:
        """Executes HTTP GET with exponential backoff and jitter, recording latency."""
        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                self.telemetry["api_calls_made"] += 1
                response = requests.get(url, timeout=self.timeout)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.telemetry["api_latency_ms_total"] += elapsed_ms

                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    # OpenFDA returns 404 when no records match a page or query
                    return {"meta": {"results": {"total": 0}}, "results": []}
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"Transient HTTP {response.status_code}")
                response.raise_for_status()
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"API request failed after {self.max_retries} attempts: {exc}") from exc
                self.telemetry["retry_count"] += 1
                sleep_sec = (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning("Attempt %d failed (%s). Retrying in %.2fs...", attempt, exc, sleep_sec)
                time.sleep(sleep_sec)
        return {}

    def extract_enforcement_recalls(self, incremental: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Paginates through OpenFDA enforcement endpoint and persists raw JSON."""
        checkpoint = self._load_checkpoint().get(self.config.domain, {})
        start_date = self.config.start_date
        if incremental and checkpoint.get("high_water_report_date"):
            start_date = checkpoint["high_water_report_date"]
            logger.info("Incremental mode active: extracting from high-water mark %s", start_date)

        all_records: List[Dict[str, Any]] = []
        raw_cache_path = RAW_DATA_DIR / f"{self.config.domain}_enforcement_raw.json"

        try:
            for page in range(self.config.max_pages):
                skip = page * self.config.page_limit
                url = (
                    f"{self.config.enforcement_url}"
                    f"?search={self.config.date_field}:[{start_date}+TO+{self.config.end_date}]"
                    f"&limit={self.config.page_limit}&skip={skip}"
                )
                payload = self._request_with_backoff(url)
                meta = payload.get("meta", {})
                if not self.telemetry["api_meta_last_updated"]:
                    self.telemetry["api_meta_last_updated"] = meta.get("last_updated")

                batch = payload.get("results", [])
                if not batch:
                    break

                all_records.extend(batch)
                total_available = meta.get("results", {}).get("total", 0)
                logger.info(
                    "Fetched page %d (%d records, %d/%d total)",
                    page + 1, len(batch), len(all_records), total_available
                )
                if skip + self.config.page_limit >= total_available:
                    break

            # Merge with existing raw cache if incremental
            if incremental and raw_cache_path.exists():
                existing = json.loads(raw_cache_path.read_text(encoding="utf-8")).get("results", [])
                all_records = existing + all_records

            # Persist raw extract for reproducibility and offline fallback
            serialized = json.dumps(
                {
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "source_endpoint": self.config.enforcement_url,
                    "telemetry": self.telemetry,
                    "results": all_records,
                },
                indent=2,
            )
            raw_cache_path.write_text(serialized, encoding="utf-8")

            # Update checkpoint state
            report_dates = [r.get("report_date", "") for r in all_records if r.get("report_date")]
            high_water = max(report_dates) if report_dates else self.config.end_date
            payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            self._save_checkpoint(self.config.domain, high_water, len(all_records), payload_hash)

        except Exception as exc:
            logger.error("Live API extraction failed: %s", exc)
            if raw_cache_path.exists():
                logger.info("Falling back to cached raw file: %s", raw_cache_path)
                self.telemetry["source_mode"] = "cached_raw_fallback"
                cached = json.loads(raw_cache_path.read_text(encoding="utf-8"))
                all_records = cached.get("results", [])
            else:
                raise

        return all_records, self.telemetry

    def extract_contextual_adverse_events(self) -> List[Dict[str, Any]]:
        """Extracts aggregated adverse event counts by manufacturer as contextual metadata."""
        cache_path = RAW_DATA_DIR / f"{self.config.domain}_adverse_events_counts.json"
        url = (
            f"{self.config.adverse_event_url}"
            f"?search=receivedate:[{self.config.start_date}+TO+{self.config.end_date}]"
            f"&count=patient.drug.openfda.manufacturer_name.exact&limit=1000"
        )
        try:
            payload = self._request_with_backoff(url)
            results = payload.get("results", [])
            cache_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
            return results
        except Exception as exc:
            logger.warning("Contextual adverse event fetch failed (%s). Checking cache.", exc)
            if cache_path.exists():
                return json.loads(cache_path.read_text(encoding="utf-8")).get("results", [])
            return []