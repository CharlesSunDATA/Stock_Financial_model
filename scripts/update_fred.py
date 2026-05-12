#!/usr/bin/env python3
"""
Fetch FRED (Federal Reserve Economic Data) macro time-series into the local
SQLite database.

Tables populated:
  fred_series       — daily observations for each tracked series
  fred_series_meta  — title, units, frequency per series

API key:
  Set FRED_API_KEY as an env var or in .streamlit/secrets.toml.
  Free keys: https://fred.stlouisfed.org/docs/api/api_key.html

Run:
  python3 -m scripts.update_fred
  python3 -m scripts.update_fred --lookback 90   # fetch last 90 days
  python3 -m scripts.update_fred --series DGS10,DFF --lookback 365
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

try:
    from scripts.init_db import init_db
except ModuleNotFoundError:
    from init_db import init_db  # type: ignore


FRED_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_CALLS_PER_MINUTE = 120  # FRED free tier is generous; 120 is conservative

# Series to track, grouped by theme.
# Format: { series_id: human-readable label }
DEFAULT_SERIES: dict[str, str] = {
    # ── Interest rates ──────────────────────────────────────────────────────
    "DFF":     "Fed Funds Rate (effective)",
    "DGS1MO":  "1-Month Treasury Yield",
    "DGS3MO":  "3-Month Treasury Yield",
    "DGS6MO":  "6-Month Treasury Yield",
    "DGS1":    "1-Year Treasury Yield",
    "DGS2":    "2-Year Treasury Yield",
    "DGS5":    "5-Year Treasury Yield",
    "DGS10":   "10-Year Treasury Yield",
    "DGS30":   "30-Year Treasury Yield",
    # ── Yield curve spreads ─────────────────────────────────────────────────
    "T10Y2Y":  "10-Year minus 2-Year Treasury Spread",
    "T10Y3M":  "10-Year minus 3-Month Treasury Spread",
    # ── Inflation ───────────────────────────────────────────────────────────
    "CPIAUCSL": "CPI (All Urban, All Items)",
    "CPILFESL": "Core CPI (ex Food & Energy)",
    "PCEPI":    "PCE Price Index",
    "PCEPILFE": "Core PCE Price Index (Fed target)",
    # ── Labor market ────────────────────────────────────────────────────────
    "UNRATE":  "Unemployment Rate",
    "ICSA":    "Initial Jobless Claims (weekly)",
    # ── Growth ──────────────────────────────────────────────────────────────
    "GDPC1":   "Real GDP (quarterly)",
    # ── Credit & liquidity ──────────────────────────────────────────────────
    "BAMLH0A0HYM2": "HY Corporate Spread (OAS)",
    "M2SL":    "M2 Money Supply",
    "BOGMBASE": "Monetary Base",
}


def _utc_now_iso() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _db_path() -> Path:
    candidates = [
        Path.cwd() / "data" / "quant_data.db",
        Path(__file__).resolve().parents[1] / "data" / "quant_data.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def _read_secret(*keys: str) -> str:
    for key in keys:
        v = os.environ.get(key, "").strip()
        if v:
            return v
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(path.read_text())
            for key in keys:
                v = str(data.get(key, "") or "").strip()
                if v:
                    return v
        except Exception:
            for line in path.read_text().splitlines():
                for key in keys:
                    if line.strip().startswith(key):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"').strip("'")
    return ""


def get_fred_key() -> str:
    key = _read_secret("FRED_API_KEY", "FRED_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not found. Set it in .streamlit/secrets.toml or as an "
            "environment variable. Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


class FredClient:
    def __init__(self, api_key: str, *, calls_per_minute: int = DEFAULT_CALLS_PER_MINUTE) -> None:
        self._key = api_key
        self._interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        self._wait()
        params = {**params, "api_key": self._key, "file_type": "json"}
        resp = requests.get(f"{FRED_BASE_URL}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def series_info(self, series_id: str) -> dict[str, Any]:
        data = self._get("series", {"series_id": series_id})
        slist = data.get("seriess", [])
        return slist[0] if slist else {}

    def observations(
        self,
        series_id: str,
        *,
        observation_start: date,
        sort_order: str = "asc",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        data = self._get(
            "series/observations",
            {
                "series_id": series_id,
                "observation_start": observation_start.isoformat(),
                "sort_order": sort_order,
                "limit": limit,
            },
        )
        return data.get("observations", [])


def upsert_observations(
    conn: sqlite3.Connection,
    *,
    series_id: str,
    rows: list[dict[str, Any]],
) -> int:
    now = _utc_now_iso()
    inserted = 0
    for row in rows:
        raw_val = row.get("value", "")
        if raw_val in (".", "", None):
            continue
        try:
            value = float(raw_val)
        except (ValueError, TypeError):
            continue
        conn.execute(
            """
            INSERT INTO fred_series (series_id, series_date, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(series_id, series_date) DO UPDATE SET
              value      = excluded.value,
              updated_at = excluded.updated_at
            """,
            (series_id, row["date"], value, now),
        )
        inserted += 1
    return inserted


def upsert_meta(
    conn: sqlite3.Connection,
    *,
    series_id: str,
    info: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO fred_series_meta (series_id, title, units, frequency, last_fetched)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(series_id) DO UPDATE SET
          title        = excluded.title,
          units        = excluded.units,
          frequency    = excluded.frequency,
          last_fetched = excluded.last_fetched
        """,
        (
            series_id,
            info.get("title", ""),
            info.get("units_short", info.get("units", "")),
            info.get("frequency_short", info.get("frequency", "")),
            _utc_now_iso(),
        ),
    )


def run(
    *,
    series_ids: list[str] | None = None,
    lookback_days: int = 30,
    calls_per_minute: int = DEFAULT_CALLS_PER_MINUTE,
) -> None:
    api_key = get_fred_key()
    client = FredClient(api_key, calls_per_minute=calls_per_minute)

    target_series = series_ids or list(DEFAULT_SERIES.keys())
    observation_start = date.today() - timedelta(days=lookback_days)

    db = _db_path()
    init_db(db)

    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        total_rows = 0
        for i, sid in enumerate(target_series, 1):
            label = DEFAULT_SERIES.get(sid, sid)
            print(f"[{i}/{len(target_series)}] {sid}: {label}", flush=True)
            try:
                info = client.series_info(sid)
                upsert_meta(conn, series_id=sid, info=info)

                obs = client.observations(sid, observation_start=observation_start)
                n = upsert_observations(conn, series_id=sid, rows=obs)
                total_rows += n
                print(f"  → {n} observations upserted", flush=True)
                conn.commit()
            except Exception as exc:
                print(f"  ✗ {sid} failed: {exc}", flush=True)

    print(f"FRED update complete — {total_rows} total rows upserted.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch FRED macro series into SQLite.")
    ap.add_argument(
        "--series",
        type=lambda s: [x.strip().upper() for x in s.split(",")],
        default=None,
        help="Comma-separated FRED series IDs (default: all DEFAULT_SERIES).",
    )
    ap.add_argument(
        "--lookback",
        type=int,
        default=30,
        metavar="DAYS",
        help="Fetch observations going back this many days (default: 30).",
    )
    ap.add_argument(
        "--calls-per-minute",
        type=int,
        default=DEFAULT_CALLS_PER_MINUTE,
        help=f"Rate limit (default: {DEFAULT_CALLS_PER_MINUTE}).",
    )
    args = ap.parse_args()
    run(
        series_ids=args.series,
        lookback_days=args.lookback,
        calls_per_minute=args.calls_per_minute,
    )


if __name__ == "__main__":
    main()
