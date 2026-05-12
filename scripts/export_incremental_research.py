"""
Export recently updated research tables into a portable delta package.

The package is designed for scheduled GitHub Actions runs. It contains CSV
files for machine import, JSON files for inspection, a manifest, and a short
Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.init_db import init_db
except ModuleNotFoundError:
    from init_db import init_db  # type: ignore


TABLES: dict[str, dict[str, str]] = {
    "prices_eod": {"watermark": "price_date", "label": "EOD prices"},
    "stock_news": {"watermark": "updated_at", "label": "Stock news"},
    "earnings_calendar": {"watermark": "updated_at", "label": "Earnings calendar"},
    "economic_calendar": {"watermark": "updated_at", "label": "Economic calendar"},
    "mergers_acquisitions": {"watermark": "updated_at", "label": "Mergers and acquisitions"},
    "fred_series": {"watermark": "updated_at", "label": "FRED macro series"},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def fetch_rows(
    conn: sqlite3.Connection,
    table: str,
    watermark_column: str,
    since_value: str,
) -> list[dict[str, Any]]:
    if watermark_column not in table_columns(conn, table):
        return []
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE {watermark_column} >= ?
        ORDER BY {watermark_column} ASC
        """,
        (since_value,),
    ).fetchall()
    return [dict(row) for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Daily Research Delta",
        "",
        f"- Package ID: `{manifest['package_id']}`",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Since: `{manifest['since']}`",
        "",
        "| Table | Rows | CSV | JSON |",
        "|---|---:|---|---|",
    ]
    for table in manifest["tables"]:
        lines.append(
            f"| {table['name']} | {table['row_count']} | "
            f"`{table['csv_file']}` | `{table['json_file']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_delta(output_dir: Path, lookback_days: int) -> Path:
    db_path = init_db()
    generated_at = utc_now()
    since = generated_at - timedelta(days=lookback_days)
    package_id = generated_at.strftime("research_delta_%Y%m%dT%H%M%SZ")
    package_dir = output_dir / package_id
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "package_id": package_id,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "since": since.isoformat().replace("+00:00", "Z"),
        "db_path": str(db_path),
        "tables": [],
    }

    since_sql = since.replace(tzinfo=None).isoformat(timespec="seconds")
    since_date = since.date().isoformat()

    with sqlite3.connect(str(db_path)) as conn:
        for table, config in TABLES.items():
            if not table_exists(conn, table):
                continue
            watermark = config["watermark"]
            since_value = since_date if watermark.endswith("_date") else since_sql
            rows = fetch_rows(conn, table, watermark, since_value)
            csv_file = f"{table}.csv"
            json_file = f"{table}.json"
            write_csv(package_dir / csv_file, rows)
            (package_dir / json_file).write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["tables"].append(
                {
                    "name": table,
                    "label": config["label"],
                    "watermark": config["watermark"],
                    "row_count": len(rows),
                    "csv_file": csv_file,
                    "json_file": json_file,
                }
            )

    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(package_dir / "summary.md", manifest)
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a daily research delta package.")
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "research_deltas"),
        help="Directory where the timestamped delta package will be written.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=3,
        help="Export rows updated within this many days.",
    )
    args = parser.parse_args()

    package_dir = export_delta(Path(args.output_dir), max(1, args.lookback_days))
    print(f"Exported research delta: {package_dir}")


if __name__ == "__main__":
    main()
