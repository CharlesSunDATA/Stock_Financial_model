"""JSON-backed research database helpers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def database_dir() -> Path:
    return project_root() / "data" / "database"


def display_date(iso_date: str) -> str:
    return date.fromisoformat(iso_date).strftime("%d.%b.%Y")


def load_json(filename: str, default: Any) -> Any:
    path = database_dir() / filename
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(filename: str, payload: Any) -> None:
    path = database_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_stocks() -> dict[str, dict[str, Any]]:
    return load_json("stocks.json", {})


def load_updates() -> list[dict[str, Any]]:
    return load_json("updates.json", [])


def load_sources() -> list[dict[str, Any]]:
    return load_json("sources.json", [])


def append_update(
    *,
    ticker: str,
    update_date: date,
    update_type: str,
    text: str,
    source_name: str = "",
    source_url: str = "",
    update_summary: bool = False,
) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    iso_date = update_date.isoformat()
    shown_date = display_date(iso_date)

    sources = load_sources()
    source_id = ""
    if source_name.strip() or source_url.strip():
        source_id = f"src_{iso_date.replace('-', '')}_{len(sources) + 1:03d}"
        sources.append(
            {
                "source_id": source_id,
                "ticker": symbol,
                "date": iso_date,
                "source_name": source_name.strip(),
                "url": source_url.strip(),
                "notes": "Added from Streamlit Research Database form.",
            }
        )
        save_json("sources.json", sources)

    entry = {
        "ticker": symbol,
        "date": iso_date,
        "display_date": shown_date,
        "type": update_type,
        "text": text.strip(),
        "display_text": f"{shown_date} {update_type}: {text.strip()}",
        "source_id": source_id,
    }
    updates = load_updates()
    updates.append(entry)
    save_json("updates.json", updates)

    if update_summary:
        stocks = load_stocks()
        if symbol in stocks:
            stocks[symbol]["latest_change"] = text.strip()
            stocks[symbol]["last_updated"] = iso_date
            save_json("stocks.json", stocks)

    return entry
