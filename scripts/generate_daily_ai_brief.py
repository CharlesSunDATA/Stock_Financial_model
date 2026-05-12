"""
Generate a Traditional Chinese daily brief for AI data center supply chain research.

Outputs:
  research/output/daily/AI_Data_Center_Daily_YYYY-MM-DD.md
  research/data/daily_sources/daily_sources_YYYY-MM-DD.md

The report body intentionally excludes external URLs. Source URLs are written
only to the daily_sources file for internal traceability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


TIER_1_TICKERS = [
    "NVDA", "AMD", "AVGO", "MRVL", "MU", "LITE", "COHR", "APH", "CRDO", "ALAB",
    "DELL", "SMCI", "ANET", "VRT", "ETN", "PWR", "MSFT", "AMZN", "GOOGL", "META", "ORCL",
]

CATEGORY_QUERIES = [
    "GPU CPU ASIC AI data center supply chain NVIDIA AMD Broadcom Marvell",
    "memory HBM storage AI data center Micron SK Hynix Samsung Western Digital Seagate",
    "optical communications photonics AI data center Lumentum Coherent Fabrinet Corning",
    "connectors interconnect retimers AI server Credo Astera Labs Amphenol TE Connectivity",
    "AI server ODM EMS Dell Supermicro Celestica Jabil Flex backlog margin",
    "networking switching AI data center Arista Cisco Broadcom Marvell Ciena",
    "power cooling data center equipment Vertiv Eaton Quanta Services EMCOR GE Vernova",
    "semiconductor equipment advanced packaging ASML Applied Materials Lam Research KLA TSMC",
    "cloud hyperscaler AI capex Microsoft Amazon Google Meta Oracle",
]


@dataclass
class SourceRecord:
    source: str
    title: str
    date: str
    url: str
    note: str
    ticker: str = ""
    category: str = ""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_research_root() -> Path:
    env_path = os.getenv("RESEARCH_LIBRARY_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()

    sibling = project_root().parent / "Codex" / "research"
    if sibling.exists():
        return sibling
    return project_root() / "research"


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw) if raw else utc_today()


def fmp_get(path: str, params: dict[str, Any], api_key: str) -> Any:
    url = f"https://financialmodelingprep.com/stable/{path.lstrip('/')}"
    response = requests.get(url, params={**params, "apikey": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_fmp_news(tickers: list[str], api_key: str, *, limit: int) -> tuple[list[SourceRecord], list[str]]:
    records: list[SourceRecord] = []
    failures: list[str] = []
    if not api_key:
        return records, ["FMP_API_KEY missing; skipped FMP stock news."]

    for ticker in tickers:
        try:
            rows = fmp_get("news/stock", {"symbols": ticker, "limit": limit, "page": 0}, api_key)
            if not isinstance(rows, list):
                failures.append(f"FMP stock news returned non-list payload for {ticker}.")
                continue
            for row in rows[:limit]:
                if not isinstance(row, dict):
                    continue
                records.append(
                    SourceRecord(
                        source=str(row.get("site") or row.get("publisher") or "FMP Stock News"),
                        title=str(row.get("title") or "").strip(),
                        date=str(row.get("publishedDate") or "")[:10],
                        url=str(row.get("url") or "").strip(),
                        note="FMP stock news retrieval.",
                        ticker=ticker,
                    )
                )
        except Exception as exc:
            failures.append(f"FMP stock news failed for {ticker}: {type(exc).__name__}: {exc}")
    return records, failures


def fetch_fmp_events(api_key: str, report_date: date) -> tuple[list[SourceRecord], list[str]]:
    records: list[SourceRecord] = []
    failures: list[str] = []
    if not api_key:
        return records, ["FMP_API_KEY missing; skipped FMP event calendars."]

    start = (report_date - timedelta(days=2)).isoformat()
    end = (report_date + timedelta(days=14)).isoformat()
    for endpoint, label in [
        ("earnings-calendar", "FMP earnings calendar"),
        ("economic-calendar", "FMP economic calendar"),
    ]:
        try:
            rows = fmp_get(endpoint, {"from": start, "to": end}, api_key)
            if not isinstance(rows, list):
                failures.append(f"{label} returned non-list payload.")
                continue
            for row in rows[:80]:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").upper()
                if symbol and symbol not in TIER_1_TICKERS:
                    continue
                title = row.get("event") or row.get("symbol") or endpoint
                records.append(
                    SourceRecord(
                        source=label,
                        title=str(title),
                        date=str(row.get("date") or "")[:10],
                        url="",
                        note=json.dumps(row, ensure_ascii=False, separators=(",", ":"))[:800],
                        ticker=symbol,
                    )
                )
        except Exception as exc:
            failures.append(f"{label} failed: {type(exc).__name__}: {exc}")
    return records, failures


def tavily_search(query: str, api_key: str, *, max_results: int) -> tuple[list[SourceRecord], str]:
    if not api_key:
        return [], "TAVILY_API_KEY missing; skipped Tavily search."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        records = []
        for row in payload.get("results", [])[:max_results]:
            records.append(
                SourceRecord(
                    source=str(row.get("source") or row.get("url") or "Tavily Search"),
                    title=str(row.get("title") or "").strip(),
                    date=str(row.get("published_date") or row.get("date") or ""),
                    url=str(row.get("url") or "").strip(),
                    note=str(row.get("content") or "")[:900],
                    category=query,
                )
            )
        return records, ""
    except Exception as exc:
        return [], f"Tavily query failed: {query} | {type(exc).__name__}: {exc}"


def fetch_tavily_records(api_key: str) -> tuple[list[SourceRecord], list[str]]:
    records: list[SourceRecord] = []
    failures: list[str] = []

    ticker_query = (
        "AI data center supply chain latest news "
        + " ".join(TIER_1_TICKERS)
        + " earnings orders capex backlog margins export controls"
    )
    queries = [ticker_query] + CATEGORY_QUERIES
    for query in queries:
        rows, failure = tavily_search(query, api_key, max_results=5)
        records.extend(rows)
        if failure:
            failures.append(failure)
    return records, failures


def source_date(row: SourceRecord, fallback: date) -> str:
    return row.date[:10] if row.date else fallback.isoformat()


def source_name(row: SourceRecord) -> str:
    value = row.source.strip()
    if value.startswith("http"):
        value = value.replace("https://", "").replace("http://", "").split("/")[0]
    return value or "Unknown source"


def unique_records(records: list[SourceRecord]) -> list[SourceRecord]:
    seen: set[tuple[str, str]] = set()
    out: list[SourceRecord] = []
    for row in records:
        key = (row.url, row.title)
        if key in seen or not row.title:
            continue
        seen.add(key)
        out.append(row)
    return out


def records_for_ticker(records: list[SourceRecord], ticker: str) -> list[SourceRecord]:
    ticker_upper = ticker.upper()
    ticker_pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(ticker_upper)}(?![A-Z0-9])")
    return [
        row for row in records
        if row.ticker.upper() == ticker_upper
        or ticker_pattern.search(row.title.upper())
    ][:4]


def records_for_keywords(records: list[SourceRecord], keywords: list[str], limit: int = 8) -> list[SourceRecord]:
    out = []
    for row in records:
        haystack = f"{row.title} {row.note} {row.category}".lower()
        if any(keyword.lower() in haystack for keyword in keywords):
            out.append(row)
    return out[:limit]


def bullet_from_record(row: SourceRecord, report_date: date) -> str:
    return f"- {row.title}（{source_name(row)}，{source_date(row, report_date)}）"


def build_report(records: list[SourceRecord], failures: list[str], report_date: date) -> str:
    records = unique_records(records)
    catalyst_records = records_for_keywords(
        records,
        ["order", "backlog", "capex", "partnership", "raise", "growth", "demand", "AI infrastructure", "data center"],
        limit=8,
    )
    risk_records = records_for_keywords(
        records,
        ["risk", "export", "control", "tariff", "margin", "valuation", "delay", "shortage", "restriction"],
        limit=8,
    )
    chain_records = records_for_keywords(
        records,
        ["HBM", "optical", "photonics", "cooling", "power", "server", "networking", "packaging", "interconnect"],
        limit=10,
    )

    lines = [
        f"# AI Data Center Daily Brief - {report_date.isoformat()}",
        "",
        "用途：每日 AI data center 供應鏈研究快報，不構成投資建議。",
        "輸出規則：正文不放外部連結；來源 URL 保存在 daily_sources 檔案。",
        "",
        "## 今日總結",
        "",
        f"今日自動檢查 Tier 1 個股 {len(TIER_1_TICKERS)} 檔，並掃描 GPU/CPU/ASIC、memory/HBM/storage、optical/photonics、interconnect、AI server、networking、power/cooling、semicap/packaging 與 cloud demand 等類別。",
        f"本次可用來源共 {len(records)} 筆；若部分 Tavily 或 FMP 查詢失敗，已記錄於內部來源檔，不中斷報告產生。",
        "今日主線請優先看三件事：hyperscaler CapEx 是否延續、供應鏈瓶頸是否從 GPU 擴散到 memory/optics/power、以及 server/networking/power 類股的收入品質與 margin 是否同步改善。",
        "",
        "## 今日主要催化劑",
        "",
    ]
    lines.extend([bullet_from_record(row, report_date) for row in catalyst_records] or ["- 今日未抓到明確新增催化劑；維持追蹤 AI infrastructure 訂單、CapEx 與 backlog。"])

    lines.extend(["", "## 今日主要風險", ""])
    lines.extend([bullet_from_record(row, report_date) for row in risk_records] or ["- 今日未抓到明確新增風險；仍需追蹤出口管制、估值、margin、供應瓶頸與 CapEx 節奏。"])

    lines.extend(["", "## 個股重點", ""])
    for ticker in TIER_1_TICKERS:
        ticker_rows = records_for_ticker(records, ticker)
        if not ticker_rows:
            continue
        lines.append(f"### {ticker}")
        lines.extend([bullet_from_record(row, report_date) for row in ticker_rows])
        lines.append("")
    if lines[-1] == "## 個股重點":
        lines.append("- 今日未抓到 Tier 1 個股的可用新增資料。")

    lines.extend(["## 產業鏈變化", ""])
    lines.extend([bullet_from_record(row, report_date) for row in chain_records] or ["- 今日未抓到明確產業鏈變化；繼續追蹤 memory、optics、power/cooling、server 與 networking bottlenecks。"])

    lines.extend(
        [
            "",
            "## 明日/下次追蹤",
            "",
            "- Hyperscaler 是否持續上修或維持 AI CapEx。",
            "- NVDA/AMD/AVGO/MRVL 的 data center 與 networking commentary。",
            "- MU/HBM、LITE/COHR/optics、VRT/ETN/PWR/power cooling 的訂單與 margin 訊號。",
            "- AI server/ODM/EMS 的 backlog 轉 revenue、gross margin 與 cash conversion。",
            "- 出口管制、關稅、電力接入與 data center 交付延遲風險。",
            "",
            "## 來源摘要",
            "",
        ]
    )
    source_summary = []
    for row in records[:18]:
        source_summary.append(f"- {source_name(row)}，{source_date(row, report_date)}：{row.title}")
    lines.extend(source_summary or ["- 無可用來源。"])
    if failures:
        lines.extend(["", "### Failed Attempts", ""])
        lines.extend([f"- {failure}" for failure in failures[:10]])
    lines.append("")
    return "\n".join(lines)


def build_sources(records: list[SourceRecord], failures: list[str], report_date: date) -> str:
    lines = [
        f"# Daily Sources - {report_date.isoformat()}",
        "",
        "用途：每日快報內部來源紀錄。報告正文不放外部連結，本檔保留 URL 與 retrieval notes 供追溯。",
        "",
        "## Source URLs",
        "",
    ]
    for row in unique_records(records):
        lines.extend(
            [
                f"### {row.title}",
                f"- Source: {source_name(row)}",
                f"- Date: {source_date(row, report_date)}",
                f"- Ticker: {row.ticker or 'N/A'}",
                f"- Category/Query: {row.category or 'N/A'}",
                f"- URL: {row.url or 'N/A'}",
                f"- Notes: {row.note or 'N/A'}",
                "",
            ]
        )
    lines.extend(["## Retrieval Failures", ""])
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(research_root: Path, report_date: date, report: str, sources: str) -> tuple[Path, Path]:
    daily_dir = research_root / "output" / "daily"
    sources_dir = research_root / "data" / "daily_sources"
    daily_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    report_path = daily_dir / f"AI_Data_Center_Daily_{report_date.isoformat()}.md"
    sources_path = sources_dir / f"daily_sources_{report_date.isoformat()}.md"
    report_path.write_text(report, encoding="utf-8")
    sources_path.write_text(sources, encoding="utf-8")
    return report_path, sources_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily AI data center supply chain brief.")
    parser.add_argument("--date", default="")
    parser.add_argument("--research-root", default=str(default_research_root()))
    parser.add_argument("--fmp-limit", type=int, default=3)
    args = parser.parse_args()

    report_date = parse_date(args.date)
    fmp_key = os.getenv("FMP_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    records: list[SourceRecord] = []
    failures: list[str] = []

    fmp_news, fmp_failures = fetch_fmp_news(TIER_1_TICKERS, fmp_key, limit=max(1, args.fmp_limit))
    records.extend(fmp_news)
    failures.extend(fmp_failures)

    fmp_events, fmp_event_failures = fetch_fmp_events(fmp_key, report_date)
    records.extend(fmp_events)
    failures.extend(fmp_event_failures)

    tavily_records, tavily_failures = fetch_tavily_records(tavily_key)
    records.extend(tavily_records)
    failures.extend(tavily_failures)

    report = build_report(records, failures, report_date)
    sources = build_sources(records, failures, report_date)
    report_path, sources_path = write_outputs(Path(args.research_root), report_date, report, sources)

    print(f"Daily brief written: {report_path}")
    print(f"Daily sources written: {sources_path}")
    print(f"Sources collected: {len(unique_records(records))}")
    if failures:
        print(f"Failures recorded: {len(failures)}")


if __name__ == "__main__":
    main()
