"""AI infrastructure research database."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.research_database import append_update, load_sources, load_stocks, load_updates


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _research_root() -> Path:
    sibling = _project_root().parent / "Codex" / "research"
    if sibling.exists():
        return sibling
    local = _project_root() / "research"
    return local


def _chips(items: list[str]) -> None:
    if not items:
        st.caption("No items yet.")
        return
    cols = st.columns(min(3, max(1, len(items))))
    for idx, item in enumerate(items):
        cols[idx % len(cols)].markdown(f"- {item}")


def _updates_for(ticker: str, updates: list[dict]) -> list[dict]:
    return sorted(
        [u for u in updates if str(u.get("ticker", "")).upper() == ticker],
        key=lambda row: str(row.get("date", "")),
        reverse=True,
    )


def _sources_for(ticker: str, sources: list[dict]) -> list[dict]:
    return sorted(
        [s for s in sources if str(s.get("ticker", "")).upper() == ticker],
        key=lambda row: str(row.get("date", "")),
        reverse=True,
    )


def _daily_brief_files() -> list[Path]:
    daily_dir = _research_root() / "output" / "daily"
    if not daily_dir.exists():
        return []
    return sorted(daily_dir.glob("AI_Data_Center_Daily_*.md"), reverse=True)


def _latest_daily_brief() -> tuple[Path | None, str]:
    files = _daily_brief_files()
    if not files:
        return None, ""
    path = files[0]
    return path, path.read_text(encoding="utf-8")


def _ticker_daily_mentions(ticker: str, brief_text: str, *, limit: int = 12) -> list[str]:
    if not brief_text:
        return []
    symbol = ticker.upper()
    lines = []
    for raw_line in brief_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if symbol in line.upper():
            lines.append(line)
    return lines[:limit]


def _format_bullets(items: list[str], *, fallback: str = "No current items.") -> str:
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in clean)


def _build_single_stock_report(ticker: str, stock: dict, updates: list[dict], sources: list[dict]) -> str:
    ticker_updates = _updates_for(ticker, updates)
    ticker_sources = _sources_for(ticker, sources)
    daily_path, daily_text = _latest_daily_brief()
    daily_mentions = _ticker_daily_mentions(ticker, daily_text)
    latest_updates = ticker_updates[:8]
    source_lines = [
        f"- {src.get('date', '')}: {src.get('source_name', '')} ({src.get('notes', 'Stored in sources database.')})"
        for src in ticker_sources[:6]
    ]
    update_lines = [
        f"- {row.get('display_text', '')}"
        for row in latest_updates
        if str(row.get("display_text", "")).strip()
    ]

    return "\n".join(
        [
            f"# {ticker} Current Research Report",
            "",
            f"**Company:** {stock.get('company', ticker)}",
            f"**Last Updated:** {stock.get('last_updated', '')}",
            f"**Supply Chain Category:** {', '.join(stock.get('category', []))}",
            "",
            "## Model Note",
            "This report is generated from the local JSON research database and GitHub Actions research deltas. It is a research tracking document, not investment advice.",
            "",
            "## Current Status",
            f"**Latest Change:** {stock.get('latest_change', 'No update yet.')}",
            "",
            f"**Positioning:** {stock.get('positioning', '')}",
            "",
            f"**Investment Thesis:** {stock.get('investment_thesis', '')}",
            "",
            "## Why It Benefits",
            stock.get("benefit_reason", "No current benefit reason."),
            "",
            "## What Can Pressure The Stock",
            stock.get("pressure_reason", "No current pressure reason."),
            "",
            "## Key Catalysts",
            _format_bullets(stock.get("catalysts", [])),
            "",
            "## Key Risks",
            _format_bullets(stock.get("risks", [])),
            "",
            "## Must-Track Items",
            _format_bullets(stock.get("tracking_items", [])),
            "",
            "## Earnings Highlights",
            _format_bullets(stock.get("earnings_notes", []), fallback="No earnings highlights yet."),
            "",
            "## News Notes",
            _format_bullets(stock.get("news_notes", []), fallback="No news notes yet."),
            "",
            "## Latest Daily Brief Mentions",
            "\n".join(daily_mentions) if daily_mentions else "- No ticker-specific mention in the latest daily brief.",
            f"\nLatest daily brief file: {daily_path.name if daily_path else 'No daily brief found.'}",
            "",
            "## Latest Update Log",
            "\n".join(update_lines) if update_lines else "- No updates yet.",
            "",
            "## Source Notes",
            "\n".join(source_lines) if source_lines else "- No sources yet.",
            "",
        ]
    )


def _render_report_view(ticker: str, stock: dict, updates: list[dict], sources: list[dict]) -> None:
    report = _build_single_stock_report(ticker, stock, updates, sources)
    st.markdown(report)
    st.download_button(
        "Download Markdown Report",
        data=report,
        file_name=f"{ticker}_current_research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_daily_briefs(ticker: str) -> None:
    st.subheader("Daily Briefs")
    files = _daily_brief_files()
    if not files:
        st.info("No daily brief files found. Run the Daily AI Data Center Brief GitHub Action or sync Google Drive.")
        return

    labels = [path.name for path in files]
    selected_name = st.selectbox("Daily Brief", labels, index=0)
    selected_path = files[labels.index(selected_name)]
    text = selected_path.read_text(encoding="utf-8")

    mentions = _ticker_daily_mentions(ticker, text, limit=20)
    with st.expander(f"{ticker} mentions in this brief", expanded=True):
        if mentions:
            for line in mentions:
                st.markdown(line)
        else:
            st.caption("No ticker-specific mentions in this brief.")

    st.markdown(text)


def _render_ticker_detail(ticker: str, stock: dict, updates: list[dict], sources: list[dict]) -> None:
    st.title(f"{ticker} Research")
    st.caption(stock.get("company", ticker))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Updated", stock.get("last_updated", ""))
    c2.metric("Categories", len(stock.get("category", [])))
    c3.metric("Updates", len(_updates_for(ticker, updates)))
    c4.metric("Sources", len(_sources_for(ticker, sources)))

    st.divider()
    st.subheader("Single Stock Current Report")
    _render_report_view(ticker, stock, updates, sources)
    st.divider()

    st.subheader("Dashboard View")
    h1, h2, h3 = st.columns(3)
    h1.info(f"Latest Change\n\n{stock.get('latest_change', 'No update yet.')}")
    h2.success(f"Main Catalyst\n\n{stock.get('weekly_catalyst', 'No catalyst yet.')}")
    h3.warning(f"Main Risk\n\n{stock.get('weekly_risk', 'No risk yet.')}")

    h4, h5, h6 = st.columns(3)
    h4.markdown(f"**Benefit Reason**\n\n{stock.get('benefit_reason', '')}")
    h5.markdown(f"**Pressure Reason**\n\n{stock.get('pressure_reason', '')}")
    h6.markdown(f"**Next Watch**\n\n{stock.get('next_watch', '')}")

    st.subheader("Investment Profile")
    st.markdown(f"**Positioning:** {stock.get('positioning', '')}")
    st.markdown(f"**Investment Thesis:** {stock.get('investment_thesis', '')}")
    st.markdown("**Supply Chain Categories:** " + ", ".join(stock.get("category", [])))

    tab_catalysts, tab_risks, tab_tracking, tab_notes, tab_daily, tab_updates, tab_sources = st.tabs(
        ["Catalysts", "Risks", "Tracking Items", "Earnings & News", "Daily Briefs", "Update Log", "Sources"]
    )

    with tab_catalysts:
        _chips(stock.get("catalysts", []))
    with tab_risks:
        _chips(stock.get("risks", []))
    with tab_tracking:
        _chips(stock.get("tracking_items", []))
    with tab_notes:
        st.markdown("**Earnings Notes**")
        _chips(stock.get("earnings_notes", []))
        st.markdown("**News Notes**")
        _chips(stock.get("news_notes", []))
    with tab_daily:
        _render_daily_briefs(ticker)
    with tab_updates:
        ticker_updates = _updates_for(ticker, updates)
        if ticker_updates:
            st.dataframe(
                pd.DataFrame(ticker_updates)[["display_date", "type", "text"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No updates yet.")
    with tab_sources:
        ticker_sources = _sources_for(ticker, sources)
        if ticker_sources:
            st.dataframe(
                pd.DataFrame(ticker_sources)[["date", "source_name", "notes"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption("Source URLs are stored in data/database/sources.json and are not shown in the main report body.")
        else:
            st.info("No sources yet.")


def _render_universe(stocks: dict[str, dict]) -> None:
    rows = []
    for ticker, stock in stocks.items():
        for category in stock.get("category", []):
            rows.append(
                {
                    "Category": category,
                    "Ticker": ticker,
                    "Company": stock.get("company", ticker),
                    "Positioning": stock.get("positioning", ""),
                }
            )
    df = pd.DataFrame(rows)
    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist()) if not df.empty else ["All"]
    selected = st.selectbox("Category Filter", categories)
    view = df if selected == "All" else df[df["Category"] == selected]
    st.dataframe(view.sort_values(["Category", "Ticker"]), hide_index=True, use_container_width=True)


def _render_add_update(stocks: dict[str, dict]) -> None:
    st.subheader("Add Update")
    tickers = sorted(stocks.keys())
    with st.form("research_update_form", clear_on_submit=True):
        ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
        update_date = st.date_input("Date", value=date.today())
        update_type = st.selectbox(
            "Update Type",
            ["Catalyst", "Risk", "Order", "Backlog", "Earnings", "CapEx", "Margin", "Valuation", "Source", "Other"],
        )
        text = st.text_area("Update Text", height=120)
        source_name = st.text_input("Source Name")
        source_url = st.text_input("Source URL")
        update_summary = st.checkbox("Update latest change summary")
        submitted = st.form_submit_button("Save Update", type="primary")

    if submitted:
        if not text.strip():
            st.error("Update text is required.")
            return
        entry = append_update(
            ticker=ticker,
            update_date=update_date,
            update_type=update_type,
            text=text,
            source_name=source_name,
            source_url=source_url,
            update_summary=update_summary,
        )
        st.success("Update saved.")
        st.code(entry["display_text"], language="text")
        st.cache_data.clear()


def main() -> None:
    st.title("AI Infrastructure Research Database")
    st.caption("JSON-backed local research notes for AI data center supply chain stocks.")

    stocks = load_stocks()
    updates = load_updates()
    sources = load_sources()
    if not stocks:
        st.error("No research database found. Run `python3 scripts/seed_research_database.py --overwrite` first.")
        return

    tab_detail, tab_universe, tab_update = st.tabs(["Ticker Research", "Supply Chain Universe", "Add Update"])

    with tab_detail:
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()
        if ticker not in stocks:
            st.warning("Ticker not found in the research database.")
        else:
            _render_ticker_detail(ticker, stocks[ticker], updates, sources)

    with tab_universe:
        _render_universe(stocks)

    with tab_update:
        _render_add_update(stocks)


main()
