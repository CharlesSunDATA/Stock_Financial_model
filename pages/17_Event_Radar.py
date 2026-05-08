"""Event Radar — news, earnings surprise, and price-reaction signals."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly, table_exists


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
_GRID = dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

POSITIVE_TERMS = (
    "beat",
    "beats",
    "raise",
    "raises",
    "raised",
    "upgrade",
    "upgraded",
    "outperform",
    "strong",
    "record",
    "growth",
    "accelerate",
    "partnership",
    "contract",
    "deal",
    "approval",
    "launch",
    "buyback",
    "expands",
    "surge",
)
NEGATIVE_TERMS = (
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "downgraded",
    "underperform",
    "weak",
    "slowdown",
    "delay",
    "lawsuit",
    "probe",
    "investigation",
    "recall",
    "breach",
    "warning",
    "layoff",
    "bankruptcy",
    "falls",
)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt_pct(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):+.1f}%"
    except Exception:
        return ""


def _fmt_score(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.0f}"
    except Exception:
        return ""


def _payload_provider(payload_json: object, site: object) -> str:
    site_text = str(site or "").strip().lower()
    if site_text == "finnhub":
        return "Finnhub"
    if not payload_json:
        return "FMP"
    try:
        payload = json.loads(str(payload_json))
        provider = str(payload.get("provider") or "").strip().lower()
        if provider == "finnhub":
            return "Finnhub"
    except Exception:
        pass
    return "FMP"


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_news(db_path: str, lookback_days: int, source: str) -> pd.DataFrame:
    db = Path(db_path)
    if not db.exists():
        return pd.DataFrame()
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    with connect_readonly(db) as conn:
        if not table_exists(conn, "stock_news"):
            return pd.DataFrame()
        news = pd.read_sql_query(
            """
            SELECT ticker, published_date, publisher, title, site, text, url, payload_json
            FROM stock_news
            WHERE substr(published_date, 1, 10) >= ?
            ORDER BY published_date DESC
            """,
            conn,
            params=(cutoff,),
        )

        profiles = pd.DataFrame()
        if table_exists(conn, "company_profile"):
            profiles = pd.read_sql_query(
                """
                SELECT ticker, company_name, sector, industry
                FROM company_profile
                """,
                conn,
            )

    if news.empty:
        return news

    news["ticker"] = news["ticker"].astype(str).str.upper()
    news["published_at"] = pd.to_datetime(news["published_date"], errors="coerce", utc=True)
    news["source"] = [
        _payload_provider(payload, site)
        for payload, site in zip(news["payload_json"], news["site"], strict=False)
    ]
    if source != "All":
        news = news[news["source"] == source].copy()
    news["headline_text"] = (
        news["title"].fillna("").astype(str) + " " + news["text"].fillna("").astype(str)
    )
    news["positive_hits"] = news["headline_text"].apply(lambda x: _term_hits(x, POSITIVE_TERMS))
    news["negative_hits"] = news["headline_text"].apply(lambda x: _term_hits(x, NEGATIVE_TERMS))

    if not profiles.empty:
        profiles["ticker"] = profiles["ticker"].astype(str).str.upper()
        news = news.merge(profiles.drop_duplicates("ticker"), on="ticker", how="left")
    return news


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_earnings(db_path: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    db = Path(db_path)
    placeholders = ",".join("?" * len(tickers))
    with connect_readonly(db) as conn:
        if not table_exists(conn, "earnings_surprises"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            f"""
            SELECT ticker, surprise_date, actual_eps, estimated_eps, surprise_percent, payload_json
            FROM earnings_surprises
            WHERE ticker IN ({placeholders})
            ORDER BY surprise_date DESC
            """,
            conn,
            params=list(tickers),
        )
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["surprise_percent"] = _num(df["surprise_percent"])
    df["surprise_date"] = pd.to_datetime(df["surprise_date"], errors="coerce")
    return df.sort_values("surprise_date", ascending=False).drop_duplicates("ticker")


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_price_reactions(db_path: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    db = Path(db_path)
    placeholders = ",".join("?" * len(tickers))
    cutoff = (date.today() - timedelta(days=45)).isoformat()
    with connect_readonly(db) as conn:
        if not table_exists(conn, "prices_eod"):
            return pd.DataFrame()
        prices = pd.read_sql_query(
            f"""
            SELECT ticker, price_date, COALESCE(adj_close, close) AS price
            FROM prices_eod
            WHERE ticker IN ({placeholders})
              AND price_date >= ?
            ORDER BY ticker, price_date
            """,
            conn,
            params=[*tickers, cutoff],
        )
    if prices.empty:
        return prices

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["price"] = _num(prices["price"])
    rows: list[dict[str, object]] = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("price_date").dropna(subset=["price"])
        if len(grp) < 2:
            continue
        latest = float(grp.iloc[-1]["price"])
        row: dict[str, object] = {"ticker": ticker, "latest_price": latest, "price_date": grp.iloc[-1]["price_date"]}
        for label, periods in (("ret_5d", 5), ("ret_20d", 20)):
            if len(grp) > periods:
                base = float(grp.iloc[-periods - 1]["price"])
                row[label] = (latest / base - 1) * 100 if base else np.nan
            else:
                row[label] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_event_scores(news: pd.DataFrame, earnings: pd.DataFrame, prices: pd.DataFrame, min_articles: int) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame()

    agg = (
        news.groupby("ticker")
        .agg(
            article_count=("title", "size"),
            latest_news=("published_at", "max"),
            positive_hits=("positive_hits", "sum"),
            negative_hits=("negative_hits", "sum"),
            company_name=("company_name", "first"),
            sector=("sector", "first"),
            industry=("industry", "first"),
        )
        .reset_index()
    )
    agg = agg[agg["article_count"] >= min_articles].copy()
    if agg.empty:
        return agg

    agg["news_intensity_score"] = agg["article_count"].rank(pct=True) * 100
    net_hits = agg["positive_hits"] - agg["negative_hits"]
    agg["news_tone_score"] = (50 + net_hits / agg["article_count"].clip(lower=1) * 25).clip(0, 100)

    if not earnings.empty:
        agg = agg.merge(
            earnings[["ticker", "surprise_date", "actual_eps", "estimated_eps", "surprise_percent"]],
            on="ticker",
            how="left",
        )
    if "surprise_percent" not in agg:
        agg["surprise_percent"] = np.nan
    agg["earnings_score"] = (50 + agg["surprise_percent"].fillna(0).clip(-50, 50)).clip(0, 100)

    if not prices.empty:
        agg = agg.merge(prices, on="ticker", how="left")
    for col in ["ret_5d", "ret_20d"]:
        if col not in agg:
            agg[col] = np.nan
    agg["price_reaction_score"] = (50 + agg["ret_20d"].fillna(0).clip(-50, 50)).clip(0, 100)

    agg["event_score"] = (
        agg["news_intensity_score"] * 0.35
        + agg["news_tone_score"] * 0.25
        + agg["earnings_score"] * 0.25
        + agg["price_reaction_score"] * 0.15
    )
    agg["event_signal"] = np.select(
        [
            (agg["event_score"] >= 75) & (agg["news_tone_score"] >= 55),
            (agg["event_score"] >= 65) & (agg["surprise_percent"].fillna(0) > 0),
            (agg["news_intensity_score"] >= 80) & (agg["news_tone_score"] < 45),
            (agg["event_score"] <= 40),
        ],
        ["Positive Catalyst", "Earnings-Backed Watch", "Headline Risk", "Negative Event Risk"],
        default="Monitor",
    )
    agg["action_note"] = np.select(
        [
            agg["event_signal"].eq("Positive Catalyst"),
            agg["event_signal"].eq("Earnings-Backed Watch"),
            agg["event_signal"].eq("Headline Risk"),
            agg["event_signal"].eq("Negative Event Risk"),
        ],
        [
            "Prioritize due diligence; confirm valuation and position sizing.",
            "Check earnings quality and whether guidance supports the move.",
            "Read primary headlines before adding exposure.",
            "Avoid new exposure until the event risk is understood.",
        ],
        default="Track for follow-through or additional confirmation.",
    )
    return agg.sort_values("event_score", ascending=False).reset_index(drop=True)


def _score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.head(top_n).iloc[::-1].copy()
    colors = np.where(sub["news_tone_score"] >= 55, "#22c55e", np.where(sub["news_tone_score"] < 45, "#ef4444", "#60a5fa"))
    fig = go.Figure(
        go.Bar(
            x=sub["event_score"],
            y=sub["ticker"],
            orientation="h",
            marker_color=colors,
            text=sub["event_score"].map(lambda x: f"{x:.0f}"),
            textposition="outside",
            customdata=np.stack(
                [
                    sub["article_count"],
                    sub["news_tone_score"].map(lambda x: f"{x:.0f}"),
                    sub["event_signal"],
                ],
                axis=-1,
            ),
            hovertemplate="<b>%{y}</b><br>Event score: %{x:.1f}<br>Articles: %{customdata[0]}<br>Tone: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 34),
        margin=dict(t=20, b=30, l=90, r=80),
        **_DARK,
        xaxis=dict(range=[0, max(100, float(sub["event_score"].max()) * 1.15)], **_GRID),
        yaxis=dict(showgrid=False),
    )
    return fig


def _render_metric_row(scores: pd.DataFrame, news: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    top = scores.iloc[0] if not scores.empty else pd.Series(dtype=object)
    c1.metric("Tickers With Events", f"{scores['ticker'].nunique():,}" if not scores.empty else "0")
    c2.metric("Articles Analyzed", f"{len(news):,}")
    c3.metric("Top Event", str(top.get("ticker", "-")))
    c4.metric("Top Score", _fmt_score(top.get("event_score", np.nan)))
    c5.metric("Positive Catalysts", f"{int(scores['event_signal'].eq('Positive Catalyst').sum()) if not scores.empty else 0:,}")


def main() -> None:
    st.title("Event Radar")
    st.caption(
        "Ranks stocks by recent news intensity, keyword tone, earnings surprise confirmation, and price reaction."
    )

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        return

    with st.sidebar:
        st.header("Event Radar")
        lookback_days = st.slider("News lookback days", 3, 60, 14, 1)
        min_articles = st.slider("Minimum articles", 1, 20, 3, 1)
        source = st.selectbox("News source", ["All", "Finnhub", "FMP"], index=0)
        top_n = st.slider("Top names shown", 5, 30, 15, 1)

    news = _load_news(str(DB_PATH), lookback_days, source)
    if news.empty:
        st.warning("No recent news found for the selected filters.")
        return

    tickers = tuple(news["ticker"].dropna().astype(str).str.upper().unique().tolist())
    earnings = _load_earnings(str(DB_PATH), tickers)
    prices = _load_price_reactions(str(DB_PATH), tickers)
    scores = _build_event_scores(news, earnings, prices, min_articles)

    if scores.empty:
        st.warning("No tickers passed the minimum article filter.")
        return

    _render_metric_row(scores, news)

    st.subheader("Highest-Scoring Event Setups")
    st.plotly_chart(_score_bar(scores, top_n), width="stretch")

    st.subheader("Action Board")
    display_cols = [
        "ticker",
        "company_name",
        "sector",
        "event_signal",
        "event_score",
        "article_count",
        "news_tone_score",
        "surprise_percent",
        "ret_5d",
        "ret_20d",
        "action_note",
    ]
    table = scores[[c for c in display_cols if c in scores.columns]].copy()
    table = table.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "event_signal": "Signal",
            "event_score": "Event Score",
            "article_count": "Articles",
            "news_tone_score": "Tone Score",
            "surprise_percent": "EPS Surprise",
            "ret_5d": "5D Return",
            "ret_20d": "20D Return",
            "action_note": "Suggested Check",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Event Score": st.column_config.ProgressColumn("Event Score", min_value=0, max_value=100, format="%.0f"),
            "Tone Score": st.column_config.ProgressColumn("Tone Score", min_value=0, max_value=100, format="%.0f"),
            "EPS Surprise": st.column_config.NumberColumn("EPS Surprise", format="%+.1f%%"),
            "5D Return": st.column_config.NumberColumn("5D Return", format="%+.1f%%"),
            "20D Return": st.column_config.NumberColumn("20D Return", format="%+.1f%%"),
        },
    )

    selected = st.selectbox("Inspect ticker", scores["ticker"].tolist(), index=0)
    selected_news = news[news["ticker"] == selected].sort_values("published_at", ascending=False).head(20)
    selected_score = scores[scores["ticker"] == selected].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.subheader(f"{selected} Event Snapshot")
        st.metric("Event Score", _fmt_score(selected_score.get("event_score")))
        st.metric("News Tone", _fmt_score(selected_score.get("news_tone_score")))
        st.metric("Articles", f"{int(selected_score.get('article_count', 0)):,}")
        st.metric("EPS Surprise", _fmt_pct(selected_score.get("surprise_percent")))
        st.metric("20D Return", _fmt_pct(selected_score.get("ret_20d")))
        st.info(str(selected_score.get("action_note") or "Track for confirmation."))

    with right:
        st.subheader("Latest Headlines")
        for _, row in selected_news.iterrows():
            title = str(row.get("title") or "").strip()
            publisher = str(row.get("publisher") or row.get("source") or "").strip()
            published = row.get("published_at")
            when = published.strftime("%Y-%m-%d %H:%M") if pd.notna(published) else ""
            url = str(row.get("url") or "").strip()
            label = f"{title}"
            if url:
                st.markdown(f"**[{label}]({url})**")
            else:
                st.markdown(f"**{label}**")
            st.caption(f"{publisher} | {when}")
            summary = str(row.get("text") or "").strip()
            if summary:
                st.write(summary[:420] + ("..." if len(summary) > 420 else ""))

    with st.expander("Earnings Surprise Detail", expanded=False):
        if earnings.empty:
            st.write("No earnings surprise rows found for the current event universe.")
        else:
            earnings_view = earnings.copy()
            earnings_view["surprise_date"] = earnings_view["surprise_date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                earnings_view[["ticker", "surprise_date", "actual_eps", "estimated_eps", "surprise_percent"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "surprise_date": "Date",
                    "actual_eps": st.column_config.NumberColumn("Actual EPS", format="%.2f"),
                    "estimated_eps": st.column_config.NumberColumn("Estimated EPS", format="%.2f"),
                    "surprise_percent": st.column_config.NumberColumn("Surprise", format="%+.1f%%"),
                },
            )


if __name__ == "__main__":
    main()
