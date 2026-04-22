"""
Earnings Call NLP Analyzer — Streamlit page module.

Primary input is manual transcript upload (.txt).
Optional: OpenAI API for LLM summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


GROWTH_KEYWORDS = ["growth", "ai", "accelerate", "record", "demand", "opportunity"]
RISK_KEYWORDS = ["headwind", "inflation", "cautious", "delay", "supply chain", "inventory"]


@dataclass
class TranscriptParts:
    full_text: str
    prepared: str
    qa: str


def _safe_decode(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def split_prepared_vs_qa(text: str) -> TranscriptParts:
    """
    Heuristic split. Looks for a Q&A marker. If not found, put everything in prepared.
    """
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Common markers seen in transcripts
    qa_markers = [
        r"\bquestions\s+and\s+answers\b",
        r"\bq\s*&\s*a\b",
        r"\bq\s+and\s+a\b",
        r"\bquestion\-and\-answer\b",
        r"\boperator\b.*\bquestion\b",
    ]

    for pat in qa_markers:
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            cut = m.start()
            prepared = t[:cut].strip()
            qa = t[cut:].strip()
            return TranscriptParts(full_text=t, prepared=prepared, qa=qa)

    return TranscriptParts(full_text=t, prepared=t.strip(), qa="")


def count_keywords(text: str, keywords: list[str]) -> dict[str, int]:
    t = (text or "").lower()
    out: dict[str, int] = {}
    for kw in keywords:
        # allow multi-word tokens like "supply chain"
        pat = r"\b" + re.escape(kw.lower()) + r"\b"
        out[kw] = len(re.findall(pat, t))
    return out


def sentence_split(text: str) -> list[str]:
    """
    Lightweight sentence splitter (no external NLP deps).
    Splits on punctuation + newlines; drops very short fragments.
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # split on ., !, ?, or newline boundaries
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", t)
    sents = [p.strip() for p in parts if p and len(p.strip()) >= 8]
    return sents


def sentiment_series(sentences: list[str]) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(sentences, start=1):
        sc = analyzer.polarity_scores(s)
        rows.append({"idx": i, "sentence": s, "compound": float(sc["compound"])})
    return pd.DataFrame(rows)


def _avg_compound(df_sent: pd.DataFrame) -> float:
    if df_sent.empty:
        return float("nan")
    return float(df_sent["compound"].mean())


def llm_summary_openai(api_key: str, model: str, text: str) -> str:
    """
    Uses the OpenAI Python SDK if an API key is provided.
    """
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a senior Wall Street NLP analyst.\n"
        "Given the following earnings call transcript, return:\n"
        "1) Three business highlights (bullet list)\n"
        "2) Three potential risks (bullet list)\n"
        "Be concise and avoid hallucinating numbers. If info is missing, say so.\n\n"
        "Transcript:\n"
        f"{text}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return Markdown only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _strip_html(s: str) -> str:
    # minimal HTML → text (no extra deps)
    s = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", s)
    s = re.sub(r"(?is)<br\\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\\s*>", "\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"\\s+", " ", s)
    return s.strip()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def sec_cik_map() -> dict[str, int]:
    """
    Load SEC ticker→CIK mapping. Requires a proper User-Agent.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    # This endpoint is public, but SEC still expects a descriptive UA.
    r = requests.get(url, headers={"User-Agent": "StockFinancialModel (contact: example@example.com)"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: dict[str, int] = {}
    if isinstance(data, dict):
        for _k, row in data.items():
            try:
                t = str(row.get("ticker", "")).upper()
                cik = int(row.get("cik_str"))
                if t:
                    out[t] = cik
            except Exception:
                continue
    return out


@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_latest_transcript_sec_best_effort(symbol: str, user_agent: str) -> tuple[str, str]:
    """
    Best-effort: search recent 8-K filings via SEC submissions JSON, download the primary document,
    and return its text. Some companies include transcript-like content in 8-K exhibits; many do not.
    """
    sym = (symbol or "").strip().upper()
    ua = (user_agent or "").strip()
    if not sym:
        raise ValueError("Ticker is required.")
    if not ua or "@" not in ua:
        raise ValueError("SEC requires a descriptive User-Agent including contact email (e.g., you@domain.com).")

    cik = sec_cik_map().get(sym)
    if cik is None:
        raise ValueError(f"Ticker not found in SEC mapping: {sym}")

    cik10 = f"{cik:010d}"
    sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    sub = requests.get(sub_url, headers={"User-Agent": ua}, timeout=30)
    sub.raise_for_status()
    sj = sub.json()

    recent = (sj.get("filings", {}) or {}).get("recent", {}) or {}
    forms = recent.get("form", []) or []
    accs = recent.get("accessionNumber", []) or []
    prim = recent.get("primaryDocument", []) or []
    dates = recent.get("filingDate", []) or []

    # pick most recent 8-K
    pick = None
    for i in range(min(len(forms), len(accs), len(prim), len(dates))):
        if str(forms[i]).strip().upper() == "8-K":
            pick = (str(dates[i]), str(accs[i]), str(prim[i]))
            break
    if pick is None:
        raise ValueError("No recent 8-K found for this ticker.")

    filing_date, accession, primary_doc = pick
    acc_nodash = accession.replace("-", "")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"
    r = requests.get(doc_url, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()
    raw = r.text if "text" in content_type or "html" in content_type or "xml" in content_type else r.content.decode("utf-8", errors="ignore")
    txt = _strip_html(raw)

    label = f"{sym} 8-K {filing_date}"
    # Very rough check for transcript-like content; still return text either way
    if len(txt) < 2000:
        raise ValueError("Downloaded 8-K content too short (no transcript content found).")
    return label, txt


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_transcript_motley_fool_url(url: str, user_agent: str) -> tuple[str, str]:
    """
    Best-effort scraper for Motley Fool call-transcript pages. User provides the URL.
    """
    u = (url or "").strip()
    ua = (user_agent or "").strip()
    if not u or not u.startswith(("http://", "https://")):
        raise ValueError("Please provide a valid Motley Fool URL.")
    if not ua:
        raise ValueError("User-Agent is required (browser-like UA is fine).")

    r = requests.get(u, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Primary selector used by Fool transcripts
    node = soup.find("div", class_="article-content")
    if node is None:
        # common fallbacks
        node = soup.find("article") or soup.find("main") or soup.body
    if node is None:
        raise ValueError("Could not parse page content.")

    text = node.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 1500:
        raise ValueError("Parsed text too short (page may be paywalled or structure changed).")

    title = soup.title.get_text(strip=True) if soup.title else "Motley Fool transcript"
    label = re.sub(r"\s+", " ", title).strip()
    return label, text


def main() -> None:
    st.title("Earnings Call NLP Analyzer")
    st.caption(
        "Upload an earnings call transcript (.txt). We split Prepared vs Q&A, track keyword frequency, "
        "and compute sentence-level sentiment (VADER). Optional: OpenAI summary."
    )

    with st.sidebar:
        st.header("Input")
        source = st.radio(
            "Transcript source",
            ["Paste text", "Upload .txt", "Auto-fetch (SEC EDGAR - best effort)", "Auto-fetch (Motley Fool URL)"],
            index=1,
        )
        ticker_q = st.text_input("Label (e.g., NVDA 2026 Q1)", value="NVDA 2026 Q1")

        transcript_text = ""
        if source == "Paste text":
            transcript_text = st.text_area("Transcript text", value="", height=220, placeholder="Paste the transcript here…")
        elif source == "Upload .txt":
            uploaded = st.file_uploader("Upload transcript (.txt)", type=["txt"])
            if uploaded is not None:
                transcript_text = _safe_decode(uploaded)
        elif source == "Auto-fetch (SEC EDGAR - best effort)":
            st.caption("Free source via SEC filings. Not guaranteed to contain a transcript.")
            ua = st.text_input("SEC User-Agent (must include email)", value="your_email@example.com")
            sym = st.text_input("Ticker", value="NVDA").strip().upper()
            if st.button("Fetch latest (8-K)", type="primary", width="stretch"):
                with st.spinner("Fetching from SEC EDGAR…"):
                    label, txt = fetch_latest_transcript_sec_best_effort(sym, ua)
                    st.session_state["sec_label"] = label
                    st.session_state["sec_text"] = txt
            if "sec_text" in st.session_state:
                transcript_text = str(st.session_state.get("sec_text", ""))
                ticker_q = st.session_state.get("sec_label", ticker_q)
                st.success(f"Loaded: {ticker_q}")
                with st.expander("Preview fetched text", expanded=False):
                    st.text(transcript_text[:20_000])
        else:
            st.caption("Paste a Motley Fool transcript URL (HTML). Some pages may be restricted.")
            mf_url = st.text_input(
                "Motley Fool URL",
                value="https://www.fool.com/earnings/call-transcripts/",
                help="Open a specific transcript page and paste its URL here.",
            )
            ua = st.text_input("User-Agent", value="Mozilla/5.0")
            if st.button("Fetch from URL", type="primary", width="stretch"):
                with st.spinner("Fetching from Motley Fool…"):
                    label, txt = fetch_transcript_motley_fool_url(mf_url, ua)
                    st.session_state["mf_label"] = label
                    st.session_state["mf_text"] = txt
            if "mf_text" in st.session_state:
                transcript_text = str(st.session_state.get("mf_text", ""))
                ticker_q = st.session_state.get("mf_label", ticker_q)
                st.success(f"Loaded: {ticker_q}")
                with st.expander("Preview fetched text", expanded=False):
                    st.text(transcript_text[:20_000])

        st.divider()
        st.subheader("Keywords")
        st.caption("Defaults can be edited (comma-separated).")
        growth_in = st.text_area("Growth / optimistic keywords", value=", ".join(GROWTH_KEYWORDS), height=80)
        risk_in = st.text_area("Risk / cautious keywords", value=", ".join(RISK_KEYWORDS), height=80)

        st.divider()
        st.subheader("LLM summary (optional)")
        api_key = st.text_input("OpenAI API key", type="password")
        model = st.text_input("Model", value="gpt-4.1-mini")
        run_llm = st.checkbox("Generate AI summary", value=False, help="Requires OpenAI key")

    if not (transcript_text or "").strip():
        st.info("Provide a transcript via paste/upload, or auto-fetch one to begin.")
        return

    parts = split_prepared_vs_qa(transcript_text)

    growth_kw = [k.strip() for k in growth_in.split(",") if k.strip()]
    risk_kw = [k.strip() for k in risk_in.split(",") if k.strip()]

    kw_growth = count_keywords(parts.full_text, growth_kw)
    kw_risk = count_keywords(parts.full_text, risk_kw)
    growth_total = int(sum(kw_growth.values()))
    risk_total = int(sum(kw_risk.values()))

    # Sentiment
    s_full = sentence_split(parts.full_text)
    s_pre = sentence_split(parts.prepared)
    s_qa = sentence_split(parts.qa)

    df_full = sentiment_series(s_full)
    df_pre = sentiment_series(s_pre)
    df_qa = sentiment_series(s_qa) if parts.qa else pd.DataFrame(columns=["idx", "sentence", "compound"])

    avg_full = _avg_compound(df_full)
    avg_pre = _avg_compound(df_pre)
    avg_qa = _avg_compound(df_qa)

    st.subheader("1) Sentiment dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Overall sentiment", f"{avg_full:.3f}" if np.isfinite(avg_full) else "—")
    with c2:
        st.metric("Prepared sentiment", f"{avg_pre:.3f}" if np.isfinite(avg_pre) else "—")
    with c3:
        st.metric("Q&A sentiment", f"{avg_qa:.3f}" if np.isfinite(avg_qa) else "—")
    with c4:
        st.metric("Growth keywords", f"{growth_total:,}")
    with c5:
        st.metric("Risk keywords", f"{risk_total:,}")

    st.caption(f"Label: **{ticker_q}** • Sentences: {len(df_full):,} • Prepared: {len(df_pre):,} • Q&A: {len(df_qa):,}")

    st.subheader("2) Keyword frequency")
    kw_df = pd.DataFrame(
        [{"keyword": k, "count": v, "group": "Growth"} for k, v in kw_growth.items()]
        + [{"keyword": k, "count": v, "group": "Risk"} for k, v in kw_risk.items()]
    ).sort_values(["group", "count"], ascending=[True, False])

    fig_kw = px.bar(
        kw_df,
        x="count",
        y="keyword",
        color="group",
        orientation="h",
        title="Keyword Frequency",
        height=420,
    )
    fig_kw.update_layout(yaxis_title="", xaxis_title="Count", legend_title="")
    st.plotly_chart(fig_kw, use_container_width=True)

    st.subheader("3) Sentiment trend (by sentence)")
    if df_full.empty:
        st.info("No sentences detected to score sentiment.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_full["idx"], y=df_full["compound"], mode="lines", name="Compound"))
        fig.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Sentence-level sentiment (VADER compound)",
            xaxis_title="Sentence index",
            yaxis_title="Compound (-1 to 1)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Lowest sentiment sentences (debug)", expanded=False):
            show = df_full.sort_values("compound").head(10)[["compound", "sentence"]]
            st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("4) AI summary (optional)")
    if run_llm and api_key.strip():
        with st.spinner("Calling OpenAI…"):
            try:
                summary = llm_summary_openai(api_key.strip(), model.strip() or "gpt-4.1-mini", parts.full_text[:120_000])
                st.markdown(summary if summary else "_No content returned._")
            except Exception as e:
                st.error("OpenAI call failed.")
                st.exception(e)
    else:
        st.info("Enable **Generate AI summary** and provide an OpenAI API key to see the AI section.")

    with st.expander("Parsed sections (Prepared vs Q&A)", expanded=False):
        st.markdown("#### Prepared remarks")
        st.text(parts.prepared[:20_000] if parts.prepared else "")
        st.markdown("#### Q&A")
        st.text(parts.qa[:20_000] if parts.qa else "")


main()

