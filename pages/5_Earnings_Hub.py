"""
Earnings Hub — combines:
  1. Earnings Call Analyzer (NLP: 8-K / 10-K / 10-Q sentiment)
  2. Earnings Calendar (upcoming report schedule + estimates)
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.local_data import connect_readonly

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


# ═════════════════════════════════════════════════════════════════════════════
# EARNINGS CALL ANALYZER — NLP logic
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _ensure_nltk_data() -> None:
    packages = ["punkt", "punkt_tab", "wordnet", "omw-1.4",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


_ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()

GROWTH_KEYWORDS = [
    "accelerate", "achieve", "advancement", "agile", "amplify",
    "awesome", "beat", "booming", "bolster", "breakthrough",
    "capture", "catalyst", "climb", "confident", "conviction",
    "demand", "disrupt", "dividend", "dominate", "double",
    "elevate", "encourage", "enhance", "escalate", "exceed",
    "exceptional", "execution", "expand", "fantastic", "favorable",
    "flexible", "flourish", "fortify", "gain", "growth",
    "highlight", "impress", "incredible", "influx", "innovate",
    "integration", "jump", "leverage", "lucrative", "maximize",
    "milestone", "momentum", "navigate", "opportunity", "optimize",
    "optimistic", "outmaneuver", "outpace", "outperform", "outstanding",
    "peak", "penetrate", "phenomenal", "pioneer", "pipeline",
    "premium", "productivity", "profitability", "progress", "promising",
    "propel", "proud", "reaffirm", "record", "recovery",
    "reinforce", "resilient", "revitalize", "revolutionize", "robust",
    "scalable", "scale", "soar", "solid", "spectacular",
    "stellar", "streamline", "strengthen", "structural", "superb",
    "surge", "sustainable", "synergy", "tailwind", "thrive",
    "traction", "transform", "tremendous", "triple", "unparalleled",
    "upgrade", "upside", "visionary", "visibility", "yield",
]
RISK_KEYWORDS = [
    "abandon", "adverse", "backlog", "bankrupt", "bottleneck",
    "burden", "cancel", "cautious", "challenge", "closure",
    "collapse", "complicate", "concern", "conservative", "constrain",
    "constraint", "contract", "contraction", "crisis", "cut",
    "damage", "decline", "decrease", "default", "defer",
    "deficit", "delay", "depress", "deteriorate", "difficult",
    "dip", "disappoint", "disaster", "disrupt", "disruption",
    "dive", "drag", "drop", "erode", "erosion",
    "excess", "fail", "fluctuate", "freeze", "friction",
    "gloomy", "halt", "headwind", "hurdle", "impair",
    "impede", "inflation", "instability", "layoff", "liability",
    "loss", "macroeconomic", "miss", "obstacle", "offset",
    "oversupply", "pause", "pessimistic", "plunge", "postpone",
    "pressure", "prudent", "pullback", "recession", "reduce",
    "reduction", "restructure", "risk", "setback", "severe",
    "shortage", "shrink", "sluggish", "soft", "soften",
    "shortfall", "squeeze", "stall", "struggle", "suffer",
    "suspend", "tariff", "tension", "threat", "tough",
    "uncertainty", "underperform", "unexpected", "unfavorable", "unpredictable",
    "volatile", "warning", "weak", "weakness", "worsen",
]

PESSIMISTIC_CATEGORY_WEIGHTS: dict[str, float] = {
    "Financial Miss & Contraction": 3.0,
    "Operational Bottlenecks":      2.5,
    "Macro & External Headwinds":   2.0,
    "Management Code Words":        1.5,
}
_PESSIMISTIC_KEYWORD_CATEGORIES: dict[str, str] = {
    "miss": "Financial Miss & Contraction", "decline": "Financial Miss & Contraction",
    "drop": "Financial Miss & Contraction", "shortfall": "Financial Miss & Contraction",
    "underperform": "Financial Miss & Contraction", "contract": "Financial Miss & Contraction",
    "shrink": "Financial Miss & Contraction", "loss": "Financial Miss & Contraction",
    "squeeze": "Financial Miss & Contraction", "decrease": "Financial Miss & Contraction",
    "contraction": "Financial Miss & Contraction", "deficit": "Financial Miss & Contraction",
    "plunge": "Financial Miss & Contraction", "collapse": "Financial Miss & Contraction",
    "dive": "Financial Miss & Contraction",
    "bottleneck": "Operational Bottlenecks", "delay": "Operational Bottlenecks",
    "shortage": "Operational Bottlenecks", "disrupt": "Operational Bottlenecks",
    "disruption": "Operational Bottlenecks", "oversupply": "Operational Bottlenecks",
    "layoff": "Operational Bottlenecks", "restructure": "Operational Bottlenecks",
    "backlog": "Operational Bottlenecks", "closure": "Operational Bottlenecks",
    "cancel": "Operational Bottlenecks", "suspend": "Operational Bottlenecks",
    "halt": "Operational Bottlenecks", "freeze": "Operational Bottlenecks",
    "headwind": "Macro & External Headwinds", "inflation": "Macro & External Headwinds",
    "macroeconomic": "Macro & External Headwinds", "recession": "Macro & External Headwinds",
    "tariff": "Macro & External Headwinds", "volatile": "Macro & External Headwinds",
    "fluctuate": "Macro & External Headwinds", "tension": "Macro & External Headwinds",
    "instability": "Macro & External Headwinds", "crisis": "Macro & External Headwinds",
    "cautious": "Management Code Words", "prudent": "Management Code Words",
    "uncertainty": "Management Code Words", "soft": "Management Code Words",
    "sluggish": "Management Code Words", "unpredictable": "Management Code Words",
    "offset": "Management Code Words", "conservative": "Management Code Words",
    "concern": "Management Code Words", "challenge": "Management Code Words",
}

CATEGORY_WEIGHTS: dict[str, float] = {
    "Financial Outperformance": 3.0,
    "Innovation & Moat":        2.5,
    "Market Momentum & Demand": 2.0,
    "Management Conviction":    1.5,
    "Operational Excellence":   1.5,
}
_KEYWORD_CATEGORIES: dict[str, str] = {
    "beat": "Financial Outperformance", "exceed": "Financial Outperformance",
    "outperform": "Financial Outperformance", "record": "Financial Outperformance",
    "surge": "Financial Outperformance", "soar": "Financial Outperformance",
    "triple": "Financial Outperformance", "upside": "Financial Outperformance",
    "lucrative": "Financial Outperformance", "premium": "Financial Outperformance",
    "breakthrough": "Innovation & Moat", "disrupt": "Innovation & Moat",
    "innovate": "Innovation & Moat", "pioneer": "Innovation & Moat",
    "revolutionize": "Innovation & Moat", "visionary": "Innovation & Moat",
    "transform": "Innovation & Moat",
    "accelerate": "Market Momentum & Demand", "catalyst": "Market Momentum & Demand",
    "demand": "Market Momentum & Demand", "momentum": "Market Momentum & Demand",
    "tailwind": "Market Momentum & Demand", "traction": "Market Momentum & Demand",
    "influx": "Market Momentum & Demand", "booming": "Market Momentum & Demand",
    "capture": "Market Momentum & Demand",
    "confident": "Management Conviction", "conviction": "Management Conviction",
    "reaffirm": "Management Conviction", "proud": "Management Conviction",
    "optimistic": "Management Conviction", "visibility": "Management Conviction",
    "promising": "Management Conviction",
    "agile": "Operational Excellence", "execution": "Operational Excellence",
    "leverage": "Operational Excellence", "resilient": "Operational Excellence",
    "streamline": "Operational Excellence", "sustainable": "Operational Excellence",
    "synergy": "Operational Excellence", "yield": "Operational Excellence",
}

HEDGE_WORDS = ["maybe", "might", "could", "possibly", "perhaps", "likely", "unlikely",
               "approximately", "around", "somewhat", "kind of", "sort of",
               "we think", "we believe", "we expect", "we hope", "we plan", "we aim", "guidance"]
CERTAINTY_WORDS = ["will", "definitely", "certainly", "clearly", "confident", "strong",
                   "committed", "always", "never", "proven", "guarantee"]

_NEGATION_TOKENS: frozenset[str] = frozenset({
    "no", "not", "never", "neither", "nor", "without", "n't",
    "less", "few", "little", "barely", "hardly", "scarcely",
})
_NEGATION_WINDOW = 3

_SEC_UA   = "StockFinancialModel/1.0 research@example.com"
_SEC_HDRS = {"User-Agent": _SEC_UA}
_MAX_8K_SCAN = 10


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
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    qa_markers = [r"\bquestions\s+and\s+answers\b", r"\bq\s*&\s*a\b", r"\bq\s+and\s+a\b",
                  r"\bquestion\-and\-answer\b", r"\boperator\b.*\bquestion\b"]
    for pat in qa_markers:
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            cut = m.start()
            return TranscriptParts(full_text=t, prepared=t[:cut].strip(), qa=t[cut:].strip())
    return TranscriptParts(full_text=t, prepared=t.strip(), qa="")


def _wn_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def _lemma_tokens(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    result: list[str] = []
    for word, pos in pos_tags:
        if word == "n't":
            result.append("not")
        elif word.isalpha():
            result.append(_lemmatizer.lemmatize(word, _wn_pos(pos)))
    return result


def _lemma_keyword(kw: str) -> list[str]:
    words = kw.lower().split()
    pos_tags = nltk.pos_tag(words)
    return [_lemmatizer.lemmatize(w, _wn_pos(pos)) for w, pos in pos_tags]


def _is_negated(tokens: list[str], pos: int) -> bool:
    window = tokens[max(0, pos - _NEGATION_WINDOW): pos]
    return bool(_NEGATION_TOKENS.intersection(window))


def count_keywords(text: str, keywords: list[str]) -> dict[str, int]:
    doc = _lemma_tokens(text or "")
    out: dict[str, int] = {}
    for kw in keywords:
        kw_lemmas = _lemma_keyword(kw)
        n = len(kw_lemmas)
        count = 0
        if n == 1:
            target = kw_lemmas[0]
            for i, token in enumerate(doc):
                if token == target and not _is_negated(doc, i):
                    count += 1
        else:
            for i in range(len(doc) - n + 1):
                if doc[i: i + n] == kw_lemmas and not _is_negated(doc, i):
                    count += 1
        out[kw] = count
    return out


def weighted_keyword_score(text: str, keywords: list[str], category_map: dict[str, str],
                            weight_map: dict[str, float]) -> tuple[float, pd.DataFrame]:
    counts = count_keywords(text, keywords)
    rows: list[dict] = []
    total = 0.0
    for kw, cnt in counts.items():
        if cnt == 0:
            continue
        cat = category_map.get(kw.lower().strip(), "General")
        weight = weight_map.get(cat, 1.0)
        score = cnt * weight
        total += score
        rows.append({"keyword": kw, "category": cat, "weight": weight, "count": cnt, "score": score})
    detail_df = pd.DataFrame(rows, columns=["keyword", "category", "weight", "count", "score"])
    if not detail_df.empty:
        detail_df = detail_df.sort_values("score", ascending=False)
    return total, detail_df


def sentence_split(text: str) -> list[str]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", t)
    return [p.strip() for p in parts if p and len(p.strip()) >= 8]


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


def confidence_index(text: str) -> float:
    t = (text or "").lower()
    if not t.strip():
        return float("nan")
    hedge = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in HEDGE_WORDS)
    sure = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in CERTAINTY_WORDS)
    total = hedge + sure
    if total == 0:
        return 50.0
    return float(max(0.0, min(100.0, 50.0 + 50.0 * (sure - hedge) / float(total))))


def _bs4_clean(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        if tag.name and tag.name.lower() in {"ix:header", "ix:references", "ix:resources", "ix:hidden"}:
            tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        if tag.parent is not None:
            tag.unwrap()
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(["p", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        tag.insert_before("\n")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p>", "\n", s)
    s = re.sub(r"(?i)</?(tr|li|div|h[1-6])[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"&lt;", "<", s)
    s = re.sub(r"&gt;", ">", s)
    s = re.sub(r"&#\d+;", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_mda_from_filing(html: str, filing_type: str) -> str:
    if "10-K" in filing_type.upper():
        start_re = re.compile(r"item\s*7[\s\.\-—–]", re.IGNORECASE)
        end_re   = re.compile(r"item\s*7\s*a[\s\.\-—]|item\s*8[\s\.\-—]", re.IGNORECASE)
        item_label = "Item 7"
    else:
        start_re = re.compile(r"item\s*2[\s\.\-—–]", re.IGNORECASE)
        end_re   = re.compile(r"item\s*3[\s\.\-—]|item\s*4[\s\.\-—]", re.IGNORECASE)
        item_label = "Item 2"
    text = _bs4_clean(html)
    start_matches = list(start_re.finditer(text))
    if not start_matches:
        raise ValueError(f"Could not locate '{item_label}' in this filing.")
    MIN_MDA_CHARS = 1_500
    for m_start in start_matches:
        pos = m_start.start()
        m_end = end_re.search(text, pos + 500)
        end_pos = m_end.start() if m_end else len(text)
        candidate = re.sub(r"\n{3,}", "\n\n", text[pos:end_pos]).strip()
        if len(candidate) >= MIN_MDA_CHARS:
            return candidate
    raise ValueError(f"Found '{item_label}' markers but could not extract a full MD&A section.")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def sec_cik_map() -> dict[str, int]:
    url = "https://www.sec.gov/files/company_tickers.json"
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


def _load_local_filing_text(symbol: str, filing_type: str) -> tuple[str, str] | None:
    sym = (symbol or "").strip().upper()
    kind = filing_type.strip().upper().replace("-", "")
    candidates = [
        _project_root() / "data" / f"{sym}_{kind}_latest.txt",
        _project_root() / "data" / f"{sym}_{filing_type.strip().upper()}_latest.txt",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 300:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) >= 300:
                return f"{sym} {filing_type.upper()} — Local text file", text
    return None


def _find_exhibit_991(cik: int, accession: str, acc_nodash: str) -> str | None:
    index_url = (f"https://www.sec.gov/Archives/edgar/data/{cik}"
                 f"/{acc_nodash}/{accession}-index.htm")
    try:
        r = requests.get(index_url, headers=_SEC_HDRS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            row_text = " ".join(c.get_text(strip=True) for c in cells).upper()
            if "EX-99" not in row_text:
                continue
            link = row.find("a", href=True)
            if link:
                href = str(link["href"])
                if not href.startswith("http"):
                    href = "https://www.sec.gov" + href
                return href
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_8k_sec(symbol: str) -> tuple[str, str]:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")
    local = _load_local_filing_text(sym, "8K")
    if local is not None:
        return local
    cik = sec_cik_map().get(sym)
    if cik is None:
        raise ValueError(f"Ticker '{sym}' not found in SEC EDGAR.")
    cik10 = f"{cik:010d}"
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik10}.json",
                       headers=_SEC_HDRS, timeout=20)
    sub.raise_for_status()
    recent = (sub.json().get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accs  = recent.get("accessionNumber") or []
    docs  = recent.get("primaryDocument") or []
    dates = recent.get("filingDate") or []
    candidates: list[tuple[str, str, str]] = []
    for form, acc, doc, date in zip(forms, accs, docs, dates):
        if str(form).strip().upper() == "8-K":
            candidates.append((str(date), str(acc), str(doc)))
        if len(candidates) >= _MAX_8K_SCAN:
            break
    if not candidates:
        raise ValueError(f"No recent 8-K filings found for '{sym}'.")
    ex991_url: str | None = None
    chosen_date, chosen_acc, chosen_doc = candidates[0]
    for date, acc, doc in candidates:
        acc_nd = acc.replace("-", "")
        url = _find_exhibit_991(cik, acc, acc_nd)
        if url:
            ex991_url, chosen_date, chosen_acc, chosen_doc = url, date, acc, doc
            break
    acc_nodash = chosen_acc.replace("-", "")
    if ex991_url:
        r = requests.get(ex991_url, headers=_SEC_HDRS, timeout=60)
        r.raise_for_status()
        text  = _bs4_clean(r.text)
        label = f"{sym} 8-K {chosen_date} — Exhibit 99.1 (Earnings Press Release)"
    else:
        doc_url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{chosen_doc}")
        r = requests.get(doc_url, headers=_SEC_HDRS, timeout=60)
        r.raise_for_status()
        text  = _bs4_clean(r.text)
        label = f"{sym} 8-K {chosen_date} (no Exhibit 99.1 in last {_MAX_8K_SCAN} filings)"
    if len(text) < 300:
        raise ValueError("The downloaded document is too short to analyse.")
    return label, text


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_mda_sec(symbol: str, filing_type: str = "10-K") -> tuple[str, str]:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")
    local = _load_local_filing_text(sym, filing_type)
    if local is not None:
        return local
    cik = sec_cik_map().get(sym)
    if cik is None:
        raise ValueError(f"Ticker '{sym}' not found in SEC EDGAR mapping.")
    cik10 = f"{cik:010d}"
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik10}.json",
                       headers=_SEC_HDRS, timeout=30)
    sub.raise_for_status()
    sj = sub.json()
    recent = (sj.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accs  = recent.get("accessionNumber") or []
    prim  = recent.get("primaryDocument") or []
    dates = recent.get("filingDate") or []
    pick = None
    target = filing_type.strip().upper()
    for i in range(min(len(forms), len(accs), len(prim), len(dates))):
        if str(forms[i]).strip().upper() == target:
            pick = (str(dates[i]), str(accs[i]), str(prim[i]))
            break
    if pick is None:
        raise ValueError(f"No recent {filing_type} filing found for '{sym}'.")
    filing_date, accession, primary_doc = pick
    acc_nodash = accession.replace("-", "")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"
    r = requests.get(doc_url, headers=_SEC_HDRS, timeout=90)
    r.raise_for_status()
    mda_text = _extract_mda_from_filing(r.text, filing_type)
    return f"{sym} {filing_type} {filing_date} — MD&A", mda_text


def _render_analysis(transcript_text: str, ticker_q: str) -> None:
    if not (transcript_text or "").strip():
        return
    parts    = split_prepared_vs_qa(transcript_text)
    qa_focus = parts.qa.strip() if (parts.qa or "").strip() else parts.full_text.strip()
    ci       = confidence_index(qa_focus)
    kw_growth    = count_keywords(qa_focus, list(GROWTH_KEYWORDS))
    kw_risk      = count_keywords(qa_focus, list(RISK_KEYWORDS))
    growth_total = int(sum(kw_growth.values()))
    risk_total   = int(sum(kw_risk.values()))
    weighted_total, weighted_df = weighted_keyword_score(qa_focus, list(GROWTH_KEYWORDS),
                                                          _KEYWORD_CATEGORIES, CATEGORY_WEIGHTS)
    risk_weighted_total, risk_weighted_df = weighted_keyword_score(qa_focus, list(RISK_KEYWORDS),
                                                                    _PESSIMISTIC_KEYWORD_CATEGORIES,
                                                                    PESSIMISTIC_CATEGORY_WEIGHTS)
    net_score = weighted_total - risk_weighted_total
    s_full = sentence_split(qa_focus)
    s_pre  = sentence_split(parts.prepared)
    s_qa   = sentence_split(parts.qa)
    df_full = sentiment_series(s_full)
    df_pre  = sentiment_series(s_pre)
    df_qa   = sentiment_series(s_qa) if parts.qa else pd.DataFrame(columns=["idx", "sentence", "compound"])
    avg_full = _avg_compound(df_full)
    avg_pre  = _avg_compound(df_pre)
    avg_qa   = _avg_compound(df_qa)

    st.subheader("1) Sentiment dashboard")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1: st.metric("Full sentiment",     f"{avg_full:.3f}" if np.isfinite(avg_full) else "—")
    with c2: st.metric("Prepared sentiment", f"{avg_pre:.3f}"  if np.isfinite(avg_pre)  else "—")
    with c3: st.metric("Q&A sentiment",      f"{avg_qa:.3f}"   if np.isfinite(avg_qa)   else "—")
    with c4: st.metric("Optimistic hits",    f"{growth_total:,}")
    with c5: st.metric("Risk hits",          f"{risk_total:,}")
    with c6:
        bull_delta = f"+{net_score:.1f}" if net_score >= 0 else f"{net_score:.1f}"
        st.metric("Net Bullish Score 🎯", f"{net_score:+.1f}", delta=bull_delta, delta_color="normal",
                  help="Weighted Optimism − Weighted Risk.")
    with c7:
        st.metric("Optimism ⚡", f"{weighted_total:.1f}",
                  help="Σ (hit × weight): Outperformance ×3.0 · Innovation ×2.5 · Momentum ×2.0 · Conviction ×1.5 · Operations ×1.5")
    st.caption(f"**Confidence index (Q&A language):** {ci:.0f}/100" if np.isfinite(ci) else "**Confidence index:** —")
    st.caption(f"Label: **{ticker_q}** • Sentences: {len(df_full):,} • Prepared: {len(df_pre):,} • Q&A: {len(df_qa):,}")

    st.subheader("2) Keyword frequency & weighted score")
    tab_weighted, tab_freq = st.tabs(["⚡ Weighted by category", "📊 Raw frequency"])
    with tab_freq:
        kw_df = pd.DataFrame(
            [{"keyword": k, "count": v, "group": "Optimistic"} for k, v in kw_growth.items()]
            + [{"keyword": k, "count": v, "group": "Risk"} for k, v in kw_risk.items()]
        ).sort_values(["group", "count"], ascending=[True, False])
        kw_df = kw_df[kw_df["count"] > 0]
        if kw_df.empty:
            st.info("No keyword hits found.")
        else:
            fig_kw = px.bar(kw_df, x="count", y="keyword", color="group", orientation="h",
                            title="Keyword Frequency", height=max(300, len(kw_df) * 22),
                            color_discrete_map={"Optimistic": "#2ecc71", "Risk": "#e74c3c"})
            fig_kw.update_layout(yaxis_title="", xaxis_title="Count", legend_title="")
            st.plotly_chart(fig_kw, use_container_width=True)
    with tab_weighted:
        st.markdown(f"**Net Bullish Score = {weighted_total:.1f} (Optimism) − "
                    f"{risk_weighted_total:.1f} (Risk) = "
                    f"{'🟢' if net_score >= 0 else '🔴'} {net_score:+.1f}**")
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("#### 🟢 Optimism by Category")
            if weighted_df.empty:
                st.info("No optimistic keyword hits.")
            else:
                bull_cat = weighted_df.groupby("category")["score"].sum().reset_index().sort_values("score", ascending=True)
                bull_color_map = {"Financial Outperformance": "#e74c3c", "Innovation & Moat": "#9b59b6",
                                  "Market Momentum & Demand": "#3498db", "Management Conviction": "#f39c12",
                                  "Operational Excellence": "#2ecc71", "General": "#95a5a6"}
                fig_bull = px.bar(bull_cat, x="score", y="category", orientation="h",
                                  color="category", color_discrete_map=bull_color_map,
                                  title=f"Optimism total = {weighted_total:.1f}", height=280)
                fig_bull.update_layout(showlegend=False, yaxis_title="", xaxis_title="Score")
                st.plotly_chart(fig_bull, use_container_width=True)
        with col_bear:
            st.markdown("#### 🔴 Risk by Category")
            if risk_weighted_df.empty:
                st.info("No risk keyword hits.")
            else:
                bear_cat = risk_weighted_df.groupby("category")["score"].sum().reset_index().sort_values("score", ascending=True)
                bear_color_map = {"Financial Miss & Contraction": "#c0392b", "Operational Bottlenecks": "#e67e22",
                                  "Macro & External Headwinds": "#d35400", "Management Code Words": "#f39c12",
                                  "General": "#95a5a6"}
                fig_bear = px.bar(bear_cat, x="score", y="category", orientation="h",
                                  color="category", color_discrete_map=bear_color_map,
                                  title=f"Risk total = {risk_weighted_total:.1f}", height=280)
                fig_bear.update_layout(showlegend=False, yaxis_title="", xaxis_title="Score")
                st.plotly_chart(fig_bear, use_container_width=True)
        with st.expander("Optimistic keyword detail", expanded=False):
            if not weighted_df.empty:
                st.dataframe(weighted_df.style.format({"weight": "{:.1f}", "score": "{:.1f}"}),
                             use_container_width=True, hide_index=True)
        with st.expander("Risk keyword detail (negation-corrected)", expanded=False):
            if not risk_weighted_df.empty:
                st.dataframe(risk_weighted_df.style.format({"weight": "{:.1f}", "score": "{:.1f}"}),
                             use_container_width=True, hide_index=True)

    st.subheader("3) Sentiment trend (by sentence)")
    if df_full.empty:
        st.info("No sentences detected.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_full["idx"], y=df_full["compound"], mode="lines", name="Compound"))
        fig.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="gray")
        fig.update_layout(title="Sentence-level sentiment (VADER compound)",
                          xaxis_title="Sentence index", yaxis_title="Compound (-1 to 1)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Lowest sentiment sentences (debug)", expanded=False):
            st.dataframe(df_full.sort_values("compound").head(10)[["compound", "sentence"]],
                         use_container_width=True, hide_index=True)

    with st.expander("Parsed sections (Prepared vs Q&A)", expanded=False):
        st.markdown("#### Prepared remarks")
        st.text(parts.prepared[:20_000] if parts.prepared else "")
        st.markdown("#### Q&A")
        st.text(parts.qa[:20_000] if parts.qa else "")


def _fetch_into_state(cache_key: str, fetch_fn) -> None:
    if cache_key not in st.session_state:
        with st.spinner("Fetching from SEC EDGAR…"):
            try:
                label, txt = fetch_fn()
                st.session_state[cache_key] = (label, txt)
            except Exception as e:
                st.error(str(e))


# ═════════════════════════════════════════════════════════════════════════════
# EARNINGS CALENDAR — data logic
# ═════════════════════════════════════════════════════════════════════════════
_TZ_ET = ZoneInfo("America/New_York")

_CAL_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/earnings",
}


def _trading_days(start: datetime.date, n: int) -> list[datetime.date]:
    days: list[datetime.date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"—", "-", "N/A", "n/a"}:
        return None
    text = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _surprise(actual: Any, estimate: Any) -> float | None:
    a = _safe_number(actual)
    e = _safe_number(estimate)
    if a is None or e is None or e == 0:
        return None
    return (a - e) / abs(e) * 100.0


def _money_short(value: Any) -> str:
    x = _safe_number(value)
    if x is None:
        return "—"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.1f}M"
    return f"{x:,.0f}"


def _normalize_when(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("before", "pre", "bmo", "morning")):
        return "Pre-Market"
    if any(token in text for token in ("after", "amc", "close", "evening")):
        return "After Hours"
    return "TBD"


def _when_icon(when: str) -> str:
    if when == "Pre-Market": return "🌅 Pre-Market"
    if when == "After Hours": return "🌙 After Hours"
    return "❓ TBD"


def _parse_payload_symbol(payload_json: Any) -> str:
    if not payload_json:
        return ""
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return ""
    return str(payload.get("symbol") or "").strip().upper()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_nasdaq_earnings(date_str: str) -> list[dict[str, Any]]:
    try:
        r = requests.get("https://api.nasdaq.com/api/calendar/earnings",
                         headers=_CAL_HEADERS, params={"date": date_str}, timeout=20)
        r.raise_for_status()
        data = r.json()
        return (data.get("data") or {}).get("rows") or []
    except Exception:
        return []


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_company_profiles() -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame(columns=["Symbol", "Company", "Sector", "Industry"])
    with connect_readonly(db) as conn:
        try:
            return pd.read_sql_query(
                "SELECT ticker AS Symbol, company_name AS Company, sector AS Sector, industry AS Industry FROM company_profile",
                conn)
        except Exception:
            return pd.DataFrame(columns=["Symbol", "Company", "Sector", "Industry"])


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_watchlists() -> dict[str, set[str]]:
    db = _db_path()
    if not db.exists():
        return {}
    with connect_readonly(db) as conn:
        try:
            df = pd.read_sql_query("SELECT watchlist_name, ticker FROM fmp_watchlist", conn)
        except Exception:
            return {}
    if df.empty:
        return {}
    return {str(name): set(group["ticker"].dropna().astype(str).str.upper())
            for name, group in df.groupby("watchlist_name")}


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_local_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame()
    with connect_readonly(db) as conn:
        try:
            raw = pd.read_sql_query(
                """SELECT ticker, event_date, eps, eps_estimated, time, revenue, revenue_estimated,
                          fiscal_date_ending, payload_json, updated_at
                   FROM earnings_calendar WHERE event_date BETWEEN ? AND ?
                   ORDER BY event_date ASC, ticker ASC""",
                conn, params=(start_date, end_date))
        except Exception:
            return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    payload_symbols = raw["payload_json"].map(_parse_payload_symbol)
    raw["Symbol"] = payload_symbols.where(payload_symbols.astype(bool), raw["ticker"].astype(str).str.upper())
    raw = raw.drop_duplicates(["Symbol", "event_date"], keep="last")
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw["event_date"], errors="coerce").dt.date,
        "Symbol": raw["Symbol"].astype(str).str.upper(),
        "When": raw["time"].map(_normalize_when),
        "EPS Est.": pd.to_numeric(raw["eps_estimated"], errors="coerce"),
        "EPS Actual": pd.to_numeric(raw["eps"], errors="coerce"),
        "Revenue Est.": pd.to_numeric(raw["revenue_estimated"], errors="coerce"),
        "Revenue Actual": pd.to_numeric(raw["revenue"], errors="coerce"),
        "Quarter": raw["fiscal_date_ending"].fillna(""),
        "Source": "Local FMP DB",
        "Updated": raw["updated_at"].fillna(""),
    })
    profiles = load_company_profiles()
    if not profiles.empty:
        out = out.merge(profiles, on="Symbol", how="left")
    else:
        out["Company"] = ""
        out["Sector"] = ""
        out["Industry"] = ""
    out["EPS Surprise %"] = [_surprise(a, e) for a, e in zip(out["EPS Actual"], out["EPS Est."])]
    out["Revenue Surprise %"] = [_surprise(a, e) for a, e in zip(out["Revenue Actual"], out["Revenue Est."])]
    return out


def load_nasdaq_calendar(days: list[datetime.date]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    profiles = load_company_profiles()
    profile_map = profiles.set_index("Symbol").to_dict("index") if not profiles.empty else {}
    for day in days:
        for row in fetch_nasdaq_earnings(day.isoformat()):
            symbol = str(row.get("symbol") or "").strip().upper()
            profile = profile_map.get(symbol, {})
            records.append({"Date": day, "Symbol": symbol,
                             "Company": str(row.get("name") or profile.get("Company") or "").strip(),
                             "Sector": profile.get("Sector", ""), "Industry": profile.get("Industry", ""),
                             "When": _normalize_when(row.get("marketTime")),
                             "EPS Est.": _safe_number(row.get("epsForecast")), "EPS Actual": None,
                             "EPS Last Yr": _safe_number(row.get("lastYearEPS")),
                             "Revenue Est.": None, "Revenue Actual": None,
                             "Quarter": row.get("fiscalQuarterEnding") or "",
                             "Source": "Nasdaq Live", "Updated": ""})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).drop_duplicates(["Date", "Symbol"], keep="last")
    df["EPS Surprise %"] = None
    df["Revenue Surprise %"] = None
    return df


def _apply_filters(df: pd.DataFrame, *, query: str, watchlist: set[str],
                   only_watchlist: bool, sectors: list[str],
                   show_missing_estimates: bool) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["In Watchlist"] = out["Symbol"].isin(watchlist) if watchlist else False
    q = query.strip().lower()
    if q:
        haystack = (out["Symbol"].fillna("").astype(str) + " "
                    + out["Company"].fillna("").astype(str) + " "
                    + out["Sector"].fillna("").astype(str)).str.lower()
        out = out[haystack.str.contains(re.escape(q), na=False)]
    if only_watchlist and watchlist:
        out = out[out["In Watchlist"]]
    if sectors:
        out = out[out["Sector"].fillna("").isin(sectors)]
    if not show_missing_estimates and "EPS Est." in out.columns:
        out = out[out["EPS Est."].notna()]
    when_order = {"Pre-Market": 0, "After Hours": 1, "TBD": 2}
    out["_when_order"] = out["When"].map(when_order).fillna(9)
    out["_watch_order"] = (~out["In Watchlist"]).astype(int)
    out = out.sort_values(["Date", "_watch_order", "_when_order", "Symbol"])
    return out.drop(columns=["_when_order", "_watch_order"])


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = df.copy()
    show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%Y-%m-%d")
    show["When"] = show["When"].map(_when_icon)
    show["Watch"] = show["In Watchlist"].map(lambda x: "Yes" if bool(x) else "")
    show["Revenue Est."] = show["Revenue Est."].map(_money_short)
    show["Revenue Actual"] = show["Revenue Actual"].map(_money_short)
    cols = ["Date", "Watch", "Symbol", "Company", "Sector", "When", "EPS Est.", "EPS Actual",
            "EPS Surprise %", "EPS Last Yr", "Revenue Est.", "Revenue Actual",
            "Revenue Surprise %", "Quarter", "Source"]
    return show[[c for c in cols if c in show.columns]]


def _height(n_rows: int, max_height: int = 520) -> int:
    return min(max(120, 38 + n_rows * 35), max_height)


def _summary(df: pd.DataFrame) -> None:
    total = len(df)
    pre   = int((df["When"] == "Pre-Market").sum()) if total else 0
    after = int((df["When"] == "After Hours").sum()) if total else 0
    watch = int(df.get("In Watchlist", pd.Series(dtype=bool)).sum()) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports", f"{total:,}")
    c2.metric("Pre-Market", f"{pre:,}")
    c3.metric("After Hours", f"{after:,}")
    c4.metric("Watchlist", f"{watch:,}")


def _show_table(df: pd.DataFrame, *, max_height: int = 520) -> None:
    if df.empty:
        st.info("No earnings match the current filters.")
        return
    st.dataframe(_display_df(df), hide_index=True, use_container_width=True,
                 height=_height(len(df), max_height=max_height),
                 column_config={
                     "EPS Est.": st.column_config.NumberColumn("EPS Est.", format="%.2f"),
                     "EPS Actual": st.column_config.NumberColumn("EPS Actual", format="%.2f"),
                     "EPS Surprise %": st.column_config.NumberColumn("EPS Surprise %", format="%.1f%%"),
                     "EPS Last Yr": st.column_config.NumberColumn("EPS Last Yr", format="%.2f"),
                     "Revenue Surprise %": st.column_config.NumberColumn("Revenue Surprise %", format="%.1f%%"),
                 })


def _show_day(day: datetime.date, df: pd.DataFrame) -> None:
    day_df = df[df["Date"] == day]
    if day_df.empty:
        st.info(f"No earnings found for {day.strftime('%A, %B %d')}.")
        return
    _summary(day_df)
    st.markdown("")
    pre   = day_df[day_df["When"] == "Pre-Market"]
    after = day_df[day_df["When"] == "After Hours"]
    tbd   = day_df[day_df["When"] == "TBD"]
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"#### 🌅 Pre-Market ({len(pre)})")
        _show_table(pre, max_height=430)
    with col_r:
        st.markdown(f"#### 🌙 After Hours ({len(after)})")
        _show_table(after, max_height=430)
    if not tbd.empty:
        with st.expander(f"❓ Time Not Confirmed ({len(tbd)})", expanded=False):
            _show_table(tbd)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def render_call_analyzer() -> None:
    st.title("Earnings Call Analyzer")
    st.caption("Tracks keyword frequency, weighted sentiment score, and sentence-level VADER analysis. "
               "Data fetched automatically from SEC EDGAR.")

    with st.sidebar:
        st.header("Input")
        ticker = st.text_input("Ticker", value="NVDA", key="nlp_ticker").strip().upper()
        st.divider()
        with st.expander("📎 Manual text override", expanded=False):
            manual_src = st.radio("Source", ["Paste text", "Upload .txt"], key="nlp_manual_src")
            manual_text = ""
            if manual_src == "Paste text":
                manual_text = st.text_area("Paste transcript / MD&A", height=220,
                                           placeholder="Paste text here…", key="nlp_manual_paste")
            else:
                uploaded = st.file_uploader("Upload .txt", type=["txt"], key="nlp_upload")
                if uploaded is not None:
                    manual_text = _safe_decode(uploaded)

    tab_8k, tab_10k, tab_10q = st.tabs([
        "📰  8-K  Earnings Release",
        "📊  10-K  Annual MD&A",
        "📋  10-Q  Quarterly MD&A",
    ])

    def _resolve(cache_key: str) -> tuple[str, str]:
        if (manual_text or "").strip():
            return ticker or "Manual Input", manual_text.strip()
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        return "", ""

    with tab_8k:
        st.caption("**8-K** — Current event report filed within 4 business days. "
                   "Contains the earnings press release (Exhibit 99.1) and financial tables.")
        if ticker:
            ck = f"edgar_8k_{ticker}"
            _fetch_into_state(ck, lambda: fetch_8k_sec(ticker))
            lbl, txt = _resolve(ck)
            if lbl:
                st.success(f"Loaded: {lbl}")
            _render_analysis(txt, lbl)
        else:
            st.info("Enter a ticker in the sidebar to auto-fetch.")

    with tab_10k:
        st.caption("**10-K MD&A** — Annual filing. MD&A carries legal liability; "
                   "more rigorous signal than a live earnings call.")
        if ticker:
            ck = f"edgar_10k_{ticker}"
            _fetch_into_state(ck, lambda: fetch_mda_sec(ticker, "10-K"))
            lbl, txt = _resolve(ck)
            if lbl:
                st.success(f"Loaded: {lbl}")
            _render_analysis(txt, lbl)
        else:
            st.info("Enter a ticker in the sidebar to auto-fetch.")

    with tab_10q:
        st.caption("**10-Q MD&A** — Quarterly filing. Captures intra-year pivots in management tone.")
        if ticker:
            ck = f"edgar_10q_{ticker}"
            _fetch_into_state(ck, lambda: fetch_mda_sec(ticker, "10-Q"))
            lbl, txt = _resolve(ck)
            if lbl:
                st.success(f"Loaded: {lbl}")
            _render_analysis(txt, lbl)
        else:
            st.info("Enter a ticker in the sidebar to auto-fetch.")


def render_calendar() -> None:
    now_et = datetime.datetime.now(tz=_TZ_ET)
    today  = now_et.date()

    st.title("Earnings Calendar")
    st.caption(f"{now_et.strftime('%Y-%m-%d %H:%M %Z')} · Track report timing, estimates, and watchlist names.")

    watchlists      = load_watchlists()
    watchlist_names = sorted(watchlists)

    with st.sidebar:
        st.header("Calendar Controls")
        source = st.radio("Data source",
                          ["Auto: local then Nasdaq", "Local FMP database", "Nasdaq live"],
                          horizontal=False, key="cal_source")
        start_input = st.date_input("Start date", value=today, key="cal_start")
        days_count  = st.slider("Trading days", min_value=3, max_value=20, value=5, step=1, key="cal_days")
        query       = st.text_input("Search symbol / company / sector", "", key="cal_query")

        selected_watchlist = ""
        watchlist: set[str] = set()
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "Watchlist", watchlist_names,
                index=watchlist_names.index("default") if "default" in watchlist_names else 0,
                key="cal_wl")
            watchlist      = watchlists.get(selected_watchlist, set())
            only_watchlist = st.checkbox("Only show watchlist", value=False, key="cal_only_wl")
        else:
            only_watchlist = False
            st.caption("No local watchlist table found.")

        show_missing_estimates = st.checkbox("Show rows without EPS estimate", value=True, key="cal_missing")
        if st.button("Refresh data", type="primary", use_container_width=True, key="cal_refresh"):
            st.cache_data.clear()
            st.rerun()

    start_base   = start_input if isinstance(start_input, datetime.date) else today
    trading_days = _trading_days(start_base, days_count)
    start_date   = trading_days[0]
    end_date     = trading_days[-1]

    with st.spinner("Loading earnings calendar..."):
        if source == "Local FMP database":
            calendar = load_local_calendar(start_date.isoformat(), end_date.isoformat())
        elif source == "Nasdaq live":
            calendar = load_nasdaq_calendar(trading_days)
        else:
            calendar = load_local_calendar(start_date.isoformat(), end_date.isoformat())
            if calendar.empty:
                calendar = load_nasdaq_calendar(trading_days)

    sector_options = sorted(s for s in calendar.get("Sector", pd.Series(dtype=str)).dropna().unique() if str(s))
    with st.sidebar:
        sectors = st.multiselect("Sector", sector_options, default=[], key="cal_sectors")
        st.caption(f"Window: {start_date.isoformat()} to {end_date.isoformat()}")
        if selected_watchlist:
            st.caption(f"Watchlist symbols: {len(watchlist):,}")

    filtered = _apply_filters(calendar, query=query, watchlist=watchlist,
                               only_watchlist=only_watchlist, sectors=sectors,
                               show_missing_estimates=show_missing_estimates)

    if calendar.empty:
        st.warning("No earnings data loaded. Try the other data source, refresh the cache, "
                   "or update the local FMP calendar data.")
        return

    _summary(filtered)

    focus = filtered[filtered.get("In Watchlist", pd.Series(dtype=bool))]
    if not focus.empty:
        with st.expander(f"Watchlist focus ({len(focus)} reports)", expanded=True):
            _show_table(focus, max_height=360)

    st.divider()

    tab_labels = [
        f"{d.strftime('%a')} {d.strftime('%m/%d')}" + (" · Today" if d == today else "")
        for d in trading_days
    ]
    tabs = st.tabs(tab_labels)
    for tab, day in zip(tabs, trading_days):
        with tab:
            _show_day(day, filtered)

    st.divider()
    st.subheader("All Matching Reports")
    _show_table(filtered, max_height=650)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
st.title("Earnings Analyst")

tool = st.sidebar.radio(
    "Select tool",
    ["Call Analyzer", "Earnings Calendar"],
    key="eh_tool",
)
st.sidebar.divider()

if tool == "Call Analyzer":
    render_call_analyzer()
else:
    render_calendar()
