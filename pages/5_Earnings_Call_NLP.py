"""
Earnings Call NLP Analyzer — Streamlit page module.

Primary input is manual transcript upload (.txt).
Optional: Google Gemini API for LLM summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import streamlit as st
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@st.cache_resource(show_spinner=False)
def _ensure_nltk_data() -> None:
    """Download required NLTK corpora once per deployment (cached globally)."""
    packages = [
        "punkt",
        "punkt_tab",                      # needed by newer NLTK versions
        "wordnet",
        "omw-1.4",                        # Open Multilingual Wordnet (lemmatizer)
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng", # newer NLTK alias
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass   # best-effort; missing optional variants are harmless


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()


# Keywords in base/lemma form — lemmatisation auto-matches inflected variants.
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
    "Financial Miss & Contraction": 3.0,   # miss / decline → direct earnings miss
    "Operational Bottlenecks":      2.5,   # bottleneck / layoff → supply-chain red flags
    "Macro & External Headwinds":   2.0,   # inflation / tariff → blaming macro
    "Management Code Words":        1.5,   # cautious / prudent → CEO-speak for scared
}

_PESSIMISTIC_KEYWORD_CATEGORIES: dict[str, str] = {
    # Financial Miss & Contraction ──────────────────────────────────────────
    "miss":         "Financial Miss & Contraction",
    "decline":      "Financial Miss & Contraction",
    "drop":         "Financial Miss & Contraction",
    "shortfall":    "Financial Miss & Contraction",
    "underperform": "Financial Miss & Contraction",
    "contract":     "Financial Miss & Contraction",
    "shrink":       "Financial Miss & Contraction",
    "loss":         "Financial Miss & Contraction",
    "squeeze":      "Financial Miss & Contraction",
    "decrease":     "Financial Miss & Contraction",
    "contraction":  "Financial Miss & Contraction",
    "deficit":      "Financial Miss & Contraction",
    "plunge":       "Financial Miss & Contraction",
    "collapse":     "Financial Miss & Contraction",
    "dive":         "Financial Miss & Contraction",
    # Operational Bottlenecks ───────────────────────────────────────────────
    "bottleneck":   "Operational Bottlenecks",
    "delay":        "Operational Bottlenecks",
    "shortage":     "Operational Bottlenecks",
    "disrupt":      "Operational Bottlenecks",
    "disruption":   "Operational Bottlenecks",
    "oversupply":   "Operational Bottlenecks",
    "layoff":       "Operational Bottlenecks",
    "restructure":  "Operational Bottlenecks",
    "backlog":      "Operational Bottlenecks",
    "closure":      "Operational Bottlenecks",
    "cancel":       "Operational Bottlenecks",
    "suspend":      "Operational Bottlenecks",
    "halt":         "Operational Bottlenecks",
    "freeze":       "Operational Bottlenecks",
    # Macro & External Headwinds ────────────────────────────────────────────
    "headwind":     "Macro & External Headwinds",
    "inflation":    "Macro & External Headwinds",
    "macroeconomic":"Macro & External Headwinds",
    "recession":    "Macro & External Headwinds",
    "tariff":       "Macro & External Headwinds",
    "volatile":     "Macro & External Headwinds",
    "fluctuate":    "Macro & External Headwinds",
    "tension":      "Macro & External Headwinds",
    "instability":  "Macro & External Headwinds",
    "crisis":       "Macro & External Headwinds",
    # Management Code Words ─────────────────────────────────────────────────
    "cautious":     "Management Code Words",
    "prudent":      "Management Code Words",
    "uncertainty":  "Management Code Words",
    "soft":         "Management Code Words",
    "sluggish":     "Management Code Words",
    "unpredictable":"Management Code Words",
    "offset":       "Management Code Words",
    "conservative": "Management Code Words",
    "concern":      "Management Code Words",
    "challenge":    "Management Code Words",
}

# ── Weighted scoring system ───────────────────────────────────────────────────
# Five categories with distinct multipliers reflecting their market impact.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Financial Outperformance": 3.0,   # beat / exceed → directly beats Wall St. estimates
    "Innovation & Moat":        2.5,   # breakthrough / disrupt → future premium (tech stocks)
    "Market Momentum & Demand": 2.0,   # demand / tailwind → macro / product traction
    "Management Conviction":    1.5,   # confident / reaffirm → reassuring analysts
    "Operational Excellence":   1.5,   # resilient / synergy → defensive strength
}

_KEYWORD_CATEGORIES: dict[str, str] = {
    # Financial Outperformance ──────────────────────────────────────────────
    "beat":         "Financial Outperformance",
    "exceed":       "Financial Outperformance",
    "outperform":   "Financial Outperformance",
    "record":       "Financial Outperformance",
    "surge":        "Financial Outperformance",
    "soar":         "Financial Outperformance",
    "triple":       "Financial Outperformance",
    "upside":       "Financial Outperformance",
    "lucrative":    "Financial Outperformance",
    "premium":      "Financial Outperformance",
    # Innovation & Moat ─────────────────────────────────────────────────────
    "breakthrough": "Innovation & Moat",
    "disrupt":      "Innovation & Moat",
    "innovate":     "Innovation & Moat",
    "pioneer":      "Innovation & Moat",
    "revolutionize":"Innovation & Moat",
    "visionary":    "Innovation & Moat",
    "transform":    "Innovation & Moat",
    # Market Momentum & Demand ──────────────────────────────────────────────
    "accelerate":   "Market Momentum & Demand",
    "catalyst":     "Market Momentum & Demand",
    "demand":       "Market Momentum & Demand",
    "momentum":     "Market Momentum & Demand",
    "tailwind":     "Market Momentum & Demand",
    "traction":     "Market Momentum & Demand",
    "influx":       "Market Momentum & Demand",
    "booming":      "Market Momentum & Demand",
    "capture":      "Market Momentum & Demand",
    # Management Conviction ─────────────────────────────────────────────────
    "confident":    "Management Conviction",
    "conviction":   "Management Conviction",
    "reaffirm":     "Management Conviction",
    "proud":        "Management Conviction",
    "optimistic":   "Management Conviction",
    "visibility":   "Management Conviction",
    "promising":    "Management Conviction",
    # Operational Excellence ────────────────────────────────────────────────
    "agile":        "Operational Excellence",
    "execution":    "Operational Excellence",
    "leverage":     "Operational Excellence",
    "resilient":    "Operational Excellence",
    "streamline":   "Operational Excellence",
    "sustainable":  "Operational Excellence",
    "synergy":      "Operational Excellence",
    "yield":        "Operational Excellence",
}


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


def _wn_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to its WordNet equivalent."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN   # default (covers nouns + unknowns)


# Negation tokens that flip the sentiment of the next keyword.
# Includes contracted forms because word_tokenize splits "don't" → ["do", "n't"].
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "no", "not", "never", "neither", "nor", "without",
    "n't",                       # contraction fragment from word_tokenize
    "less", "few", "little",
    "barely", "hardly", "scarcely",
})
_NEGATION_WINDOW = 3            # tokens to look back for a negation word


def _lemma_tokens(text: str) -> list[str]:
    """
    Tokenise and lemmatise every word in *text*.

    - Alphabetic words are POS-tagged then lemmatised.
    - Negation fragments ("n't") are normalised to "not" so that
      "don't", "can't", "won't" etc. are correctly detected by the
      negation-window check in count_keywords().
    """
    tokens = word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    result: list[str] = []
    for word, pos in pos_tags:
        if word == "n't":
            result.append("not")          # normalise contraction
        elif word.isalpha():
            result.append(_lemmatizer.lemmatize(word, _wn_pos(pos)))
    return result


def _lemma_keyword(kw: str) -> list[str]:
    """Lemmatise each word in a (possibly multi-word) keyword phrase."""
    words = kw.lower().split()
    pos_tags = nltk.pos_tag(words)
    return [_lemmatizer.lemmatize(w, _wn_pos(pos)) for w, pos in pos_tags]


def _is_negated(tokens: list[str], pos: int) -> bool:
    """
    Return True if any of the _NEGATION_WINDOW tokens before *pos*
    is a negation word (e.g. "no", "not", "n't" → "not").
    Handles: "no headwinds", "not declining", "don't see any weakness".
    """
    window = tokens[max(0, pos - _NEGATION_WINDOW) : pos]
    return bool(_NEGATION_TOKENS.intersection(window))


def count_keywords(text: str, keywords: list[str]) -> dict[str, int]:
    """
    Count keyword occurrences using POS-aware lemmatisation + negation detection.

    Lemmatisation catches inflected forms:
      'growth'     → grew, growing, grown, growth
      'accelerate' → accelerated, accelerating
      'risk'       → risks, risky
      'delay'      → delayed, delays, delaying
      'supply chain' → multi-word sliding-window match

    Negation detection prevents false positives:
      "We see no headwinds"     → headwind NOT counted
      "We did not miss guidance" → miss NOT counted
      "Don't see any weakness"   → weakness NOT counted  (n't → not)
    """
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
                if doc[i : i + n] == kw_lemmas and not _is_negated(doc, i):
                    count += 1

        out[kw] = count

    return out


def weighted_keyword_score(
    text: str,
    keywords: list[str],
    category_map: dict[str, str],
    weight_map: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    """
    Score the text using per-category keyword weights.

    Parameters
    ----------
    category_map : keyword → category name
    weight_map   : category name → multiplier

    Returns
    -------
    total_score : float
    detail_df   : DataFrame with columns keyword | category | weight | count | score
    """
    counts = count_keywords(text, keywords)
    rows: list[dict] = []
    total = 0.0

    for kw, cnt in counts.items():
        if cnt == 0:
            continue
        cat    = category_map.get(kw.lower().strip(), "General")
        weight = weight_map.get(cat, 1.0)
        score  = cnt * weight
        total += score
        rows.append({"keyword": kw, "category": cat,
                     "weight": weight, "count": cnt, "score": score})

    detail_df = pd.DataFrame(
        rows, columns=["keyword", "category", "weight", "count", "score"]
    )
    if not detail_df.empty:
        detail_df = detail_df.sort_values("score", ascending=False)

    return total, detail_df


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




def _strip_html(s: str) -> str:
    """Convert HTML to plain text, preserving paragraph breaks."""
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
    """
    Extract the MD&A section from a 10-K (Item 7) or 10-Q (Item 2) HTML/iXBRL filing.

    Uses character-position search across ALL occurrences of the start marker,
    skipping Table-of-Contents entries (which are immediately followed by the next
    section) and returning the first candidate with enough content.
    """
    if "10-K" in filing_type.upper():
        start_re = re.compile(r"item\s*7[\s\.\-\u2014\u2013]", re.IGNORECASE)
        end_re   = re.compile(r"item\s*7\s*a[\s\.\-\u2014]|item\s*8[\s\.\-\u2014]", re.IGNORECASE)
        item_label = "Item 7"
    else:
        start_re = re.compile(r"item\s*2[\s\.\-\u2014\u2013]", re.IGNORECASE)
        end_re   = re.compile(r"item\s*3[\s\.\-\u2014]|item\s*4[\s\.\-\u2014]", re.IGNORECASE)
        item_label = "Item 2"

    text = _bs4_clean(html)

    # Collect every position where the start pattern appears
    start_matches = list(start_re.finditer(text))
    if not start_matches:
        raise ValueError(
            f"Could not locate '{item_label}' in this filing. "
            "The layout may be unusual — try pasting the text manually."
        )

    # Anything shorter is a TOC entry or a stray heading — not the real section
    MIN_MDA_CHARS = 1_500

    for m_start in start_matches:
        pos = m_start.start()
        # Look for the end marker at least 500 chars after the start
        m_end = end_re.search(text, pos + 500)
        end_pos = m_end.start() if m_end else len(text)

        candidate = re.sub(r"\n{3,}", "\n\n", text[pos:end_pos]).strip()

        if len(candidate) >= MIN_MDA_CHARS:
            return candidate

    raise ValueError(
        f"Found '{item_label}' markers but could not extract a full MD&A section — "
        "all candidates were too short (the filing may use image-based text). "
        "Try pasting the MD&A text manually."
    )


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


_SEC_UA   = "StockFinancialModel/1.0 research@example.com"
_SEC_HDRS = {"User-Agent": _SEC_UA}


def _bs4_clean(html: str) -> str:
    """
    Convert SEC filing HTML (including iXBRL) to clean plain text.

    iXBRL (inline XBRL) wraps financial data in custom tags like <ix:nonfraction>.
    We strip the metadata sections and unwrap the data tags so the underlying
    text is preserved for NLP analysis.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Pass 1: remove iXBRL metadata blocks entirely (no readable content)
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        if tag.name and tag.name.lower() in {
            "ix:header", "ix:references", "ix:resources", "ix:hidden",
        }:
            tag.decompose()

    # Pass 2: unwrap remaining ix: data tags (still in the tree after pass 1)
    # ix:nonfraction / ix:nonnumeric etc wrap real text — keep the text, drop the tag
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        if tag.parent is not None:   # guard: skip any orphaned tags
            tag.unwrap()

    # Remove other noise
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    # Insert newlines at structural boundaries
    for tag in soup.find_all(["p", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        tag.insert_before("\n")

    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_exhibit_991(cik: int, accession: str, acc_nodash: str) -> str | None:
    """
    Parse the SEC EDGAR filing index page to locate the Exhibit 99.1 document URL.

    The index HTML lists every file in the submission (main form + exhibits).
    We scan every table row for "EX-99" in the Type or Description column,
    then return the absolute URL of the first match.

    Returns None if the exhibit is not found or the index cannot be fetched.
    """
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}"
        f"/{acc_nodash}/{accession}-index.htm"
    )
    try:
        r = requests.get(index_url, headers=_SEC_HDRS, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            # Check every cell for "EX-99" (covers EX-99.1, EX-99.2, …)
            row_text = " ".join(c.get_text(strip=True) for c in cells).upper()
            if "EX-99" not in row_text:
                continue

            # Return the URL from the first hyperlink in this row
            link = row.find("a", href=True)
            if link:
                href = str(link["href"])
                if not href.startswith("http"):
                    href = "https://www.sec.gov" + href
                return href

    except Exception:
        pass

    return None


_MAX_8K_SCAN = 10   # how many recent 8-Ks to scan for an EX-99.1 exhibit


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_8k_sec(symbol: str) -> tuple[str, str]:
    """
    Fetch the Exhibit 99.1 (earnings press release) from a recent 8-K filing.

    Pipeline:
      1. Resolve ticker → CIK.
      2. Pull submissions JSON; iterate the last _MAX_8K_SCAN 8-K filings.
      3. For each, parse the filing index to look for an EX-99.1 exhibit.
         → Return the FIRST filing that contains one (usually the earnings 8-K).
      4. If none of the recent 8-Ks have EX-99.1, fall back to the most recent
         8-K primary document.

    Why scan multiple 8-Ks?
    Companies file many 8-Ks (compensation changes, board updates, etc.) that
    have no press release. The earnings 8-K — the one that matters — contains
    Exhibit 99.1 with hundreds of KB of financial narrative.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")

    cik = sec_cik_map().get(sym)
    if cik is None:
        raise ValueError(f"Ticker '{sym}' not found in SEC EDGAR.")

    cik10 = f"{cik:010d}"
    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers=_SEC_HDRS, timeout=20,
    )
    sub.raise_for_status()
    recent = (sub.json().get("filings") or {}).get("recent") or {}

    forms = recent.get("form")            or []
    accs  = recent.get("accessionNumber") or []
    docs  = recent.get("primaryDocument") or []
    dates = recent.get("filingDate")      or []

    # Collect up to _MAX_8K_SCAN recent 8-K entries
    candidates: list[tuple[str, str, str]] = []   # (date, accession, primary_doc)
    for form, acc, doc, date in zip(forms, accs, docs, dates):
        if str(form).strip().upper() == "8-K":
            candidates.append((str(date), str(acc), str(doc)))
        if len(candidates) >= _MAX_8K_SCAN:
            break

    if not candidates:
        raise ValueError(f"No recent 8-K filings found for '{sym}'.")

    # ── Scan for first 8-K that contains an EX-99.1 ───────────────────────
    ex991_url: str | None = None
    chosen_date: str = candidates[0][0]
    chosen_acc:  str = candidates[0][1]
    chosen_doc:  str = candidates[0][2]

    for date, acc, doc in candidates:
        acc_nd = acc.replace("-", "")
        url = _find_exhibit_991(cik, acc, acc_nd)
        if url:
            ex991_url    = url
            chosen_date  = date
            chosen_acc   = acc
            chosen_doc   = doc
            break

    acc_nodash = chosen_acc.replace("-", "")

    if ex991_url:
        r = requests.get(ex991_url, headers=_SEC_HDRS, timeout=60)
        r.raise_for_status()
        text  = _bs4_clean(r.text)
        label = f"{sym} 8-K {chosen_date} — Exhibit 99.1 (Earnings Press Release)"
    else:
        # Fallback: primary document of the most recent 8-K
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}"
            f"/{acc_nodash}/{chosen_doc}"
        )
        r = requests.get(doc_url, headers=_SEC_HDRS, timeout=60)
        r.raise_for_status()
        text  = _bs4_clean(r.text)
        label = f"{sym} 8-K {chosen_date} (no Exhibit 99.1 in last {_MAX_8K_SCAN} filings)"

    if len(text) < 300:
        raise ValueError(
            "The downloaded document is too short to analyse. "
            "The filing may not contain a text-based press release."
        )

    return label, text


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_mda_sec(symbol: str, filing_type: str = "10-K") -> tuple[str, str]:
    """
    Fetch the MD&A section (Item 7 for 10-K, Item 2 for 10-Q) from SEC EDGAR.
    MD&A carries legal liability — management cannot hide risks or exaggerate here.
    Returns (label, mda_text).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")

    cik = sec_cik_map().get(sym)
    if cik is None:
        raise ValueError(f"Ticker '{sym}' not found in SEC EDGAR mapping.")

    cik10 = f"{cik:010d}"

    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers=_SEC_HDRS, timeout=30,
    )
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
    label = f"{sym} {filing_type} {filing_date} — MD&A"
    return label, mda_text


HEDGE_WORDS = [
    "maybe",
    "might",
    "could",
    "possibly",
    "perhaps",
    "likely",
    "unlikely",
    "approximately",
    "around",
    "somewhat",
    "kind of",
    "sort of",
    "we think",
    "we believe",
    "we expect",
    "we hope",
    "we plan",
    "we aim",
    "guidance",
]
CERTAINTY_WORDS = [
    "will",
    "definitely",
    "certainly",
    "clearly",
    "confident",
    "strong",
    "committed",
    "always",
    "never",
    "proven",
    "guarantee",
]


def confidence_index(text: str) -> float:
    """
    Heuristic 0–100 confidence index from language.
    Higher when certainty words outweigh hedge words.
    """
    t = (text or "").lower()
    if not t.strip():
        return float("nan")
    hedge = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in HEDGE_WORDS)
    sure = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in CERTAINTY_WORDS)
    total = hedge + sure
    if total == 0:
        return 50.0
    score = 50.0 + 50.0 * (sure - hedge) / float(total)
    return float(max(0.0, min(100.0, score)))


def _render_analysis(transcript_text: str, ticker_q: str) -> None:
    """Full NLP analysis panel for one filing's text."""
    if not (transcript_text or "").strip():
        return

    growth_kw = list(GROWTH_KEYWORDS)
    risk_kw   = list(RISK_KEYWORDS)

    parts    = split_prepared_vs_qa(transcript_text)
    qa_focus = parts.qa.strip() if (parts.qa or "").strip() else parts.full_text.strip()
    ci       = confidence_index(qa_focus)

    kw_growth    = count_keywords(qa_focus, growth_kw)
    kw_risk      = count_keywords(qa_focus, risk_kw)
    growth_total = int(sum(kw_growth.values()))
    risk_total   = int(sum(kw_risk.values()))

    weighted_total, weighted_df = weighted_keyword_score(
        qa_focus, growth_kw, _KEYWORD_CATEGORIES, CATEGORY_WEIGHTS
    )
    risk_weighted_total, risk_weighted_df = weighted_keyword_score(
        qa_focus, risk_kw, _PESSIMISTIC_KEYWORD_CATEGORIES, PESSIMISTIC_CATEGORY_WEIGHTS
    )
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
    with c1:
        st.metric("Full sentiment",     f"{avg_full:.3f}" if np.isfinite(avg_full) else "—")
    with c2:
        st.metric("Prepared sentiment", f"{avg_pre:.3f}"  if np.isfinite(avg_pre)  else "—")
    with c3:
        st.metric("Q&A sentiment",      f"{avg_qa:.3f}"   if np.isfinite(avg_qa)   else "—")
    with c4:
        st.metric("Optimistic hits",    f"{growth_total:,}")
    with c5:
        st.metric("Risk hits",          f"{risk_total:,}")
    with c6:
        bull_delta = f"+{net_score:.1f}" if net_score >= 0 else f"{net_score:.1f}"
        st.metric(
            "Net Bullish Score 🎯",
            f"{net_score:+.1f}",
            delta=bull_delta,
            delta_color="normal",
            help="Weighted Optimism − Weighted Risk.  "
                 "Positive = bullish tone dominant; Negative = bearish language dominates.",
        )
    with c7:
        st.metric(
            "Optimism ⚡", f"{weighted_total:.1f}",
            help="Σ (hit × weight): Outperformance ×3.0 · Innovation ×2.5 · Momentum ×2.0 · Conviction ×1.5 · Operations ×1.5",
        )

    st.caption(
        f"**Confidence index (Q&A language):** {ci:.0f}/100"
        if np.isfinite(ci)
        else "**Confidence index (Q&A language):** —"
    )
    st.caption(
        f"Label: **{ticker_q}** • "
        f"Sentences: {len(df_full):,} • Prepared: {len(df_pre):,} • Q&A: {len(df_qa):,}"
    )

    st.subheader("2) Keyword frequency & weighted score")
    tab_weighted, tab_freq = st.tabs(["⚡ Weighted by category", "📊 Raw frequency"])

    with tab_freq:
        kw_df = pd.DataFrame(
            [{"keyword": k, "count": v, "group": "Optimistic"} for k, v in kw_growth.items()]
            + [{"keyword": k, "count": v, "group": "Risk"} for k, v in kw_risk.items()]
        ).sort_values(["group", "count"], ascending=[True, False])
        kw_df = kw_df[kw_df["count"] > 0]
        if kw_df.empty:
            st.info("No keyword hits found in this text.")
        else:
            fig_kw = px.bar(
                kw_df, x="count", y="keyword", color="group",
                orientation="h", title="Keyword Frequency",
                height=max(300, len(kw_df) * 22),
                color_discrete_map={"Optimistic": "#2ecc71", "Risk": "#e74c3c"},
            )
            fig_kw.update_layout(yaxis_title="", xaxis_title="Count", legend_title="")
            st.plotly_chart(fig_kw, use_container_width=True)

    with tab_weighted:
        st.markdown(
            f"**Net Bullish Score = {weighted_total:.1f} (Optimism) − "
            f"{risk_weighted_total:.1f} (Risk) = "
            f"{'🟢' if net_score >= 0 else '🔴'} {net_score:+.1f}**"
        )
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("#### 🟢 Optimism by Category")
            if weighted_df.empty:
                st.info("No optimistic keyword hits.")
            else:
                bull_cat = (
                    weighted_df.groupby("category")["score"]
                    .sum().reset_index().sort_values("score", ascending=True)
                )
                bull_color_map = {
                    "Financial Outperformance": "#e74c3c",
                    "Innovation & Moat":        "#9b59b6",
                    "Market Momentum & Demand": "#3498db",
                    "Management Conviction":    "#f39c12",
                    "Operational Excellence":   "#2ecc71",
                    "General":                  "#95a5a6",
                }
                fig_bull = px.bar(
                    bull_cat, x="score", y="category", orientation="h",
                    color="category", color_discrete_map=bull_color_map,
                    title=f"Optimism total = {weighted_total:.1f}", height=280,
                )
                fig_bull.update_layout(showlegend=False, yaxis_title="", xaxis_title="Score")
                st.plotly_chart(fig_bull, use_container_width=True)
        with col_bear:
            st.markdown("#### 🔴 Risk by Category")
            if risk_weighted_df.empty:
                st.info("No risk keyword hits.")
            else:
                bear_cat = (
                    risk_weighted_df.groupby("category")["score"]
                    .sum().reset_index().sort_values("score", ascending=True)
                )
                bear_color_map = {
                    "Financial Miss & Contraction": "#c0392b",
                    "Operational Bottlenecks":      "#e67e22",
                    "Macro & External Headwinds":   "#d35400",
                    "Management Code Words":        "#f39c12",
                    "General":                      "#95a5a6",
                }
                fig_bear = px.bar(
                    bear_cat, x="score", y="category", orientation="h",
                    color="category", color_discrete_map=bear_color_map,
                    title=f"Risk total = {risk_weighted_total:.1f}", height=280,
                )
                fig_bear.update_layout(showlegend=False, yaxis_title="", xaxis_title="Score")
                st.plotly_chart(fig_bear, use_container_width=True)
        with st.expander("Optimistic keyword detail", expanded=False):
            if not weighted_df.empty:
                st.dataframe(
                    weighted_df.style.format({"weight": "{:.1f}", "score": "{:.1f}"}),
                    width="stretch", hide_index=True,
                )
        with st.expander("Risk keyword detail (negation-corrected)", expanded=False):
            if not risk_weighted_df.empty:
                st.dataframe(
                    risk_weighted_df.style.format({"weight": "{:.1f}", "score": "{:.1f}"}),
                    width="stretch", hide_index=True,
                )

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
            st.dataframe(show, width="stretch", hide_index=True)

    with st.expander("Parsed sections (Prepared vs Q&A)", expanded=False):
        st.markdown("#### Prepared remarks")
        st.text(parts.prepared[:20_000] if parts.prepared else "")
        st.markdown("#### Q&A")
        st.text(parts.qa[:20_000] if parts.qa else "")


def _fetch_into_state(cache_key: str, fetch_fn) -> None:
    """Fetch once and store in session_state; show spinner while loading."""
    if cache_key not in st.session_state:
        with st.spinner("Fetching from SEC EDGAR…"):
            try:
                label, txt = fetch_fn()
                st.session_state[cache_key] = (label, txt)
            except Exception as e:
                st.error(str(e))


def main() -> None:
    st.title("Earnings Call Analyzer")
    st.caption(
        "Select a filing tab — data is fetched automatically from SEC EDGAR. "
        "Tracks keyword frequency, weighted sentiment score, and sentence-level VADER analysis."
    )

    # ── Sidebar: ticker + optional manual text ─────────────────────────────
    with st.sidebar:
        st.header("Input")
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()

        st.divider()
        with st.expander("📎 Manual text override", expanded=False):
            manual_src = st.radio("Source", ["Paste text", "Upload .txt"], key="manual_src")
            manual_text = ""
            if manual_src == "Paste text":
                manual_text = st.text_area(
                    "Paste transcript / MD&A",
                    height=220,
                    placeholder="Paste text here…",
                    key="manual_paste",
                )
            else:
                uploaded = st.file_uploader("Upload .txt", type=["txt"])
                if uploaded is not None:
                    manual_text = _safe_decode(uploaded)

    # ── Filing-type tabs ────────────────────────────────────────────────────
    tab_8k, tab_10k, tab_10q = st.tabs([
        "📰  8-K  Earnings Release",
        "📊  10-K  Annual MD&A",
        "📋  10-Q  Quarterly MD&A",
    ])

    # Helper: resolve text — always returns (label, txt)
    # manual override wins over any EDGAR cache
    def _resolve(cache_key: str) -> tuple[str, str]:
        if (manual_text or "").strip():
            return ticker or "Manual Input", manual_text.strip()
        if cache_key in st.session_state:
            return st.session_state[cache_key]  # stored as (label, txt)
        return "", ""

    with tab_8k:
        st.caption(
            "**8-K** — Current event report filed within 4 business days. "
            "Contains the earnings press release (Exhibit 99.1) and financial tables."
        )
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
        st.caption(
            "**10-K MD&A** — Annual filing. Management's Discussion & Analysis carries "
            "legal liability; more rigorous signal than a live earnings call."
        )
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
        st.caption(
            "**10-Q MD&A** — Quarterly filing. Faster cadence than 10-K; "
            "captures intra-year pivots in management tone."
        )
        if ticker:
            ck = f"edgar_10q_{ticker}"
            _fetch_into_state(ck, lambda: fetch_mda_sec(ticker, "10-Q"))
            lbl, txt = _resolve(ck)
            if lbl:
                st.success(f"Loaded: {lbl}")
            _render_analysis(txt, lbl)
        else:
            st.info("Enter a ticker in the sidebar to auto-fetch.")


main()

