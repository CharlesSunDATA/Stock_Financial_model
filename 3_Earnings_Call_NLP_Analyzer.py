"""
Earnings Call NLP Analyzer — Streamlit page module.

Primary input is manual transcript upload (.txt).
Optional: OpenAI API for LLM summary.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
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


def llm_summary_gemini(api_key: str, model: str, text: str) -> str:
    """
    Google Gemini summary via Generative Language API (HTTP), no extra deps.
    """
    api_key = (api_key or "").strip()
    model = (model or "").strip() or "gemini-2.0-flash"
    if not api_key:
        raise ValueError("Google API key is required.")

    prompt = (
        "You are a senior Wall Street NLP analyst.\n"
        "Given the following earnings call transcript, return:\n"
        "1) Three business highlights (bullet list)\n"
        "2) Three potential risks (bullet list)\n"
        "Be concise and avoid hallucinating numbers. If info is missing, say so.\n\n"
        "Return Markdown only.\n\n"
        "Transcript:\n"
        f"{text}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    # Retry on 429 / transient 5xx. Never include the API key in exceptions.
    data: dict[str, Any] | None = None
    for attempt in range(5):
        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload,
                timeout=90,
            )
            if r.status_code == 429 or (500 <= int(r.status_code) <= 599):
                wait_s = min(16.0, 1.0 * (2**attempt))
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt < 4:
                wait_s = min(16.0, 1.0 * (2**attempt))
                time.sleep(wait_s)
                continue
            raise RuntimeError("Gemini request failed (rate limited or network error). Please try again shortly.") from e

    if data is None:
        raise RuntimeError("Gemini request failed (rate limited). Please try again shortly.")

    cands = data.get("candidates") or []
    if not cands:
        return ""
    content = (cands[0].get("content") or {}).get("parts") or []
    out = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return out.strip()


@st.cache_data(show_spinner=False, ttl=60 * 15)
def cached_llm_summary(provider: str, api_key: str, model: str, text: str) -> str:
    # Cache only on a short prefix so we don't store giant blobs in the cache key.
    snippet = (text or "")[:20_000]
    if provider.startswith("Google"):
        return llm_summary_gemini(api_key, model, snippet)
    return llm_summary_openai(api_key, model, snippet)


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
        raise ValueError(
            "Seeking Alpha: parsed text too short. The page is likely paywalled, requires login, or blocks scraping. "
            "Try copying the Q&A text into **Paste text** or upload a .txt instead."
        )

    title = soup.title.get_text(strip=True) if soup.title else "Motley Fool transcript"
    label = re.sub(r"\s+", " ", title).strip()
    return label, text


def _sa_fetch_by_article_id(art_id: str, ua: str) -> tuple[str, str] | None:
    """Fetch transcript via Seeking Alpha internal API using article id."""
    api_url = f"https://seekingalpha.com/api/v3/articles/{art_id}"
    try:
        rj = requests.get(
            api_url,
            headers={"User-Agent": ua, "Accept": "application/json"},
            timeout=60,
        )
        if rj.status_code != 200:
            return None
        js = rj.json()
        attrs = (js.get("data") or {}).get("attributes") or {}
        title = str(attrs.get("title") or "Seeking Alpha transcript").strip()
        html = str(attrs.get("content") or "")
        if html and len(html) >= 200:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text("\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) >= 400:
                return re.sub(r"\s+", " ", title).strip(), text
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=60 * 10)
def _sa_rss_find_transcript_url(sym: str, ua: str) -> str | None:
    """Scan the public Seeking Alpha transcripts RSS for the latest match for sym."""
    try:
        r = requests.get(
            "https://seekingalpha.com/sector/transcripts.xml",
            headers={"User-Agent": ua},
            timeout=30,
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for it in root.findall("./channel/item"):
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            if not link:
                continue
            if re.search(rf"\({re.escape(sym)}\)", title) and "Transcript" in title:
                return link
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=60 * 10)
def _motley_fool_search_transcript_url(sym: str, ua: str) -> str | None:
    """Search Motley Fool for the latest earnings call transcript for sym."""
    try:
        search_url = f"https://www.fool.com/earnings/call-transcripts/?search={sym}"
        r = requests.get(search_url, headers={"User-Agent": ua}, timeout=30)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        sym_lower = sym.lower()
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            if "earnings-call-transcript" in href.lower() and sym_lower in href.lower():
                if not href.startswith("http"):
                    href = "https://www.fool.com" + href
                return href
        # Broader fallback: first transcript link on the page
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            if "earnings-call-transcript" in href.lower():
                if not href.startswith("http"):
                    href = "https://www.fool.com" + href
                return href
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=60 * 15)
def search_and_fetch_transcript(ticker: str, ua: str) -> tuple[str, str]:
    """
    Multi-source auto-fetch:
      1. Seeking Alpha RSS → article API (no login needed, but only last ~20)
      2. Motley Fool search page scrape
    Raises ValueError with a clear message if nothing is found.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        raise ValueError("Please enter a ticker symbol.")

    errors: list[str] = []

    # ── Source 1: Seeking Alpha RSS ───────────────────────────────────────────
    try:
        sa_url = _sa_rss_find_transcript_url(sym, ua)
        if sa_url:
            # Extract article id from URL
            m = re.search(r"/article/(\d+)", sa_url)
            if m:
                result = _sa_fetch_by_article_id(m.group(1), ua)
                if result:
                    return result
    except Exception as e:
        errors.append(f"Seeking Alpha: {e}")

    # ── Source 2: Motley Fool ──────────────────────────────────────────────────
    try:
        mf_url = _motley_fool_search_transcript_url(sym, ua)
        if mf_url:
            label, txt = fetch_transcript_motley_fool_url(mf_url, ua)
            return label, txt
    except Exception as e:
        errors.append(f"Motley Fool: {e}")

    detail = "; ".join(errors) if errors else "ticker not found in recent RSS or Motley Fool search"
    raise ValueError(
        f"No recent earnings call transcript found for **{sym}**. "
        f"({detail})\n\n"
        "**Tip:** Use **Paste text** or **Upload .txt** to paste the transcript manually."
    )


_KOYFIN_TICKERS = "https://app.koyfin.com/api/v3/tickers"
_KOYFIN_PUBHUB = "https://app.koyfin.com/api/v1/pubhub"
_KOYFIN_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://app.koyfin.com",
    "Referer": "https://app.koyfin.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


@st.cache_data(show_spinner=False, ttl=60 * 60)
def koyfin_search_ticker(ticker: str) -> str | None:
    """Resolve ticker → Koyfin KID (e.g. 'NVDA' → 'eq-212q1o'). No auth needed."""
    sym = ticker.strip().upper()
    r = requests.post(
        f"{_KOYFIN_TICKERS}/search",
        json={"q": sym, "limit": 20},
        headers=_KOYFIN_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    for item in (r.json().get("data") or []):
        if item.get("ticker", "").upper() == sym and item.get("category") == "Equity":
            return str(item["KID"])
    return None


@st.cache_data(show_spinner=False, ttl=60 * 15)
def koyfin_transcript_list(kid: str) -> list[dict]:
    """Fetch all transcript events for a KID. No auth needed."""
    r = requests.get(
        f"{_KOYFIN_PUBHUB}/transcript/list/{kid}",
        headers=_KOYFIN_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return r.json() or []


def _koyfin_paragraphs_to_text(content: object) -> str:
    """Recursively flatten the Koyfin transcript content tree into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(_koyfin_paragraphs_to_text(c) for c in content if c)
    if isinstance(content, dict):
        # Common keys: 'text', 'body', 'paragraphs', 'content', 'sections'
        for key in ("text", "body", "content", "paragraphs", "sections"):
            val = content.get(key)
            if val:
                return _koyfin_paragraphs_to_text(val)
        # Try all string values as last resort
        parts = [str(v) for v in content.values() if isinstance(v, str) and len(v) > 20]
        return "\n\n".join(parts)
    return str(content) if content else ""


@st.cache_data(show_spinner=False, ttl=60 * 30)
def koyfin_fetch_transcript_content(event_id: str, auth_token: str) -> tuple[str, str]:
    """
    Fetch full transcript for a Koyfin event_id (keyDevId).
    Requires an auth_token from your logged-in Koyfin session.

    How to get your auth token:
      1. Log in to app.koyfin.com in Chrome.
      2. Press F12 → Application → Local Storage → https://app.koyfin.com
      3. Find the key 'auth_token' and copy its value.
    """
    if not auth_token:
        raise ValueError("auth_token is required for fetching transcript content from Koyfin.")

    headers = {
        **_KOYFIN_HEADERS,
        "Authorization": f"Bearer {auth_token.strip()}",
    }
    r = requests.get(
        f"{_KOYFIN_PUBHUB}/v2/transcript/{event_id}",
        headers=headers,
        timeout=30,
    )
    if r.status_code == 401:
        raise ValueError(
            "Koyfin returned 401 Unauthorized. Your auth token may be expired. "
            "Log in again and copy a fresh token from DevTools → Application → "
            "Local Storage → auth_token."
        )
    r.raise_for_status()
    data = r.json()

    # Extract title
    header = data.get("header") or {}
    title = str(header.get("title") or header.get("formattedTitle") or f"Koyfin transcript {event_id}")

    # Extract body – try known keys
    raw = None
    for key in ("body", "content", "paragraphs", "sections", "transcript"):
        raw = data.get(key)
        if raw:
            break
    if raw is None:
        # Dump everything except header
        raw = {k: v for k, v in data.items() if k != "header"}

    text = _koyfin_paragraphs_to_text(raw).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 200:
        raise ValueError(
            f"Koyfin returned a very short transcript body ({len(text)} chars). "
            "The content structure may have changed. "
            "Try the 'Paste text' source instead."
        )
    return title.strip(), text


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_tikr_transcript(url: str, cookie: str) -> tuple[str, str]:
    """
    Fetch an earnings call transcript from TIKR using a browser session cookie.

    TIKR is a Next.js SPA; logged-in pages embed all initial data in a
    <script id="__NEXT_DATA__"> tag, which we parse directly – no JavaScript
    execution required, as long as the cookie is valid.

    How to get your cookie:
      1. Log in to app.tikr.com in Chrome/Firefox.
      2. Open DevTools (F12) → Network tab.
      3. Reload the page, click any request to app.tikr.com, and copy the
         full value of the "Cookie:" request header.
      4. Paste that value into the Cookie field below.
    """
    import json as _json

    url = (url or "").strip()
    cookie = (cookie or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("Please paste the full TIKR transcript URL.")
    if not cookie:
        raise ValueError(
            "A session cookie is required. See instructions below the input field."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Cookie": cookie,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://app.tikr.com/",
    }

    # ── 1. Try TIKR's internal transcript API (reverse-engineered) ─────────
    from urllib.parse import urlparse, parse_qs
    params = parse_qs(urlparse(url).query)
    tid = (params.get("tid") or [""])[0]
    cid = (params.get("cid") or [""])[0]

    if tid:
        for api_url in [
            f"https://app.tikr.com/api/transcript?tid={tid}",
            f"https://app.tikr.com/api/transcripts/{tid}",
            f"https://api.tikr.com/transcript?tid={tid}&cid={cid}",
        ]:
            try:
                rj = requests.get(
                    api_url,
                    headers={**headers, "Accept": "application/json"},
                    timeout=30,
                )
                if rj.status_code == 200:
                    data = rj.json()
                    # Common shapes: {"transcript": "..."} or {"content": [...]} or {"body": "..."}
                    raw = (
                        data.get("transcript")
                        or data.get("content")
                        or data.get("body")
                        or data.get("text")
                    )
                    if isinstance(raw, str) and len(raw) > 400:
                        title = str(data.get("title") or data.get("name") or f"TIKR transcript (tid={tid})")
                        return title.strip(), raw.strip()
                    if isinstance(raw, list):
                        # Array of sections/paragraphs
                        pieces: list[str] = []
                        for item in raw:
                            if isinstance(item, str):
                                pieces.append(item)
                            elif isinstance(item, dict):
                                pieces.append(
                                    item.get("text") or item.get("content") or item.get("body") or ""
                                )
                        combined = "\n\n".join(p for p in pieces if p.strip())
                        if len(combined) > 400:
                            title = str(data.get("title") or f"TIKR transcript (tid={tid})")
                            return title.strip(), combined.strip()
            except Exception:
                continue

    # ── 2. Fetch the HTML page and extract __NEXT_DATA__ ───────────────────
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code in (401, 403):
        raise ValueError(
            "TIKR returned 'Unauthorized' – your cookie may have expired or be incomplete. "
            "Please copy a fresh Cookie header from DevTools."
        )
    if r.status_code != 200:
        raise ValueError(f"TIKR returned HTTP {r.status_code}. Check the URL and cookie.")

    soup = BeautifulSoup(r.text, "html.parser")

    # Check for login redirect (page title or redirect marker)
    page_title = soup.title.get_text(strip=True) if soup.title else ""
    if any(kw in page_title.lower() for kw in ("sign in", "login", "log in")):
        raise ValueError(
            "TIKR redirected to the login page – your cookie is expired or invalid. "
            "Please copy a fresh Cookie header from DevTools after logging in."
        )

    # Parse __NEXT_DATA__
    nd_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if nd_tag and nd_tag.string:
        try:
            nd = _json.loads(nd_tag.string)
            page_props = nd.get("props", {}).get("pageProps", {})
            # Walk common key names used by TIKR
            for key in ("transcript", "transcriptData", "data", "earningsCall"):
                obj = page_props.get(key)
                if not obj:
                    continue
                if isinstance(obj, str) and len(obj) > 400:
                    label = page_title or f"TIKR transcript (tid={tid})"
                    return label, re.sub(r"\n{3,}", "\n\n", obj).strip()
                if isinstance(obj, dict):
                    raw = (
                        obj.get("body")
                        or obj.get("content")
                        or obj.get("text")
                        or obj.get("transcript")
                    )
                    if isinstance(raw, str) and len(raw) > 400:
                        label = str(obj.get("title") or page_title or f"TIKR transcript (tid={tid})")
                        return label.strip(), re.sub(r"\n{3,}", "\n\n", raw).strip()
                    if isinstance(raw, list):
                        combined = "\n\n".join(
                            (item if isinstance(item, str) else item.get("text") or item.get("content") or "")
                            for item in raw
                        )
                        if len(combined) > 400:
                            label = str(obj.get("title") or page_title or f"TIKR transcript (tid={tid})")
                            return label.strip(), combined.strip()
        except Exception:
            pass

    # ── 3. Last-resort: dump visible text from <main> / <article> ─────────
    node = soup.find("main") or soup.find("article") or soup.body
    if node:
        text = node.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 800:
            label = page_title or f"TIKR transcript (tid={tid or 'unknown'})"
            return label.strip(), text

    raise ValueError(
        "Could not extract the transcript from the TIKR page. "
        "The page may require JavaScript rendering beyond what static fetching supports. "
        "As a workaround, open the transcript on TIKR, select all text (Ctrl+A), copy, "
        "and use the **Paste text** source."
    )


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
            [
                "Paste text",
                "Upload .txt",
                "Auto-fetch by ticker",
                "Koyfin (auth token)",
                "TIKR (cookie login)",
                "Auto-fetch (SEC EDGAR - best effort)",
            ],
            index=2,
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
                    try:
                        label, txt = fetch_latest_transcript_sec_best_effort(sym, ua)
                        st.session_state["sec_label"] = label
                        st.session_state["sec_text"] = txt
                    except Exception as e:
                        st.error(str(e))
            if "sec_text" in st.session_state:
                transcript_text = str(st.session_state.get("sec_text", ""))
                ticker_q = st.session_state.get("sec_label", ticker_q)
                st.success(f"Loaded: {ticker_q}")
                with st.expander("Preview fetched text", expanded=False):
                    st.text(transcript_text[:20_000])
        elif source == "Koyfin (auth token)":
            st.caption("Fetch the latest earnings call transcript from Koyfin. Ticker lookup is free; full content requires your auth token.")
            kf_ticker = st.text_input("Ticker symbol", value="NVDA", key="kf_ticker").strip().upper()
            kf_token = st.text_input(
                "Koyfin auth token",
                value="",
                type="password",
                placeholder="Paste auth_token from DevTools",
                help=(
                    "Log in at app.koyfin.com → F12 → Application → Local Storage "
                    "→ https://app.koyfin.com → auth_token"
                ),
            )
            kf_event_type = st.selectbox(
                "Event type",
                ["Earnings Calls", "Shareholder/Analyst Calls", "Any"],
                index=0,
            )
            with st.expander("How to get your Koyfin auth token", expanded=False):
                st.markdown(
                    """
1. Log in to [app.koyfin.com](https://app.koyfin.com).
2. Press **F12** to open DevTools.
3. Go to **Application** tab → **Local Storage** → `https://app.koyfin.com`.
4. Find the row with key **`auth_token`** and copy its value.

The token is only sent to Koyfin's own API and is never stored.
                    """
                )
            if st.button("Fetch latest transcript", type="primary", width="stretch", key="kf_fetch"):
                with st.spinner(f"Fetching Koyfin transcript for {kf_ticker}…"):
                    try:
                        kid = koyfin_search_ticker(kf_ticker)
                        if not kid:
                            st.error(f"Ticker **{kf_ticker}** not found on Koyfin.")
                        else:
                            events = koyfin_transcript_list(kid)
                            # Filter by event type
                            if kf_event_type != "Any":
                                filtered = [e for e in events if e.get("eventType") == kf_event_type]
                            else:
                                filtered = events
                            if not filtered:
                                st.warning(
                                    f"No '{kf_event_type}' transcripts found for {kf_ticker}. "
                                    f"Total events available: {len(events)}. "
                                    f"Try changing the Event type filter."
                                )
                            else:
                                latest = filtered[0]
                                event_id = str(latest["keyDevId"])
                                default_title = latest.get("transcriptTitle") or latest.get("formattedTitle") or f"{kf_ticker} transcript"
                                if not kf_token:
                                    st.info(
                                        f"Found: **{default_title}** (event ID: {event_id})\n\n"
                                        "Paste your **auth token** above and click again to fetch the full text."
                                    )
                                else:
                                    label, txt = koyfin_fetch_transcript_content(event_id, kf_token)
                                    st.session_state["kf_label"] = label
                                    st.session_state["kf_text"] = txt
                    except Exception as e:
                        st.error(str(e))
            if "kf_text" in st.session_state:
                transcript_text = str(st.session_state.get("kf_text", ""))
                ticker_q = st.session_state.get("kf_label", ticker_q)
                st.success(f"Loaded: {ticker_q}")
                with st.expander("Preview fetched text", expanded=False):
                    st.text(transcript_text[:20_000])
        elif source == "TIKR (cookie login)":
            st.caption("Fetch a specific transcript from TIKR using your account session.")
            tikr_url = st.text_input(
                "TIKR transcript URL",
                value="",
                placeholder="https://app.tikr.com/stock/transcript?cid=...&tid=...&e=...&ts=...",
                help="Copy the URL from your TIKR browser tab.",
            )
            tikr_cookie = st.text_input(
                "Session cookie",
                value="",
                type="password",
                placeholder="Paste the full Cookie: header value from DevTools",
            )
            with st.expander("How to get your TIKR cookie", expanded=False):
                st.markdown(
                    """
1. Log in to [app.tikr.com](https://app.tikr.com) and open the transcript page you want.
2. Press **F12** (or right-click → Inspect) to open DevTools.
3. Go to the **Network** tab, then reload the page.
4. Click any request to `app.tikr.com`, scroll to **Request Headers**.
5. Find the **Cookie:** line — copy its entire value and paste it above.

Your cookie is only stored in this browser session and is never sent anywhere except to TIKR.
                    """
                )
            if st.button("Fetch transcript", type="primary", width="stretch", key="tikr_fetch"):
                with st.spinner("Fetching from TIKR…"):
                    try:
                        label, txt = fetch_tikr_transcript(tikr_url, tikr_cookie)
                        st.session_state["tikr_label"] = label
                        st.session_state["tikr_text"] = txt
                    except Exception as e:
                        st.error(str(e))
            if "tikr_text" in st.session_state:
                transcript_text = str(st.session_state.get("tikr_text", ""))
                ticker_q = st.session_state.get("tikr_label", ticker_q)
                st.success(f"Loaded: {ticker_q}")
                with st.expander("Preview fetched text", expanded=False):
                    st.text(transcript_text[:20_000])
        elif source == "Auto-fetch by ticker":
            st.caption(
                "Enter a ticker to auto-fetch the latest earnings call transcript.\n\n"
                "Sources tried in order: **Seeking Alpha** (public RSS) → **Motley Fool** (search)."
            )
            at_ticker = st.text_input("Ticker symbol", value="NVDA", key="at_ticker").strip().upper()
            if st.button("Fetch latest transcript", type="primary", width="stretch", key="at_fetch"):
                with st.spinner(f"Searching for latest earnings call transcript for {at_ticker}…"):
                    try:
                        label, txt = search_and_fetch_transcript(at_ticker, "Mozilla/5.0")
                        st.session_state["at_label"] = label
                        st.session_state["at_text"] = txt
                    except Exception as e:
                        st.error(str(e))
            if "at_text" in st.session_state:
                transcript_text = str(st.session_state.get("at_text", ""))
                ticker_q = st.session_state.get("at_label", ticker_q)
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
        provider = st.radio("Provider", ["Google (Gemini)", "OpenAI"], index=0, horizontal=True)
        if provider.startswith("Google"):
            api_key = st.text_input("Google API key", type="password")
            model = st.text_input("Model", value="gemini-2.0-flash")
            gen_click = st.button("Generate AI summary", type="primary", width="stretch", key="llm_go")
        else:
            api_key = st.text_input("OpenAI API key", type="password")
            model = st.text_input("Model", value="gpt-4.1-mini")
            gen_click = st.button("Generate AI summary", type="primary", width="stretch", key="llm_go")

    if not (transcript_text or "").strip():
        st.info("Provide a transcript via paste/upload, or auto-fetch one to begin.")
        return

    parts = split_prepared_vs_qa(transcript_text)
    qa_focus = parts.qa.strip() if (parts.qa or "").strip() else parts.full_text.strip()
    ci = confidence_index(qa_focus)

    growth_kw = [k.strip() for k in growth_in.split(",") if k.strip()]
    risk_kw = [k.strip() for k in risk_in.split(",") if k.strip()]

    # Focus analysis on Q&A when available
    kw_growth = count_keywords(qa_focus, growth_kw)
    kw_risk = count_keywords(qa_focus, risk_kw)
    growth_total = int(sum(kw_growth.values()))
    risk_total = int(sum(kw_risk.values()))

    # Sentiment
    s_full = sentence_split(qa_focus)
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
        st.metric("Q&A sentiment", f"{avg_full:.3f}" if np.isfinite(avg_full) else "—")
    with c2:
        st.metric("Prepared sentiment", f"{avg_pre:.3f}" if np.isfinite(avg_pre) else "—")
    with c3:
        st.metric("Q&A sentiment", f"{avg_qa:.3f}" if np.isfinite(avg_qa) else "—")
    with c4:
        st.metric("Growth keywords", f"{growth_total:,}")
    with c5:
        st.metric("Risk keywords", f"{risk_total:,}")

    st.caption(
        f"**Confidence index (Q&A language):** {ci:.0f}/100"
        if np.isfinite(ci)
        else "**Confidence index (Q&A language):** —"
    )

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
    if gen_click:
        if not api_key.strip():
            st.error("Please provide an API key first.")
        else:
            calling = "Calling Gemini…" if provider.startswith("Google") else "Calling OpenAI…"
            with st.spinner(calling):
                try:
                    summary = cached_llm_summary(provider, api_key.strip(), model.strip(), parts.full_text)
                    st.session_state["llm_summary_last"] = summary
                    st.session_state["llm_summary_provider"] = provider
                    st.session_state["llm_summary_model"] = model
                except Exception as e:
                    st.error(str(e))

    last = st.session_state.get("llm_summary_last", "")
    if last:
        st.markdown(last)
    else:
        st.info("Click **Generate AI summary** to run the LLM once (results are cached briefly to avoid rate limits).")

    with st.expander("Parsed sections (Prepared vs Q&A)", expanded=False):
        st.markdown("#### Prepared remarks")
        st.text(parts.prepared[:20_000] if parts.prepared else "")
        st.markdown("#### Q&A")
        st.text(parts.qa[:20_000] if parts.qa else "")


main()

