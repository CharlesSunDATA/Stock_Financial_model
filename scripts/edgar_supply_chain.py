"""
edgar_supply_chain.py

Fetch major-customer relationships from SEC EDGAR 10-K filings and upsert
into the `suppliers` table (ticker=customer, supplier_symbol=filer).

Strategy
--------
For each ticker in the watchlist:
  1. Look up its CIK from the SEC company_tickers.json map.
  2. Fetch the latest 10-K accession via data.sec.gov/submissions/.
  3. Download the primary 10-K HTML document.
  4. Extract customer concentration sentences using two strategies:
     A. Verb-first: "[Company] … [verb] N% of net revenue"
     B. List form: "revenues from [X], [Y] and [Z] each comprised N%"
  5. Map company name → ticker via SEC name lookup + curated alias table.
  6. Upsert rows: (ticker=customer_ticker, supplier_symbol=filer_ticker).

Rate limit: SEC enforces ~10 req/s.  We default to 0.12 s/req.
Local cache at .cache/edgar/ avoids re-downloading for 30 days.

Usage (standalone)
------------------
  python3 scripts/edgar_supply_chain.py --watchlist-name quant_core_union
  python3 scripts/edgar_supply_chain.py --tickers SWKS,QCOM,AVGO --max-tickers 50

Usage (as library)
------------------
  from edgar_supply_chain import run_edgar_supply_chain
  rows = run_edgar_supply_chain(conn, tickers=["SWKS","QCOM"], ...)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEC_UA = "StockFinancialModel/1.0 research@example.com"
SEC_HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip"}
_SEC_RATE_S = 0.12          # ~8 req/s (under the 10/s limit)
_DOC_MAX_BYTES = 8_000_000  # skip filings > 8 MB (some are huge)
_CACHE_TTL_DAYS = 30        # re-download 10-K after this many days

# ---------------------------------------------------------------------------
# Regex patterns for customer concentration sentences
# ---------------------------------------------------------------------------
# Anchor: finds "X % of [net] revenue" (the percentage is always near the anchor)
_REVENUE_ANCHOR = re.compile(
    r'(\d{1,3}(?:\.\d+)?)\s*%\s+of\s+'
    r'(?:the\s+)?'
    r'(?:Company[\'s]*\s+|our\s+(?:total\s+)?|total\s+|combined\s+)?'
    r'(?:net\s+)?'
    r'(?:revenues?|sales|net\s+revenue)',
    re.IGNORECASE,
)
# Verbs that connect company name to the percentage (full-text scan, no $ anchor).
# Handles: "in the aggregate accounted for" and "accounted for approximately"
_VERB_RE = re.compile(
    r'(?:in the aggregate\s+)?'
    r'(?:accounted for|constituted|represented|comprised|generated)\s+'
    r'(?:(?:more than|at least|approximately)\s+)?',
    re.IGNORECASE,
)
# A valid company name token (starts with capital, ≥ 3 chars, no pure digit)
_COMPANY_NAME_RE = re.compile(
    r'^[A-Z][A-Za-z0-9][A-Za-z0-9\s,\.\-&\(\)\']{1,70}$'
)

# ---------------------------------------------------------------------------
# Curated alias map: normalised company name fragment → US ticker
# (covers common variations that differ from the SEC name field)
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta platforms": "META",
    "facebook": "META",
    "samsung": "",            # foreign – no US ticker; blank = skip
    "huawei": "",             # private
    "xiaomi": "",             # HK-listed; no direct US ticker
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "intel": "INTC",
    "intel corporation": "INTC",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "marvell": "MRVL",
    "skyworks": "SWKS",
    "qorvo": "QRVO",
    "cirrus logic": "CRUS",
    "texas instruments": "TXN",
    "analog devices": "ADI",
    "on semiconductor": "ON",
    "onsemi": "ON",
    "microchip technology": "MCHP",
    "lam research": "LRCX",
    "applied materials": "AMAT",
    "kla": "KLAC",
    "asml": "ASML",
    "cisco": "CSCO",
    "cisco systems": "CSCO",
    "motorola": "MSI",
    "ericsson": "ERIC",
    "nokia": "NOK",
    "hp": "HPQ",
    "hewlett": "HPQ",
    "dell": "DELL",
    "lenovo": "",             # HK-listed
    "sony": "SONY",
    "lg electronics": "",     # KR-listed
    "general electric": "GE",
    "honeywell": "HON",
    "ge": "GE",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "boeing": "BA",
    "northrop grumman": "NOC",
    "at&t": "T",
    "verizon": "VZ",
    "t-mobile": "TMUS",
    "comcast": "CMCSA",
    "netflix": "NFLX",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


def _cache_dir() -> Path:
    d = _project_root() / ".cache" / "edgar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# SEC helpers
# ---------------------------------------------------------------------------
def _sec_get(url: str, *, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


def load_cik_map(*, force_refresh: bool = False) -> dict[str, int]:
    """Download (and cache) the SEC ticker→CIK mapping. Returns {TICKER: cik_int}."""
    cache_file = _cache_dir() / "company_tickers.json"
    if not force_refresh and cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            return json.loads(cache_file.read_text())

    time.sleep(_SEC_RATE_S)
    data = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
    out: dict[str, int] = {}
    for row in data.values():
        t = str(row.get("ticker", "")).strip().upper()
        try:
            cik = int(row["cik_str"])
        except (KeyError, ValueError):
            continue
        if t:
            out[t] = cik
    cache_file.write_text(json.dumps(out))
    return out


def build_name_to_ticker(cik_map: dict[str, int]) -> dict[str, str]:
    """Build normalised company-name → ticker lookup from SEC company_tickers.json."""
    cache_file = _cache_dir() / "company_tickers_full.json"
    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            return json.loads(cache_file.read_text())

    time.sleep(_SEC_RATE_S)
    data = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
    name_to_ticker: dict[str, str] = {}
    for row in data.values():
        t = str(row.get("ticker", "")).strip().upper()
        name = str(row.get("title", "")).strip()
        if t and name:
            name_to_ticker[_norm_name(name)] = t
    cache_file.write_text(json.dumps(name_to_ticker))
    return name_to_ticker


def _norm_name(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|co\.?|plc\.?|n\.v\.?|s\.a\.?|ag|holdings?|group|international|technologies?)\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_name_to_ticker(
    name: str,
    name_to_ticker: dict[str, str],
) -> str:
    """Try to map a company name string to a US ticker symbol. Returns "" if not found."""
    norm = _norm_name(name)
    for alias, ticker in _ALIAS_MAP.items():
        if alias in norm or norm in alias:
            return ticker  # may be "" for foreign/private

    if norm in name_to_ticker:
        return name_to_ticker[norm]

    if len(norm) >= 8:
        for sec_name, ticker in name_to_ticker.items():
            if sec_name.startswith(norm[:8]) or norm.startswith(sec_name[:8]):
                return ticker

    return ""


# ---------------------------------------------------------------------------
# 10-K fetching
# ---------------------------------------------------------------------------
def _get_latest_10k_accession(cik: int) -> tuple[str, str, str] | None:
    """
    Returns (accession_number, filed_date, primary_doc_name) for the most recent 10-K, or None.
    """
    cik10 = str(cik).zfill(10)
    cache_file = _cache_dir() / f"subs_{cik10}.json"
    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            subs = json.loads(cache_file.read_text())
        else:
            subs = None
    else:
        subs = None

    if subs is None:
        time.sleep(_SEC_RATE_S)
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        try:
            subs = _sec_get(url).json()
        except Exception:
            return None
        cache_file.write_text(json.dumps(subs))

    filings = subs.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accns = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", []) or filings.get("filed", [])
    primary_docs = filings.get("primaryDocument", [])

    idx = next((i for i, f in enumerate(forms) if f == "10-K"), None)
    if idx is None:
        return None
    filed_date = dates[idx] if idx < len(dates) else ""
    primary_doc = primary_docs[idx] if idx < len(primary_docs) else ""
    return accns[idx], filed_date, primary_doc


def _get_primary_doc_url(cik: int, accession: str, primary_doc: str = "") -> str | None:
    """
    Return the direct URL for the primary 10-K HTML document.
    Uses primary_doc from the submissions JSON when available (no extra HTTP request).
    Falls back to parsing the filing index HTML.
    """
    acc_nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"

    if primary_doc and primary_doc.lower().endswith((".htm", ".html")):
        return f"{base}/{primary_doc}"

    idx_url = f"{base}/{accession}-index.htm"
    time.sleep(_SEC_RATE_S)
    try:
        r = _sec_get(idx_url)
    except Exception:
        return None

    if not _HAS_BS4:
        m = re.search(r'href="(/Archives/edgar/data/[^"]+\.htm)"[^>]*>\s*\w+-\d+\.htm', r.text)
        return f"https://www.sec.gov{m.group(1)}" if m else None

    soup = BeautifulSoup(r.text, "html.parser")
    for row in soup.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) >= 4 and tds[3].get_text(strip=True) == "10-K":
            link = tds[2].find("a", href=True)
            if link:
                href = link["href"]
                if "?doc=" in href:
                    href = href.split("?doc=")[-1]
                if not href.startswith("http"):
                    href = "https://www.sec.gov" + href.lstrip("/")
                return href
    return None


def _download_10k_text(cik: int, accession: str, doc_url: str) -> str | None:
    """Download and cache the 10-K HTML, return plain text."""
    acc_nodash = accession.replace("-", "")
    cache_file = _cache_dir() / f"10k_{cik}_{acc_nodash}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")

    time.sleep(_SEC_RATE_S)
    try:
        r = requests.get(doc_url, headers=SEC_HEADERS, timeout=90, stream=True)
        r.raise_for_status()
        content = b""
        for chunk in r.iter_content(chunk_size=65536):
            content += chunk
            if len(content) > _DOC_MAX_BYTES:
                break
    except Exception:
        return None

    if _HAS_BS4:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    else:
        text = re.sub(r"<[^>]+>", " ", content.decode("utf-8", errors="replace"))

    text = re.sub(r"\s+", " ", text)
    # Normalise Unicode quotation marks so regex patterns match reliably
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    cache_file.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Customer extraction
# ---------------------------------------------------------------------------
def extract_customers_from_text(text: str) -> list[dict[str, Any]]:
    """
    Two extraction strategies for customer concentration sentences:

    Strategy A (verb-first): "[Company] … [verb] N% of [net] revenue"
      e.g. "Apple accounted for 67% of the Company's net revenue"

    Strategy B (list form): "revenues from [X], [Y] and [Z] each [verb] N%"
      e.g. "revenues from Apple, Samsung and Xiaomi each comprised 10%"

    Returns deduplicated [{"name": str, "pct": float}].
    """
    found: dict[str, float] = {}
    _stop_words = {"the", "our", "this", "that", "each", "any", "fiscal", "total", "net",
                   "during", "for", "in", "by", "such"}
    _generic_words = {"sales", "revenue", "revenues", "distributor", "distributors",
                      "customer", "customers", "aggregate", "believe", "direct", "indirect",
                      "reseller", "resellers", "partner", "partners", "licensee", "licensees"}

    _post_pct_re = re.compile(
        r'(\d{1,3}(?:\.\d+)?)\s*%'
        r'(?:[^%]{0,80}?(?:\d{1,3}(?:\.\d+)?)\s*%)*'
        r"[^%]{0,80}?of\s+(?:the\s+)?(?:Company['s]*\s+|our\s+(?:total\s+)?|total\s+|consolidated\s+)?"
        r'(?:net\s+)?(?:revenues?|sales|net\s+revenue)',
        re.IGNORECASE,
    )

    def _is_valid_company_name(name: str) -> bool:
        if len(name) < 3 or len(name) > 80:
            return False
        if not name[0].isupper():
            return False
        if name.lower() in _stop_words:
            return False
        if re.match(
            r'^(?:fiscal|calendar|during|for|in|by|such|our|the|we|net|total|'
            r'sales|revenues?|direct|indirect|aggregate|each|no\s+single)\s',
            name, re.IGNORECASE,
        ):
            return False
        first_word = name.split()[0].rstrip(".,").lower()
        if first_word in _generic_words:
            return False
        name_lower = name.lower()
        generic_count = sum(1 for w in _generic_words if re.search(r'\b' + w + r'\b', name_lower))
        if generic_count >= 2:
            return False
        if name.isupper() and len(name) <= 4 and " " not in name:
            return False
        return True

    # --- Strategy A: verb-first ---
    for verb_m in _VERB_RE.finditer(text):
        look_back = max(0, verb_m.start() - 500)
        pre = text[look_back: verb_m.start()].strip()

        frags = re.split(r'(?<=[.!?])\s+', pre)
        candidate = frags[-1].strip() if frags else pre.strip()

        candidate = re.sub(
            r'^(?:During|In|For|Throughout|Each of)\s+'
            r'(?:(?:fiscal(?:\s+year)?|calendar\s+year|the\s+year(?:\s+ended)?)'
            r'(?:\s+\d{4}|\s+\(?\w+\s+\d{4}\)?)[\s,]*(?:and\s+)?){1,5}'
            r',?\s*',
            '',
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        comma_parts = re.split(r',\s*(?!(?:Inc|Corp|Ltd|LLC|Co|plc|N\.V|S\.A|AG)\b)', candidate)
        company_name = comma_parts[0].strip()

        if not _is_valid_company_name(company_name):
            continue

        post = text[verb_m.end(): verb_m.end() + 300]
        pct_m = _post_pct_re.search(post)
        if pct_m is None:
            continue
        try:
            pct = float(pct_m.group(1))
        except ValueError:
            continue
        if pct <= 0 or pct > 100:
            continue

        if company_name not in found or pct > found[company_name]:
            found[company_name] = pct

    # --- Strategy B: list form ---
    _list_pattern = re.compile(
        r'(?:revenues?|sales)\s+from\s+'
        r'([A-Z][A-Za-z0-9\s,\.\-&]+?)'
        r'\s+each\s+'
        r'(?:accounted for|represented|comprised|constituted)\s+'
        r'(?:approximately\s+)?(\d{1,3}(?:\.\d+)?)\s*%\s+or\s+more\s+of',
        re.IGNORECASE,
    )
    for m in _list_pattern.finditer(text):
        company_list_raw = m.group(1)
        try:
            pct = float(m.group(2))
        except ValueError:
            continue
        if pct <= 0 or pct > 100:
            continue
        names = re.split(r',\s*(?:and\s+)?|(?:\s+and\s+)', company_list_raw)
        for name in names:
            name = name.strip().rstrip(",")
            if _is_valid_company_name(name):
                if name not in found or pct > found[name]:
                    found[name] = pct

    return [{"name": k, "pct": v} for k, v in found.items()]


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------
def upsert_supply_chain_rows(
    conn: sqlite3.Connection,
    *,
    supplier_ticker: str,
    customers: list[dict[str, Any]],
    filed_date: str,
) -> int:
    """
    For each customer, insert a row where:
      ticker          = customer_ticker  (the company that buys)
      supplier_symbol = supplier_ticker  (the company that sells)
    """
    now = _utc_now_iso()
    n = 0
    for c in customers:
        cust_ticker = c.get("customer_ticker", "")
        if not cust_ticker:
            continue
        payload = {
            "source": "edgar_10k",
            "filed_date": filed_date,
            "pct_of_revenue": c["pct"],
            "customer_name_raw": c["name"],
        }
        conn.execute(
            """
            INSERT OR REPLACE INTO suppliers
              (ticker, supplier_symbol, supplier_name, relationship, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cust_ticker,
                supplier_ticker,
                c["name"],
                "supplier_from_edgar_10k",
                json.dumps(payload, ensure_ascii=False),
                now,
            ),
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main entry point (library API)
# ---------------------------------------------------------------------------
def run_edgar_supply_chain(
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    max_tickers: int = 0,
    progress_every: int = 25,
    verbose: bool = True,
) -> int:
    """
    Process each ticker in *tickers*, download its 10-K, extract customers,
    and upsert into the suppliers table.  Returns total rows upserted.
    """
    if not _HAS_BS4:
        print("ERROR: beautifulsoup4 is required. Install with: pip install beautifulsoup4 lxml")
        return 0

    cik_map = load_cik_map()
    name_to_ticker = build_name_to_ticker(cik_map)

    if verbose:
        print(f"SEC CIK map loaded: {len(cik_map):,} tickers")

    work = tickers if max_tickers <= 0 else tickers[:max_tickers]
    total = len(work)
    total_rows = 0
    skipped_no_cik = 0
    skipped_no_filing = 0
    skipped_no_doc = 0
    skipped_no_text = 0

    for i, ticker in enumerate(work, start=1):
        if verbose and (i == 1 or i % progress_every == 0 or i == total):
            print(f"  edgar-supply-chain: {i:,}/{total:,} ({ticker}) — {total_rows:,} rows so far")

        cik = cik_map.get(ticker.upper())
        if not cik:
            skipped_no_cik += 1
            continue

        result = _get_latest_10k_accession(cik)
        if not result:
            skipped_no_filing += 1
            continue
        accession, filed_date, primary_doc = result

        doc_url = _get_primary_doc_url(cik, accession, primary_doc)
        if not doc_url:
            skipped_no_doc += 1
            continue

        text = _download_10k_text(cik, accession, doc_url)
        if not text:
            skipped_no_text += 1
            continue

        raw_customers = extract_customers_from_text(text)

        resolved: list[dict[str, Any]] = []
        for c in raw_customers:
            ct = resolve_name_to_ticker(c["name"], name_to_ticker)
            if ct and ct != ticker.upper():
                resolved.append({**c, "customer_ticker": ct})

        if resolved:
            rows = upsert_supply_chain_rows(
                conn,
                supplier_ticker=ticker.upper(),
                customers=resolved,
                filed_date=filed_date,
            )
            total_rows += rows

    conn.commit()

    if verbose:
        print(
            f"edgar-supply-chain done: {total_rows:,} rows upserted "
            f"(skipped: {skipped_no_cik} no-CIK, {skipped_no_filing} no-10K, "
            f"{skipped_no_doc} no-doc, {skipped_no_text} no-text)"
        )
    return total_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_watchlist_tickers(watchlist_name: str, db_path: Path) -> list[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM fmp_watchlist WHERE watchlist_name=? ORDER BY ticker",
            (watchlist_name,),
        ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch SEC EDGAR supply chain data into suppliers table.")
    ap.add_argument("--watchlist-name", default="", help="Load tickers from fmp_watchlist with this name.")
    ap.add_argument("--tickers", default="", help="Comma-separated list of tickers (overrides --watchlist-name).")
    ap.add_argument("--max-tickers", type=int, default=0, help="Process at most N tickers (0=all).")
    ap.add_argument("--progress-every", type=int, default=25, help="Log progress every N tickers.")
    ap.add_argument("--db-path", default="", help="Override default DB path.")
    args = ap.parse_args()

    db_path = Path(args.db_path) if args.db_path else _db_path()
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.watchlist_name:
        tickers = _load_watchlist_tickers(args.watchlist_name, db_path)
        print(f"Loaded {len(tickers):,} tickers from watchlist '{args.watchlist_name}'")
    else:
        print("Provide --watchlist-name or --tickers")
        return

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        run_edgar_supply_chain(
            conn,
            tickers,
            max_tickers=args.max_tickers,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
