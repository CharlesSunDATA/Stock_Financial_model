"""Export local SQLite data to Tableau-friendly CSV files."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "quant_data.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tableau_exports"
DEFAULT_PRICE_LOOKBACK_DAYS = 365 * 3

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ExportSpec:
    table: str
    date_column: str | None = None
    default_lookback_days: int | None = None
    order_by: str | None = None


EXPORT_SPECS: tuple[ExportSpec, ...] = (
    ExportSpec("company_profile", order_by="ticker"),
    ExportSpec("prices_eod", "price_date", DEFAULT_PRICE_LOOKBACK_DAYS, "ticker, price_date"),
    ExportSpec("fundamental_data", "report_date", None, "ticker, report_date"),
    ExportSpec("balance_sheet", "report_date", None, "ticker, report_date"),
    ExportSpec("financial_ratios", "report_date", None, "ticker, report_date"),
    ExportSpec("income_statement", "report_date", None, "ticker, report_date"),
    ExportSpec("cash_flow_statement", "report_date", None, "ticker, report_date"),
    ExportSpec("key_metrics_ttm", "as_of_date", None, "ticker, as_of_date"),
    ExportSpec("enterprise_values", "report_date", None, "ticker, report_date"),
    ExportSpec("historical_market_cap", "price_date", DEFAULT_PRICE_LOOKBACK_DAYS, "ticker, price_date"),
    ExportSpec("analyst_estimates", "estimated_date", None, "ticker, estimated_date"),
    ExportSpec("price_target_consensus", None, None, "ticker"),
    ExportSpec("price_target_summary", None, None, "ticker"),
    ExportSpec("dcf", "as_of_date", None, "ticker, as_of_date"),
    ExportSpec("earnings_calendar", "event_date", None, "ticker, event_date"),
    ExportSpec("earnings_surprises", "surprise_date", None, "ticker, surprise_date"),
    ExportSpec("institutional_ownership", "as_of_date", None, "ticker, as_of_date"),
    ExportSpec("insider_trading", "filing_date", None, "ticker, filing_date"),
    ExportSpec("stock_peers", None, None, "ticker, peer_ticker"),
    ExportSpec("suppliers", None, None, "ticker, supplier_symbol"),
    ExportSpec("stock_news", "published_date", 365, "ticker, published_date"),
    ExportSpec("fmp_watchlist", None, None, "watchlist_name, ticker"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export quant_data.db tables and computed scores to CSV files for Tableau."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to the SQLite database.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--price-start",
        type=str,
        default=None,
        help="Export prices and market cap from this date, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Keep payload_json columns. By default they are removed for Tableau compatibility.",
    )
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Skip computed Opportunity Score and Risk Score exports.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Optional table names to export instead of the default Tableau set.",
    )
    return parser.parse_args()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')]


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def write_csv(df: pd.DataFrame, path: Path) -> int:
    if df.empty:
        df.to_csv(path, index=False, encoding="utf-8")
        return 0
    normalized = df.copy()
    for col in normalized.columns:
        if pd.api.types.is_bool_dtype(normalized[col]):
            normalized[col] = normalized[col].map({True: "true", False: "false"})
    normalized.to_csv(path, index=False, encoding="utf-8")
    return len(normalized)


def normalize_public_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in normalized.select_dtypes(include=["number"]).columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").round(4)
    for col in normalized.columns:
        if not pd.api.types.is_numeric_dtype(normalized[col]):
            normalized[col] = normalized[col].fillna("").astype(str).str.slice(0, 240)
    return normalized


def export_table(
    conn: sqlite3.Connection,
    spec: ExportSpec,
    out_dir: Path,
    *,
    include_json: bool,
    price_start: str | None,
) -> dict[str, object] | None:
    if not table_exists(conn, spec.table):
        print(f"Skipped missing table: {spec.table}")
        return None

    columns = table_columns(conn, spec.table)
    selected_columns = columns if include_json else [col for col in columns if col != "payload_json"]
    if not selected_columns:
        print(f"Skipped table without exportable columns: {spec.table}")
        return None

    select_sql = ", ".join(quote_identifier(col) for col in selected_columns)
    sql = f"SELECT {select_sql} FROM {quote_identifier(spec.table)}"
    params: list[object] = []

    if spec.date_column and spec.date_column in columns:
        start_date = price_start
        if not start_date and spec.default_lookback_days:
            start_date = (date.today() - timedelta(days=spec.default_lookback_days)).isoformat()
        if start_date:
            sql += f" WHERE {quote_identifier(spec.date_column)} >= ?"
            params.append(start_date)

    if spec.order_by:
        sortable = [part.strip() for part in spec.order_by.split(",")]
        existing_sort = [quote_identifier(col) for col in sortable if col in columns]
        if existing_sort:
            sql += " ORDER BY " + ", ".join(existing_sort)

    df = pd.read_sql_query(sql, conn, params=params)
    path = out_dir / f"{spec.table}.csv"
    row_count = write_csv(df, path)
    print(f"Exported {spec.table}: {row_count:,} rows -> {path}")
    return {
        "name": spec.table,
        "kind": "table",
        "file": path.name,
        "rows": row_count,
        "columns": len(df.columns),
    }


def export_score_frames(db_path: Path, out_dir: Path) -> list[dict[str, object]]:
    from utils.opportunity_score import ScoreInputs, compute_opportunity_scores
    from utils.risk_score import RiskInputs, compute_risk_scores
    from utils.scoring_common import available_watchlists

    manifest_rows: list[dict[str, object]] = []
    watchlists = [None] + available_watchlists(db_path)

    opportunity_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    for watchlist_name in watchlists:
        label = watchlist_name or "All Watchlists"
        try:
            opportunity = compute_opportunity_scores(db_path, ScoreInputs(watchlist_name=watchlist_name))
        except Exception as exc:
            print(f"Skipped Opportunity Score for {label}: {exc}")
            opportunity = pd.DataFrame()
        if not opportunity.empty:
            opportunity.insert(0, "watchlist_name", label)
            opportunity_frames.append(opportunity)

        try:
            risk = compute_risk_scores(db_path, RiskInputs(watchlist_name=watchlist_name))
        except Exception as exc:
            print(f"Skipped Risk Score for {label}: {exc}")
            risk = pd.DataFrame()
        if not risk.empty:
            risk.insert(0, "watchlist_name", label)
            risk_frames.append(risk)

    score_exports = {
        "opportunity_scores": opportunity_frames,
        "risk_scores": risk_frames,
    }
    combined_scores: dict[str, pd.DataFrame] = {}
    for name, frames in score_exports.items():
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined_scores[name] = df
        path = out_dir / f"{name}.csv"
        row_count = write_csv(df, path)
        print(f"Exported {name}: {row_count:,} rows -> {path}")
        manifest_rows.append(
            {
                "name": name,
                "kind": "computed_score",
                "file": path.name,
                "rows": row_count,
                "columns": len(df.columns),
            }
        )
    dashboard_row = export_tableau_public_opportunities(
        combined_scores.get("opportunity_scores", pd.DataFrame()),
        combined_scores.get("risk_scores", pd.DataFrame()),
        out_dir,
    )
    if dashboard_row:
        manifest_rows.append(dashboard_row)
    safe_dashboard_row = export_tableau_public_opportunities_safe(
        combined_scores.get("opportunity_scores", pd.DataFrame()),
        combined_scores.get("risk_scores", pd.DataFrame()),
        out_dir,
    )
    if safe_dashboard_row:
        manifest_rows.append(safe_dashboard_row)
    return manifest_rows


def _all_watchlists_or_first(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "watchlist_name" not in df:
        return df
    if (df["watchlist_name"] == "All Watchlists").any():
        return df[df["watchlist_name"] == "All Watchlists"].copy()
    first_watchlist = str(df["watchlist_name"].dropna().iloc[0])
    return df[df["watchlist_name"] == first_watchlist].copy()


def export_tableau_public_opportunities(
    opportunity: pd.DataFrame,
    risk: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object] | None:
    opportunity = _all_watchlists_or_first(opportunity)
    risk = _all_watchlists_or_first(risk)
    if opportunity.empty:
        return None

    if not risk.empty:
        risk_cols = [
            col
            for col in [
                "ticker",
                "risk_score",
                "risk_level",
                "risk_drivers",
                "cash_flow_risk",
                "debt_risk",
                "valuation_risk",
                "technical_risk",
                "fundamental_risk",
                "systemic_risk",
            ]
            if col in risk.columns
        ]
        opportunity = opportunity.merge(
            risk[risk_cols].drop_duplicates("ticker"),
            on="ticker",
            how="left",
        )

    source_cols = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "primary_market_theme",
        "market_themes",
        "type",
        "judgment",
        "score",
        "risk_score",
        "risk_level",
        "risk_drivers",
        "latest_price",
        "price_date",
        "target_consensus",
        "target_upside_pct",
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "revenue_yoy_pct",
        "eps_yoy_pct",
        "fcf_yoy_pct",
        "ev_to_sales",
        "free_cash_flow_yield",
        "net_debt_to_ebitda",
        "price_momentum",
        "revenue_growth",
        "eps_fcf_improvement",
        "valuation_reasonableness",
        "financial_safety",
        "industry_strength",
        "risk_protection_score",
        "upside_momentum_score",
        "industry_opportunity_score",
        "industry_rank",
        "industry_candidate_type",
        "industry_thesis",
        "top_industry_candidate",
    ]
    rename_map = {
        "ticker": "Ticker",
        "company_name": "Company Name",
        "sector": "Sector",
        "industry": "Industry",
        "primary_market_theme": "Primary Market Theme",
        "market_themes": "Market Themes",
        "type": "Opportunity Type",
        "judgment": "Opportunity Judgment",
        "score": "Opportunity Score",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "risk_drivers": "Risk Drivers",
        "latest_price": "Latest Price",
        "price_date": "Price Date",
        "target_consensus": "Target Consensus",
        "target_upside_pct": "Target Upside %",
        "ret_1m": "Return 1M %",
        "ret_3m": "Return 3M %",
        "ret_6m": "Return 6M %",
        "ret_12m": "Return 12M %",
        "revenue_yoy_pct": "Revenue YoY %",
        "eps_yoy_pct": "EPS YoY %",
        "fcf_yoy_pct": "FCF YoY %",
        "ev_to_sales": "EV to Sales",
        "free_cash_flow_yield": "Free Cash Flow Yield",
        "net_debt_to_ebitda": "Net Debt to EBITDA",
        "price_momentum": "Price Momentum",
        "revenue_growth": "Revenue Growth",
        "eps_fcf_improvement": "EPS FCF Improvement",
        "valuation_reasonableness": "Valuation Reasonableness",
        "financial_safety": "Financial Safety",
        "industry_strength": "Industry Strength",
        "risk_protection_score": "Risk Protection Score",
        "upside_momentum_score": "Upside Momentum Score",
        "industry_opportunity_score": "Industry Opportunity Score",
        "industry_rank": "Industry Rank",
        "industry_candidate_type": "Industry Candidate Type",
        "industry_thesis": "Industry Thesis",
        "top_industry_candidate": "Top Industry Candidate",
    }

    existing_cols = [col for col in source_cols if col in opportunity.columns]
    df = opportunity[existing_cols].drop_duplicates("ticker").rename(columns=rename_map)
    if "Opportunity Score" in df:
        df = df.sort_values("Opportunity Score", ascending=False, na_position="last")

    path = out_dir / "tableau_public_opportunities.csv"
    row_count = write_csv(df, path)
    print(f"Exported tableau_public_opportunities: {row_count:,} rows -> {path}")
    return {
        "name": "tableau_public_opportunities",
        "kind": "dashboard_dataset",
        "file": path.name,
        "rows": row_count,
        "columns": len(df.columns),
    }


def export_tableau_public_opportunities_safe(
    opportunity: pd.DataFrame,
    risk: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object] | None:
    opportunity = _all_watchlists_or_first(opportunity)
    risk = _all_watchlists_or_first(risk)
    if opportunity.empty:
        return None

    if not risk.empty:
        risk_cols = [col for col in ["ticker", "risk_score", "risk_level"] if col in risk.columns]
        opportunity = opportunity.merge(
            risk[risk_cols].drop_duplicates("ticker"),
            on="ticker",
            how="left",
        )

    source_cols = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "type",
        "judgment",
        "score",
        "risk_score",
        "risk_level",
        "latest_price",
        "price_date",
        "target_upside_pct",
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "revenue_yoy_pct",
        "eps_yoy_pct",
        "fcf_yoy_pct",
        "ev_to_sales",
        "net_debt_to_ebitda",
        "price_momentum",
        "revenue_growth",
        "eps_fcf_improvement",
        "valuation_reasonableness",
        "financial_safety",
        "industry_strength",
        "risk_protection_score",
        "upside_momentum_score",
        "industry_opportunity_score",
    ]
    rename_map = {
        "ticker": "ticker",
        "company_name": "company_name",
        "sector": "sector",
        "industry": "industry",
        "type": "opportunity_type",
        "judgment": "opportunity_judgment",
        "score": "opportunity_score",
        "risk_score": "risk_score",
        "risk_level": "risk_level",
        "latest_price": "latest_price",
        "price_date": "price_date",
        "target_upside_pct": "target_upside_pct",
        "ret_1m": "return_1m_pct",
        "ret_3m": "return_3m_pct",
        "ret_6m": "return_6m_pct",
        "ret_12m": "return_12m_pct",
        "revenue_yoy_pct": "revenue_yoy_pct",
        "eps_yoy_pct": "eps_yoy_pct",
        "fcf_yoy_pct": "fcf_yoy_pct",
        "ev_to_sales": "ev_to_sales",
        "net_debt_to_ebitda": "net_debt_to_ebitda",
        "price_momentum": "price_momentum",
        "revenue_growth": "revenue_growth",
        "eps_fcf_improvement": "eps_fcf_improvement",
        "valuation_reasonableness": "valuation_reasonableness",
        "financial_safety": "financial_safety",
        "industry_strength": "industry_strength",
        "risk_protection_score": "risk_protection_score",
        "upside_momentum_score": "upside_momentum_score",
        "industry_opportunity_score": "industry_opportunity_score",
    }

    existing_cols = [col for col in source_cols if col in opportunity.columns]
    df = opportunity[existing_cols].drop_duplicates("ticker").rename(columns=rename_map)
    df = normalize_public_frame(df)
    if "opportunity_score" in df:
        df = df.sort_values("opportunity_score", ascending=False, na_position="last")

    path = out_dir / "tableau_public_opportunities_safe.csv"
    row_count = write_csv(df, path)
    print(f"Exported tableau_public_opportunities_safe: {row_count:,} rows -> {path}")
    return {
        "name": "tableau_public_opportunities_safe",
        "kind": "dashboard_dataset",
        "file": path.name,
        "rows": row_count,
        "columns": len(df.columns),
    }


def selected_specs(table_names: Iterable[str] | None) -> list[ExportSpec]:
    if not table_names:
        return list(EXPORT_SPECS)
    known = {spec.table: spec for spec in EXPORT_SPECS}
    specs: list[ExportSpec] = []
    for table_name in table_names:
        specs.append(known.get(table_name, ExportSpec(table_name)))
    return specs


def main() -> int:
    args = parse_args()
    db_path = args.db.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for spec in selected_specs(args.tables):
            row = export_table(
                conn,
                spec,
                out_dir,
                include_json=args.include_json,
                price_start=args.price_start,
            )
            if row:
                manifest_rows.append(row)

    if not args.skip_scores:
        manifest_rows.extend(export_score_frames(db_path, out_dir))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    write_csv(manifest, manifest_path)
    print(f"Exported manifest: {len(manifest):,} rows -> {manifest_path}")
    print("Tableau export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
