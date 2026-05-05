#!/usr/bin/env python3
"""動能排名報告 — 計算 watchlist 股票的 1/3/6/12 個月報酬率並排名"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from utils.local_data import connect_readonly

DB_PATH = Path(__file__).parent / "data" / "quant_data.db"
OUTPUT_PATH = Path(__file__).parent / "momentum_report.md"

def get_price_on_or_before(conn, ticker, target_date):
    row = conn.execute(
        "SELECT adj_close, close FROM prices_eod WHERE ticker=? AND price_date<=? ORDER BY price_date DESC LIMIT 1",
        (ticker, target_date)
    ).fetchone()
    if row is None:
        return None
    return row[0] if row[0] else row[1]

def main():
    conn = connect_readonly(DB_PATH)

    # 最新交易日
    latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")

    # 各時間點（交易日近似）
    periods = {
        "1M":  (latest_dt - timedelta(days=30)).strftime("%Y-%m-%d"),
        "3M":  (latest_dt - timedelta(days=91)).strftime("%Y-%m-%d"),
        "6M":  (latest_dt - timedelta(days=182)).strftime("%Y-%m-%d"),
        "12M": (latest_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
    }

    print(f"最新日期：{latest_date}")
    print(f"計算期間：{periods}")

    # 取 watchlist 股票
    tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM fmp_watchlist").fetchall()]
    print(f"Watchlist 股票數：{len(tickers)}")

    # 批次查詢最新收盤價
    print("載入價格資料...")
    price_df = pd.read_sql(
        f"SELECT ticker, price_date, COALESCE(adj_close, close) as price "
        f"FROM prices_eod "
        f"WHERE ticker IN ({','.join('?'*len(tickers))}) "
        f"AND price_date >= ? "
        f"ORDER BY ticker, price_date",
        conn,
        params=tickers + [periods["12M"]]
    )

    print(f"載入 {len(price_df):,} 筆價格記錄")

    # 取每個 ticker 在各時間點最近的價格
    results = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("price_date").dropna(subset=["price"])
        if grp.empty:
            continue
        price_now = grp.iloc[-1]["price"]
        actual_date = grp.iloc[-1]["price_date"]

        row = {"ticker": ticker, "latest_date": actual_date, "price": price_now}
        for period, date_str in periods.items():
            past = grp[grp["price_date"] <= date_str].dropna(subset=["price"])
            if len(past) > 0 and price_now and past.iloc[-1]["price"]:
                p0 = past.iloc[-1]["price"]
                row[f"ret_{period}"] = (price_now - p0) / p0 * 100
            else:
                row[f"ret_{period}"] = None
        results.append(row)

    df = pd.DataFrame(results).dropna(subset=["ret_1M", "ret_3M"])

    # 加入公司名稱
    names = pd.read_sql(
        "SELECT ticker, company_name, sector FROM company_profile",
        conn
    ).drop_duplicates("ticker")
    df = df.merge(names, on="ticker", how="left")

    # 加入市值（最新）
    mcap = pd.read_sql(
        "SELECT ticker, market_cap FROM key_metrics_ttm WHERE as_of_date=(SELECT MAX(as_of_date) FROM key_metrics_ttm)",
        conn
    ).drop_duplicates("ticker")
    df = df.merge(mcap, on="ticker", how="left")

    conn.close()

    # 動能綜合分數（等權 1M+3M+6M+12M 排名百分位）
    for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M"]:
        df[f"rank_{col}"] = df[col].rank(pct=True, na_option="keep")
    rank_cols = [c for c in ["rank_ret_1M", "rank_ret_3M", "rank_ret_6M", "rank_ret_12M"] if c in df.columns]
    df["momentum_score"] = df[rank_cols].mean(axis=1) * 100

    df_sorted = df.sort_values("momentum_score", ascending=False)

    # ── 輸出 Markdown 報告 ──
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 動能排名報告",
        f"**產生時間：** {now_str}  ",
        f"**資料截至：** {latest_date}  ",
        f"**股票數量：** {len(df_sorted):,}",
        "",
        "---",
        "",
        "## 🏆 Top 30 — 最強動能",
        "",
        "| 排名 | 股票 | 公司名稱 | 1M% | 3M% | 6M% | 12M% | 動能分數 |",
        "|------|------|----------|-----|-----|-----|------|----------|",
    ]
    def _name(r) -> str:
        v = r.get("company_name")
        return str(v) if pd.notna(v) and v else ""

    def fmt(v):
        return f"{v:+.1f}%" if pd.notna(v) else "—"

    for i, (_, r) in enumerate(df_sorted.head(30).iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {_name(r)} | "
            f"{fmt(r.get('ret_1M'))} | {fmt(r.get('ret_3M'))} | {fmt(r.get('ret_6M'))} | {fmt(r.get('ret_12M'))} | "
            f"{r['momentum_score']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 💀 Bottom 30 — 最弱動能",
        "",
        "| 排名 | 股票 | 公司名稱 | 1M% | 3M% | 6M% | 12M% | 動能分數 |",
        "|------|------|----------|-----|-----|-----|------|----------|",
    ]
    for i, (_, r) in enumerate(df_sorted.tail(30).iloc[::-1].iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {_name(r)} | "
            f"{fmt(r.get('ret_1M'))} | {fmt(r.get('ret_3M'))} | {fmt(r.get('ret_6M'))} | {fmt(r.get('ret_12M'))} | "
            f"{r['momentum_score']:.1f} |"
        )

    # 各期間 Top 10
    for period, label in [("ret_1M", "1 個月"), ("ret_3M", "3 個月"), ("ret_6M", "6 個月"), ("ret_12M", "12 個月")]:
        top = df_sorted.dropna(subset=[period]).nlargest(10, period)
        lines += [
            "",
            f"## 📈 {label}漲幅 Top 10",
            "",
            f"| 股票 | 公司名稱 | {label}報酬 |",
            "|------|----------|------------|",
        ]
        for _, r in top.iterrows():
            lines.append(f"| **{r['ticker']}** | {_name(r)} | {r[period]:+.1f}% |")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 報告已輸出：{OUTPUT_PATH}")

    # 終端機預覽
    print("\n" + "="*60)
    print("🏆 Top 10 動能股票")
    print("="*60)
    cols = ["ticker", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]
    print(df_sorted[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

if __name__ == "__main__":
    main()
