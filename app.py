"""
Finance Dashboard — Streamlit multipage app.

Pages:
- Market Overview (homepage dashboard)
- Stock valuation (DCF + P/E zone)
- Quarterly financials
- Markowitz portfolio optimization
- Technical strategy backtester
- Earnings call NLP analyzer
- Earnings Calendar
- Pullback Analyzer (bias ratio, run-up stats, RSI overbought duration)
- Market Trend Analysis (breadth, moving averages, new highs/lows)
- Momentum Rankings (1/3/6/12-month composite momentum, top/bottom movers)
- Stock Opportunity Score (daily weighted research ranking)
- Stock Risk Score (investment risk ranking)
- Industry Ranking (industry and peer ranking system)
- Smart Money (insider purchases, buy/sell ratio, institutional 13-F ownership flows)
- Portfolio Watchlist Dashboard (combined score, risk, momentum, targets, earnings)
- Optical Communications Sector Tracker (leading indicators: hyperscaler capex, momentum)
- Memory Industry Sector Tracker (HBM demand, hyperscaler capex, equipment proxy, DRAM pricing)
- AI Data Center Research (imported NVDA report and supply-chain workbook)
- Supply Chain Explorer (interactive upstream/downstream/peer navigator for tech companies)
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📊",
    layout="wide",
)

# Global full-width layout (remove default Streamlit paddings)
st.markdown(
    """
    <style>
      section.main > div.block-container {
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

pg = st.navigation(
    [
        st.Page("pages/0_Dashboard.py",          title="Market Overview",       icon="📊", default=True),
        st.Page("pages/1_Stock_Analysis.py",       title="Stock Analysis",        icon="💹"),
        st.Page("pages/3_Technical_Analyser.py",   title="Technical Analyser",    icon="📐"),
        st.Page("pages/5_Earnings_Hub.py",         title="Earnings Analyst",      icon="📰"),
        st.Page("pages/9_Market_Trend.py",        title="Market Trend",          icon="🌐"),
        st.Page("pages/10_Ranking_Suite.py",      title="Ranking Suite",         icon="🏭"),
        st.Page("pages/14_Smart_Money.py",        title="Smart Money",           icon="🏦"),
        st.Page("pages/15_Portfolio_Watchlist.py", title="Portfolio Watchlist",  icon="🧭"),
        st.Page("pages/16_Sector_Tracker.py",       title="Sector Tracker",        icon="💡"),
        st.Page("pages/18_Supply_Chain.py",         title="Supply Chain Explorer",  icon="🔗"),
    ]
)

pg.run()
