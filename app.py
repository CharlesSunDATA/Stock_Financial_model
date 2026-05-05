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
- Smart Money (insider purchases, buy/sell ratio, institutional 13-F ownership flows)
- Portfolio Watchlist Dashboard (combined score, risk, momentum, targets, earnings)
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
        st.Page("pages/1_Stock_valuation.py",     title="Stock Valuation",       icon="💹"),
        st.Page("pages/2_Quarterly_financials.py", title="Quarterly Financials",  icon="📋"),
        st.Page("pages/3_Markowitz_opt.py",       title="Markowitz Optimization", icon="⚖️"),
        st.Page("pages/4_Technical_backtester.py", title="Technical Backtester",  icon="📈"),
        st.Page("pages/5_Earnings_Call_NLP.py",   title="Earnings Call Analyzer",     icon="🎙️"),
        st.Page("pages/6_Earnings_Calendar.py",   title="Earnings Calendar",     icon="📅"),
        st.Page("pages/7_Pullback_Analyzer.py",   title="Pullback Analyzer",     icon="📉"),
        st.Page("pages/8_Analyst_Sentiment.py",   title="Analyst Sentiment",     icon="🧠"),
        st.Page("pages/9_Market_Trend.py",        title="Market Trend",          icon="🌐"),
        st.Page("pages/10_Momentum_Report.py",    title="Momentum Rankings",     icon="🚀"),
        st.Page("pages/11_Fundamentals.py",       title="Fundamentals",          icon="📊"),
        st.Page("pages/12_Opportunity_Score.py",  title="Opportunity Score",     icon="🎯"),
        st.Page("pages/13_Risk_Score.py",         title="Risk Score",            icon="🛡️"),
        st.Page("pages/14_Smart_Money.py",        title="Smart Money",           icon="🏦"),
        st.Page("pages/15_Portfolio_Watchlist.py", title="Portfolio Watchlist",  icon="🧭"),
    ]
)

pg.run()
