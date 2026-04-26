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
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📊",
    layout="wide",
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
    ]
)

pg.run()

