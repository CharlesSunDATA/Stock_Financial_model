"""
Stock Financial Model — Streamlit multipage app.

Pages:
- Stock valuation (DCF + P/E zone)
- Quarterly financials
- Markowitz portfolio optimization
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Stock Financial Model", layout="wide")

pg = st.navigation(
    [
        st.Page("valuation_dashboard.py", title="Stock valuation", default=True),
        st.Page("pages/1_Quarterly_financials.py", title="Quarterly financials"),
        st.Page("markowitz_portfolio.py", title="Markowitz optimization"),
        st.Page("pages/2_Technical_Strategy_Backtester.py", title="Technical backtester"),
        st.Page("pages/3_Earnings_Call_NLP_Analyzer.py", title="Earnings call NLP"),
    ]
)

pg.run()

