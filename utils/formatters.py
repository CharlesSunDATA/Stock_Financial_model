"""Shared display formatters for Streamlit pages."""

from __future__ import annotations

import pandas as pd


def fmt_score(value, *, missing: str = "-") -> str:
    if pd.isna(value):
        return missing
    return f"{float(value):.0f}"


def fmt_pct(value, *, missing: str = "-") -> str:
    if pd.isna(value):
        return missing
    return f"{float(value):+.1f}%"


def fmt_money(value, *, missing: str = "-") -> str:
    if pd.isna(value):
        return missing
    x = float(value)
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:,.2f}T"
    if ax >= 1e9:
        return f"{x / 1e9:,.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.1f}M"
    return f"{x:,.0f}"
