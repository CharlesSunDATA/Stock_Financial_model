"""Ranking Suite page.

Combines momentum, opportunity, risk, and industry ranking tools behind one
sidebar entry while reusing the existing page implementations.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import streamlit as st


PAGE_DIR = Path(__file__).resolve().parent

RANKING_TOOLS = {
    "Momentum Rankings": {
        "file": PAGE_DIR / "10_Momentum_Report.py",
        "module": "_ranking_suite_momentum",
    },
    "Opportunity Score": {
        "file": PAGE_DIR / "12_Opportunity_Score.py",
        "module": "_ranking_suite_opportunity",
    },
    "Risk Score": {
        "file": PAGE_DIR / "13_Risk_Score.py",
        "module": "_ranking_suite_risk",
    },
    "Industry Ranking": {
        "file": PAGE_DIR / "16_Industry_Ranking.py",
        "module": "_ranking_suite_industry",
    },
}


def _load_page_module(module_name: str, file_path: Path) -> ModuleType:
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load page module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    with st.sidebar:
        st.header("Ranking Suite")
        selected_tool = st.radio(
            "Tool",
            list(RANKING_TOOLS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        st.divider()

    config = RANKING_TOOLS[selected_tool]
    module = _load_page_module(config["module"], config["file"])

    if not hasattr(module, "main"):
        st.error(f"`{config['file'].name}` does not define a main() function.")
        return

    module.main()


if __name__ == "__main__":
    main()
