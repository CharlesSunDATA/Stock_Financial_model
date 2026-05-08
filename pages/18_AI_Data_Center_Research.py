"""
AI Data Center Research (Streamlit page)

Imports the NVDA AI data center Word report and supply-chain workbook into
the dashboard with English UI labels and downloadable source files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import streamlit as st


XML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

SHEET_LABELS = {
    "1.": "Supply Chain Overview",
    "2.": "US Listed Stocks",
    "3.": "Taiwan Listed Stocks",
    "4.": "Moat Scores",
    "5.": "Risks and Catalysts",
    "6.": "Portfolio Ideas",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _research_dir() -> Path:
    return _project_root() / "data" / "research"


def _report_path() -> Path:
    return _research_dir() / "NVDA_AI_DataCenter_deep_research.docx"


def _workbook_path() -> Path:
    return _research_dir() / "NVDA_AI_DataCenter_supply_chain_research.xlsx"


def _sheet_label(sheet_name: str) -> str:
    for prefix, label in SHEET_LABELS.items():
        if sheet_name.startswith(prefix):
            return label
    return sheet_name


def _cell_ref_to_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return max(index - 1, 0)


def _unique_columns(values: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    columns: list[str] = []
    for idx, value in enumerate(values, start=1):
        name = str(value).strip() if value is not None and str(value).strip() else f"Column {idx}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        columns.append(name if count == 1 else f"{name} {count}")
    return columns


@st.cache_data(show_spinner=False)
def load_docx_paragraphs(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.exists():
        return []

    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", XML_NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", XML_NS)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


@st.cache_data(show_spinner=False)
def load_workbook_tables(path_str: str) -> dict[str, pd.DataFrame]:
    path = Path(path_str)
    if not path.exists():
        return {}

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", XML_NS):
                shared_strings.append("".join(node.text or "" for node in item.findall(".//a:t", XML_NS)))

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}

        tables: dict[str, pd.DataFrame] = {}
        sheets = workbook_root.find("a:sheets", XML_NS)
        if sheets is None:
            return tables

        for sheet in sheets:
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{XML_NS['r']}}}id"]
            target = rel_targets[rel_id].lstrip("/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            sheet_root = ET.fromstring(zf.read(sheet_path))

            rows: list[list[Any]] = []
            for row in sheet_root.findall(".//a:sheetData/a:row", XML_NS):
                row_values: list[Any] = []
                for cell in row.findall("a:c", XML_NS):
                    col_idx = _cell_ref_to_col_index(cell.attrib.get("r", "A1"))
                    while len(row_values) < col_idx:
                        row_values.append("")

                    value = ""
                    cell_type = cell.attrib.get("t")
                    raw_value = cell.find("a:v", XML_NS)
                    if cell_type == "s" and raw_value is not None and raw_value.text is not None:
                        value = shared_strings[int(raw_value.text)]
                    elif cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//a:t", XML_NS))
                    elif raw_value is not None and raw_value.text is not None:
                        value = raw_value.text
                    row_values.append(value)

                if any(str(value).strip() for value in row_values):
                    rows.append(row_values)

            if not rows:
                tables[_sheet_label(sheet_name)] = pd.DataFrame()
                continue

            header_index = 0
            for idx, row_values in enumerate(rows):
                filled = sum(1 for value in row_values if str(value).strip())
                if filled >= 3:
                    header_index = idx
                    break

            width = max(len(row) for row in rows[header_index:])
            columns = _unique_columns(rows[header_index] + [""] * (width - len(rows[header_index])))
            data_rows = [row + [""] * (width - len(row)) for row in rows[header_index + 1 :]]
            tables[_sheet_label(sheet_name)] = pd.DataFrame(data_rows, columns=columns)

        return tables


def _render_report_preview(paragraphs: list[str]) -> None:
    if not paragraphs:
        st.warning("The Word report could not be loaded from the research folder.")
        return

    st.subheader("Report Preview")
    max_paragraphs = st.slider("Preview length", min_value=10, max_value=120, value=35, step=5)
    for paragraph in paragraphs[:max_paragraphs]:
        st.write(paragraph)

    if len(paragraphs) > max_paragraphs:
        st.caption(f"Showing {max_paragraphs} of {len(paragraphs)} paragraphs.")


def _render_workbook_tables(tables: dict[str, pd.DataFrame]) -> None:
    if not tables:
        st.warning("The Excel workbook could not be loaded from the research folder.")
        return

    st.subheader("Workbook Tables")
    selected = st.selectbox("Select a table", list(tables.keys()))
    df = tables[selected]
    st.caption(f"{len(df):,} rows x {len(df.columns):,} columns")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_downloads() -> None:
    st.subheader("Source Files")
    report = _report_path()
    workbook = _workbook_path()
    c1, c2 = st.columns(2)

    with c1:
        if report.exists():
            st.download_button(
                "Download Word Report",
                data=report.read_bytes(),
                file_name=report.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.error("Word report is missing.")

    with c2:
        if workbook.exists():
            st.download_button(
                "Download Excel Workbook",
                data=workbook.read_bytes(),
                file_name=workbook.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.error("Excel workbook is missing.")


def main() -> None:
    st.title("AI Data Center Research")
    st.caption("NVDA-focused research imported into the finance dashboard.")

    report = _report_path()
    workbook = _workbook_path()
    paragraphs = load_docx_paragraphs(str(report))
    tables = load_workbook_tables(str(workbook))

    c1, c2, c3 = st.columns(3)
    c1.metric("Report paragraphs", f"{len(paragraphs):,}")
    c2.metric("Workbook tables", f"{len(tables):,}")
    c3.metric("Research files", f"{int(report.exists()) + int(workbook.exists())}/2")

    st.divider()
    tab_report, tab_workbook, tab_files = st.tabs(["Word Report", "Excel Workbook", "Downloads"])
    with tab_report:
        _render_report_preview(paragraphs)
    with tab_workbook:
        _render_workbook_tables(tables)
    with tab_files:
        _render_downloads()


if __name__ == "__main__":
    main()
