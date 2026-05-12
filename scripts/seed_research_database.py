"""
Seed the local JSON research database for AI infrastructure stock tracking.

The Streamlit research page reads these files:
  data/database/stocks.json
  data/database/updates.json
  data/database/sources.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


CATEGORIES: dict[str, list[str]] = {
    "GPU / CPU / ASIC": ["NVDA", "AMD", "INTC", "AVGO", "MRVL", "TSM"],
    "Memory / HBM / Storage": ["MU", "WDC", "STX", "SSNLF", "HXSCL"],
    "Optical Communications / Photonics": ["LITE", "COHR", "FN", "AAOI", "CIEN", "GLW", "NOK"],
    "Connector / Interconnect": ["APH", "TEL", "CRDO", "ALAB"],
    "AI Server / ODM / EMS": ["DELL", "SMCI", "HPE", "CLS", "JBL", "FLEX", "QUCCF", "WICOF", "HNHPF"],
    "Networking": ["ANET", "CSCO", "AVGO", "MRVL", "CIEN"],
    "Power / Cooling / Data Center Equipment": [
        "VRT", "ETN", "GEV", "PWR", "EME", "FIX", "TT", "CARR", "JCI", "NVT", "BE", "GNRC", "CMI", "CAT",
    ],
    "Semiconductor Equipment / Packaging": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "AEHR"],
    "Cloud Demand Drivers": ["MSFT", "AMZN", "GOOGL", "META", "ORCL"],
}


COMPANIES: dict[str, str] = {
    "AAOI": "Applied Optoelectronics",
    "AEHR": "Aehr Test Systems",
    "ALAB": "Astera Labs",
    "AMAT": "Applied Materials",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "ANET": "Arista Networks",
    "APH": "Amphenol",
    "ASML": "ASML",
    "AVGO": "Broadcom",
    "BE": "Bloom Energy",
    "CARR": "Carrier Global",
    "CAT": "Caterpillar",
    "CIEN": "Ciena",
    "CLS": "Celestica",
    "CMI": "Cummins",
    "COHR": "Coherent",
    "CRDO": "Credo Technology",
    "CSCO": "Cisco",
    "DELL": "Dell Technologies",
    "EME": "EMCOR",
    "ETN": "Eaton",
    "FIX": "Comfort Systems USA",
    "FLEX": "Flex",
    "FN": "Fabrinet",
    "GEV": "GE Vernova",
    "GLW": "Corning",
    "GNRC": "Generac",
    "GOOGL": "Alphabet",
    "HNHPF": "Hon Hai Precision",
    "HPE": "Hewlett Packard Enterprise",
    "HXSCL": "SK Hynix",
    "INTC": "Intel",
    "JBL": "Jabil",
    "JCI": "Johnson Controls",
    "KLAC": "KLA",
    "LITE": "Lumentum",
    "LRCX": "Lam Research",
    "META": "Meta Platforms",
    "MRVL": "Marvell Technology",
    "MSFT": "Microsoft",
    "MU": "Micron Technology",
    "NOK": "Nokia",
    "NVDA": "NVIDIA",
    "NVT": "nVent Electric",
    "ORCL": "Oracle",
    "PANW": "Palo Alto Networks",
    "PLTR": "Palantir",
    "PWR": "Quanta Services",
    "QUCCF": "Quanta Computer",
    "SMCI": "Super Micro Computer",
    "SNOW": "Snowflake",
    "SSNLF": "Samsung Electronics",
    "STX": "Seagate",
    "TEL": "TE Connectivity",
    "TER": "Teradyne",
    "TSM": "Taiwan Semiconductor Manufacturing",
    "TSLA": "Tesla",
    "TT": "Trane Technologies",
    "VRT": "Vertiv",
    "WDC": "Western Digital",
    "WICOF": "Wistron",
}


SPECIAL_RESEARCH: dict[str, dict[str, Any]] = {
    "NVDA": {
        "positioning": "Core AI data center platform company across GPUs, networking, rack-scale systems, and software.",
        "investment_thesis": "NVIDIA remains the demand thermometer for AI factories, but the next phase depends on networking, HBM, packaging, power, and deployment velocity.",
        "catalysts": [
            "Blackwell shipments and the Vera Rubin product cycle.",
            "Hyperscaler AI CapEx revisions.",
            "Expansion in networking, NVLink, BlueField, AI storage, and rack-scale systems.",
        ],
        "risks": [
            "High expectations amplify any growth deceleration.",
            "Export controls and geopolitical constraints.",
            "Custom ASIC adoption could affect long-term GPU mix.",
        ],
        "tracking_items": [
            "Data center revenue growth.",
            "Gross margin and Blackwell/Rubin product mix.",
            "HBM and advanced packaging capacity.",
            "CSP CapEx guidance.",
        ],
        "earnings_notes": [
            "Q4 FY2026 revenue was $68.127B, up 73% year over year.",
            "Q4 FY2026 data center revenue was $62.3B, up 75% year over year.",
            "FY2026 data center revenue was $193.7B, up 68% year over year.",
        ],
    },
    "MU": {
        "positioning": "AI data center memory and HBM beneficiary.",
        "investment_thesis": "AI training and inference increase demand for HBM, server DRAM, and high-performance SSDs, shifting the memory cycle toward data-center-driven demand.",
        "catalysts": [
            "Tight HBM supply and strong pricing.",
            "Higher HBM attach rates on NVIDIA, AMD, and custom ASIC platforms.",
            "Data center SSD and high-end DRAM demand.",
        ],
        "risks": [
            "Memory remains a cyclical industry.",
            "HBM yield, qualification, and customer concentration risk.",
            "High CapEx could pressure cash flow if demand slows.",
        ],
        "tracking_items": [
            "HBM sold-out commentary.",
            "Cloud Memory and Core Data Center revenue growth.",
            "Gross margin guidance.",
            "CapEx and inventory.",
        ],
        "earnings_notes": [
            "FQ2 FY2026 Cloud Memory Business Unit revenue was $7.749B with 74% gross margin.",
            "FQ3 FY2026 revenue guidance was $33.5B plus or minus $0.75B.",
        ],
    },
    "LITE": {
        "positioning": "Optical communications and laser component supplier for AI data center scaling.",
        "investment_thesis": "As GPU clusters scale, high-speed optical modules, optical engines, lasers, CPO, and OCS become more important bottlenecks.",
        "catalysts": [
            "800G, 1.6T, and future 3.2T optical upgrade cycles.",
            "Cloud data center interconnect and AI cluster scale-out.",
            "CPO, OCS, and high-speed laser design wins.",
        ],
        "risks": [
            "Competitive pricing pressure in optical components.",
            "Customer concentration and order lumpiness.",
            "Fast technology transitions can pressure product mix.",
        ],
        "tracking_items": [
            "Datacom and cloud revenue growth.",
            "Gross margin and operating margin.",
            "Backlog and customer commitments.",
            "New product ramp in OCS, CPO, and high-speed lasers.",
        ],
    },
    "DELL": {
        "positioning": "AI server and enterprise AI factory systems supplier.",
        "investment_thesis": "The key question is whether AI-optimized server orders and backlog convert into high-quality revenue with improving margins and cash conversion.",
        "catalysts": [
            "AI server backlog conversion.",
            "NVIDIA Blackwell/Rubin rack-scale deployments.",
            "Enterprise AI factory adoption.",
        ],
        "risks": [
            "AI server gross margins may lag traditional server margins.",
            "Dependency on GPU, HBM, and high-end component supply.",
            "Large-customer concentration can create quarterly volatility.",
        ],
        "tracking_items": [
            "AI-optimized server orders, shipments, and backlog.",
            "ISG revenue growth and operating margin.",
            "Working capital and cash conversion.",
        ],
        "earnings_notes": [
            "FY2026 AI-optimized server orders exceeded $64B.",
            "FY2027 AI-optimized server revenue guidance was about $50B.",
        ],
    },
    "VRT": {
        "positioning": "Critical AI data center power and thermal infrastructure supplier.",
        "investment_thesis": "Higher AI rack power density makes power management, liquid cooling, thermal systems, UPS, modular deployment, and services key deployment bottlenecks.",
        "catalysts": [
            "AI rack power density growth.",
            "Liquid cooling, power train, and modular infrastructure orders.",
            "Faster backlog-to-revenue conversion.",
        ],
        "risks": [
            "Supply-chain and field deployment delays.",
            "Raw material, tariff, and labor cost pressure.",
            "Data center CapEx slowdown or customer acceptance delays.",
        ],
        "tracking_items": [
            "Organic orders growth.",
            "Backlog and book-to-bill.",
            "Americas growth.",
            "Adjusted operating margin.",
            "Liquid cooling adoption.",
        ],
        "earnings_notes": [
            "Q1 2026 net sales were $2.65B, up 30%.",
            "The company raised 2026 net sales guidance to $13.5B-$14.0B.",
        ],
    },
    "GLW": {
        "positioning": "Optical fiber and connectivity supplier for AI data center scale-out.",
        "investment_thesis": "Optical connectivity can become a bottleneck as AI clusters and data center interconnect demand rises.",
        "catalysts": [
            "AI data center optical fiber and connectivity demand.",
            "Potential customer partnerships around high-bandwidth optical infrastructure.",
        ],
        "risks": [
            "Telecom and enterprise demand cycles can dilute AI-driven growth.",
            "Pricing and mix matter if optical demand rises without margin expansion.",
        ],
        "tracking_items": [
            "Optical communications revenue.",
            "AI data center customer commentary.",
            "Margin contribution from optical connectivity products.",
        ],
    },
}


SOURCE_URLS: dict[str, tuple[str, str]] = {
    "NVDA": ("NVIDIA Q4 FY2026 results", "https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2026/"),
    "MU": ("Micron FQ2 FY2026 results", "https://investors.micron.com/news-releases/news-release-details/micron-technology-inc-reports-results-second-quarter-fiscal-2026"),
    "LITE": ("Lumentum FY2026 Q3 results", "https://investor.lumentum.com/financial-news-releases/news-details/2026/Lumentum-Announces-Third-Quarter-of-Fiscal-Year-2026-Financial-Results/default.aspx"),
    "DELL": ("Dell FY2026 Q4 and full-year results", "https://www.dell.com/en-us/dt/corporate/newsroom/announcements/detailpage.press-releases~usa~2026~2~dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-2026-results.htm"),
    "VRT": ("Vertiv Q1 2026 results", "https://investors.vertiv.com/news/news-details/2026/Vertiv-Reports-Strong-First-Quarter-with-Diluted-EPS-Growth-of-136-Adjusted-Diluted-EPS-Growth-of-83-Raises-Full-Year-Guidance/default.aspx"),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def display_date(iso_date: str) -> str:
    parsed = date.fromisoformat(iso_date)
    return parsed.strftime("%d.%b.%Y")


def category_map() -> dict[str, list[str]]:
    by_ticker: dict[str, list[str]] = {}
    for category, tickers in CATEGORIES.items():
        for ticker in tickers:
            by_ticker.setdefault(ticker, []).append(category)
    return by_ticker


def build_stocks(seed_date: str) -> dict[str, dict[str, Any]]:
    by_ticker = category_map()
    stocks: dict[str, dict[str, Any]] = {}
    for ticker in sorted(by_ticker):
        category_text = ", ".join(by_ticker[ticker])
        base = {
            "ticker": ticker,
            "company": COMPANIES.get(ticker, ticker),
            "category": by_ticker[ticker],
            "positioning": f"AI infrastructure company tracked under {category_text}.",
            "investment_thesis": "Tracked as part of the AI data center supply chain where bottlenecks are moving beyond GPUs into memory, optical connectivity, servers, networking, power, cooling, and deployment capacity.",
            "latest_change": "Initial research database seed.",
            "weekly_catalyst": "Monitor AI infrastructure demand, order conversion, and margin quality.",
            "weekly_risk": "Watch valuation, supply bottlenecks, customer concentration, and demand digestion risk.",
            "benefit_reason": "Potential beneficiary of AI factory buildout and hyperscaler infrastructure spending.",
            "pressure_reason": "May face cyclical demand, component constraints, pricing pressure, or execution risk.",
            "next_watch": "Track earnings commentary, backlog, margin, CapEx, and customer demand signals.",
            "catalysts": [],
            "risks": [],
            "tracking_items": [],
            "earnings_notes": [],
            "news_notes": [],
            "last_updated": seed_date,
        }
        base.update(SPECIAL_RESEARCH.get(ticker, {}))
        stocks[ticker] = base
    return stocks


def build_sources(seed_date: str) -> list[dict[str, Any]]:
    sources = []
    for idx, (ticker, (source_name, url)) in enumerate(SOURCE_URLS.items(), start=1):
        sources.append(
            {
                "source_id": f"src_{seed_date.replace('-', '')}_{idx:03d}",
                "ticker": ticker,
                "date": seed_date,
                "source_name": source_name,
                "url": url,
                "notes": "Imported during initial research database seed.",
            }
        )
    return sources


def build_updates(seed_date: str) -> list[dict[str, Any]]:
    display = display_date(seed_date)
    updates = [
        ("GLW", "Catalyst", "Added Corning to the AI optical and fiber connectivity watchlist."),
        ("NVDA", "Catalyst", "NVIDIA remains the central AI platform, but the trade is moving toward the next bottlenecks."),
        ("DELL", "Risk", "AI server revenue must be evaluated alongside gross margin, working capital, and cash conversion."),
        ("MU", "Risk", "Memory and HBM upside must be balanced against cyclical supply response and valuation risk."),
        ("VRT", "Catalyst", "Power and cooling are becoming core deployment-speed bottlenecks for AI data centers."),
    ]
    return [
        {
            "ticker": ticker,
            "date": seed_date,
            "display_date": display,
            "type": update_type,
            "text": text,
            "display_text": f"{display} {update_type}: {text}",
            "source_id": "",
        }
        for ticker, update_type, text in updates
    ]


def write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local JSON research database.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    db_dir = project_root() / "data" / "database"
    write_json(db_dir / "stocks.json", build_stocks(args.date), overwrite=args.overwrite)
    write_json(db_dir / "sources.json", build_sources(args.date), overwrite=args.overwrite)
    write_json(db_dir / "updates.json", build_updates(args.date), overwrite=args.overwrite)
    print(f"Seeded research database at {db_dir}")


if __name__ == "__main__":
    main()
