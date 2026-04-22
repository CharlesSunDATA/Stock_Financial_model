# Stock_Financial_model

Single Streamlit app that bundles three core features:

- **Stock valuation**: two-stage DCF + trailing P/E zone vs historical range
- **Quarterly financials**: live pull from Yahoo Finance (yfinance), with margins/growth/leverage shortcuts
- **Markowitz optimization**: max Sharpe + min volatility with **per-asset weight cap** (0–20%) to avoid corner solutions

## Run locally

```bash
cd Stock_Financial_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

## Deploy (Streamlit Community Cloud)

- Set the repo root to this project
- Set the main file to `app.py`
- Install requirements from `requirements.txt`

