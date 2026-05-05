"""Single source of truth for the FMP API key.

Resolution order:
1. Environment variable FMP_API_KEY (good for CLI / CI)
2. Streamlit secrets FMP_API_KEY (good for deployed app)
3. .streamlit/secrets.toml parsed directly (good for CLI when running from project root)
"""

import os
from pathlib import Path


def _read_secrets_toml() -> str:
    """Parse .streamlit/secrets.toml directly — works from any cwd up to project root."""
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[3] / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if path.exists():
            try:
                try:
                    import tomllib  # Python 3.11+
                    data = tomllib.loads(path.read_text())
                except ImportError:
                    import tomli as tomllib  # type: ignore
                    data = tomllib.loads(path.read_text())
                return data.get("FMP_API_KEY", "")
            except Exception:
                # fall back to naive line scan
                for line in path.read_text().splitlines():
                    if line.strip().startswith("FMP_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"').strip("'")
    return ""


def get_fmp_key() -> str:
    key = os.getenv("FMP_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("FMP_API_KEY", "")
        except Exception:
            pass
    if not key:
        key = _read_secrets_toml()
    return key.strip()
