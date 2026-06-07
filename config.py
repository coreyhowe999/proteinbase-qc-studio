"""Shared config + secrets for ProteinBase QC Studio."""
from pathlib import Path
import os

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CURVES = DATA / "curves"          # cached kinetic-curve JSONs
DB = DATA / "measurements.db"
CHECKS = ROOT / "checks"          # saved QC checks ("skills")
CSV = DATA / "proteinbase_all.csv"

CURVES.mkdir(parents=True, exist_ok=True)
CHECKS.mkdir(parents=True, exist_ok=True)

CURVE_BASE = "https://proteinbase-pub.t3.storage.dev/kinetic-curves/"

# Targets we pull curve data for (keeps the demo bounded + rich).
# PD-L1 is the example called out in the task brief; il7r adds a 2nd target.
DEMO_TARGETS = ["pd-l1", "il7r"]

# Anthropic model for the natural-language -> check builder.
ANTHROPIC_MODEL = "claude-opus-4-8"

# .env locations to scan for ANTHROPIC_API_KEY (Windows view of the WSL file).
_ENV_PATHS = [
    Path(r"\\wsl$\Ubuntu\home\corey\seqdesign\.env"),
    ROOT / ".env",
]


def anthropic_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    # Streamlit Cloud / hosted: read from st.secrets if the deployer opted in
    try:
        import streamlit as st
        if "ANTHROPIC_API_KEY" in st.secrets:
            v = st.secrets["ANTHROPIC_API_KEY"]
            if v:
                return v
    except Exception:
        pass
    for p in _ENV_PATHS:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            continue
    return None
