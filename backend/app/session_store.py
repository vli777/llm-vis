# backend/app/session_store.py
import pandas as pd
from typing import Dict
from pathlib import Path

# Simple in-memory session store: {session_id: {table_name: DataFrame}}
SESSIONS: Dict[str, Dict[str, pd.DataFrame]] = {}

# Sample tables loaded from backend/data/*.csv at startup
SAMPLE_TABLES: Dict[str, pd.DataFrame] = {}

def get_session(session_id: str) -> Dict[str, pd.DataFrame]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    return SESSIONS[session_id]

def load_samples() -> None:
    """Load all CSVs from backend/data into SAMPLE_TABLES."""
    SAMPLE_TABLES.clear()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not data_dir.exists():
        return
    for p in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            SAMPLE_TABLES[p.stem] = df
        except Exception:
            # ignore unreadable files
            pass

def ensure_seeded(session_id: str) -> Dict[str, pd.DataFrame]:
    """If the session has no tables yet, copy the samples into it."""
    sess = get_session(session_id)
    if not sess and SAMPLE_TABLES:
        for name, df in SAMPLE_TABLES.items():
            sess[name] = df.copy()
    return sess
