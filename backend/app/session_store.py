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