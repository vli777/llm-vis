# backend/app/session_store.py
import pandas as pd
from typing import Dict

# In-memory session store: {session_id: {table_name: DataFrame}}
SESSIONS: Dict[str, Dict[str, pd.DataFrame]] = {}

# Per-session map of file content hash -> table name
SESS_HASHES: Dict[str, Dict[str, str]] = {}

# Per-session map of table name -> metadata dict
SESS_META: Dict[str, Dict[str, dict]] = {}

# Sample tables loaded from backend/data/*.csv at startup
SAMPLE_TABLES: Dict[str, pd.DataFrame] = {}


def get_session(session_id: str) -> Dict[str, pd.DataFrame]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    return SESSIONS[session_id]


def get_session_hashes(session_id: str) -> Dict[str, str]:
    if session_id not in SESS_HASHES:
        SESS_HASHES[session_id] = {}
    return SESS_HASHES[session_id]


def get_session_meta(session_id: str) -> Dict[str, dict]:
    """Return (and lazily init) the metadata map for this session."""
    if session_id not in SESS_META:
        SESS_META[session_id] = {}
    return SESS_META[session_id]
