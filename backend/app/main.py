from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.session_store import get_session, get_session_hashes, get_session_meta
from app.models import NLQRequest
from app.nlq_llm import handle_llm_nlq
import pandas as pd
import io
from dotenv import load_dotenv
import logging
import hashlib
import json
import time
from datetime import datetime

logger = logging.getLogger("uvicorn.error")
load_dotenv()
app = FastAPI(title="AI Data Vis", description="Turn prompts into charts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_session_id(request: Request) -> str:
    sid = request.headers.get("X-Session-Id")
    if not sid:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")
    return sid


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _log_response(ctx: str, payload) -> None:
    """Pretty-print JSON-able payloads; fall back to str()."""
    try:
        logger.info("%s response: %s", ctx, json.dumps(payload, indent=2, default=str))
    except Exception:
        logger.info("%s response (non-serializable): %s", ctx, str(payload))


def human_dtype(s: pd.Series) -> str:
    dt = s.dtype
    # straightforward cases
    if pd.api.types.is_string_dtype(dt):
        return "string"
    if pd.api.types.is_bool_dtype(dt):
        return "boolean"
    if pd.api.types.is_integer_dtype(dt):
        return "integer"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_categorical_dtype(dt):
        return "category"
    # object -> check if it's actually all strings
    if pd.api.types.is_object_dtype(dt):
        nonna = s.dropna()
        if len(nonna) == 0 or (nonna.map(type) == str).all():
            return "string"
        return "object"
    # fallback (arrow-backed or other extension types)
    return str(dt)


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    sid = require_session_id(request)
    content = await file.read()

    file_size = len(content)
    filename = file.filename or "table.csv"
    ext = (filename.rsplit(".", 1)[1].lower() if "." in filename else "").strip()

    # duplicate detection (by content hash)
    file_hash = _sha256_bytes(content)
    sess_hashes = get_session_hashes(sid)
    if file_hash in sess_hashes:
        existing_name = sess_hashes[file_hash]
        dup_resp = {
            "ok": False,
            "duplicate": True,
            "table": existing_name,
            "detail": "Duplicate upload: this file was already uploaded for this session.",
        }
        _log_response("UPLOAD (duplicate)", dup_resp)
        return JSONResponse(status_code=409, content=dup_resp)

    # --- read CSV with pyarrow if available; fallback otherwise ---
    try:
        df = pd.read_csv(io.BytesIO(content), engine="pyarrow", dtype_backend="pyarrow")
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logger.exception("Failed to read CSV")
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # --- normalize to pandas nullable dtypes (so text -> 'string', ints -> 'Int64', etc.) ---
    df = df.convert_dtypes(
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=True,
    )

    # If any columns remain object but are all strings, explicitly cast to string dtype
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s.dtype):
            nonna = s.dropna()
            if len(nonna) == 0 or (nonna.map(type) == str).all():
                df[c] = s.astype("string")

    # --- session + unique name ---
    sess = get_session(sid)
    meta_store = get_session_meta(sid)

    base = filename.rsplit(".", 1)[0] if filename else "table"
    name = base
    i = 1
    while name in sess:
        i += 1
        name = f"{base}_{i}"

    # store dataframe and remember hash -> name
    sess[name] = df
    sess_hashes[file_hash] = name

    # --- metadata ---
    created_at = datetime.utcnow().isoformat() + "Z"
    meta_store[name] = {
        "file_name": filename,
        "file_ext": ext,
        "file_size": file_size,
        "created_at": created_at,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: human_dtype(df[c]) for c in df.columns},  # human-friendly types
    }

    resp = {
        "ok": True,
        "table": name,
        "rows": len(df),
        "columns": list(df.columns),
        "meta": meta_store[name],  # handy for FE display
    }
    _log_response("UPLOAD", resp)
    return resp


@app.get("/tables")
async def tables(request: Request):
    sid = require_session_id(request)
    sess = get_session(sid)
    meta_store = get_session_meta(sid)

    tables_info = []
    for name, df in sess.items():
        # compute on the fly if missing (e.g., sample tables)
        meta = meta_store.get(name) or {
            "file_name": name,
            "file_ext": "",
            "file_size": None,
            "created_at": None,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
        tables_info.append(
            {
                "name": name,
                **meta,
            }
        )

    resp = {"tables": tables_info}
    _log_response("TABLES", resp)
    return resp


@app.post("/nlq", response_model=None)
async def nlq(request: Request, body: NLQRequest):
    sid = require_session_id(request)
    sess = get_session(sid)
    meta_store = get_session_meta(sid)
    t0 = time.perf_counter()
    try:
        result = handle_llm_nlq(
            body.prompt,
            sess,
            client_ctx=body.clientContext or {},
            meta_store=meta_store,
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)

        # Log prompt + result (pretty JSON where possible)
        meta = {
            "prompt": body.prompt,
            "client_ctx_keys": list((body.clientContext or {}).keys()),
            "duration_ms": dt_ms,
        }
        try:
            logger.info("NLQ meta: %s", json.dumps(meta, indent=2))
        except Exception:
            logger.info("NLQ meta: %s", meta)

        _log_response("NLQ", result)
        return result
    except Exception:
        logger.exception("NLQ failed")
        raise HTTPException(status_code=400, detail="Request failed")
