from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.storage import get_session, get_session_hashes, get_session_meta
from core.models import NLQRequest
from app.nlq_llm import handle_llm_nlq
from server.api import router as eda_router
import pandas as pd
import io
from dotenv import load_dotenv
import logging
import hashlib
import json
import time
from datetime import datetime
from collections import OrderedDict

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

# Mount the EDA API router
app.include_router(eda_router)


PREVIEW_CACHE_MAX = 512
_preview_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _preview_cache_get(key: tuple):
    cached = _preview_cache.get(key)
    if cached is not None:
        _preview_cache.move_to_end(key)
    return cached


def _preview_cache_set(key: tuple, value: dict) -> None:
    _preview_cache[key] = value
    _preview_cache.move_to_end(key)
    if len(_preview_cache) > PREVIEW_CACHE_MAX:
        _preview_cache.popitem(last=False)


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
    # Handle PyArrow date types that convert_dtypes doesn't support
    try:
        df = df.convert_dtypes(
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=True,
        )
    except (KeyError, ValueError) as e:
        # PyArrow date types (date32, date64) aren't handled by convert_dtypes
        # Convert them to datetime64 first, then retry
        logger.warning(f"convert_dtypes failed with PyArrow types: {e}. Converting manually.")
        for col in df.columns:
            # Check for PyArrow date types
            if hasattr(df[col].dtype, 'pyarrow_dtype'):
                import pyarrow as pa
                pa_type = df[col].dtype.pyarrow_dtype
                if pa.types.is_date(pa_type) or pa.types.is_timestamp(pa_type):
                    df[col] = pd.to_datetime(df[col])
        # Retry conversion after fixing date columns
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
    # Clear any stale previews for this table name (defensive).
    if _preview_cache:
        keys_to_drop = [k for k in _preview_cache.keys() if k[0] == sid and k[1] == name]
        for k in keys_to_drop:
            _preview_cache.pop(k, None)

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


@app.get("/table/{table_name}/preview")
async def table_preview(request: Request, table_name: str, offset: int = 0, limit: int = 50):
    """Get a preview of the table data with cursor pagination."""
    sid = require_session_id(request)
    sess = get_session(sid)

    if table_name not in sess:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    # Cap limit at 100 rows per request
    limit = min(limit, 100)

    cache_key = (sid, table_name, offset, limit)
    cached = _preview_cache_get(cache_key)
    if cached is not None:
        return cached

    df = sess[table_name]
    total_rows = len(df)

    # Get the slice of data
    start = offset
    end = min(offset + limit, total_rows)
    preview_df = df.iloc[start:end]

    # Convert to JSON-safe format
    def df_json_safe_preview(df: pd.DataFrame) -> pd.DataFrame:
        """Replace ±inf -> NaN, then NaN -> None for JSON serialization."""
        if df.empty:
            return df
        tmp = df.replace([float('inf'), float('-inf')], None)
        tmp = tmp.astype(object)
        return tmp.where(pd.notna(tmp), None)

    safe_df = df_json_safe_preview(preview_df)

    has_more = end < total_rows
    next_offset = end if has_more else None

    resp = {
        "table": table_name,
        "columns": list(safe_df.columns),
        "rows": safe_df.to_dict(orient="records"),
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
        "returned_rows": len(safe_df),
        "has_more": has_more,
        "next_offset": next_offset,
    }

    _preview_cache_set(cache_key, resp)
    _log_response("PREVIEW", resp)
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
