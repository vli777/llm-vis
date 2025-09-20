from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.session_store import get_session, get_session_hashes
from app.models import NLQRequest, NLQResponse, TableInfo
from app.nlq_llm import handle_llm_nlq
import pandas as pd
import io
from dotenv import load_dotenv
import logging
import hashlib

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

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    sid = require_session_id(request)
    content = await file.read()

    # --- NEW: duplicate detection by content hash ---
    file_hash = _sha256_bytes(content)
    sess_hashes = get_session_hashes(sid)
    if file_hash in sess_hashes:
        # Already uploaded in this session; point to existing table
        existing_name = sess_hashes[file_hash]
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "duplicate": True,
                "table": existing_name,
                "detail": "Duplicate upload: this file was already uploaded for this session.",
            },
        )
    # -------------------------------------------------

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    sess = get_session(sid)

    base = file.filename.rsplit(".", 1)[0] if file.filename else "table"
    name = base
    i = 1
    while name in sess:
        i += 1
        name = f"{base}_{i}"

    sess[name] = df

    # NEW: remember the content hash -> table name
    sess_hashes[file_hash] = name

    return {"ok": True, "table": name, "rows": len(df), "columns": list(df.columns)}

@app.get("/tables")
async def tables(request: Request):
    sid = require_session_id(request)
    sess = get_session(sid)
    info = [TableInfo(name=k, rows=len(v)).model_dump() for k, v in sess.items()]
    return {"tables": info}

@app.post("/nlq", response_model=None)
async def nlq(request: Request, body: NLQRequest):
    sid = require_session_id(request)
    sess = get_session(sid)
    try:
        return handle_llm_nlq(body.prompt, sess, client_ctx=body.clientContext or {})
    except Exception as e:
        logger.exception("NLQ failed: %s", e)
        raise HTTPException(status_code=400, detail="Request failed")