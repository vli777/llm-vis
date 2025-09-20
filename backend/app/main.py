from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.session_store import get_session
from app.models import NLQRequest, NLQResponse, TableInfo
from app.nlq_llm import handle_llm_nlq
import pandas as pd
import io
from dotenv import load_dotenv
import logging

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

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    sid = require_session_id(request)
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    sess = get_session(sid)
    # simple unique table name
    base = file.filename.rsplit(".", 1)[0]
    name = base
    i = 1
    while name in sess:
        i += 1
        name = f"{base}_{i}"
    sess[name] = df
    return {"ok": True, "table": name, "rows": len(df), "columns": list(df.columns)}

@app.get("/tables")
async def tables(request: Request):
    sid = require_session_id(request)
    sess = get_session(sid)
    info = [TableInfo(name=k, rows=len(v)).model_dump() for k, v in sess.items()]
    return {"tables": info}

@app.post("/nlq", response_model=NLQResponse)
async def nlq(request: Request, body: NLQRequest):
    sid = require_session_id(request)
    sess = get_session(sid)
    try:
        return handle_llm_nlq(body.prompt, sess, client_ctx=body.clientContext or {})
    except Exception as e:
        logger.exception("NLQ failed: %s", e)
        raise HTTPException(status_code=400, detail="Request failed")

