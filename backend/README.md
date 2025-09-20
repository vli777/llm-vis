# Backend (FastAPI)
## Setup
- Create a virtualenv, then:
```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Env
```
LLM_BASE_URL=https://integrate.api.nvidia.com/v1    
LLM_API_KEY=***
LLM_MODEL=openai/gpt-oss-120b
```

## Endpoints
- `POST /upload` (multipart) with header `X-Session-Id`
- `GET /tables` with header `X-Session-Id`
- `POST /nlq` with JSON `{ "prompt": "..." }` and header `X-Session-Id`

Data is stored in-memory per session. This skeleton includes rule-based handling for:
- Industry pie
- Founded year vs valuation scatter
- Top investors table
- ARR vs valuation correlation chart

## LLM configuration

Set `LLM_PROVIDER` to switch between providers (defaults to `nvidia`).

- **NVIDIA** (default): requires `LLM_API_KEY` or `NVIDIA_API_KEY`. Optional overrides:
  - `LLM_MODEL` (defaults to `meta/llama-3.1-8b-instruct`)
  - `LLM_BASE_URL` (defaults to `https://integrate.api.nvidia.com/v1`)
- **OpenAI**: set `LLM_PROVIDER=openai` and `LLM_API_KEY` (or `OPENAI_API_KEY`). Optional overrides:
  - `LLM_MODEL` (defaults to `gpt-4o-mini`)
  - `LLM_BASE_URL` / `OPENAI_BASE_URL` for Azure-compatible endpoints

The backend uses LangChain chat models, so installing `langchain-openai` is necessary when targeting OpenAI.
