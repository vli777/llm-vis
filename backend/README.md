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
