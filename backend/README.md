# Backend (FastAPI)
## Setup
- Create a virtualenv, then:
```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
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
