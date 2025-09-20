# AI Data Vis (Next.js + FastAPI + Vega-Lite)

## Quickstart
### Backend
```
cd backend
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```
cd frontend
npm install
cp .env.local.example .env.local   # ensure NEXT_PUBLIC_API_BASE matches backend URL
npm run dev
```
Open http://localhost:3000, upload your CSV, and try prompts like:
- "Create a pie chart representing industry breakdown"
- "Create a scatter plot of founded year and valuation"
- "Create a table to see which investors appear most frequently"
- "Give me the best representation of data if I want to understand the correlation of ARR and Valuation"
