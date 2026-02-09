# Auto-EDA

Agent-driven, autonomous exploratory data analysis. Upload a dataset, describe what you want to learn, and the system profiles the data, reasons about intent, and returns the best visual explanation — all streamed in real time.

![Recording 2026-02-08 223201 (1)](https://github.com/user-attachments/assets/970026c2-8785-47d0-8632-07a8e633432f)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Visualization | Recharts, Vega-Lite |
| Backend | FastAPI, Python 3, Pandas, Pydantic |
| LLM | LangChain (Groq, OpenAI, NVIDIA) |
| Streaming | Server-Sent Events (SSE) |

## Features

- **Autonomous EDA Pipeline**: Upload a CSV and get a full analysis — profiling, intent detection, chart recommendations, and narrated insights — with no manual steps
- **Real-Time Streaming**: SSE-powered progress updates render charts and findings as they're generated
- **Smart Chart Recommendations**: Deterministic recommendation engine picks chart types based on column roles, cardinality, and data distributions
- **Semantic Column Detection**: Automatically classifies columns as temporal, geographic, measure, categorical, or identifier
- **String-Encoded Numeric Parsing**: Recognizes financial shorthand like "$1.3B", "1,234.56" and converts to proper numbers
- **Multi-Provider LLM Support**: Works with Groq, OpenAI, or NVIDIA — falls back to deterministic logic on LLM failure
- **Follow-Up Queries**: Ask natural language questions to drill deeper into your data after the initial analysis
- **Interactive Charts**: Bar, line, scatter, histogram, box, pie, heatmap, and table views with tooltips and responsive layout

## How It Works

1. **Upload** — User uploads a CSV via the drag-and-drop UI
2. **Profile** — Backend computes statistics, detects column roles, and builds a data profile
3. **Classify** — LLM refines column roles using semantic understanding
4. **Recommend** — Deterministic engine generates chart candidates scored by relevance
5. **Build** — Selected plans are rendered into chart specs with pre-aggregated data
6. **Narrate** — LLM generates explanations and insights for each visualization
7. **Stream** — Results are streamed to the frontend via SSE as each step completes

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # ensure NEXT_PUBLIC_API_BASE matches backend URL
npm run dev
```

Open http://localhost:3000, upload a CSV, and try prompts like:
- "Create a pie chart representing industry breakdown"
- "Create a scatter plot of founded year and valuation"
- "Create a table to see which investors appear most frequently"
- "Give me the best representation of data if I want to understand the correlation of ARR and Valuation"

## Configuration

Create a `.env` file in the `backend/` directory:

```env
LLM_PROVIDER=groq          # Options: groq, openai, nvidia
LLM_API_KEY=your_api_key
LLM_MODEL=llama-3.1-8b-instant
```

| Provider | Notes |
|----------|-------|
| **Groq** (default) | Fast inference with Llama models |
| **OpenAI** | GPT-4o-mini or other models with strict schema validation |
| **NVIDIA** | Access to NVIDIA-hosted models |

Frontend config lives in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Project Structure

```
backend/
  main.py                  # FastAPI entry point
  app/                     # LLM integration (chat, loader, models, prompts)
  core/                    # Domain models, in-memory storage, utilities
  server/                  # API routes, SSE streaming, orchestrator
  skills/                  # Analysis pipeline modules
  test_llm_features.py     # Test suite

frontend/
  app/                     # Next.js pages and layout
  components/              # UI components (upload, prompt bar, charts)
  components/charts/       # Chart renderers (bar, line, scatter, histogram, etc.)
  lib/                     # API client, SSE hook, utilities
  types/                   # TypeScript type definitions
```

## Testing

```bash
cd backend
python -m pytest test_llm_features.py -v
```

Covers column role detection, numeric parsing, dataset profiling, chart recommendations, model validation, and provider capabilities.
