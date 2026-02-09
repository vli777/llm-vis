# CLAUDE.md

## Project Overview

Auto-EDA: an agent-driven exploratory data analysis tool. Users upload CSVs, optionally describe what they want to learn, and the system profiles the data, reasons about intent via LLMs, and returns interactive charts with explanations. Built with Next.js (frontend) + FastAPI (backend) + Recharts/Vega-Lite (visualization).

## Architecture

```
backend/
  main.py              – FastAPI app entry point
  app/                 – LLM integration (chat, loader, models, prompts)
  core/                – Domain models (Pydantic), in-memory storage, utils
  server/              – API routes, SSE streaming, orchestrator
  skills/              – Modular analysis pipeline steps (profile, classify, recommend, build_view, narrate, etc.)

frontend/
  app/                 – Next.js pages (page.tsx is the main UI)
  components/          – React components (UploadZone, PromptBar, charts/)
  components/charts/   – Chart renderers (Bar, Line, Scatter, Histogram, Box, Pie, Heatmap, DataTable)
  lib/                 – API client, SSE hook, utilities
  types/               – TypeScript type definitions (mirrors backend Pydantic models)
```

### Data Flow

1. CSV upload → `POST /upload` → backend parses & stores in-memory per session
2. EDA run → `POST /api/runs?stream=1` → returns run ID
3. SSE stream → `GET /api/runs/{run_id}/events` → profile → classify → recommend → build views → narrate
4. Follow-up queries → `POST /api/runs` with `{ query: "..." }` → appends results to existing report

### Key Patterns

- **Skills architecture**: each analysis step (`skills/`) is a standalone module (profile, classify, intent, recommend, build_view, narrate, validate, summary)
- **SSE streaming**: real-time progress via Server-Sent Events (`server/sse.py`)
- **Multi-provider LLM**: Groq/OpenAI/NVIDIA abstracted in `app/llm_loader.py` with deterministic fallbacks on LLM failure
- **Session-based storage**: in-memory, keyed by session ID from `X-Session-Id` header
- **Chart dispatch**: `RechartsCard.tsx` routes chart specs to type-specific renderers

## Development Commands

### Backend
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev          # dev server on :3000
npm run build        # production build
npm run lint         # ESLint
```

### Tests
```bash
cd backend
python -m pytest test_llm_features.py -v
```

## Configuration

- Backend: `backend/.env` — set `LLM_PROVIDER` (groq/openai/nvidia), `LLM_API_KEY`, `LLM_MODEL`
- Frontend: `frontend/.env.local` — set `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`)

## Code Conventions

### Python (backend)
- snake_case for functions/variables, PascalCase for classes, ALL_CAPS for constants
- Full type annotations on all functions
- Pydantic models with `Field()` validators for all domain objects
- Logging via `logging.getLogger(__name__)`
- Graceful fallback to deterministic logic when LLM calls fail

### TypeScript (frontend)
- camelCase for variables/functions, PascalCase for components/types
- `"use client"` directive on interactive components
- Tailwind CSS for styling with custom theme variables (`--theme-body`, `--color-accent`, etc.)
- Custom theme classes: `theme-muted`, `theme-panel`, `theme-card`, `theme-chip`
- Recharts for most chart types; Vega-Lite available for advanced specs

## Type Alignment

Frontend types in `types/chart.ts` mirror backend Pydantic models in `core/models.py`. When changing chart specs or API shapes, update both sides.
