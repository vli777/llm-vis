# Auto-EDA (Next.js + FastAPI + Vega-Lite)

Agent-driven, autonomous EDA from a data upload. Upload your dataset, describe what you want to learn, and the system profiles the data, reasons about intent, and returns the best visual explanation.

![Recording 2026-02-08 223201 (1)](https://github.com/user-attachments/assets/970026c2-8785-47d0-8632-07a8e633432f)

## Features

- **Smart Chart Recommendations**: Automatically suggests appropriate chart types based on your data characteristics (temporal, categorical, numeric)
- **Semantic Data Understanding**: Detects column roles (dates, locations, measures, identifiers) for better visualization choices
- **String-Encoded Numeric Parsing**: Automatically recognizes and parses financial data like "$1.3B", "1,234.56" for proper charting
- **Structured Outputs**: Pydantic-driven validation ensures reliable, type-safe responses from LLMs
- **Multi-Provider Support**: Works with Groq, OpenAI, or NVIDIA with provider-specific optimizations
- **Rich Context**: LLM receives statistical profiles, data distributions, and visualization hints for smarter suggestions
- **Interactive Charts**: Powered by Vega-Lite with tooltips, zooming, and responsive design

## High-Level Flow

1. **Upload**: User uploads a CSV in the UI.
2. **Profile**: Backend computes dataset stats, data types, distributions, and column roles.
3. **Agent Reasoning**: An LLM agent interprets the request, uses the profile as context, and chooses a visualization approach.
4. **Spec Generation**: The agent outputs a validated Vega-Lite spec plus any data transforms needed.
5. **Render**: Frontend renders the interactive chart and supports follow-up prompts for iterative EDA.

## Quickstart
### Backend
```
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
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

## Configuration

Create a `.env` file in the backend directory:

```env
LLM_PROVIDER=groq          # Options: groq, openai, nvidia
LLM_API_KEY=your_api_key
LLM_MODEL=llama-3.1-8b-instant
```

**Supported Providers:**
- **Groq** (recommended): Fast inference with Llama models
- **OpenAI**: GPT-4o-mini or other OpenAI models with strict schema validation
- **NVIDIA**: Access to NVIDIA-hosted models

## Testing

```bash
cd backend
python -m pytest test_llm_features.py -v
```

The test suite covers:
- Column role detection (temporal, geographic, measure, categorical)
- String-encoded numeric parsing ($1.3B, 1,234.56, etc.)
- Dataset profiling with visualization hints
- Chart type recommendations
- Pydantic model validation
- Provider capability detection

