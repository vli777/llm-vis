# AI Data Vis (Next.js + FastAPI + Vega-Lite)

Generate beautiful, interactive visualizations from natural language using LLMs. Upload your data, describe what you want to see, and get Vega-Lite charts instantly.

## Features

- **Smart Chart Recommendations**: Automatically suggests appropriate chart types based on your data characteristics (temporal, categorical, numeric)
- **Semantic Data Understanding**: Detects column roles (dates, locations, measures, identifiers) for better visualization choices
- **Structured Outputs**: Pydantic-driven validation ensures reliable, type-safe responses from LLMs
- **Multi-Provider Support**: Works with Groq, OpenAI, or NVIDIA with provider-specific optimizations
- **Rich Context**: LLM receives statistical profiles, data distributions, and visualization hints for smarter suggestions
- **Interactive Charts**: Powered by Vega-Lite with tooltips, zooming, and responsive design

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
- Dataset profiling with visualization hints
- Chart type recommendations
- Pydantic model validation
- Provider capability detection

### Examples

Prompt: "bar chart of founding year vs valuation"
<img width="937" height="1126" alt="image" src="https://github.com/user-attachments/assets/59ebc42b-c62a-4f91-95b1-352db2f69ca9" />

Prompt: "count of top investors table"
<img width="623" height="280" alt="image" src="https://github.com/user-attachments/assets/6f28588d-5984-40f9-ab4c-07c4afc38f65" />
<img width="925" height="1117" alt="image" src="https://github.com/user-attachments/assets/14b06bfd-59b6-4cbb-bf16-9043e5dde6ef" />



