# Deconstruct

**AI-Powered Document Deconstructor with a Tiered Reasoning Architecture**

Transform unstructured PDFs into high-fidelity structured JSON using intelligent System 1 vs System 2 model routing.

## Overview

Deconstruct is an extraction engine that uses a **two-tier reasoning system** to balance cost and accuracy for content extraction:

- **System 1 (Fast):** Llama 3.2 Vision for simple documents (~$0.0001/doc)
- **System 2 (Deep):** Claude Opus 4.5 for complex documents (~$0.05-0.20/doc)
- **Auto-Routing:** Complexity detection determines which system to use
- **Verification Loop:** Critic agent re-extracts failed extractions

## Features

### Intelligence
- **Complexity Detection:** Automatic routing based on document difficulty
- **Semantic Search:** pgvector-powered similarity search across extractions
- **Auto-Verification:** Critic agent validates extraction quality
- **Analytics:** Built-in views for cost analysis and model performance

### Infrastructure
- **Modal GPU:** Parallel batch processing across 20+ GPU containers
- **Supabase:** Local Postgres with pgvector for embeddings
- **React Dashboard:** Real-time job monitoring with persistent state

## Project Structure

```
deconstruct/
├── gpu/                    # Modal functions (Python)
│   ├── schemas.py          # Pydantic models (source of truth)
│   ├── modal_app.py        # Extraction pipeline & reasoning router
│   └── requirements.txt
│
├── web/                    # React dashboard
│   ├── src/
│   │   ├── store/          # Zustand state management
│   │   ├── schemas/        # Zod validation (mirrors Pydantic)
│   │   └── components/     # FileUpload, JobMonitor
│   └── package.json
│
├── supabase/               # Database
│   ├── migrations/         # SQL migrations (mirrors Pydantic)
│   └── config.toml

```

## Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Modal account** (sign up at modal.com)
- **Supabase CLI** (`brew install supabase/tap/supabase`)
- **Anthropic API key** (for Claude Opus)

### 1. Set Up Supabase (Local)

```bash
cd supabase
supabase start

# Note the API URL and anon key printed by this command
```

### 2. Set Up Modal

```bash
cd gpu
pip install -r requirements.txt

# Authenticate with Modal
modal token set

# Create Modal secrets
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# Deploy to Modal
modal deploy modal_app.py
```

### 3. Set Up Web Dashboard

```bash
cd web
npm install
npm run dev

# Open http://localhost:3000
```

## Usage

### Basic Workflow

1. **Upload Documents**
   - Drag & drop PDFs into the web dashboard
   - Files are validated with Zod before upload

2. **Batch Processing**
   - Documents are sent to Modal in batches
   - Each document is scanned for complexity
   - Routed to System 1 or System 2 automatically

3. **Monitor Jobs**
   - Real-time status updates in the dashboard
   - Track: reasoning tier, cost, processing time, confidence

4. **Query Results**
   - Structured JSON stored in Supabase
   - Semantic search via embeddings
   - Export to JSON or connect to your app

### Manual Trigger (Python)

```python
import modal

app = modal.App.lookup("deconstruct-shredder")
batch_extract = modal.Function.lookup("deconstruct-shredder", "batch_extract")

# Prepare batch request
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

request = {
    "batch_id": "test_batch_001",
    "documents": [pdf_bytes],
    "complexity_threshold": 0.8,
    "enable_verification": True,
}

# Run extraction
results = batch_extract.remote(request)
print(results[0].model_dump_json(indent=2))
```

## Configuration

### Complexity Threshold

Adjust the threshold for System 1 vs System 2 routing in `gpu/modal_app.py`:

```python
COMPLEXITY_THRESHOLD = 0.8  # 0.0 (always System 1) to 1.0 (always System 2)
```

Or override per-batch in the web UI.

### Model Selection

To use a different System 2 model:

```python
# In gpu/modal_app.py, system2_extract function
response = client.messages.create(
    model="claude-opus-4-5-20251101",  # Change this
    # ...
)
```

## Architecture Deep Dive

### System 1: Fast Extraction

1. Convert PDF → images (200 DPI)
2. Run Llama 3.2 Vision (11B) on A10G GPU
3. Parse structured JSON from output
4. Validate with Pydantic
5. Store in Supabase

**Average:** 5-10 seconds, $0.0001 per document

### System 2: Deep Reasoning

1. Convert PDF → high-res images (300 DPI)
2. Send to Claude Opus 4.5 with chain-of-thought prompt
3. Model explains reasoning before returning JSON
4. Extract JSON from reasoning trace
5. Validate and store

**Average:** 30-60 seconds, $0.05-0.20 per document

### Complexity Detection

Uses heuristics to score documents (0.0 to 1.0):

- Nested tables: +0.15
- Multi-column layout: +0.10
- Handwriting detected: +0.20
- Low scan quality: +0.15
- Ambiguous language: +0.20
- Complex formulas: +0.10
- Mixed languages: +0.05
- Page count: +0.05 per 200 pages

**Threshold:** Documents with score ≥ 0.8 go to System 2

## Database Schema

Key tables (see `supabase/migrations/` for full schema):

- `batches`: Batch metadata and statistics
- `documents`: Document processing records
- `complexity_markers`: Complexity detection results
- `legal_clauses`, `financial_tables`, `technical_specifications`: Extracted entities
- `embeddings`: Vector embeddings for semantic search
- `reasoning_logs`: Chain-of-thought traces

## Analytics

Query built-in views for insights:

```sql
-- Batch performance
SELECT * FROM batch_performance ORDER BY created_at DESC;

-- Complexity distribution
SELECT * FROM complexity_distribution;

-- Model comparison
SELECT * FROM model_performance;

-- Legal risk analysis
SELECT * FROM legal_risk_analysis WHERE overall_risk = 'Critical';
```

## Cost Optimization

1. **Tune Complexity Threshold**
   - Monitor `complexity_distribution` view
   - Adjust threshold to balance cost vs accuracy

2. **Batch Aggressively**
   - Process 50+ documents per batch
   - Amortize cold start costs

3. **Cache Models**
   - Modal volumes cache Llama weights (~40GB)
   - Saves 30s per cold start

4. **Use Critic Sparingly**
   - Only verify low-confidence extractions
   - Disable for high-volume, low-stakes use cases

## Development

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines.

### Running Tests

```bash
# GPU tests (Modal)
cd gpu
modal run modal_app.py::test_extraction

# Web tests
cd web
npm run test

# Database tests
cd supabase
supabase db test
```

### Adding a New Document Type

1. Define Pydantic schema in `gpu/schemas.py`
2. Add extraction logic to `gpu/modal_app.py`
3. Create migration in `supabase/migrations/`
4. Add Zod schema in `web/src/schemas/validation.ts`
5. Create UI component in `web/src/components/`

See CLAUDE.md for detailed workflow.

## Troubleshooting

### Jobs stuck in "processing"
- Check Modal dashboard for errors
- Increase timeout in function decorator
- Verify API keys in Modal secrets

### "Invalid JSON" errors
- Check `reasoning_logs` table for raw output
- Model may be mixing reasoning with JSON
- Update `extract_json_from_response()`

### Embeddings search returns nothing
- Verify embeddings are generated after extraction
- Check vector dimension (should be 1536)
- Rebuild ivfflat index

## Roadmap

- [ ] Support for more document types (invoices, forms, etc.)
- [ ] Fine-tuned complexity detector (replace heuristics)
- [ ] Streaming extraction for large PDFs
- [ ] Web app authentication (Supabase Auth)
- [ ] Export to common formats (CSV, Excel, etc.)
- [ ] API endpoints for external integrations

## License

MIT

## Contributing

This is a research project. Contributions welcome, but expect rapid iteration and breaking changes.

---

**Built with:** Modal • Claude Opus 4.5 • Llama 3.2 Vision • Supabase • React • Zustand • Zod
