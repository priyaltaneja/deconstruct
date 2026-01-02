# Deconstruct - Claude Code Instructions

## Project Overview

Deconstruct is an AI-heavy extraction engine that transforms unstructured documents (PDFs, images) into high-fidelity structured JSON using a tiered reasoning architecture. The core innovation is a **System 1 vs System 2 reasoning router** that intelligently chooses between fast (cheap) and deep (expensive) models based on document complexity.

### Architecture

```
/gpu          → Python/Modal: GPU-accelerated document processing
/web          → React/Zustand/Zod: Lean monitoring dashboard
/supabase     → PostgreSQL + pgvector: Structured data storage & search
```

## Core Principles

When working on this project, Claude Code should follow these principles:

### 1. Prioritize Reasoning
- **Always implement a "Chain-of-Thought" step before final output**
- When extracting data, the model should explain its reasoning before returning JSON
- Use structured prompts that explicitly request step-by-step analysis
- Log all reasoning traces to the `reasoning_logs` table

### 2. GPU First
- **All heavy processing MUST be offloaded to `gpu/modal_app.py`**
- Never run ML models, PDF processing, or vision tasks in the web frontend
- The web app is a thin client that triggers GPU jobs and displays results
- Use Modal's `modal.map()` for parallel batch processing across multiple GPUs

### 3. Fail Fast
- **Use Pydantic (backend) and Zod (frontend) to crash early if LLMs return malformed JSON**
- Validate all data at system boundaries before it enters the database
- If extraction fails validation, immediately retry with System 2
- Never store partial or invalid extraction results

### 4. Functionality Over Beauty
- **The UI should be a lean dashboard for monitoring the GPU pipeline, not a polished product**
- Focus on real-time job status, cost tracking, and result inspection
- Avoid premature optimization, animations, or design polish
- Ship working features fast, iterate on UX later

## System 1 vs System 2 Design

### System 1: Fast & Cheap (Llama 3.2 Vision)
- **When to use:** Simple documents (complexity_score < 0.8)
- **Model:** Llama 3.2 11B Vision (runs on A10G GPU)
- **Cost:** ~$0.0001 per document
- **Use cases:** Clean PDFs, simple contracts, single-page documents

### System 2: Deep Reasoning (Claude Opus / DeepSeek-R1)
- **When to use:** Complex documents (complexity_score ≥ 0.8) OR System 1 verification fails
- **Model:** Claude Opus 4.5 (via API)
- **Cost:** ~$0.05-0.20 per document (depending on length)
- **Use cases:** Nested tables, legal jargon, multi-column layouts, poor scans

### Routing Logic (in `gpu/modal_app.py`)

```python
# 1. Fast scan with System 1 (Llama 8B) to detect complexity
markers = system1_scan(pdf_bytes)
complexity_score = markers.complexity_score  # 0.0 to 1.0

# 2. Route based on threshold
if complexity_score >= THRESHOLD or force_system2:
    result = system2_extract(pdf_bytes)  # Claude Opus
else:
    result = system1_extract(pdf_bytes)  # Llama Vision

# 3. Verify extraction
if enable_verification and result.confidence_score < 0.7:
    result = verify_extraction(pdf_bytes, result)  # Critic agent
```

## Code Guidelines

### When Modifying `/gpu` (Modal Functions)

1. **Always use type hints and Pydantic models**
   ```python
   def extract_document(pdf_bytes: bytes, doc_id: str) -> ExtractionResult:
       # ... extraction logic
       return ExtractionResult(**result_dict)  # Pydantic validation
   ```

2. **Parallelize everything with `modal.map()`**
   ```python
   results = list(system1_extract.map(pdf_list, doc_ids))
   ```

3. **Cache models on Modal volumes**
   ```python
   @app.function(volumes={"/models": models_volume})
   def load_model():
       model = AutoModel.from_pretrained("...", cache_dir="/models")
   ```

4. **Always calculate and log costs**
   ```python
   cost_usd = (input_tokens * PRICE_PER_INPUT_TOKEN) + (output_tokens * PRICE_PER_OUTPUT_TOKEN)
   ```

### When Modifying `/web` (React/Zustand)

1. **Use Zustand for all state management**
   - Store extraction jobs in `useExtractionStore`
   - Persist to IndexedDB with localForage (survives tab closures)
   - Never use React state for job tracking

2. **Validate all inputs with Zod before sending to backend**
   ```typescript
   const validation = validateBatchRequest(formData);
   if (!validation.valid) {
       alert(validation.error);
       return;
   }
   ```

3. **Display real-time job status**
   - Poll backend every 2-5 seconds for active jobs
   - Show: status, reasoning tier, cost, processing time
   - Use color coding: green (completed), blue (processing), red (failed)

4. **Functionality-first components**
   - Don't add unnecessary animations or transitions
   - Focus on data density and clarity
   - Use simple inline styles for now (no CSS frameworks)

### When Modifying `/supabase` (Database)

1. **Mirror Pydantic schemas in SQL**
   - If you add a field to `schemas.py`, add it to the migration
   - Use CHECK constraints to enforce valid values
   - Use JSONB for nested/dynamic data

2. **Use pgvector for embeddings**
   ```sql
   CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
   ```

3. **Update batch stats with triggers**
   - The `update_batch_stats()` trigger auto-updates batch statistics
   - Never manually update `batches.completed_jobs` from app code

4. **Use analytical views for dashboards**
   - Query `batch_performance` instead of aggregating manually
   - Views are pre-optimized and easier to maintain

## Workflow: Adding a New Document Type

Example: Adding support for "Medical Records"

### 1. Define Pydantic Schema (`gpu/schemas.py`)

```python
class MedicalRecord(BaseModel):
    record_type: Literal["lab_result", "prescription", "diagnosis"]
    patient_id: str
    date: str
    findings: List[str]
    physician_notes: str
```

### 2. Update Extraction Prompts (`gpu/modal_app.py`)

```python
def get_schema_prompt(document_type: str) -> str:
    if document_type == "medical":
        return """Extract medical data matching the MedicalRecord schema.
        Include all lab results, prescriptions, and diagnoses."""
```

### 3. Create Database Tables (`supabase/migrations/`)

```sql
CREATE TABLE medical_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id),
    record_type TEXT CHECK (record_type IN ('lab_result', 'prescription', 'diagnosis')),
    -- ... other fields
);
```

### 4. Update Frontend Types (`web/src/schemas/validation.ts`)

```typescript
export const medicalRecordSchema = z.object({
    record_type: z.enum(['lab_result', 'prescription', 'diagnosis']),
    // ... mirror Pydantic schema
});
```

### 5. Display in UI (`web/src/components/`)

Create `MedicalRecordView.tsx` to render extracted medical data.

## Testing & Deployment

### Local Development

```bash
# Start Supabase (local Postgres + pgvector)
cd supabase
supabase start

# Run Modal locally (requires Modal account)
cd gpu
modal run modal_app.py

# Start web dev server
cd web
npm install
npm run dev
```

### Deploying to Modal

```bash
cd gpu
modal deploy modal_app.py
```

This creates a persistent endpoint at `https://{your-workspace}.modal.run/batch_extract`

## Cost Optimization Tips

1. **Set appropriate complexity thresholds**
   - Start with 0.8, tune based on actual performance
   - Monitor `complexity_distribution` view to see routing decisions

2. **Batch documents aggressively**
   - Use `modal.map()` to process 20+ docs in parallel
   - Amortize cold start costs across batches

3. **Cache model weights on Modal volumes**
   - Llama Vision model is ~40GB, caching saves 30s per cold start

4. **Use System 1 for retry attempts**
   - If System 2 fails, don't retry with System 2 again
   - Use critic agent to identify specific issues

## Reasoning Quality Checklist

When implementing extraction logic, ensure:

- [ ] Prompt includes "Think step-by-step" instruction
- [ ] Model outputs reasoning trace before JSON
- [ ] Reasoning trace is stored in `reasoning_logs` table
- [ ] Critic agent compares output to source (not just validates JSON)
- [ ] Confidence score is based on cross-reference verification
- [ ] Low confidence triggers System 2 re-extraction

## Anti-Patterns to Avoid

❌ **Don't:**
- Run ML models in the browser
- Store API keys in frontend code
- Manually update batch statistics (use triggers)
- Create new tables without corresponding Pydantic models
- Skip Zod validation on user input
- Add UI polish before core functionality works
- Use `git commit --amend` without explicit user request

✅ **Do:**
- Offload all heavy work to Modal
- Validate at every boundary (Zod → API → Pydantic → SQL)
- Log reasoning traces for debugging
- Use parallel processing whenever possible
- Keep UI lean and functional
- Fail fast on validation errors

## Key Files Reference

| File | Purpose |
|------|---------|
| `gpu/schemas.py` | Pydantic models (source of truth) |
| `gpu/modal_app.py` | Main extraction pipeline |
| `web/src/store/extractionStore.ts` | Zustand state management |
| `web/src/schemas/validation.ts` | Zod validation (mirrors Pydantic) |
| `supabase/migrations/*.sql` | Database schema (mirrors Pydantic) |

## Debugging Common Issues

### "Extraction returned invalid JSON"
- Check `reasoning_logs` table for the raw model output
- Look for reasoning text mixed with JSON
- Update `extract_json_from_response()` to handle new format

### "Complexity score always 1.0"
- Check `system1_scan()` heuristics are running correctly
- Verify PDF → image conversion is working
- Test OCR on sample pages

### "Jobs stuck in 'processing' status"
- Check Modal dashboard for failed function calls
- Look for timeout errors (increase `timeout` parameter)
- Verify API keys are set in Modal secrets

### "Embeddings search returns no results"
- Check that `embeddings` table is populated
- Verify vector dimension matches model (1536 for bge-large)
- Rebuild ivfflat index if data was bulk-inserted

---

**Remember:** This is a research tool focused on extraction quality and cost-efficiency, not a production SaaS. Prioritize functionality, reasoning quality, and GPU optimization over UI polish.
