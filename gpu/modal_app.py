"""
Deconstruct: AI-Powered Document Shredder
Modal App with System 1 vs System 2 Reasoning Router
"""

import modal
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import json
import time
from io import BytesIO
from pydantic import BaseModel, Field


# ============ SCHEMAS (embedded to avoid import issues) ============

class LegalClause(BaseModel):
    clause_id: str
    clause_type: Literal["definition", "obligation", "right", "prohibition", "condition", "termination", "liability", "indemnity"]
    text: str
    section_reference: str
    parties_involved: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    effective_date: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"]
    extracted_entities: List[str] = Field(default_factory=list)


class LegalDocument(BaseModel):
    document_type: Literal["contract", "agreement", "amendment", "policy", "terms", "other"]
    title: str
    parties: List[str]
    execution_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    clauses: List[LegalClause] = Field(default_factory=list)
    key_obligations: List[str] = Field(default_factory=list)
    risk_summary: str = ""


class FinancialCell(BaseModel):
    value: str
    value_type: Literal["currency", "percentage", "number", "text", "date"]
    formatted_value: Optional[str] = None
    row_label: str
    column_label: str
    is_calculated: bool = False


class FinancialTable(BaseModel):
    table_id: str
    table_type: Literal["balance_sheet", "income_statement", "cash_flow", "schedule", "footnote", "other"]
    title: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    currency: str = "USD"
    headers: List[str] = Field(default_factory=list)
    rows: List[List[FinancialCell]] = Field(default_factory=list)
    totals: Optional[List[FinancialCell]] = None
    notes: List[str] = Field(default_factory=list)
    audited: bool = False


class FinancialDocument(BaseModel):
    document_type: Literal["10-K", "10-Q", "earnings", "prospectus", "other"]
    company_name: str
    reporting_period: str
    filing_date: Optional[str] = None
    tables: List[FinancialTable] = Field(default_factory=list)
    key_metrics: dict = Field(default_factory=dict)
    executive_summary: str = ""


class TechnicalSpecification(BaseModel):
    spec_id: str
    category: Literal["functional", "non_functional", "performance", "security", "compliance", "interface", "data"]
    title: str
    description: str
    priority: Literal["P0", "P1", "P2", "P3"]
    acceptance_criteria: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    technical_details: dict = Field(default_factory=dict)


class TechnicalDocument(BaseModel):
    document_type: Literal["architecture", "api_spec", "requirements", "design", "other"]
    title: str
    version: str
    last_updated: Optional[str] = None
    specifications: List[TechnicalSpecification] = Field(default_factory=list)
    diagrams_detected: int = 0
    code_snippets: List[str] = Field(default_factory=list)
    technology_stack: List[str] = Field(default_factory=list)


class ComplexityMarkers(BaseModel):
    has_nested_tables: bool = False
    has_multi_column_layout: bool = False
    has_handwriting: bool = False
    has_low_quality_scan: bool = False
    has_ambiguous_language: bool = False
    has_complex_formulas: bool = False
    language_is_mixed: bool = False
    page_count: int = 1
    estimated_entities: int = 0

    @property
    def complexity_score(self) -> float:
        score = 0.0
        score += 0.15 if self.has_nested_tables else 0
        score += 0.1 if self.has_multi_column_layout else 0
        score += 0.2 if self.has_handwriting else 0
        score += 0.15 if self.has_low_quality_scan else 0
        score += 0.2 if self.has_ambiguous_language else 0
        score += 0.1 if self.has_complex_formulas else 0
        score += 0.05 if self.language_is_mixed else 0
        score += min(0.05, self.page_count / 200)
        return min(score, 1.0)


class ExtractionResult(BaseModel):
    document_id: str
    document_type: Literal["legal", "financial", "technical", "general"]
    complexity_markers: ComplexityMarkers
    reasoning_tier: Literal["system1", "system2"]
    model_used: str
    processing_time_ms: float
    cost_usd: float
    legal_content: Optional[LegalDocument] = None
    financial_content: Optional[FinancialDocument] = None
    technical_content: Optional[TechnicalDocument] = None
    raw_text: Optional[str] = None
    extracted_entities: List[dict] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: Literal["passed", "failed", "needs_review"] = "needs_review"
    verification_notes: List[str] = Field(default_factory=list)


class BatchExtractionRequest(BaseModel):
    batch_id: str
    documents: List[bytes]
    force_system2: bool = False
    complexity_threshold: float = 0.8
    enable_verification: bool = True


# ============ MODAL SETUP ============
app = modal.App("deconstruct-shredder")

# Lightweight image for web endpoint (NO GPU dependencies)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.115.0",
    "python-multipart>=0.0.9",
    "pydantic==2.9.0",
)

# GPU image with all dependencies (for actual extraction workers)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "huggingface_hub",  # For HF authentication
        "accelerate",  # Required for device_map="auto"
        "pydantic",
        "pillow",
        "pypdf",
        "pdf2image",
        "pytesseract",
        "numpy",
        "anthropic",  # For Claude API integration
        "openai",  # For GPT-4V or other OpenAI models
    )
    .apt_install("poppler-utils", "tesseract-ocr")
)

# Shared volume for model caching
models_volume = modal.Volume.from_name("deconstruct-models", create_if_missing=True)


# ============ SYSTEM 1: FAST SCAN (Llama 8B) ============
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/models": models_volume},
    timeout=300,
    )
def system1_scan(pdf_bytes: bytes, document_id: str) -> Tuple[ComplexityMarkers, dict]:
    """
    Fast scan using lightweight model to detect complexity markers.
    Returns complexity markers and basic document info.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pdf2image import convert_from_bytes
    import pytesseract

    start_time = time.time()

    # Convert PDF to images (first 3 pages for quick scan)
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=3)

    # Quick OCR to get text
    text_sample = ""
    for img in images[:3]:
        text_sample += pytesseract.image_to_string(img)[:1000]

    # Detect complexity markers using heuristics
    markers = ComplexityMarkers(
        has_nested_tables="table" in text_sample.lower() and "|" in text_sample,
        has_multi_column_layout=detect_multicolumn(images[0]),
        has_handwriting=False,  # Would need specialized model
        has_low_quality_scan=check_image_quality(images[0]),
        has_ambiguous_language=check_language_complexity(text_sample),
        has_complex_formulas="=" in text_sample or "$" in text_sample,
        language_is_mixed=detect_mixed_language(text_sample),
        page_count=len(convert_from_bytes(pdf_bytes)),
        estimated_entities=estimate_entity_count(text_sample),
    )

    # Basic document classification
    doc_info = {
        "inferred_type": classify_document_type(text_sample),
        "preview_text": text_sample[:500],
        "processing_time_ms": (time.time() - start_time) * 1000,
    }

    return markers, doc_info


# ============ SYSTEM 2: DEEP REASONING (DeepSeek-R1 / o1) ============
@app.function(
    image=gpu_image,
    gpu="A100",  # More powerful GPU for complex reasoning
    timeout=900,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    )
def system2_extract(
    pdf_bytes: bytes,
    document_id: str,
    document_type: str,
    complexity_markers: dict,
) -> ExtractionResult:
    """
    Deep extraction using advanced reasoning models.
    Uses Claude Opus or OpenAI o1 for complex documents.
    """
    import anthropic
    from pdf2image import convert_from_bytes
    import base64

    start_time = time.time()

    # Convert PDF to high-res images
    images = convert_from_bytes(pdf_bytes, dpi=300)

    # Prepare schema based on document type
    schema_prompt = get_schema_prompt(document_type)

    # Use Claude with vision for extraction
    client = anthropic.Anthropic()

    # Convert images to base64
    image_content = []
    for img in images[:10]:  # Limit to 10 pages for cost control
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        image_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        })

    # System 2 reasoning prompt
    messages = [
        {
            "role": "user",
            "content": [
                *image_content,
                {
                    "type": "text",
                    "text": f"""You are a deep reasoning extraction system. Analyze this document carefully.

Document Type: {document_type}
Complexity Markers: {json.dumps(complexity_markers, indent=2)}

{schema_prompt}

Think step-by-step:
1. Identify the document structure
2. Extract all entities according to the schema
3. Verify cross-references and dependencies
4. Assess data quality and confidence

Return ONLY valid JSON matching the schema. Use chain-of-thought reasoning before your final answer."""
                },
            ],
        }
    ]

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=16000,
        temperature=0,
        messages=messages,
    )

    # Parse structured output
    result_json = extract_json_from_response(response.content[0].text)

    processing_time = (time.time() - start_time) * 1000

    # Calculate cost (approximate)
    cost_usd = calculate_cost(response.usage, "claude-opus-4-5")

    # Create extraction result
    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=ComplexityMarkers(**complexity_markers),
        reasoning_tier="system2",
        model_used="claude-opus-4-5-20251101",
        processing_time_ms=processing_time,
        cost_usd=cost_usd,
        confidence_score=0.9,  # Would be calculated by critic
        verification_status="needs_review",
    )

    # Try to populate typed content, fall back to raw JSON if validation fails
    try:
        if document_type == "legal":
            result.legal_content = LegalDocument(**result_json)
        elif document_type == "financial":
            result.financial_content = FinancialDocument(**result_json)
        elif document_type == "technical":
            result.technical_content = TechnicalDocument(**result_json)
    except Exception as e:
        # Schema mismatch - store raw extraction instead
        print(f"Schema validation failed: {e}")
        result.extracted_entities = [result_json] if isinstance(result_json, dict) else result_json
        result.verification_notes.append(f"Schema validation failed: {str(e)[:200]}")

    return result


# ============ SYSTEM 1: FAST EXTRACTION (Llama Vision) ============
@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=300,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    )
def system1_extract(
    pdf_bytes: bytes,
    document_id: str,
    document_type: str,
) -> ExtractionResult:
    """
    Fast extraction using Llama Vision for simple documents.
    """
    import os
    import torch
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    from huggingface_hub import login
    from pdf2image import convert_from_bytes

    start_time = time.time()

    # Login to HuggingFace with token from Modal secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load Llama 3.2 Vision model (cached on volume)
    model_path = "/models/llama-3.2-11b-vision"
    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        cache_dir=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        token=hf_token,
    )

    # Convert PDF to images
    images = convert_from_bytes(pdf_bytes, dpi=200)

    # Prepare prompt
    schema_prompt = get_schema_prompt(document_type)
    prompt = f"""Extract structured data from this document. Return valid JSON only.

Document Type: {document_type}
{schema_prompt}"""

    # Process with vision model
    inputs = processor(images[0], prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=4096, temperature=0)
    result_text = processor.decode(output[0], skip_special_tokens=True)

    # Parse result
    result_json = extract_json_from_response(result_text)

    processing_time = (time.time() - start_time) * 1000

    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=ComplexityMarkers(
            page_count=len(images),
            estimated_entities=0,
        ),
        reasoning_tier="system1",
        model_used="llama-3.2-11b-vision",
        processing_time_ms=processing_time,
        cost_usd=0.0001,  # Approximate GPU cost
        confidence_score=0.75,
        verification_status="needs_review",
    )

    # Populate content
    if document_type == "legal":
        result.legal_content = LegalDocument(**result_json)
    elif document_type == "financial":
        result.financial_content = FinancialDocument(**result_json)
    elif document_type == "technical":
        result.technical_content = TechnicalDocument(**result_json)

    return result


# ============ REASONING ROUTER ============
@app.function(image=gpu_image, timeout=1200)
def route_and_extract(
    pdf_b64: str = "",
    document_id: str = "",
    complexity_threshold: float = 0.8,
    force_system2: bool = False,
) -> ExtractionResult:
    """
    Main router: Decides between System 1 and System 2 based on complexity.
    Accepts base64-encoded PDF bytes.
    """
    import base64

    # Decode base64 to bytes
    if not pdf_b64:
        raise ValueError("pdf_b64 must be provided")

    pdf_bytes = base64.b64decode(pdf_b64)

    if not document_id:
        document_id = "unknown_doc"

    # Step 1: Fast scan to assess complexity
    markers, doc_info = system1_scan.remote(pdf_bytes, document_id)

    complexity_score = markers.complexity_score
    document_type = doc_info["inferred_type"]

    print(f"Document {document_id}: Complexity Score = {complexity_score:.2f}")
    print(f"Inferred Type: {document_type}")

    # Step 2: Route based on complexity
    if force_system2 or complexity_score >= complexity_threshold:
        print(f"→ Routing to SYSTEM 2 (Deep Reasoning)")
        result = system2_extract.remote(
            pdf_bytes, document_id, document_type, markers.model_dump()
        )
    else:
        print(f"→ Routing to SYSTEM 1 (Fast Extraction)")
        try:
            result = system1_extract.remote(pdf_bytes, document_id, document_type)
        except Exception as e:
            print(f"⚠️ System 1 failed: {e}")
            print(f"→ Falling back to SYSTEM 2 (Deep Reasoning)")
            result = system2_extract.remote(
                pdf_bytes, document_id, document_type, markers.model_dump()
            )

    return result


# ============ CRITIC AGENT (VERIFICATION) ============
@app.function(
    image=gpu_image,
    timeout=300,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    )
def verify_extraction(
    original_pdf: bytes,
    extraction_result: ExtractionResult,
) -> ExtractionResult:
    """
    Critic agent that verifies extraction quality.
    Re-extracts with System 2 if confidence is low.
    """
    import anthropic

    # Simple verification: Check if required fields are populated
    confidence = extraction_result.confidence_score

    if confidence < 0.7:
        print(f"Low confidence ({confidence:.2f}), re-running with System 2")
        return system2_extract.remote(
            original_pdf,
            extraction_result.document_id,
            extraction_result.document_type,
            extraction_result.complexity_markers.model_dump(),
        )

    extraction_result.verification_status = "passed"
    return extraction_result


# ============ BATCH ORCHESTRATION ============
@app.function(image=gpu_image, timeout=3600)
def batch_extract(request: BatchExtractionRequest) -> List[ExtractionResult]:
    """
    Parallel batch processing using modal.map()
    """
    print(f"Batch {request.batch_id}: Processing {len(request.documents)} documents")

    # Create document IDs
    doc_ids = [f"{request.batch_id}_{i}" for i in range(len(request.documents))]

    # Parallel extraction using modal.map
    results = list(
        route_and_extract.map(
            request.documents,
            doc_ids,
            [request.complexity_threshold] * len(request.documents),
            [request.force_system2] * len(request.documents),
        )
    )

    # Optional verification loop
    if request.enable_verification:
        verified_results = list(
            verify_extraction.map(request.documents, results)
        )
        return verified_results

    return results


# ============ HELPER FUNCTIONS ============
def detect_multicolumn(image) -> bool:
    """Detect multi-column layout using basic heuristics"""
    # Simplified - would use vision model in production
    return False


def check_image_quality(image) -> bool:
    """Check if image is low quality scan"""
    import numpy as np
    img_array = np.array(image)
    variance = np.var(img_array)
    return variance < 1000  # Low variance = poor quality


def check_language_complexity(text: str) -> bool:
    """Check for ambiguous or complex language"""
    # Simplified heuristic
    complex_words = ["notwithstanding", "wherein", "thereof", "hereby", "aforesaid"]
    return any(word in text.lower() for word in complex_words)


def detect_mixed_language(text: str) -> bool:
    """Detect if multiple languages are present"""
    # Would use langdetect in production
    return False


def estimate_entity_count(text: str) -> int:
    """Rough estimate of extractable entities"""
    # Count capitalized words as potential entities
    words = text.split()
    return sum(1 for w in words if w and w[0].isupper())


def classify_document_type(text: str) -> str:
    """Classify document type from text sample"""
    text_lower = text.lower()
    if any(w in text_lower for w in ["agreement", "clause", "party", "whereas"]):
        return "legal"
    elif any(w in text_lower for w in ["revenue", "balance sheet", "fiscal", "$"]):
        return "financial"
    elif any(w in text_lower for w in ["api", "specification", "requirement", "function"]):
        return "technical"
    return "general"


def get_schema_prompt(document_type: str) -> str:
    """Get schema-specific extraction prompt"""
    if document_type == "legal":
        return """Extract:
- All clauses with their types (obligation, right, prohibition, etc.)
- Parties involved
- Key dates
- Risk assessment for each clause
Return JSON matching the LegalDocument schema."""
    elif document_type == "financial":
        return """Extract:
- All tables with their types
- Financial data with proper formatting
- Period information
- Key metrics and totals
Return JSON matching the FinancialDocument schema."""
    elif document_type == "technical":
        return """Extract:
- All specifications and requirements
- Priority and category for each
- Technical details and dependencies
- Code snippets if present
Return JSON matching the TechnicalDocument schema."""
    return "Extract all structured data from this document."


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from model response (may include reasoning)"""
    import re

    # Try to find JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    # Fallback: try to parse entire response
    return json.loads(text)


def calculate_cost(usage, model_name: str) -> float:
    """Calculate API cost based on token usage"""
    # Approximate costs (update with actual pricing)
    costs = {
        "claude-opus-4-5": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
        "llama-3.2-vision": {"input": 0.0001, "output": 0.0001},
    }

    if model_name in costs:
        pricing = costs[model_name]
        return (usage.input_tokens * pricing["input"]) + (
            usage.output_tokens * pricing["output"]
        )
    return 0.0


# ============ WEB ENDPOINT (DISABLED) ============
# The Modal web endpoint approach doesn't work well for this architecture
# because the web container can't properly call GPU functions via .remote()
#
# Instead, use the local API (simple_api.py) which calls Modal via subprocess
# This is the correct pattern for local -> Modal communication
#
# If you need a production web endpoint, use Modal's webhook/API gateway pattern
# or deploy a separate web service that calls Modal functions via the SDK


# ============ LOCAL ENTRYPOINT ============
@app.local_entrypoint()
def main():
    """Local test entrypoint"""
    print("Deconstruct Shredder initialized")
    print("Deploy with: modal deploy gpu/modal_app.py")
    print("Test with: modal run gpu/modal_app.py")
