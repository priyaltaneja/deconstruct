"""
Deconstruct: AI-Powered Document Extraction
Modal App with Vision LLM Architecture

Uses Qwen2-VL via vLLM for document extraction.
"""

import modal
from typing import List
import time
import json
import base64

# Import schemas from centralized module
from schemas import (
    LegalDocument,
    FinancialDocument,
    TechnicalDocument,
    ComplexityMarkers,
    ExtractionResult,
    BatchExtractionRequest,
)

# Import configuration
from config import (
    DEFAULT_COMPLEXITY_THRESHOLD,
    SYSTEM2_TIMEOUT,
    BATCH_TIMEOUT,
    SYSTEM2_VISION_MODEL,
    SYSTEM2_MAX_TOKENS,
    SYSTEM2_TEMPERATURE,
    SYSTEM2_DPI,
    MAX_PAGES_FOR_EXTRACTION,
    SYSTEM2_GPU,
    estimate_gpu_cost,
)


# ============ MODAL SETUP ============
app = modal.App("deconstruct-extractor")

# Vision LLM image
vision_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "poppler-utils",
    )
    .pip_install(
        "vllm>=0.4.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pillow",
        "pdf2image",
        "qwen-vl-utils",
        "pydantic>=2.0",
    )
    .add_local_python_source("schemas", "config")
)

# Shared volume for model caching
models_volume = modal.Volume.from_name("deconstruct-models", create_if_missing=True)


# ============ HELPER FUNCTIONS ============
def get_extraction_prompt(document_type: str) -> str:
    """Generate extraction prompt based on document type."""

    return f"""You are a document extraction system. Analyze the document images and extract structured data.

Return ONLY valid JSON based on document type "{document_type}":

For LEGAL documents:
{{"document_type": "contract|agreement|amendment|policy|terms|other", "title": "...", "parties": ["..."], "execution_date": "...", "effective_date": "...", "expiration_date": "...", "clauses": [{{"clause_id": "...", "clause_type": "definition|obligation|right|prohibition|condition|termination|liability|indemnity", "text": "...", "section_reference": "...", "risk_level": "low|medium|high|critical"}}], "key_obligations": ["..."], "risk_summary": "..."}}

For FINANCIAL documents:
{{"document_type": "10-K|10-Q|earnings|prospectus|other", "company_name": "...", "reporting_period": "...", "filing_date": "...", "tables": [{{"table_id": "...", "table_type": "balance_sheet|income_statement|cash_flow|schedule|footnote|other", "title": "...", "headers": ["..."], "rows": [[{{"value": "...", "value_type": "currency|percentage|number|text|date"}}]]}}], "key_metrics": {{}}, "executive_summary": "..."}}

For TECHNICAL documents:
{{"document_type": "architecture|api_spec|requirements|design|other", "title": "...", "version": "...", "specifications": [{{"spec_id": "...", "category": "functional|non_functional|performance|security|compliance|interface|data", "title": "...", "description": "...", "priority": "P0|P1|P2|P3"}}], "technology_stack": ["..."], "code_snippets": ["..."]}}

For GENERAL/OTHER documents:
{{"document_type": "general", "title": "...", "summary": "...", "key_points": ["..."], "entities": ["..."]}}

Return ONLY the JSON object, no explanation."""


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from model response."""
    import re

    # Try to find JSON in code blocks
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find standalone JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {}


def classify_document_from_images(num_pages: int) -> str:
    """Basic classification - the LLM will do the real work."""
    # Simple heuristic, vision LLM will refine this
    return "general"


def create_complexity_markers(page_count: int) -> ComplexityMarkers:
    """Create basic complexity markers based on page count."""
    return ComplexityMarkers(
        has_nested_tables=False,
        has_multi_column_layout=False,
        has_handwriting=False,
        has_low_quality_scan=False,
        has_ambiguous_language=False,
        has_complex_formulas=False,
        language_is_mixed=False,
        page_count=page_count,
        estimated_entities=0,
    )


# ============ VISION LLM SERVER (Qwen2-VL) ============
@app.cls(
    image=vision_image,
    gpu=SYSTEM2_GPU,
    volumes={"/models": models_volume},
    timeout=SYSTEM2_TIMEOUT,
    scaledown_window=300,
)
class VisionLLM:
    """vLLM-powered vision LLM for document extraction."""

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        print(f"Loading {SYSTEM2_VISION_MODEL} with vLLM...")
        self.llm = LLM(
            model=SYSTEM2_VISION_MODEL,
            download_dir="/models/vllm",
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": MAX_PAGES_FOR_EXTRACTION},
        )
        self.sampling_params = SamplingParams(
            temperature=SYSTEM2_TEMPERATURE,
            max_tokens=SYSTEM2_MAX_TOKENS,
            stop=["```\n\n", "\n\n\n"],
        )
        print("Vision LLM loaded!")

    @modal.method()
    def generate_with_images(self, prompt: str, image_data: List[dict]) -> str:
        """Generate with images. image_data is list of {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}"""

        # Build multimodal prompt
        messages = [{
            "role": "user",
            "content": image_data + [{"type": "text", "text": prompt}]
        }]

        outputs = self.llm.chat(messages, self.sampling_params)
        return outputs[0].outputs[0].text


# ============ MAIN EXTRACTION FUNCTION ============
@app.function(image=vision_image, timeout=SYSTEM2_TIMEOUT)
def extract_document(
    pdf_bytes: bytes,
    document_id: str,
    document_type: str = "general",
) -> ExtractionResult:
    """Extract structured data from PDF using Vision LLM."""
    from pdf2image import convert_from_bytes
    from PIL import Image
    import io

    start_time = time.time()

    # Convert PDF to images
    print(f"[{document_id}] Converting PDF to images...")
    images = convert_from_bytes(pdf_bytes, dpi=SYSTEM2_DPI)
    page_count = len(images)
    print(f"[{document_id}] {page_count} pages")

    # Prepare images for vLLM (base64 encoded)
    image_data = []
    for i, img in enumerate(images[:MAX_PAGES_FOR_EXTRACTION]):
        # Resize if needed
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        image_data.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    # Generate extraction prompt
    prompt = get_extraction_prompt(document_type)

    # Call vision LLM
    print(f"[{document_id}] Calling Vision LLM...")
    vision_llm = VisionLLM()
    response = vision_llm.generate_with_images.remote(prompt, image_data)
    result_json = extract_json_from_response(response)

    # Detect document type from response
    detected_type = result_json.get("document_type", "general")
    if detected_type in ["contract", "agreement", "amendment", "policy", "terms"]:
        document_type = "legal"
    elif detected_type in ["10-K", "10-Q", "earnings", "prospectus", "balance_sheet"]:
        document_type = "financial"
    elif detected_type in ["architecture", "api_spec", "requirements", "design"]:
        document_type = "technical"

    processing_time = (time.time() - start_time) * 1000

    # Build result
    complexity_markers = create_complexity_markers(page_count)

    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=complexity_markers,
        reasoning_tier="vision",
        model_used=SYSTEM2_VISION_MODEL,
        processing_time_ms=processing_time,
        cost_usd=estimate_gpu_cost(processing_time, SYSTEM2_GPU),
        confidence_score=0.85,
        verification_status="needs_review",
    )

    # Populate typed content
    try:
        if document_type == "legal" and result_json:
            result.legal_content = LegalDocument(**result_json)
        elif document_type == "financial" and result_json:
            result.financial_content = FinancialDocument(**result_json)
        elif document_type == "technical" and result_json:
            result.technical_content = TechnicalDocument(**result_json)
        else:
            result.extracted_entities = [result_json] if result_json else []
    except Exception as e:
        result.extracted_entities = [result_json] if result_json else []
        result.verification_notes.append(f"Schema: {str(e)[:200]}")

    print(f"[{document_id}] Done in {processing_time:.0f}ms")
    return result


# ============ ROUTER (simplified) ============
@app.function(image=vision_image, timeout=SYSTEM2_TIMEOUT + 60)
def route_and_extract(
    pdf_b64: str = "",
    document_id: str = "",
    complexity_threshold: float = DEFAULT_COMPLEXITY_THRESHOLD,
    force_system2: bool = False,
) -> ExtractionResult:
    """Main entry point - extracts document using Vision LLM."""
    if not pdf_b64:
        raise ValueError("pdf_b64 must be provided")

    pdf_bytes = base64.b64decode(pdf_b64)
    document_id = document_id or f"doc_{int(time.time())}"

    print(f"[{document_id}] Starting extraction...")
    return extract_document.remote(pdf_bytes, document_id)


# ============ BATCH PROCESSING ============
@app.function(image=vision_image, timeout=BATCH_TIMEOUT)
def batch_extract(request: BatchExtractionRequest) -> List[ExtractionResult]:
    """Process multiple documents in parallel."""
    print(f"Batch {request.batch_id}: {len(request.documents)} documents")

    docs_b64 = [base64.b64encode(doc).decode() for doc in request.documents]
    doc_ids = [f"{request.batch_id}_{i}" for i in range(len(request.documents))]

    results = list(route_and_extract.map(
        docs_b64,
        doc_ids,
        [request.complexity_threshold] * len(request.documents),
        [request.force_system2] * len(request.documents),
    ))

    return results


# ============ LOCAL ENTRYPOINT ============
@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Deconstruct Document Extractor")
    print("=" * 60)
    print()
    print("Architecture:")
    print(f"  Vision LLM: {SYSTEM2_VISION_MODEL} via vLLM ({SYSTEM2_GPU})")
    print()
    print("Deploy: modal deploy gpu/modal_app.py")
    print("=" * 60)
