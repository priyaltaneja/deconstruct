"""
Deconstruct: AI-Powered Document Extraction
Modal App with Dual-Model Architecture

System 1: PyMuPDF + Small Text LLM (fast, cheap)
System 2: Vision LLM (accurate, handles complex layouts)
"""

import modal
from typing import List, Tuple
import time
import json
import base64

from schemas import (
    LegalDocument,
    FinancialDocument,
    TechnicalDocument,
    ComplexityMarkers,
    ExtractionResult,
)

from config import (
    DEFAULT_COMPLEXITY_THRESHOLD,
    SYSTEM1_TIMEOUT,
    SYSTEM2_TIMEOUT,
    SYSTEM1_TEXT_MODEL,
    SYSTEM2_VISION_MODEL,
    SYSTEM1_MAX_TOKENS,
    SYSTEM2_MAX_TOKENS,
    SYSTEM1_TEMPERATURE,
    SYSTEM2_TEMPERATURE,
    SYSTEM2_DPI,
    SYSTEM1_GPU,
    SYSTEM2_GPU,
    estimate_gpu_cost,
)


# ============ MODAL SETUP ============
app = modal.App("deconstruct-extractor")

# System 1: Text extraction + small LLM
system1_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pymupdf>=1.24.0",
        "pydantic>=2.0",
    )
    .add_local_python_source("schemas", "config")
)

# System 2: Vision LLM
system2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "vllm>=0.4.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pillow",
        "pdf2image",
        "pymupdf>=1.24.0",
        "qwen-vl-utils",
        "pydantic>=2.0",
        "fastapi",
        "python-multipart",
    )
    .add_local_python_source("schemas", "config")
)

# Shared volume for model caching
models_volume = modal.Volume.from_name("deconstruct-models", create_if_missing=True)


# ============ DOCUMENT ANALYSIS ============
def analyze_pdf_complexity(pdf_bytes: bytes) -> Tuple[ComplexityMarkers, str, List[dict]]:
    """
    Analyze PDF to determine complexity and extract text if possible.
    Returns (complexity_markers, extracted_text, page_analysis)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    total_text = ""
    total_images = 0
    total_tables = 0
    has_multi_column = False
    text_coverage_low = False
    page_analysis = []

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        total_text += text + "\n\n"

        # Count images
        images = page.get_images()
        total_images += len(images)

        # Detect tables (look for grid-like structures)
        tables = page.find_tables()
        total_tables += len(tables.tables) if tables else 0

        # Check text blocks for multi-column layout
        blocks = page.get_text("blocks")
        if len(blocks) > 1:
            x_positions = [b[0] for b in blocks if b[4].strip()]  # x0 of text blocks
            if len(set(int(x / 50) for x in x_positions)) > 2:  # Multiple column zones
                has_multi_column = True

        # Check if page has low text coverage (might be scanned)
        page_area = page.rect.width * page.rect.height
        text_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in blocks if b[4].strip())
        coverage = text_area / page_area if page_area > 0 else 0

        page_analysis.append({
            "page": page_num + 1,
            "text_length": len(text),
            "images": len(images),
            "tables": len(tables.tables) if tables else 0,
            "text_coverage": round(coverage, 2),
        })

        if coverage < 0.1 and len(images) > 0:
            text_coverage_low = True

    doc.close()

    # Build complexity markers
    markers = ComplexityMarkers(
        has_nested_tables=total_tables > 2,
        has_multi_column_layout=has_multi_column,
        has_handwriting=False,  # Can't detect without vision
        has_low_quality_scan=text_coverage_low,
        has_ambiguous_language=False,
        has_complex_formulas=total_images > 5,
        language_is_mixed=False,
        page_count=page_count,
        estimated_entities=len(total_text.split()) // 50,
    )

    return markers, total_text.strip(), page_analysis


def should_use_vision(markers: ComplexityMarkers, text: str, threshold: float) -> Tuple[bool, str]:
    """
    Decide whether to use Vision LLM based on complexity.
    Returns (use_vision, reasoning)
    """
    reasons = []

    # Check text quality
    if len(text) < 100:
        reasons.append("Insufficient extractable text (<100 chars)")

    if markers.has_low_quality_scan:
        reasons.append("Low text coverage detected (possibly scanned)")

    if markers.has_nested_tables:
        reasons.append(f"Complex table structures detected")

    if markers.has_multi_column_layout:
        reasons.append("Multi-column layout detected")

    if markers.has_complex_formulas:
        reasons.append("Many images/formulas detected")

    # Calculate score
    score = markers.complexity_score

    if score >= threshold:
        reasons.append(f"Complexity score {score:.2f} >= threshold {threshold}")
        return True, " | ".join(reasons) if reasons else "High complexity score"

    if len(text) < 100:
        return True, "Insufficient text for System 1"

    return False, f"Complexity score {score:.2f} < threshold {threshold}, good text extraction"


# ============ SYSTEM 1: TEXT LLM ============
@app.cls(
    image=system1_image,
    gpu=SYSTEM1_GPU,
    volumes={"/models": models_volume},
    timeout=SYSTEM1_TIMEOUT,
    scaledown_window=300,
)
class TextLLM:
    """Fast text LLM for simple documents."""

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        print(f"Loading {SYSTEM1_TEXT_MODEL} with vLLM...")
        self.llm = LLM(
            model=SYSTEM1_TEXT_MODEL,
            download_dir="/models/vllm",
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=SYSTEM1_TEMPERATURE,
            max_tokens=SYSTEM1_MAX_TOKENS,
            stop=["```\n\n", "\n\n\n"],
        )
        print("Text LLM loaded!")

    @modal.method()
    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text


# ============ SYSTEM 2: VISION LLM ============
@app.cls(
    image=system2_image,
    gpu=SYSTEM2_GPU,
    volumes={"/models": models_volume},
    timeout=SYSTEM2_TIMEOUT,
    scaledown_window=300,
)
class VisionLLM:
    """Vision LLM for complex documents with layouts/tables."""

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        print(f"Loading {SYSTEM2_VISION_MODEL} with vLLM...")
        self.llm = LLM(
            model=SYSTEM2_VISION_MODEL,
            download_dir="/models/vllm",
            dtype="bfloat16",
            max_model_len=16384,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 6},
        )
        self.sampling_params = SamplingParams(
            temperature=SYSTEM2_TEMPERATURE,
            max_tokens=SYSTEM2_MAX_TOKENS,
            stop=["```\n\n", "\n\n\n"],
        )
        print("Vision LLM loaded!")

    @modal.method()
    def generate_with_images(self, prompt: str, image_data: List[dict]) -> str:
        messages = [{
            "role": "user",
            "content": image_data + [{"type": "text", "text": prompt}]
        }]
        outputs = self.llm.chat(messages, self.sampling_params)
        return outputs[0].outputs[0].text


# ============ PROMPTS ============
def get_text_extraction_prompt(text: str, document_type: str) -> str:
    """Prompt for text-based extraction (System 1)."""
    return f"""You are a document extraction system. Extract structured data from the following text.

DOCUMENT TEXT:
{text[:12000]}

Return ONLY valid JSON. Detect the document type and use appropriate schema:

For LEGAL documents:
{{"document_type": "contract|agreement|amendment|policy|terms|other", "title": "...", "parties": ["..."], "execution_date": "...", "clauses": [{{"clause_id": "...", "clause_type": "definition|obligation|right|prohibition|condition|termination|liability|indemnity", "text": "...", "section_reference": "...", "risk_level": "low|medium|high|critical"}}], "key_obligations": ["..."], "risk_summary": "..."}}

For FINANCIAL documents:
{{"document_type": "10-K|10-Q|earnings|prospectus|other", "company_name": "...", "reporting_period": "...", "tables": [{{"table_id": "...", "table_type": "balance_sheet|income_statement|cash_flow", "title": "...", "headers": ["..."]}}], "key_metrics": {{}}, "executive_summary": "..."}}

For TECHNICAL documents:
{{"document_type": "architecture|api_spec|requirements|design|other", "title": "...", "version": "...", "specifications": [{{"spec_id": "...", "category": "functional|non_functional|performance|security", "title": "...", "description": "...", "priority": "P0|P1|P2|P3"}}], "technology_stack": ["..."]}}

For GENERAL documents:
{{"document_type": "general", "title": "...", "summary": "...", "key_points": ["..."], "entities": ["..."]}}

Also include a "confidence" field (0.0-1.0) indicating extraction confidence.

Return ONLY the JSON object."""


def get_vision_extraction_prompt(document_type: str) -> str:
    """Prompt for vision-based extraction (System 2)."""
    return f"""You are a document extraction system. Analyze the document images and extract structured data.

Return ONLY valid JSON. Detect the document type and use appropriate schema:

For LEGAL documents:
{{"document_type": "contract|agreement|amendment|policy|terms|other", "title": "...", "parties": ["..."], "execution_date": "...", "clauses": [{{"clause_id": "...", "clause_type": "definition|obligation|right|prohibition|condition|termination|liability|indemnity", "text": "...", "section_reference": "...", "risk_level": "low|medium|high|critical"}}], "key_obligations": ["..."], "risk_summary": "..."}}

For FINANCIAL documents:
{{"document_type": "10-K|10-Q|earnings|prospectus|other", "company_name": "...", "reporting_period": "...", "tables": [{{"table_id": "...", "table_type": "balance_sheet|income_statement|cash_flow", "title": "...", "headers": ["..."]}}], "key_metrics": {{}}, "executive_summary": "..."}}

For TECHNICAL documents:
{{"document_type": "architecture|api_spec|requirements|design|other", "title": "...", "version": "...", "specifications": [{{"spec_id": "...", "category": "functional|non_functional|performance|security", "title": "...", "description": "...", "priority": "P0|P1|P2|P3"}}], "technology_stack": ["..."]}}

For GENERAL documents:
{{"document_type": "general", "title": "...", "summary": "...", "key_points": ["..."], "entities": ["..."]}}

Also include a "confidence" field (0.0-1.0) indicating extraction confidence.

Return ONLY the JSON object."""


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

    # Try to find raw JSON
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ============ MAIN EXTRACTION ============
@app.function(image=system2_image, timeout=SYSTEM2_TIMEOUT + 120)
def route_and_extract(
    pdf_b64: str = "",
    document_id: str = "",
    complexity_threshold: float = DEFAULT_COMPLEXITY_THRESHOLD,
    force_system2: bool = False,
) -> dict:
    """
    Main entry point with intelligent routing.
    Returns full result with reasoning steps.
    """
    from pdf2image import convert_from_bytes
    from PIL import Image
    import io

    if not pdf_b64:
        raise ValueError("pdf_b64 must be provided")

    pdf_bytes = base64.b64decode(pdf_b64)
    document_id = document_id or f"doc_{int(time.time())}"

    reasoning_steps = []
    start_time = time.time()

    # Step 1: Analyze document
    step_start = time.time()
    reasoning_steps.append({
        "step": 1,
        "action": "Analyzing document structure",
        "status": "running",
    })

    markers, extracted_text, page_analysis = analyze_pdf_complexity(pdf_bytes)

    reasoning_steps[-1].update({
        "status": "complete",
        "duration_ms": int((time.time() - step_start) * 1000),
        "details": {
            "pages": markers.page_count,
            "text_length": len(extracted_text),
            "tables_detected": sum(p["tables"] for p in page_analysis),
            "images_detected": sum(p["images"] for p in page_analysis),
            "complexity_score": round(markers.complexity_score, 3),
        }
    })

    # Step 2: Route decision
    step_start = time.time()
    use_vision, routing_reason = should_use_vision(markers, extracted_text, complexity_threshold)

    if force_system2:
        use_vision = True
        routing_reason = "Forced to System 2 (Vision LLM)"

    reasoning_tier = "system2" if use_vision else "system1"
    model_used = SYSTEM2_VISION_MODEL if use_vision else SYSTEM1_TEXT_MODEL
    gpu_used = SYSTEM2_GPU if use_vision else SYSTEM1_GPU

    reasoning_steps.append({
        "step": 2,
        "action": "Routing decision",
        "status": "complete",
        "duration_ms": int((time.time() - step_start) * 1000),
        "details": {
            "selected_tier": reasoning_tier,
            "model": model_used,
            "reason": routing_reason,
        }
    })

    # Step 3: Extract with appropriate model
    step_start = time.time()
    reasoning_steps.append({
        "step": 3,
        "action": f"Extracting with {reasoning_tier.upper()}",
        "status": "running",
        "details": {"model": model_used}
    })

    if use_vision:
        # System 2: Vision LLM
        images = convert_from_bytes(pdf_bytes, dpi=SYSTEM2_DPI)
        max_pages = min(6, len(images))

        image_data = []
        for img in images[:max_pages]:
            max_size = 512
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            image_data.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        prompt = get_vision_extraction_prompt("general")
        vision_llm = VisionLLM()
        response = vision_llm.generate_with_images.remote(prompt, image_data)
    else:
        # System 1: Text LLM
        prompt = get_text_extraction_prompt(extracted_text, "general")
        text_llm = TextLLM()
        response = text_llm.generate.remote(prompt)

    result_json = extract_json_from_response(response)
    extraction_time = int((time.time() - step_start) * 1000)

    reasoning_steps[-1].update({
        "status": "complete",
        "duration_ms": extraction_time,
        "details": {
            "model": model_used,
            "pages_processed": max_pages if use_vision else "N/A (text)",
            "response_length": len(response),
        }
    })

    # Step 4: Parse and validate
    step_start = time.time()
    reasoning_steps.append({
        "step": 4,
        "action": "Parsing and validating output",
        "status": "running",
    })

    # Detect document type
    detected_type = result_json.get("document_type", "general")
    if detected_type in ["contract", "agreement", "amendment", "policy", "terms"]:
        document_type = "legal"
    elif detected_type in ["10-K", "10-Q", "earnings", "prospectus", "balance_sheet"]:
        document_type = "financial"
    elif detected_type in ["architecture", "api_spec", "requirements", "design"]:
        document_type = "technical"
    else:
        document_type = "general"

    # Get confidence from model output or estimate
    confidence = result_json.get("confidence", 0.75 if use_vision else 0.85)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except:
            confidence = 0.75

    processing_time = (time.time() - start_time) * 1000

    reasoning_steps[-1].update({
        "status": "complete",
        "duration_ms": int((time.time() - step_start) * 1000),
        "details": {
            "detected_type": document_type,
            "confidence": confidence,
            "fields_extracted": len(result_json),
        }
    })

    # Build result
    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=markers,
        reasoning_tier=reasoning_tier,
        model_used=model_used,
        processing_time_ms=processing_time,
        cost_usd=estimate_gpu_cost(processing_time, gpu_used),
        confidence_score=min(1.0, max(0.0, confidence)),
        verification_status="needs_review" if confidence < 0.8 else "passed",
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
        result.verification_notes.append(f"Schema validation: {str(e)[:100]}")

    # Return with reasoning
    return {
        **result.model_dump(),
        "reasoning_steps": reasoning_steps,
        "page_analysis": page_analysis,
    }


# ============ WEB API ============
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.post("/extract")
async def api_extract(
    file: UploadFile = File(...),
    document_id: str = Form(None),
    force_vision: bool = Form(False),
):
    """Extract structured data from uploaded PDF."""
    try:
        pdf_bytes = await file.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        doc_id = document_id or file.filename or f"doc_{int(time.time())}"

        result = route_and_extract.remote(
            pdf_b64=pdf_b64,
            document_id=doc_id,
            force_system2=force_vision,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@web_app.get("/health")
async def health():
    return {
        "status": "ok",
        "system1_model": SYSTEM1_TEXT_MODEL,
        "system2_model": SYSTEM2_VISION_MODEL,
    }


@app.function(image=system2_image, timeout=600)
@modal.asgi_app()
def fastapi_app():
    return web_app


# ============ LOCAL ENTRYPOINT ============
@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Deconstruct Document Extractor")
    print("=" * 60)
    print()
    print("Dual-Model Architecture:")
    print(f"  System 1 (Fast):   {SYSTEM1_TEXT_MODEL} ({SYSTEM1_GPU})")
    print(f"  System 2 (Vision): {SYSTEM2_VISION_MODEL} ({SYSTEM2_GPU})")
    print()
    print("Deploy: modal deploy gpu/modal_app.py")
    print("=" * 60)
