"""
Deconstruct: AI-Powered Document Extraction
Modal App with Two-Tier Local LLM Architecture

System 1 (Fast): PaddleOCR → Qwen2.5 (vLLM)
System 2 (Deep): Qwen2-VL (vLLM)

All models served via vLLM on Modal GPUs for high performance.
"""

import modal
from typing import List, Tuple
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
    SYSTEM1_TIMEOUT,
    SYSTEM2_TIMEOUT,
    BATCH_TIMEOUT,
    SYSTEM1_TEXT_MODEL,
    SYSTEM1_MAX_TOKENS,
    SYSTEM1_TEMPERATURE,
    SYSTEM2_VISION_MODEL,
    SYSTEM2_MAX_TOKENS,
    SYSTEM2_TEMPERATURE,
    SYSTEM1_DPI,
    SYSTEM2_DPI,
    MAX_PAGES_FOR_EXTRACTION,
    SCAN_FIRST_N_PAGES,
    SYSTEM1_GPU,
    SYSTEM2_GPU,
    estimate_gpu_cost,
)


# ============ MODAL SETUP ============

# Get path to current directory for mounting local files
import pathlib
LOCAL_DIR = pathlib.Path(__file__).parent

# Mount for local Python modules (schemas, config)
local_modules = modal.Mount.from_local_file(LOCAL_DIR / "schemas.py", remote_path="/root/schemas.py")
config_mount = modal.Mount.from_local_file(LOCAL_DIR / "config.py", remote_path="/root/config.py")

app = modal.App(
    "deconstruct-extractor",
    mounts=[local_modules, config_mount],
)

# Base image with common dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "poppler-utils",
    )
    .pip_install(
        "paddlepaddle",
        "paddleocr",
        "pdf2image",
        "pillow",
        "numpy",
        "pydantic>=2.0",
    )
)

# vLLM image for text model serving
vllm_text_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pydantic>=2.0",
    )
)

# vLLM image for vision model serving
vllm_vision_image = (
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
)

# OCR processing image (lightweight, no LLM)
ocr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "poppler-utils",
    )
    .pip_install(
        "paddlepaddle",
        "paddleocr",
        "pdf2image",
        "pillow",
        "numpy",
        "pydantic>=2.0",
    )
)

# Shared volume for model caching
models_volume = modal.Volume.from_name("deconstruct-models", create_if_missing=True)


# ============ HELPER FUNCTIONS ============
def get_extraction_prompt(document_type: str, ocr_text: str = None) -> str:
    """Generate extraction prompt based on document type."""

    base_prompt = """You are a document extraction system. Extract structured data and return ONLY valid JSON.

Document Type: {doc_type}

{content_section}

Extract based on document type:

For LEGAL documents:
{{"document_type": "contract|agreement|amendment|policy|terms|other", "title": "...", "parties": ["..."], "execution_date": "...", "effective_date": "...", "expiration_date": "...", "clauses": [{{"clause_id": "...", "clause_type": "definition|obligation|right|prohibition|condition|termination|liability|indemnity", "text": "...", "section_reference": "...", "risk_level": "low|medium|high|critical"}}], "key_obligations": ["..."], "risk_summary": "..."}}

For FINANCIAL documents:
{{"document_type": "10-K|10-Q|earnings|prospectus|other", "company_name": "...", "reporting_period": "...", "filing_date": "...", "tables": [{{"table_id": "...", "table_type": "balance_sheet|income_statement|cash_flow|schedule|footnote|other", "title": "...", "headers": ["..."], "rows": [[{{"value": "...", "value_type": "currency|percentage|number|text|date"}}]]}}], "key_metrics": {{}}, "executive_summary": "..."}}

For TECHNICAL documents:
{{"document_type": "architecture|api_spec|requirements|design|other", "title": "...", "version": "...", "specifications": [{{"spec_id": "...", "category": "functional|non_functional|performance|security|compliance|interface|data", "title": "...", "description": "...", "priority": "P0|P1|P2|P3"}}], "technology_stack": ["..."], "code_snippets": ["..."]}}

Return ONLY the JSON object."""

    if ocr_text:
        content_section = f"Document text:\n{ocr_text[:6000]}"
    else:
        content_section = "Analyze the document images provided."

    return base_prompt.format(doc_type=document_type, content_section=content_section)


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


def classify_document(text: str) -> str:
    """Classify document type from text content."""
    text_lower = text.lower()

    legal_kw = ["agreement", "clause", "party", "whereas", "hereby", "contract"]
    financial_kw = ["revenue", "balance sheet", "fiscal", "earnings", "assets"]
    technical_kw = ["api", "specification", "requirement", "function", "architecture"]

    scores = {
        "legal": sum(1 for kw in legal_kw if kw in text_lower),
        "financial": sum(1 for kw in financial_kw if kw in text_lower),
        "technical": sum(1 for kw in technical_kw if kw in text_lower),
    }

    max_score = max(scores.values())
    if max_score == 0:
        return "general"

    return max(scores, key=scores.get)


def analyze_complexity(text: str, page_count: int) -> ComplexityMarkers:
    """Analyze document complexity for routing."""
    has_tables = "|" in text or "table" in text.lower()
    has_complex = any(w in text.lower() for w in ["notwithstanding", "whereas", "hereinafter"])
    has_formulas = any(c in text for c in ["=", "∑", "∫"])

    lines = text.split('\n')
    short_lines = sum(1 for l in lines if 10 < len(l.strip()) < 50)
    multi_col = short_lines > len(lines) * 0.4 if lines else False

    words = text.split()
    entities = sum(1 for w in words if w and w[0].isupper() and len(w) > 2)

    return ComplexityMarkers(
        has_nested_tables=has_tables and text.count("|") > 20,
        has_multi_column_layout=multi_col,
        has_handwriting=False,
        has_low_quality_scan=False,
        has_ambiguous_language=has_complex,
        has_complex_formulas=has_formulas,
        language_is_mixed=False,
        page_count=page_count,
        estimated_entities=min(entities, 500),
    )


# ============ vLLM TEXT MODEL SERVER (Qwen2.5) ============
@app.cls(
    image=vllm_text_image,
    gpu=SYSTEM1_GPU,
    volumes={"/models": models_volume},
    timeout=SYSTEM1_TIMEOUT,
    container_idle_timeout=300,  # Keep warm for 5 min
    allow_concurrent_inputs=10,  # Handle multiple requests
)
class TextLLM:
    """vLLM-powered text LLM for fast extraction."""

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


# ============ vLLM VISION MODEL SERVER (Qwen2-VL) ============
@app.cls(
    image=vllm_vision_image,
    gpu=SYSTEM2_GPU,
    volumes={"/models": models_volume},
    timeout=SYSTEM2_TIMEOUT,
    container_idle_timeout=300,
    allow_concurrent_inputs=5,
)
class VisionLLM:
    """vLLM-powered vision LLM for complex document extraction."""

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
        from vllm import SamplingParams

        # Build multimodal prompt
        messages = [{
            "role": "user",
            "content": image_data + [{"type": "text", "text": prompt}]
        }]

        outputs = self.llm.chat(messages, self.sampling_params)
        return outputs[0].outputs[0].text


# ============ OCR SERVICE (PaddleOCR) ============
@app.cls(
    image=ocr_image,
    gpu="T4",  # Lightweight GPU for OCR
    volumes={"/models": models_volume},
    timeout=120,
    container_idle_timeout=300,
    allow_concurrent_inputs=20,  # OCR is fast, can handle many
)
class OCRService:
    """PaddleOCR service for text extraction."""

    @modal.enter()
    def load_ocr(self):
        from paddleocr import PaddleOCR

        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False,
        )
        print("PaddleOCR loaded!")

    @modal.method()
    def extract_text(self, pdf_bytes: bytes, max_pages: int = 10, dpi: int = 200) -> Tuple[str, int]:
        """Extract text from PDF. Returns (text, page_count)."""
        from pdf2image import convert_from_bytes
        import numpy as np

        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        page_count = len(images)

        # OCR each page
        all_text = []
        for i, img in enumerate(images[:max_pages]):
            img_array = np.array(img)
            result = self.ocr.ocr(img_array, cls=True)

            if result and result[0]:
                page_text = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                        page_text.append(text)
                all_text.append(f"--- Page {i+1} ---\n" + "\n".join(page_text))

        return "\n\n".join(all_text), page_count

    @modal.method()
    def quick_scan(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Quick scan first few pages for complexity detection."""
        from pdf2image import convert_from_bytes
        import numpy as np

        # Low DPI for speed
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=SCAN_FIRST_N_PAGES)

        # Get total page count
        all_images = convert_from_bytes(pdf_bytes, dpi=72)
        page_count = len(all_images)

        # Quick OCR
        text_parts = []
        for img in images:
            result = self.ocr.ocr(np.array(img), cls=True)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                        text_parts.append(text)

        return "\n".join(text_parts), page_count


# ============ SYSTEM 1: OCR + Text LLM ============
@app.function(image=ocr_image, timeout=SYSTEM1_TIMEOUT)
def system1_extract(pdf_bytes: bytes, document_id: str) -> ExtractionResult:
    """Fast extraction: PaddleOCR → Qwen2.5 Text LLM via vLLM."""
    start_time = time.time()

    # Get OCR service and extract text
    ocr = OCRService()
    ocr_text, page_count = ocr.extract_text.remote(pdf_bytes, MAX_PAGES_FOR_EXTRACTION, SYSTEM1_DPI)

    ocr_time = time.time() - start_time
    print(f"[{document_id}] OCR completed in {ocr_time:.1f}s, {len(ocr_text)} chars")

    # Analyze document
    complexity_markers = analyze_complexity(ocr_text, page_count)
    document_type = classify_document(ocr_text)

    # Generate extraction prompt
    prompt = get_extraction_prompt(document_type, ocr_text)

    # Call text LLM
    text_llm = TextLLM()
    response = text_llm.generate.remote(prompt)
    result_json = extract_json_from_response(response)

    processing_time = (time.time() - start_time) * 1000

    # Build result
    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=complexity_markers,
        reasoning_tier="system1",
        model_used=f"PaddleOCR + {SYSTEM1_TEXT_MODEL}",
        processing_time_ms=processing_time,
        cost_usd=estimate_gpu_cost(processing_time, SYSTEM1_GPU),
        confidence_score=0.75,
        verification_status="needs_review",
        raw_text=ocr_text[:5000],
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

    return result


# ============ SYSTEM 2: Vision LLM ============
@app.function(image=vllm_vision_image, timeout=SYSTEM2_TIMEOUT)
def system2_extract(
    pdf_bytes: bytes,
    document_id: str,
    document_type: str,
    complexity_markers: dict,
) -> ExtractionResult:
    """Deep extraction: Qwen2-VL Vision LLM via vLLM."""
    from pdf2image import convert_from_bytes
    from PIL import Image
    import io

    start_time = time.time()

    # Convert PDF to images
    images = convert_from_bytes(pdf_bytes, dpi=SYSTEM2_DPI)

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
    vision_llm = VisionLLM()
    response = vision_llm.generate_with_images.remote(prompt, image_data)
    result_json = extract_json_from_response(response)

    processing_time = (time.time() - start_time) * 1000

    # Build result
    result = ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        complexity_markers=ComplexityMarkers(**complexity_markers),
        reasoning_tier="system2",
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

    return result


# ============ ROUTER ============
@app.function(image=ocr_image, timeout=SYSTEM2_TIMEOUT + 300)
def route_and_extract(
    pdf_b64: str = "",
    document_id: str = "",
    complexity_threshold: float = DEFAULT_COMPLEXITY_THRESHOLD,
    force_system2: bool = False,
) -> ExtractionResult:
    """Route between System 1 (OCR+Text) and System 2 (Vision) based on complexity."""
    if not pdf_b64:
        raise ValueError("pdf_b64 must be provided")

    pdf_bytes = base64.b64decode(pdf_b64)
    document_id = document_id or f"doc_{int(time.time())}"

    # Quick scan to assess complexity
    print(f"[{document_id}] Scanning...")
    ocr = OCRService()
    scan_text, page_count = ocr.quick_scan.remote(pdf_bytes)

    markers = analyze_complexity(scan_text, page_count)
    doc_type = classify_document(scan_text)
    score = markers.complexity_score

    print(f"[{document_id}] Complexity: {score:.2f}, Type: {doc_type}")

    # Route
    if force_system2 or score >= complexity_threshold:
        print(f"[{document_id}] → SYSTEM 2 (Vision)")
        return system2_extract.remote(pdf_bytes, document_id, doc_type, markers.model_dump())
    else:
        print(f"[{document_id}] → SYSTEM 1 (OCR+Text)")
        try:
            return system1_extract.remote(pdf_bytes, document_id)
        except Exception as e:
            print(f"[{document_id}] System 1 failed: {e}, falling back to System 2")
            return system2_extract.remote(pdf_bytes, document_id, doc_type, markers.model_dump())


# ============ BATCH PROCESSING ============
@app.function(image=ocr_image, timeout=BATCH_TIMEOUT)
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
    print("Deconstruct Document Extractor (vLLM)")
    print("=" * 60)
    print()
    print("Architecture:")
    print(f"  OCR:      PaddleOCR (T4 GPU)")
    print(f"  System 1: {SYSTEM1_TEXT_MODEL} via vLLM ({SYSTEM1_GPU})")
    print(f"  System 2: {SYSTEM2_VISION_MODEL} via vLLM ({SYSTEM2_GPU})")
    print()
    print("Deploy: modal deploy gpu/modal_app.py")
    print("=" * 60)
