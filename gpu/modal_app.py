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
def analyze_pdf_complexity(pdf_bytes: bytes) -> Tuple[dict, str, List[dict], float]:
    """
    Analyze PDF complexity using font/structure analysis.
    Returns (complexity_info, extracted_text, page_analysis, complexity_score)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    total_text = ""
    page_analysis = []

    # Aggregate metrics
    all_fonts = set()
    all_font_sizes = set()
    all_colors = set()
    total_images = 0
    total_drawings = 0
    total_links = 0
    total_annotations = 0
    form_fields = 0

    # Layout metrics
    column_structures = []  # Track column count per page
    text_densities = []
    block_counts = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # Extract text
        page_text = page.get_text()
        total_text += page_text + "\n\n"

        # Analyze fonts and styling from detailed dict
        page_fonts = set()
        page_sizes = set()
        page_colors = set()

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = span.get("font", "")
                        size = round(span.get("size", 0), 1)
                        color = span.get("color", 0)
                        if font:
                            page_fonts.add(font)
                            all_fonts.add(font)
                        if size > 0:
                            page_sizes.add(size)
                            all_font_sizes.add(size)
                        all_colors.add(color)
                        page_colors.add(color)

        # Count images
        images = page.get_images(full=True)
        total_images += len(images)

        # Count vector drawings
        drawings = page.get_drawings()
        total_drawings += len(drawings)

        # Count links and annotations
        links = page.get_links()
        total_links += len(links)

        annots = list(page.annots()) if page.annots() else []
        total_annotations += len(annots)

        # Check for form fields (widgets)
        widgets = list(page.widgets()) if page.widgets() else []
        form_fields += len(widgets)

        # Analyze column structure
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if len(b) > 4 and isinstance(b[4], str) and b[4].strip()]
        block_counts.append(len(text_blocks))

        # Detect columns by analyzing x-position clusters
        num_columns = 1
        if len(text_blocks) > 3:
            x_positions = sorted([b[0] for b in text_blocks])
            # Find gaps in x positions (column separators)
            gaps = []
            for i in range(1, len(x_positions)):
                gap = x_positions[i] - x_positions[i-1]
                if gap > 50:  # Significant gap
                    gaps.append(gap)
            if len(gaps) >= 2:
                num_columns = min(len(gaps) + 1, 4)
        column_structures.append(num_columns)

        # Text density (chars per page area)
        page_area = page.rect.width * page.rect.height
        text_density = len(page_text.strip()) / max(page_area, 1) * 10000
        text_densities.append(text_density)

        # Check if likely scanned
        is_scanned = len(page_text.strip()) < 100 and len(images) > 0

        page_analysis.append({
            "page": page_num + 1,
            "text_chars": len(page_text.strip()),
            "fonts_used": len(page_fonts),
            "font_sizes": len(page_sizes),
            "colors": len(page_colors),
            "images": len(images),
            "drawings": len(drawings),
            "columns": num_columns,
            "is_scanned": is_scanned,
        })

    doc.close()

    # Calculate complexity score with weighted factors
    score = 0.0
    reasons = []

    # 1. Font complexity (many fonts = complex formatting)
    font_count = len(all_fonts)
    if font_count > 5:
        contrib = min(0.15, (font_count - 5) * 0.02)
        score += contrib
        reasons.append(f"{font_count} fonts")

    # 2. Font size variety (many sizes = headings, hierarchies)
    size_count = len(all_font_sizes)
    if size_count > 6:
        contrib = min(0.1, (size_count - 6) * 0.015)
        score += contrib
        reasons.append(f"{size_count} font sizes")

    # 3. Color usage (multiple colors = styled document)
    color_count = len(all_colors)
    if color_count > 3:
        contrib = min(0.1, (color_count - 3) * 0.02)
        score += contrib
        reasons.append(f"{color_count} colors")

    # 4. Images (embedded images need vision)
    if total_images > 0:
        contrib = min(0.25, total_images * 0.03)
        score += contrib
        reasons.append(f"{total_images} images")

    # 5. Vector drawings (charts, diagrams)
    if total_drawings > 50:
        contrib = min(0.15, (total_drawings - 50) * 0.001)
        score += contrib
        reasons.append(f"{total_drawings} drawings")

    # 6. Multi-column layout
    avg_columns = sum(column_structures) / max(len(column_structures), 1)
    if avg_columns > 1.3:
        score += 0.15
        reasons.append(f"multi-column ({avg_columns:.1f} avg)")

    # 7. Form fields (need special handling)
    if form_fields > 0:
        score += min(0.2, form_fields * 0.05)
        reasons.append(f"{form_fields} form fields")

    # 8. Low text density (scanned or image-heavy)
    avg_density = sum(text_densities) / max(len(text_densities), 1)
    if avg_density < 5:
        score += 0.25
        reasons.append(f"low text density ({avg_density:.1f})")

    # 9. High block count variation (complex layout)
    if block_counts:
        block_variance = max(block_counts) - min(block_counts)
        if block_variance > 20:
            score += 0.1
            reasons.append(f"varied layout (block var: {block_variance})")

    # 10. Annotations/links (interactive document)
    if total_annotations > 5:
        score += 0.05
        reasons.append(f"{total_annotations} annotations")

    score = min(1.0, score)

    # If no complexity reasons found, document is simple
    if not reasons:
        reasons.append("simple text document")

    complexity_info = {
        "page_count": page_count,
        "total_images": total_images,
        "total_drawings": total_drawings,
        "font_count": font_count,
        "font_size_count": size_count,
        "color_count": color_count,
        "avg_columns": round(avg_columns, 2),
        "form_fields": form_fields,
        "avg_text_density": round(avg_density, 2),
        "reasons": reasons,
    }

    return complexity_info, total_text.strip(), page_analysis, score


def should_use_vision(complexity_info: dict, text: str, score: float, threshold: float) -> Tuple[bool, List[str]]:
    """
    Decide whether to use Vision LLM based on complexity.
    Returns (use_vision, reasons)
    """
    reasons = complexity_info.get("reasons", [])

    # Insufficient text always needs vision
    if len(text) < 100:
        reasons.append("insufficient extractable text")
        return True, reasons

    # Score above threshold
    if score >= threshold:
        return True, reasons

    # Good text extraction, use fast path
    return False, [f"score {score:.2f} < threshold {threshold}", "good text extraction"]


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

    complexity_info, extracted_text, page_analysis, complexity_score = analyze_pdf_complexity(pdf_bytes)

    reasoning_steps[-1].update({
        "status": "complete",
        "duration_ms": int((time.time() - step_start) * 1000),
        "details": {
            "pages": complexity_info["page_count"],
            "text_chars": len(extracted_text),
            "fonts": complexity_info["font_count"],
            "font_sizes": complexity_info["font_size_count"],
            "colors": complexity_info["color_count"],
            "images": complexity_info["total_images"],
            "drawings": complexity_info["total_drawings"],
            "avg_columns": complexity_info["avg_columns"],
            "form_fields": complexity_info["form_fields"],
            "text_density": complexity_info["avg_text_density"],
            "complexity_score": round(complexity_score, 2),
        }
    })

    # Step 2: Route decision
    step_start = time.time()
    use_vision, routing_reasons = should_use_vision(complexity_info, extracted_text, complexity_score, complexity_threshold)

    if force_system2:
        use_vision = True
        routing_reasons = ["forced to Vision LLM"]

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
            "reasons": routing_reasons,
            "threshold": complexity_threshold,
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

    # Build complexity markers for result
    markers = ComplexityMarkers(
        has_nested_tables=complexity_info["total_images"] > 2,  # Images often contain tables
        has_multi_column_layout=complexity_info["avg_columns"] > 1.3,
        has_handwriting=False,
        has_low_quality_scan=complexity_info["avg_text_density"] < 5,
        has_ambiguous_language=False,
        has_complex_formulas=complexity_info["total_drawings"] > 50,
        language_is_mixed=complexity_info["color_count"] > 5,  # Proxy: many colors = mixed content
        page_count=complexity_info["page_count"],
        estimated_entities=len(extracted_text.split()) // 50,
    )

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
        "complexity_score": round(complexity_score, 3),
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
