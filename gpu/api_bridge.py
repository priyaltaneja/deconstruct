"""
API Bridge for Deconstruct
Connects web frontend to Modal backend via HTTP endpoints
Run with: modal serve api_bridge.py
"""

import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
from pydantic import ValidationError

from schemas import (
    ExtractionResult,
    BatchExtractionRequest,
    ComplexityMarkers,
)

# Create Modal app
app = modal.App("deconstruct-api")

# Create FastAPI app
web_app = FastAPI(title="Deconstruct API", version="1.0.0")

# Add CORS middleware for local development
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Deconstruct API",
        "version": "1.0.0",
    }


@web_app.post("/api/extract/single")
async def extract_single_document(
    file: UploadFile = File(...),
    force_system2: bool = Form(False),
    complexity_threshold: float = Form(0.8),
    enable_verification: bool = Form(True),
):
    """
    Extract data from a single document
    """
    try:
        # Read file bytes
        pdf_bytes = await file.read()

        # Import Modal function
        from modal_app import route_and_extract

        # Run extraction
        result = route_and_extract.remote(
            pdf_bytes=pdf_bytes,
            document_id=file.filename or "unknown",
            complexity_threshold=complexity_threshold,
            force_system2=force_system2,
        )

        # Optional verification
        if enable_verification and result.confidence_score < 0.7:
            from modal_app import verify_extraction
            result = verify_extraction.remote(pdf_bytes, result)

        return {
            "success": True,
            "data": result.model_dump(),
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/api/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    batch_name: str = Form("Untitled Batch"),
    force_system2: bool = Form(False),
    complexity_threshold: float = Form(0.8),
    enable_verification: bool = Form(True),
):
    """
    Extract data from multiple documents in parallel
    """
    try:
        # Read all file bytes
        documents = []
        for file in files:
            pdf_bytes = await file.read()
            documents.append(pdf_bytes)

        # Create batch request
        batch_id = f"batch_{hash(batch_name)}_{len(documents)}"
        request = BatchExtractionRequest(
            batch_id=batch_id,
            documents=documents,
            force_system2=force_system2,
            complexity_threshold=complexity_threshold,
            enable_verification=enable_verification,
        )

        # Import Modal function
        from modal_app import batch_extract

        # Run batch extraction
        results = batch_extract.remote(request)

        return {
            "success": True,
            "data": {
                "batch_id": batch_id,
                "total_documents": len(results),
                "results": [r.model_dump() for r in results],
            },
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/api/complexity/analyze")
async def analyze_complexity(file: UploadFile = File(...)):
    """
    Analyze document complexity without full extraction
    """
    try:
        pdf_bytes = await file.read()

        # Import Modal function
        from modal_app import system1_scan

        markers, doc_info = system1_scan.remote(pdf_bytes, file.filename or "unknown")

        return {
            "success": True,
            "data": {
                "complexity_markers": markers.model_dump(),
                "document_info": doc_info,
                "recommended_tier": "system2" if markers.complexity_score >= 0.8 else "system1",
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.get("/api/stats")
async def get_stats():
    """
    Get extraction statistics (would query Supabase in production)
    """
    # Placeholder - in production, query Supabase views
    return {
        "success": True,
        "data": {
            "total_documents": 0,
            "total_batches": 0,
            "total_cost_usd": 0.0,
            "avg_processing_time_ms": 0.0,
            "system1_count": 0,
            "system2_count": 0,
        },
    }


# Mount FastAPI app to Modal
@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "fastapi[standard]",
        "python-multipart",
    ),
    secrets=[modal.Secret.from_name("anthropic-api-key")],
)
@modal.asgi_app()
def fastapi_app():
    """Serve FastAPI app via Modal"""
    return web_app


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Run API server locally"""
    print("Starting Deconstruct API server...")
    print("Access at: http://localhost:8000")
    print("Docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    # For local development without Modal
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
