"""
Local API server that proxies requests to Modal functions
Runs on localhost:8000 and calls Modal functions directly
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Add current directory to path so we can import modal_app
sys.path.insert(0, str(Path(__file__).parent))

# Import the Modal app and functions directly
from modal_app import route_and_extract

app = FastAPI(title="Deconstruct Local Proxy")

# CORS for local web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Deconstruct Local Proxy", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/extract")
async def extract_single(
    file: UploadFile = File(...),
    complexity_threshold: float = Form(0.8),
    force_system2: bool = Form(False),
):
    """Extract single document via Modal"""
    try:
        # Read file
        pdf_bytes = await file.read()
        document_id = file.filename.replace(".pdf", "")

        print(f"Processing {file.filename}...")
        print(f"  - Size: {len(pdf_bytes):,} bytes")
        print(f"  - Complexity threshold: {complexity_threshold}")
        print(f"  - Force System 2: {force_system2}")

        # Call Modal function remotely
        result = route_and_extract.remote(
            pdf_bytes=pdf_bytes,
            document_id=document_id,
            complexity_threshold=complexity_threshold,
            force_system2=force_system2,
        )

        print(f"✓ Extraction complete for {file.filename}")
        print(f"  - Tier: {result.reasoning_tier}")
        print(f"  - Model: {result.model_used}")
        print(f"  - Time: {result.processing_time_ms:.0f}ms")
        print(f"  - Cost: ${result.cost_usd:.4f}")

        return result.model_dump()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-extract")
async def extract_batch(
    files: list[UploadFile] = File(...),
    complexity_threshold: float = Form(0.8),
    force_system2: bool = Form(False),
):
    """Extract multiple documents via Modal"""
    try:
        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION: {len(files)} file(s)")
        print(f"{'='*70}")

        # Process each file
        results = []
        for i, file in enumerate(files, 1):
            pdf_bytes = await file.read()
            document_id = file.filename.replace(".pdf", "")

            print(f"\n[{i}/{len(files)}] Processing: {file.filename}")
            print(f"  - Size: {len(pdf_bytes):,} bytes")

            result = route_and_extract.remote(
                pdf_bytes=pdf_bytes,
                document_id=document_id,
                complexity_threshold=complexity_threshold,
                force_system2=force_system2,
            )

            print(f"  ✓ Complete - Tier: {result.reasoning_tier}, Model: {result.model_used}")
            results.append(result.model_dump())

        print(f"\n{'='*70}")
        print(f"✓ Batch complete: {len(results)} successful")
        print(f"{'='*70}\n")

        return results

    except Exception as e:
        print(f"\n❌ Batch extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 70)
    print("Deconstruct API Server")
    print("=" * 70)
    print("Running on: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print()
    print("This server proxies requests to Modal functions")
    print("Make sure Modal is authenticated and app is deployed")
    print("=" * 70)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
