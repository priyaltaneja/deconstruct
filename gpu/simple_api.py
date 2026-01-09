"""
Simple local API that calls Modal GPU functions via subprocess
Runs on localhost:8000
"""

import sys
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import json
import tempfile
from pathlib import Path

# Import Supabase persistence
try:
    from supabase_client import save_extraction_result, get_recent_extractions
    SUPABASE_ENABLED = True
    print("[SUPABASE] Client loaded - persistence enabled")
except ImportError as e:
    SUPABASE_ENABLED = False
    print(f"[SUPABASE] Not available - results will NOT be saved to database")

app = FastAPI(title="Deconstruct Simple API")

# CORS - must be configured before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Deconstruct Simple API", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "supabase_enabled": SUPABASE_ENABLED}


@app.get("/api/extractions")
async def list_extractions(limit: int = 20):
    """Get recent extractions from the database."""
    if not SUPABASE_ENABLED:
        return {"error": "Supabase not configured", "extractions": []}

    extractions = get_recent_extractions(limit)
    return {"extractions": extractions, "count": len(extractions)}


@app.post("/api/extract/batch")
@app.post("/api/batch-extract")
async def extract_batch(
    files: list[UploadFile] = File(...),
    batch_name: str = Form("Untitled Batch"),
    complexity_threshold: float = Form(0.8),
    force_system2: bool = Form(False),
    enable_verification: bool = Form(True),
):
    """Extract documents using Modal GPU functions"""
    try:
        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION: {len(files)} file(s)")
        print(f"{'='*70}")

        results = []

        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file.filename}")

            # Read file bytes
            pdf_bytes = await file.read()

            # Save to temp file for Modal extraction
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                # Call Modal via extract_json.py
                script_path = Path(__file__).parent / 'extract_json.py'
                cmd = ['python', str(script_path), tmp_path]

                print(f"  Running Modal extraction...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=900,  # 15 minute timeout for cold starts
                    encoding='utf-8',
                    errors='replace'
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    print(f"  [ERROR] Modal extraction failed")
                    print(f"  {error_msg[:300]}")

                    # Check for common issues
                    if "modal" in error_msg.lower() and "not found" in error_msg.lower():
                        error_msg = "Modal not installed. Run: pip install modal"
                    elif "authenticate" in error_msg.lower() or "token" in error_msg.lower():
                        error_msg = "Modal not authenticated. Run: modal token new"
                    elif "not found" in error_msg.lower() and "deconstruct" in error_msg.lower():
                        error_msg = "Modal app not deployed. Run: modal deploy modal_app.py"

                    results.append({
                        "document_id": file.filename,
                        "status": "failed",
                        "error": error_msg[:200]
                    })
                    continue

                # Parse JSON result
                result_data = json.loads(result.stdout.strip())

                if "error" in result_data:
                    print(f"  [ERROR] {result_data['error']}")
                    results.append({
                        "document_id": file.filename,
                        "status": "failed",
                        "error": result_data["error"]
                    })
                    continue

                print(f"  [OK] Success - Tier: {result_data.get('reasoning_tier', 'unknown')}")

                # Save to Supabase if enabled
                if SUPABASE_ENABLED:
                    doc_id = save_extraction_result(
                        result=result_data,
                        file_name=file.filename,
                        file_size=len(pdf_bytes),
                        file_bytes=pdf_bytes
                    )
                    if doc_id:
                        result_data["database_id"] = doc_id
                        print(f"  [DB] Saved: {doc_id}")

                results.append(result_data)

            except subprocess.TimeoutExpired:
                print(f"  [ERROR] Extraction timed out (15 min)")
                results.append({
                    "document_id": file.filename,
                    "status": "failed",
                    "error": "Extraction timed out. Try again (cold start can be slow)."
                })

            except json.JSONDecodeError as e:
                print(f"  [ERROR] Invalid JSON response")
                results.append({
                    "document_id": file.filename,
                    "status": "failed",
                    "error": f"Invalid JSON: {str(e)}"
                })

            finally:
                Path(tmp_path).unlink(missing_ok=True)

        successful = len([r for r in results if r.get('status') != 'failed'])
        print(f"\n{'='*70}")
        print(f"[DONE] {successful}/{len(files)} successful")
        print(f"{'='*70}\n")

        return results

    except Exception as e:
        print(f"\n[ERROR] Batch extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  DECONSTRUCT API SERVER")
    print("=" * 70)
    print()
    print("  BEFORE RUNNING THIS SERVER:")
    print()
    print("  1. Deploy Modal functions (one-time):")
    print("     cd gpu")
    print("     modal deploy modal_app.py")
    print()
    print("  2. Set up Modal authentication:")
    print("     modal token new")
    print()
    print("  3. Set Anthropic API key in Modal secrets:")
    print("     modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-...")
    print()
    print("=" * 70)
    print(f"  Server starting at: http://localhost:8000")
    print(f"  API docs at: http://localhost:8000/docs")
    print("=" * 70)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
