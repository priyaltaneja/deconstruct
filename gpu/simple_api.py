"""
Simple local API that calls Modal via subprocess
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
    print(f"[SUPABASE] Not available ({e}) - results will NOT be saved to database")

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


@app.post("/api/batch-extract")
async def extract_batch(
    files: list[UploadFile] = File(...),
    complexity_threshold: float = Form(0.8),
    force_system2: bool = Form(False),
):
    """Extract documents by calling Modal via subprocess"""
    try:
        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION: {len(files)} file(s)")
        print(f"{'='*70}")

        results = []

        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file.filename}")

            # Save file to temp location
            pdf_bytes = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                # Call the extract_json.py script that outputs JSON to stdout
                script_path = Path(__file__).parent / 'extract_json.py'
                cmd = ['python', str(script_path), tmp_path]

                print(f"  Running extraction...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=900,  # 15 minute timeout for cold start
                    encoding='utf-8',
                    errors='replace'  # Replace encoding errors instead of crashing
                )

                if result.returncode != 0:
                    print(f"  [ERROR] Extraction failed")
                    error_msg = result.stderr if result.stderr else result.stdout
                    print(f"  Output: {error_msg[:500]}")
                    raise Exception(f"Extraction failed: {error_msg[:200]}")

                # Parse JSON from stdout
                try:
                    result_data = json.loads(result.stdout.strip())

                    if "error" in result_data:
                        raise Exception(result_data["error"])

                    print(f"  [OK] Complete - Tier: {result_data.get('reasoning_tier', 'unknown')}")

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
                            print(f"  [DB] Saved to Supabase: {doc_id}")
                        else:
                            print(f"  [DB] Warning: Failed to save to database")

                    results.append(result_data)

                except json.JSONDecodeError as e:
                    print(f"  [ERROR] Failed to parse JSON output")
                    print(f"  Output: {result.stdout[:500]}")
                    raise Exception(f"Invalid JSON response: {str(e)}")

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        print(f"\n{'='*70}")
        print(f"[OK] Batch complete: {len(results)} successful")
        print(f"{'='*70}\n")

        return results

    except Exception as e:
        print(f"\n[ERROR] Batch extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 70)
    print("Deconstruct Simple API")
    print("=" * 70)
    print("Running on: http://localhost:8000")
    print("Calls Modal functions via subprocess")
    print("=" * 70)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
