"""
Extract PDF using Modal Python SDK
Usage: python extract_json.py <path_to_pdf>
"""

import sys
import json
import base64
from pathlib import Path

from config import DEFAULT_COMPLEXITY_THRESHOLD

def extract_pdf(pdf_path: str):
    """Process PDF via Modal Python SDK"""
    import modal

    document_id = Path(pdf_path).stem

    # Read PDF file
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # Encode as base64 string
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Look up the deployed function
    route_fn = modal.Function.from_name("deconstruct-shredder", "route_and_extract")

    # Call remotely with bytes
    result = route_fn.remote(
        pdf_b64=pdf_b64,
        document_id=document_id,
        complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
        force_system2=False,
    )

    # Convert Pydantic model to dict
    return result.model_dump()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No PDF path provided"}))
        sys.exit(1)

    try:
        result = extract_pdf(sys.argv[1])
        # Output ONLY the JSON to stdout
        print(json.dumps(result, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
