"""
Simple test to verify Modal deployment
Creates a minimal test document
"""

import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    """Create a simple test PDF"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Add some text
    c.drawString(100, 750, "TEST LEGAL AGREEMENT")
    c.drawString(100, 720, "")
    c.drawString(100, 690, "This Agreement is entered into on January 1, 2026")
    c.drawString(100, 670, "between Party A and Party B.")
    c.drawString(100, 650, "")
    c.drawString(100, 630, "1. OBLIGATIONS")
    c.drawString(100, 610, "Party A shall provide services as described herein.")
    c.drawString(100, 590, "")
    c.drawString(100, 570, "2. TERMINATION")
    c.drawString(100, 550, "Either party may terminate with 30 days notice.")

    c.save()
    return buffer.getvalue()

if __name__ == "__main__":
    import modal

    print("Creating test PDF...")
    pdf_bytes = create_test_pdf()
    print(f"Created test PDF ({len(pdf_bytes)} bytes)")

    print("\nTesting Modal deployment...")
    print("Looking up function...")

    try:
        route_and_extract = modal.Function.lookup("deconstruct-extractor", "route_and_extract")
        print("Found route_and_extract function")

        # Encode PDF as base64
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

        print("\nRunning extraction...")
        result = route_and_extract.remote(
            pdf_b64=pdf_b64,
            document_id="test_doc_001",
            complexity_threshold=0.7,
            force_system2=False,
        )

        print("\n" + "="*60)
        print("EXTRACTION RESULTS")
        print("="*60)
        print(f"Document Type: {result.document_type}")
        print(f"Reasoning Tier: {result.reasoning_tier}")
        print(f"Model Used: {result.model_used}")
        print(f"Processing Time: {result.processing_time_ms:.0f}ms")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Confidence: {result.confidence_score * 100:.1f}%")
        print(f"Complexity Score: {result.complexity_markers.complexity_score:.2f}")
        print("\nTEST PASSED!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
