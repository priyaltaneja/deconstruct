"""
Test script for Deconstruct extraction pipeline
Run with: modal run gpu/test_extraction.py
"""

import modal
from pathlib import Path
from schemas import BatchExtractionRequest, ExtractionResult


def test_local():
    """Test extraction with a sample PDF (local mode)"""
    print("🧪 Testing Deconstruct extraction pipeline...")

    # Load a sample PDF (you'll need to provide one)
    sample_pdf_path = Path("sample.pdf")

    if not sample_pdf_path.exists():
        print("❌ sample.pdf not found in gpu/ directory")
        print("   Please add a sample PDF to test with")
        return

    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"📄 Loaded {sample_pdf_path.name} ({len(pdf_bytes)} bytes)")

    # Create batch request
    request = BatchExtractionRequest(
        batch_id="test_batch_001",
        documents=[pdf_bytes],
        force_system2=False,  # Let the router decide
        complexity_threshold=0.8,
        enable_verification=True,
    )

    print(f"🚀 Processing batch with {len(request.documents)} document(s)...")
    print(f"   Complexity threshold: {request.complexity_threshold}")
    print(f"   Verification enabled: {request.enable_verification}")
    print("")

    # Import and run the extraction
    from modal_app import batch_extract

    try:
        # Run locally (will still use GPU if available)
        results = batch_extract.local(request)

        print(f"✅ Extraction complete!")
        print("")

        # Display results
        for i, result in enumerate(results):
            print(f"Document {i + 1}:")
            print(f"  Document Type: {result.document_type}")
            print(f"  Reasoning Tier: {result.reasoning_tier}")
            print(f"  Model Used: {result.model_used}")
            print(f"  Processing Time: {result.processing_time_ms:.0f}ms")
            print(f"  Cost: ${result.cost_usd:.4f}")
            print(f"  Confidence: {result.confidence_score * 100:.1f}%")
            print(f"  Verification: {result.verification_status}")
            print("")

            print(f"  Complexity Markers:")
            markers = result.complexity_markers
            print(f"    Score: {markers.complexity_score:.2f}")
            print(f"    Page Count: {markers.page_count}")
            print(f"    Nested Tables: {markers.has_nested_tables}")
            print(f"    Multi-Column: {markers.has_multi_column_layout}")
            print(f"    Low Quality: {markers.has_low_quality_scan}")
            print("")

            # Show extracted content summary
            if result.legal_content:
                print(f"  Legal Content:")
                print(f"    Title: {result.legal_content.title}")
                print(f"    Parties: {', '.join(result.legal_content.parties)}")
                print(f"    Clauses: {len(result.legal_content.clauses)}")
            elif result.financial_content:
                print(f"  Financial Content:")
                print(f"    Company: {result.financial_content.company_name}")
                print(f"    Period: {result.financial_content.reporting_period}")
                print(f"    Tables: {len(result.financial_content.tables)}")
            elif result.technical_content:
                print(f"  Technical Content:")
                print(f"    Title: {result.technical_content.title}")
                print(f"    Version: {result.technical_content.version}")
                print(f"    Specifications: {len(result.technical_content.specifications)}")

            print("")
            print("─" * 60)

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def test_complexity_detection():
    """Test just the complexity detection (fast)"""
    print("🧪 Testing complexity detection...")

    sample_pdf_path = Path("sample.pdf")

    if not sample_pdf_path.exists():
        print("❌ sample.pdf not found")
        return

    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    from modal_app import system1_scan

    markers, doc_info = system1_scan.local(pdf_bytes, "test_doc_001")

    print(f"📊 Complexity Analysis:")
    print(f"   Score: {markers.complexity_score:.2f}")
    print(f"   Inferred Type: {doc_info['inferred_type']}")
    print(f"   Processing Time: {doc_info['processing_time_ms']:.0f}ms")
    print("")
    print(f"   Markers:")
    print(f"     Nested Tables: {markers.has_nested_tables}")
    print(f"     Multi-Column: {markers.has_multi_column_layout}")
    print(f"     Handwriting: {markers.has_handwriting}")
    print(f"     Low Quality: {markers.has_low_quality_scan}")
    print(f"     Ambiguous Language: {markers.has_ambiguous_language}")
    print(f"     Complex Formulas: {markers.has_complex_formulas}")
    print(f"     Mixed Language: {markers.language_is_mixed}")
    print(f"     Page Count: {markers.page_count}")
    print(f"     Estimated Entities: {markers.estimated_entities}")
    print("")

    if markers.complexity_score >= 0.8:
        print("   ➡️  Would route to SYSTEM 2 (Deep Reasoning)")
    else:
        print("   ➡️  Would route to SYSTEM 1 (Fast Extraction)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "complexity":
        test_complexity_detection()
    else:
        test_local()
