"""
Process uploaded PDFs through Modal and show results
Run: python process_uploaded.py <path_to_pdf>
"""

import sys
import modal
import json
from pathlib import Path

from config import DEFAULT_COMPLEXITY_THRESHOLD

def process_pdf(pdf_path: str):
    """Process a PDF through the Modal extraction pipeline"""

    print("="*70)
    print("DECONSTRUCT - Document Extraction Pipeline")
    print("="*70)
    print()

    # Load PDF
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return

    with open(pdf_file, 'rb') as f:
        pdf_bytes = f.read()

    print(f"📄 Processing: {pdf_file.name}")
    print(f"📊 Size: {len(pdf_bytes):,} bytes")
    print()

    # Get Modal function
    try:
        print("🔗 Connecting to Modal...")
        route_and_extract = modal.Function.lookup("deconstruct-shredder", "route_and_extract")
        print("✓ Connected to Modal deployment")
        print()
    except Exception as e:
        print(f"❌ Error connecting to Modal: {e}")
        return

    # Run extraction
    print("🧠 STEP 1: Complexity Detection (System 1 Scan)")
    print("-" * 70)

    try:
        result = route_and_extract.remote(
            pdf_bytes=pdf_bytes,
            document_id=pdf_file.stem,
            complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
            force_system2=False,
        )

        # Display Complexity Analysis
        markers = result.complexity_markers
        print(f"   Complexity Score: {markers.complexity_score:.2f}")
        print(f"   Page Count: {markers.page_count}")
        print(f"   Nested Tables: {'Yes' if markers.has_nested_tables else 'No'}")
        print(f"   Multi-Column Layout: {'Yes' if markers.has_multi_column_layout else 'No'}")
        print(f"   Low Quality Scan: {'Yes' if markers.has_low_quality_scan else 'No'}")
        print(f"   Ambiguous Language: {'Yes' if markers.has_ambiguous_language else 'No'}")
        print(f"   Complex Formulas: {'Yes' if markers.has_complex_formulas else 'No'}")
        print()

        # Display Routing Decision
        print("🎯 STEP 2: Routing Decision")
        print("-" * 70)

        if result.reasoning_tier == "system2":
            print(f"   ➡️  Routed to SYSTEM 2 (Deep Reasoning)")
            print(f"   💡 Reason: Complexity score ({markers.complexity_score:.2f}) >= threshold ({DEFAULT_COMPLEXITY_THRESHOLD})")
            print(f"   🧠 Model: {result.model_used}")
        else:
            print(f"   ➡️  Routed to SYSTEM 1 (Fast Extraction)")
            print(f"   💡 Reason: Complexity score ({markers.complexity_score:.2f}) < threshold ({DEFAULT_COMPLEXITY_THRESHOLD})")
            print(f"   ⚡ Model: {result.model_used}")
        print()

        # Display Extraction Results
        print("📦 STEP 3: Extraction Results")
        print("-" * 70)
        print(f"   Document Type: {result.document_type}")
        print(f"   Processing Time: {result.processing_time_ms:.0f}ms ({result.processing_time_ms/1000:.2f}s)")
        print(f"   Cost: ${result.cost_usd:.4f}")
        print(f"   Confidence: {result.confidence_score * 100:.1f}%")
        print(f"   Verification Status: {result.verification_status}")
        print()

        # Display Structured JSON Output
        print("📋 STEP 4: Structured JSON Output")
        print("-" * 70)

        # Convert to JSON
        result_dict = result.model_dump()

        # Show relevant content based on document type
        if result.legal_content:
            print("   📜 LEGAL DOCUMENT DETECTED")
            print(f"   Title: {result.legal_content.title}")
            print(f"   Parties: {', '.join(result.legal_content.parties)}")
            print(f"   Document Type: {result.legal_content.document_type}")
            print(f"   Execution Date: {result.legal_content.execution_date or 'N/A'}")
            print(f"   Total Clauses: {len(result.legal_content.clauses)}")
            print()

            if result.legal_content.clauses:
                print("   Top 3 Clauses:")
                for i, clause in enumerate(result.legal_content.clauses[:3], 1):
                    print(f"     {i}. [{clause.clause_type.upper()}] {clause.section_reference}")
                    print(f"        Risk Level: {clause.risk_level}")
                    print(f"        Text: {clause.text[:100]}...")
                    print()

        elif result.financial_content:
            print("   💰 FINANCIAL DOCUMENT DETECTED")
            print(f"   Company: {result.financial_content.company_name}")
            print(f"   Document Type: {result.financial_content.document_type}")
            print(f"   Reporting Period: {result.financial_content.reporting_period}")
            print(f"   Total Tables: {len(result.financial_content.tables)}")
            print()

        elif result.technical_content:
            print("   🔧 TECHNICAL DOCUMENT DETECTED")
            print(f"   Title: {result.technical_content.title}")
            print(f"   Version: {result.technical_content.version}")
            print(f"   Document Type: {result.technical_content.document_type}")
            print(f"   Total Specifications: {len(result.technical_content.specifications)}")
            print()

        # Save full JSON
        output_file = Path(f"{pdf_file.stem}_extracted.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"   ✅ Full JSON saved to: {output_file}")
        print()

        # Display summary
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Reasoning Tier: {result.reasoning_tier.upper()}")
        print(f"✓ Model Used: {result.model_used}")
        print(f"✓ Document Type: {result.document_type}")
        print(f"✓ Processing Time: {result.processing_time_ms/1000:.2f}s")
        print(f"✓ Cost: ${result.cost_usd:.4f}")
        print(f"✓ Confidence: {result.confidence_score * 100:.1f}%")
        print()
        print(f"📁 Full structured output: {output_file}")
        print()

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_uploaded.py <path_to_pdf>")
        print("\nExample:")
        print("  python process_uploaded.py document.pdf")
        sys.exit(1)

    process_pdf(sys.argv[1])
