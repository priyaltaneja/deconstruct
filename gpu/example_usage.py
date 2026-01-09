"""
Example Usage of Deconstruct Extraction Pipeline

This script demonstrates various ways to use Deconstruct for document extraction.
Run sections individually to understand the workflow.
"""

import base64
from config import DEFAULT_COMPLEXITY_THRESHOLD

# ============================================
# Example 1: Basic Single Document Extraction
# ============================================

def example_single_extraction():
    """Extract data from a single PDF"""
    import modal
    from pathlib import Path

    # Load your PDF
    pdf_path = Path("your_document.pdf")
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found. Please provide a PDF to test.")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Encode as base64 for transmission
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Get the Modal function
    route_and_extract = modal.Function.lookup("deconstruct-extractor", "route_and_extract")

    # Run extraction
    result = route_and_extract.remote(
        pdf_b64=pdf_b64,
        document_id=pdf_path.stem,
        complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
        force_system2=False,
    )

    # Print results
    print(f"✅ Extraction complete!")
    print(f"Document Type: {result.document_type}")
    print(f"Reasoning Tier: {result.reasoning_tier}")
    print(f"Model: {result.model_used}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Time: {result.processing_time_ms:.0f}ms")
    print(f"Confidence: {result.confidence_score * 100:.1f}%")

    # Access extracted data
    if result.legal_content:
        print(f"\nLegal Document: {result.legal_content.title}")
        print(f"Clauses: {len(result.legal_content.clauses)}")
        for clause in result.legal_content.clauses[:3]:
            print(f"  - {clause.clause_type}: {clause.text[:100]}...")

    return result


# ============================================
# Example 2: Batch Processing Multiple PDFs
# ============================================

def example_batch_extraction():
    """Process multiple PDFs in parallel"""
    import modal
    from pathlib import Path

    # Collect PDFs from a directory
    pdf_dir = Path("./sample_pdfs")
    if not pdf_dir.exists():
        print("❌ ./sample_pdfs directory not found")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in ./sample_pdfs")
        return

    print(f"📁 Found {len(pdf_files)} PDFs")

    # Read all PDFs and encode as base64
    documents_b64 = []
    document_ids = []
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            documents_b64.append(base64.b64encode(pdf_bytes).decode('utf-8'))
            document_ids.append(pdf_file.stem)

    # Get Modal function
    route_and_extract = modal.Function.lookup("deconstruct-extractor", "route_and_extract")

    # Run batch extraction (process each document)
    print("🚀 Starting batch extraction...")
    results = []
    for pdf_b64, doc_id in zip(documents_b64, document_ids):
        result = route_and_extract.remote(
            pdf_b64=pdf_b64,
            document_id=doc_id,
            complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
            force_system2=False,
        )
        results.append(result)
        print(f"  ✓ Processed {doc_id}")

    # Aggregate statistics
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.processing_time_ms for r in results)
    system1_count = sum(1 for r in results if r.reasoning_tier == "system1")
    system2_count = sum(1 for r in results if r.reasoning_tier == "system2")

    print(f"\n✅ Batch complete!")
    print(f"Total Documents: {len(results)}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Total Time: {total_time / 1000:.1f}s")
    print(f"System 1: {system1_count} documents")
    print(f"System 2: {system2_count} documents")
    print(f"Avg Cost/Doc: ${total_cost / len(results):.4f}")

    return results


# ============================================
# Example 3: Complexity Analysis Only
# ============================================

def example_complexity_analysis():
    """Check document complexity without full extraction"""
    import modal
    from pathlib import Path

    pdf_path = Path("your_document.pdf")
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Encode as base64 for transmission
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Get Modal function
    system1_scan = modal.Function.lookup("deconstruct-extractor", "system1_scan")

    # Run complexity scan
    markers, doc_info = system1_scan.remote(pdf_b64, pdf_path.stem)

    # Display results
    print(f"📊 Complexity Analysis for {pdf_path.name}")
    print(f"─" * 50)
    print(f"Complexity Score: {markers.complexity_score:.2f}")
    print(f"Inferred Type: {doc_info['inferred_type']}")
    print(f"Scan Time: {doc_info['processing_time_ms']:.0f}ms")
    print(f"\nMarkers:")
    print(f"  Nested Tables: {markers.has_nested_tables}")
    print(f"  Multi-Column Layout: {markers.has_multi_column_layout}")
    print(f"  Handwriting: {markers.has_handwriting}")
    print(f"  Low Quality Scan: {markers.has_low_quality_scan}")
    print(f"  Ambiguous Language: {markers.has_ambiguous_language}")
    print(f"  Complex Formulas: {markers.has_complex_formulas}")
    print(f"  Mixed Language: {markers.language_is_mixed}")
    print(f"  Page Count: {markers.page_count}")
    print(f"  Estimated Entities: {markers.estimated_entities}")

    print(f"\n{'🧠 System 2 Recommended' if markers.complexity_score >= DEFAULT_COMPLEXITY_THRESHOLD else '⚡ System 1 Sufficient'}")

    return markers, doc_info


# ============================================
# Example 4: Force System 2 for High-Stakes Document
# ============================================

def example_force_system2():
    """Force high-quality extraction with System 2 (Vision LLM)"""
    import modal
    from pathlib import Path

    pdf_path = Path("important_contract.pdf")
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Encode as base64 for transmission
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Get Modal function - use route_and_extract with force_system2=True
    route_and_extract = modal.Function.lookup("deconstruct-extractor", "route_and_extract")

    # Run System 2 extraction (Qwen2-VL Vision Model)
    print("🧠 Running deep extraction with Vision LLM...")
    result = route_and_extract.remote(
        pdf_b64=pdf_b64,
        document_id=pdf_path.stem,
        complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
        force_system2=True,  # Force System 2
    )

    print(f"✅ Deep extraction complete!")
    print(f"Model: {result.model_used}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Confidence: {result.confidence_score * 100:.1f}%")

    return result


# ============================================
# Example 5: Store Results in Supabase
# ============================================

def example_store_in_supabase():
    """Extract and store results in Supabase"""
    import modal
    from pathlib import Path
    from supabase import create_client
    import os

    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL", "http://localhost:54321")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not supabase_key:
        print("❌ SUPABASE_ANON_KEY not set")
        return

    supabase = create_client(supabase_url, supabase_key)

    # Extract document
    pdf_path = Path("your_document.pdf")
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Encode as base64 for transmission
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

    route_and_extract = modal.Function.lookup("deconstruct-extractor", "route_and_extract")
    result = route_and_extract.remote(
        pdf_b64=pdf_b64,
        document_id=pdf_path.stem,
        complexity_threshold=DEFAULT_COMPLEXITY_THRESHOLD,
        force_system2=False,
    )

    # Create batch in Supabase
    batch = supabase.table("batches").insert({
        "name": "Manual Extraction",
    }).execute()
    batch_id = batch.data[0]["id"]

    # Insert document record
    doc = supabase.table("documents").insert({
        "batch_id": batch_id,
        "file_name": pdf_path.name,
        "file_size": len(pdf_bytes),
        "file_hash": hash(pdf_bytes),
        "document_type": result.document_type,
        "status": "completed",
        "reasoning_tier": result.reasoning_tier,
        "model_used": result.model_used,
        "processing_time_ms": result.processing_time_ms,
        "cost_usd": float(result.cost_usd),
        "confidence_score": float(result.confidence_score),
        "verification_status": result.verification_status,
    }).execute()
    doc_id = doc.data[0]["id"]

    # Insert complexity markers
    supabase.table("complexity_markers").insert({
        "document_id": doc_id,
        **result.complexity_markers.model_dump(),
    }).execute()

    # Insert extracted entities (example for legal)
    if result.legal_content:
        for clause in result.legal_content.clauses:
            supabase.table("legal_clauses").insert({
                "document_id": doc_id,
                **clause.model_dump(),
            }).execute()

    print(f"✅ Stored in Supabase!")
    print(f"Batch ID: {batch_id}")
    print(f"Document ID: {doc_id}")

    return doc_id


# ============================================
# Example 6: Semantic Search with Embeddings
# ============================================

def example_semantic_search():
    """Search for similar documents using embeddings"""
    from supabase import create_client
    import os

    supabase_url = os.getenv("SUPABASE_URL", "http://localhost:54321")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not supabase_key:
        print("❌ SUPABASE_ANON_KEY not set")
        return

    supabase = create_client(supabase_url, supabase_key)

    # Generate embedding for query (using a mock here)
    query_text = "indemnification clause"
    # In production, use: embedding = generate_embedding(query_text)
    query_embedding = [0.1] * 1536  # Mock embedding

    # Search similar content
    result = supabase.rpc(
        "search_similar_content",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.8,
            "match_count": 10,
        }
    ).execute()

    print(f"🔍 Found {len(result.data)} similar results")
    for item in result.data:
        print(f"  Document: {item['document_id']}")
        print(f"  Source: {item['source_type']} - {item['source_id']}")
        print(f"  Similarity: {item['similarity']:.2%}")
        print(f"  Content: {item['content'][:100]}...")
        print()

    return result.data


# ============================================
# Run Examples
# ============================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <example_number>")
        print("\nAvailable examples:")
        print("  1 - Single document extraction")
        print("  2 - Batch processing")
        print("  3 - Complexity analysis only")
        print("  4 - Force System 2 extraction")
        print("  5 - Store results in Supabase")
        print("  6 - Semantic search")
        sys.exit(1)

    example_num = int(sys.argv[1])

    examples = {
        1: example_single_extraction,
        2: example_batch_extraction,
        3: example_complexity_analysis,
        4: example_force_system2,
        5: example_store_in_supabase,
        6: example_semantic_search,
    }

    if example_num not in examples:
        print(f"❌ Example {example_num} not found")
        sys.exit(1)

    examples[example_num]()
