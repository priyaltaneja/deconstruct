"""
Supabase client for persisting extraction results.

Usage:
    from supabase_client import save_extraction_result

    # After Modal extraction completes:
    doc_id = save_extraction_result(result_dict, file_name, file_size)
"""

import os
import hashlib
import uuid
from datetime import datetime
from typing import Optional
from supabase import create_client, Client


def get_client() -> Client:
    """Get Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set. "
            "For local development, run 'supabase start' and use the output values."
        )

    return create_client(url, key)


def get_or_create_batch(
    client: Client,
    batch_name: str = "default",
    complexity_threshold: float = 0.8,
    force_system2: bool = False
) -> str:
    """Get existing batch by name or create a new one."""
    # Try to find existing batch with this name created today
    today = datetime.now().strftime("%Y-%m-%d")

    result = client.table("batches").select("id").eq("name", batch_name).gte("created_at", today).execute()

    if result.data:
        return result.data[0]["id"]

    # Create new batch
    batch = {
        "name": batch_name,
        "force_system2": force_system2,
        "complexity_threshold": complexity_threshold,
        "total_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
    }

    result = client.table("batches").insert(batch).execute()
    return result.data[0]["id"]


def save_extraction_result(
    result: dict,
    file_name: str,
    file_size: int,
    file_bytes: bytes = None,
    batch_name: str = "api-extractions"
) -> Optional[str]:
    """
    Save extraction result to Supabase.

    Args:
        result: The extraction result dict from Modal
        file_name: Original file name
        file_size: File size in bytes
        file_bytes: Raw file bytes for hash (optional)
        batch_name: Name for grouping extractions

    Returns:
        Document ID if successful, None if failed
    """
    try:
        client = get_client()

        # Get or create batch
        batch_id = get_or_create_batch(
            client,
            batch_name,
            result.get("complexity_threshold", 0.8),
            result.get("force_system2", False)
        )

        # Calculate file hash
        if file_bytes:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        else:
            file_hash = hashlib.sha256(file_name.encode()).hexdigest()

        # Determine document type from result
        document_type = result.get("document_type", "general")
        if document_type not in ("legal", "financial", "technical", "general"):
            document_type = "general"

        # Map reasoning tier
        reasoning_tier = result.get("reasoning_tier", "system2")
        if reasoning_tier not in ("system1", "system2"):
            reasoning_tier = "system2"

        # Create document record
        document = {
            "batch_id": batch_id,
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "document_type": document_type,
            "status": "completed",
            "reasoning_tier": reasoning_tier,
            "model_used": result.get("model_used", "unknown"),
            "processing_time_ms": result.get("processing_time_ms"),
            "cost_usd": result.get("cost_usd"),
            "confidence_score": result.get("confidence_score"),
            "verification_status": "passed" if result.get("confidence_score", 0) >= 0.7 else "needs_review",
        }

        doc_result = client.table("documents").insert(document).execute()
        document_id = doc_result.data[0]["id"]

        # Update batch total_jobs count
        client.table("batches").update({"total_jobs": client.table("batches").select("total_jobs").eq("id", batch_id).execute().data[0]["total_jobs"] + 1}).eq("id", batch_id).execute()

        # Save complexity markers if present
        complexity_markers = result.get("complexity_markers")
        if complexity_markers:
            markers = {
                "document_id": document_id,
                "complexity_score": complexity_markers.get("complexity_score", 0.5),
                "page_count": complexity_markers.get("page_count", 1),
                "estimated_entities": complexity_markers.get("estimated_entities", 0),
                "has_nested_tables": complexity_markers.get("has_nested_tables", False),
                "has_multi_column_layout": complexity_markers.get("has_multi_column_layout", False),
                "has_handwriting": complexity_markers.get("has_handwriting", False),
                "has_low_quality_scan": complexity_markers.get("has_low_quality_scan", False),
                "has_ambiguous_language": complexity_markers.get("has_ambiguous_language", False),
                "has_complex_formulas": complexity_markers.get("has_complex_formulas", False),
                "language_is_mixed": complexity_markers.get("language_is_mixed", False),
            }
            client.table("complexity_markers").insert(markers).execute()

        # Save type-specific content
        if document_type == "legal" and result.get("legal_content"):
            save_legal_content(client, document_id, result["legal_content"])
        elif document_type == "financial" and result.get("financial_content"):
            save_financial_content(client, document_id, result["financial_content"])
        elif document_type == "technical" and result.get("technical_content"):
            save_technical_content(client, document_id, result["technical_content"])

        # Save reasoning log if present
        reasoning_trace = result.get("reasoning_trace")
        if reasoning_trace:
            log = {
                "document_id": document_id,
                "reasoning_tier": reasoning_tier,
                "model_used": result.get("model_used", "unknown"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_tokens": result.get("total_tokens"),
                "reasoning_trace": reasoning_trace,
            }
            client.table("reasoning_logs").insert(log).execute()

        print(f"[SUPABASE] Saved document: {document_id}")
        return document_id

    except Exception as e:
        print(f"[SUPABASE ERROR] Failed to save: {e}")
        return None


def save_legal_content(client: Client, document_id: str, content: dict):
    """Save legal document content."""
    # Save legal document metadata
    legal_doc = {
        "document_id": document_id,
        "document_type": content.get("document_type", "other"),
        "title": content.get("title", "Untitled"),
        "parties": content.get("parties", []),
        "execution_date": content.get("execution_date"),
        "effective_date": content.get("effective_date"),
        "expiration_date": content.get("expiration_date"),
        "key_obligations": content.get("key_obligations", []),
        "risk_summary": content.get("risk_summary", ""),
    }

    # Validate document_type
    valid_types = ("contract", "agreement", "amendment", "policy", "terms", "other")
    if legal_doc["document_type"] not in valid_types:
        legal_doc["document_type"] = "other"

    try:
        client.table("legal_documents").insert(legal_doc).execute()
    except Exception as e:
        print(f"[SUPABASE] Warning: Could not save legal_documents: {e}")

    # Save clauses
    clauses = content.get("clauses", [])
    for clause in clauses:
        clause_type = clause.get("clause_type", "obligation")
        valid_clause_types = ("definition", "obligation", "right", "prohibition",
                             "condition", "termination", "liability", "indemnity")
        if clause_type not in valid_clause_types:
            clause_type = "obligation"

        risk_level = clause.get("risk_level", "low")
        if risk_level not in ("low", "medium", "high", "critical"):
            risk_level = "low"

        clause_record = {
            "document_id": document_id,
            "clause_id": clause.get("clause_id", str(uuid.uuid4())[:8]),
            "clause_type": clause_type,
            "text": clause.get("text", "")[:10000],  # Limit text length
            "section_reference": clause.get("section_reference", ""),
            "parties_involved": clause.get("parties_involved", []),
            "jurisdiction": clause.get("jurisdiction"),
            "effective_date": clause.get("effective_date"),
            "dependencies": clause.get("dependencies", []),
            "risk_level": risk_level,
            "extracted_entities": clause.get("extracted_entities", []),
        }
        try:
            client.table("legal_clauses").insert(clause_record).execute()
        except Exception as e:
            print(f"[SUPABASE] Warning: Could not save clause: {e}")


def save_financial_content(client: Client, document_id: str, content: dict):
    """Save financial document content."""
    # Save financial document metadata
    fin_doc = {
        "document_id": document_id,
        "document_type": content.get("document_type", "other"),
        "company_name": content.get("company_name", "Unknown"),
        "reporting_period": content.get("reporting_period", "Unknown"),
        "filing_date": content.get("filing_date"),
        "key_metrics": content.get("key_metrics", {}),
        "executive_summary": content.get("executive_summary", ""),
    }

    valid_types = ("10-K", "10-Q", "earnings", "prospectus", "other")
    if fin_doc["document_type"] not in valid_types:
        fin_doc["document_type"] = "other"

    try:
        client.table("financial_documents").insert(fin_doc).execute()
    except Exception as e:
        print(f"[SUPABASE] Warning: Could not save financial_documents: {e}")

    # Save tables
    tables = content.get("tables", [])
    for table in tables:
        table_type = table.get("table_type", "other")
        valid_table_types = ("balance_sheet", "income_statement", "cash_flow",
                           "schedule", "footnote", "other")
        if table_type not in valid_table_types:
            table_type = "other"

        table_record = {
            "document_id": document_id,
            "table_id": table.get("table_id", str(uuid.uuid4())[:8]),
            "table_type": table_type,
            "title": table.get("title", "Untitled"),
            "period_start": table.get("period_start"),
            "period_end": table.get("period_end"),
            "currency": table.get("currency", "USD"),
            "headers": table.get("headers", []),
            "rows": table.get("rows", []),
            "totals": table.get("totals"),
            "notes": table.get("notes", []),
            "audited": table.get("audited", False),
        }
        try:
            client.table("financial_tables").insert(table_record).execute()
        except Exception as e:
            print(f"[SUPABASE] Warning: Could not save table: {e}")


def save_technical_content(client: Client, document_id: str, content: dict):
    """Save technical document content."""
    # Save technical document metadata
    tech_doc = {
        "document_id": document_id,
        "document_type": content.get("document_type", "other"),
        "title": content.get("title", "Untitled"),
        "version": content.get("version", "1.0"),
        "last_updated": content.get("last_updated"),
        "diagrams_detected": content.get("diagrams_detected", 0),
        "code_snippets": content.get("code_snippets", []),
        "technology_stack": content.get("technology_stack", []),
    }

    valid_types = ("architecture", "api_spec", "requirements", "design", "other")
    if tech_doc["document_type"] not in valid_types:
        tech_doc["document_type"] = "other"

    try:
        client.table("technical_documents").insert(tech_doc).execute()
    except Exception as e:
        print(f"[SUPABASE] Warning: Could not save technical_documents: {e}")

    # Save specifications
    specs = content.get("specifications", [])
    for spec in specs:
        category = spec.get("category", "functional")
        valid_categories = ("functional", "non_functional", "performance",
                          "security", "compliance", "interface", "data")
        if category not in valid_categories:
            category = "functional"

        priority = spec.get("priority", "P2")
        if priority not in ("P0", "P1", "P2", "P3"):
            priority = "P2"

        spec_record = {
            "document_id": document_id,
            "spec_id": spec.get("spec_id", str(uuid.uuid4())[:8]),
            "category": category,
            "title": spec.get("title", "Untitled"),
            "description": spec.get("description", ""),
            "priority": priority,
            "acceptance_criteria": spec.get("acceptance_criteria", []),
            "dependencies": spec.get("dependencies", []),
            "technical_details": spec.get("technical_details", {}),
        }
        try:
            client.table("technical_specifications").insert(spec_record).execute()
        except Exception as e:
            print(f"[SUPABASE] Warning: Could not save spec: {e}")


def get_recent_extractions(limit: int = 20) -> list:
    """Get recent extraction results from the database."""
    try:
        client = get_client()
        result = client.table("documents")\
            .select("*, batches(name)")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data
    except Exception as e:
        print(f"[SUPABASE ERROR] Failed to get extractions: {e}")
        return []


def get_batch_stats() -> list:
    """Get batch performance statistics."""
    try:
        client = get_client()
        result = client.rpc("get_batch_stats").execute()
        return result.data
    except Exception as e:
        print(f"[SUPABASE ERROR] Failed to get batch stats: {e}")
        return []
