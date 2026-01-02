"""
Pydantic Schemas for Structured Document Extraction
Deep schemas for Legal, Financial, and Technical documents
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# ============ LEGAL SCHEMAS ============
class LegalClause(BaseModel):
    """Represents a single legal clause with context"""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_type: Literal[
        "definition", "obligation", "right", "prohibition",
        "condition", "termination", "liability", "indemnity"
    ]
    text: str = Field(..., description="Full text of the clause")
    section_reference: str = Field(..., description="Section number or reference")
    parties_involved: List[str] = Field(default_factory=list, description="Parties affected by this clause")
    jurisdiction: Optional[str] = Field(None, description="Applicable jurisdiction")
    effective_date: Optional[str] = Field(None, description="When the clause becomes effective")
    dependencies: List[str] = Field(default_factory=list, description="References to other clause IDs")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Risk assessment")
    extracted_entities: List[str] = Field(default_factory=list, description="Key entities mentioned")


class LegalDocument(BaseModel):
    """Complete legal document structure"""
    document_type: Literal["contract", "agreement", "amendment", "policy", "terms", "other"]
    title: str
    parties: List[str] = Field(..., description="All parties to the document")
    execution_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    clauses: List[LegalClause]
    key_obligations: List[str] = Field(..., description="Summary of main obligations")
    risk_summary: str = Field(..., description="Overall risk assessment")


# ============ FINANCIAL SCHEMAS ============
class FinancialCell(BaseModel):
    """Single cell in a financial table"""
    value: str = Field(..., description="Cell value (can be number, text, or formula)")
    value_type: Literal["currency", "percentage", "number", "text", "date"]
    formatted_value: Optional[str] = Field(None, description="Human-readable formatted value")
    row_label: str
    column_label: str
    is_calculated: bool = Field(default=False, description="Whether this is a calculated field")


class FinancialTable(BaseModel):
    """Structured financial table with metadata"""
    table_id: str = Field(..., description="Unique identifier for the table")
    table_type: Literal[
        "balance_sheet", "income_statement", "cash_flow",
        "schedule", "footnote", "other"
    ]
    title: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    currency: str = Field(default="USD", description="Currency code")
    headers: List[str] = Field(..., description="Column headers")
    rows: List[List[FinancialCell]] = Field(..., description="Table data organized by rows")
    totals: Optional[List[FinancialCell]] = Field(None, description="Total/summary row")
    notes: List[str] = Field(default_factory=list, description="Footnotes and references")
    audited: bool = Field(default=False, description="Whether this data is audited")


class FinancialDocument(BaseModel):
    """Complete financial document structure"""
    document_type: Literal["10-K", "10-Q", "earnings", "prospectus", "other"]
    company_name: str
    reporting_period: str
    filing_date: Optional[str] = None
    tables: List[FinancialTable]
    key_metrics: dict = Field(..., description="Extracted KPIs and metrics")
    executive_summary: str = Field(..., description="High-level summary of financials")


# ============ TECHNICAL SCHEMAS ============
class TechnicalSpecification(BaseModel):
    """Single technical specification or requirement"""
    spec_id: str = Field(..., description="Unique identifier")
    category: Literal[
        "functional", "non_functional", "performance",
        "security", "compliance", "interface", "data"
    ]
    title: str
    description: str
    priority: Literal["P0", "P1", "P2", "P3"]
    acceptance_criteria: List[str] = Field(..., description="Testable acceptance criteria")
    dependencies: List[str] = Field(default_factory=list, description="Dependent spec IDs")
    technical_details: dict = Field(..., description="Additional technical metadata")


class TechnicalDocument(BaseModel):
    """Technical specification or architecture document"""
    document_type: Literal["architecture", "api_spec", "requirements", "design", "other"]
    title: str
    version: str
    last_updated: Optional[str] = None
    specifications: List[TechnicalSpecification]
    diagrams_detected: int = Field(default=0, description="Number of diagrams found")
    code_snippets: List[str] = Field(default_factory=list, description="Extracted code examples")
    technology_stack: List[str] = Field(default_factory=list, description="Technologies mentioned")


# ============ UNIVERSAL EXTRACTION SCHEMA ============
class ComplexityMarkers(BaseModel):
    """Signals that determine routing to System 1 vs System 2"""
    has_nested_tables: bool = False
    has_multi_column_layout: bool = False
    has_handwriting: bool = False
    has_low_quality_scan: bool = False
    has_ambiguous_language: bool = False
    has_complex_formulas: bool = False
    language_is_mixed: bool = False
    page_count: int
    estimated_entities: int = Field(..., description="Rough count of extractable entities")

    @property
    def complexity_score(self) -> float:
        """Calculate complexity score between 0 and 1"""
        score = 0.0
        score += 0.15 if self.has_nested_tables else 0
        score += 0.1 if self.has_multi_column_layout else 0
        score += 0.2 if self.has_handwriting else 0
        score += 0.15 if self.has_low_quality_scan else 0
        score += 0.2 if self.has_ambiguous_language else 0
        score += 0.1 if self.has_complex_formulas else 0
        score += 0.05 if self.language_is_mixed else 0
        score += min(0.05, self.page_count / 200)  # More pages = more complex
        return min(score, 1.0)


class ExtractionResult(BaseModel):
    """Universal extraction result wrapper"""
    document_id: str
    document_type: Literal["legal", "financial", "technical", "general"]
    complexity_markers: ComplexityMarkers
    reasoning_tier: Literal["system1", "system2"] = Field(..., description="Which model tier was used")
    model_used: str = Field(..., description="Specific model name")
    processing_time_ms: float
    cost_usd: float

    # Polymorphic content based on document_type
    legal_content: Optional[LegalDocument] = None
    financial_content: Optional[FinancialDocument] = None
    technical_content: Optional[TechnicalDocument] = None

    # Fallback for unknown document types
    raw_text: Optional[str] = None
    extracted_entities: List[dict] = Field(default_factory=list)

    # Quality metrics
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    verification_status: Literal["passed", "failed", "needs_review"]
    verification_notes: List[str] = Field(default_factory=list)


class BatchExtractionRequest(BaseModel):
    """Request for batch processing multiple documents"""
    batch_id: str
    documents: List[bytes] = Field(..., description="List of PDF bytes")
    force_system2: bool = Field(default=False, description="Force all documents to System 2")
    complexity_threshold: float = Field(default=0.8, description="Threshold for System 2 escalation")
    enable_verification: bool = Field(default=True, description="Enable critic agent verification")
