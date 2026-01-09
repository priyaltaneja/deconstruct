"""
Prompt Templates for Document Extraction
Schema-specific prompts for different document types.
"""


def get_schema_prompt(document_type: str) -> str:
    """
    Get schema-specific extraction prompt.

    Args:
        document_type: Type of document ("legal", "financial", "technical", or "general")

    Returns:
        Extraction prompt with schema guidance
    """
    prompts = {
        "legal": """Extract:
- All clauses with their types (obligation, right, prohibition, condition, termination, liability, indemnity, definition)
- Parties involved in each clause
- Key dates (execution, effective, expiration)
- Risk assessment for each clause (low, medium, high, critical)
- Dependencies between clauses
- Jurisdiction information

Return JSON matching this schema:
{
    "document_type": "contract|agreement|amendment|policy|terms|other",
    "title": "Document title",
    "parties": ["Party A", "Party B"],
    "execution_date": "YYYY-MM-DD or null",
    "effective_date": "YYYY-MM-DD or null",
    "expiration_date": "YYYY-MM-DD or null",
    "clauses": [
        {
            "clause_id": "unique_id",
            "clause_type": "obligation|right|...",
            "text": "Full clause text",
            "section_reference": "Section 1.1",
            "parties_involved": ["Party A"],
            "jurisdiction": "State/Country or null",
            "effective_date": "YYYY-MM-DD or null",
            "dependencies": ["other_clause_id"],
            "risk_level": "low|medium|high|critical",
            "extracted_entities": ["Entity1", "Entity2"]
        }
    ],
    "key_obligations": ["Obligation 1", "Obligation 2"],
    "risk_summary": "Overall risk assessment"
}""",

        "financial": """Extract:
- All tables with their types (balance_sheet, income_statement, cash_flow, schedule, footnote)
- Financial data with proper formatting and currency
- Period information (start/end dates)
- Key metrics and totals
- Footnotes and audit status

Return JSON matching this schema:
{
    "document_type": "10-K|10-Q|earnings|prospectus|other",
    "company_name": "Company Name",
    "reporting_period": "Q1 2024 or FY 2024",
    "filing_date": "YYYY-MM-DD or null",
    "tables": [
        {
            "table_id": "unique_id",
            "table_type": "balance_sheet|income_statement|...",
            "title": "Table title",
            "period_start": "YYYY-MM-DD or null",
            "period_end": "YYYY-MM-DD or null",
            "currency": "USD",
            "headers": ["Column 1", "Column 2"],
            "rows": [[
                {
                    "value": "1000000",
                    "value_type": "currency|percentage|number|text|date",
                    "formatted_value": "$1,000,000",
                    "row_label": "Revenue",
                    "column_label": "2024",
                    "is_calculated": false
                }
            ]],
            "totals": [...],
            "notes": ["Note 1"],
            "audited": true
        }
    ],
    "key_metrics": {"metric_name": "value"},
    "executive_summary": "Summary of financials"
}""",

        "technical": """Extract:
- All specifications and requirements
- Priority and category for each (P0-P3, functional/non_functional/etc.)
- Technical details and dependencies
- Code snippets if present
- Technology stack and diagrams mentioned

Return JSON matching this schema:
{
    "document_type": "architecture|api_spec|requirements|design|other",
    "title": "Document title",
    "version": "1.0.0",
    "last_updated": "YYYY-MM-DD or null",
    "specifications": [
        {
            "spec_id": "unique_id",
            "category": "functional|non_functional|performance|security|compliance|interface|data",
            "title": "Spec title",
            "description": "Full description",
            "priority": "P0|P1|P2|P3",
            "acceptance_criteria": ["Criterion 1"],
            "dependencies": ["other_spec_id"],
            "technical_details": {"key": "value"}
        }
    ],
    "diagrams_detected": 0,
    "code_snippets": ["code here"],
    "technology_stack": ["Python", "React"]
}""",

        "general": """Extract all structured data from this document.

Identify and extract:
- Document title and type
- Key sections and their content
- Named entities (people, organizations, dates, locations)
- Numerical data and metrics
- Any tables or structured information

Return as structured JSON with clear field names."""
    }

    return prompts.get(document_type, prompts["general"])


def get_system_prompt(document_type: str) -> str:
    """
    Get system prompt for LLM extraction.

    Args:
        document_type: Type of document

    Returns:
        System prompt for the LLM
    """
    return f"""You are a deep reasoning extraction system specialized in {document_type} documents.

Your task is to:
1. Carefully analyze the document structure
2. Extract all relevant information according to the schema
3. Verify cross-references and dependencies
4. Assess data quality and confidence

Always return valid JSON matching the provided schema. Think step-by-step before providing your final answer."""


def get_verification_prompt(document_type: str) -> str:
    """
    Get prompt for verification/critic agent.

    Args:
        document_type: Type of document

    Returns:
        Verification prompt
    """
    return f"""You are a verification agent for {document_type} document extraction.

Review the extraction result and check:
1. Are all required fields populated?
2. Are the data types correct?
3. Are there any obvious errors or inconsistencies?
4. Is the confidence score appropriate?

Provide a confidence assessment from 0.0 to 1.0 and any notes about issues found."""
