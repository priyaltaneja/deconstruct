-- Deconstruct Initial Schema
-- Tables for storing extracted document data and reasoning logs

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- CORE TABLES
-- ============================================

-- Documents: Stores uploaded documents and metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID NOT NULL,
    file_name TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash TEXT NOT NULL, -- SHA256 hash for deduplication

    -- Document classification
    document_type TEXT CHECK (document_type IN ('legal', 'financial', 'technical', 'general')),

    -- Processing status
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'verifying')),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Processing metadata
    reasoning_tier TEXT CHECK (reasoning_tier IN ('system1', 'system2')),
    model_used TEXT,
    processing_time_ms NUMERIC,
    cost_usd NUMERIC(10, 6),

    -- Quality metrics
    confidence_score NUMERIC(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    verification_status TEXT CHECK (verification_status IN ('passed', 'failed', 'needs_review')),

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Indexes
    CONSTRAINT documents_batch_id_idx FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE
);

-- Batches: Groups of documents processed together
CREATE TABLE batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Batch settings
    force_system2 BOOLEAN DEFAULT FALSE,
    complexity_threshold NUMERIC(3, 2) DEFAULT 0.8,
    enable_verification BOOLEAN DEFAULT TRUE,

    -- Statistics (computed from documents)
    total_jobs INTEGER DEFAULT 0,
    completed_jobs INTEGER DEFAULT 0,
    failed_jobs INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10, 6) DEFAULT 0,
    total_processing_time_ms NUMERIC DEFAULT 0
);

-- ============================================
-- COMPLEXITY MARKERS
-- ============================================

CREATE TABLE complexity_markers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Complexity flags
    has_nested_tables BOOLEAN DEFAULT FALSE,
    has_multi_column_layout BOOLEAN DEFAULT FALSE,
    has_handwriting BOOLEAN DEFAULT FALSE,
    has_low_quality_scan BOOLEAN DEFAULT FALSE,
    has_ambiguous_language BOOLEAN DEFAULT FALSE,
    has_complex_formulas BOOLEAN DEFAULT FALSE,
    language_is_mixed BOOLEAN DEFAULT FALSE,

    -- Metrics
    page_count INTEGER NOT NULL,
    estimated_entities INTEGER DEFAULT 0,
    complexity_score NUMERIC(3, 2) NOT NULL CHECK (complexity_score >= 0 AND complexity_score <= 1),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================
-- EXTRACTED ENTITIES (Polymorphic)
-- ============================================

-- Legal Clauses
CREATE TABLE legal_clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    clause_id TEXT NOT NULL,
    clause_type TEXT NOT NULL CHECK (clause_type IN (
        'definition', 'obligation', 'right', 'prohibition',
        'condition', 'termination', 'liability', 'indemnity'
    )),
    text TEXT NOT NULL,
    section_reference TEXT NOT NULL,

    parties_involved JSONB DEFAULT '[]'::jsonb,
    jurisdiction TEXT,
    effective_date TEXT,
    dependencies JSONB DEFAULT '[]'::jsonb, -- Array of clause IDs

    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    extracted_entities JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Legal Documents Metadata
CREATE TABLE legal_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,

    document_type TEXT NOT NULL CHECK (document_type IN (
        'contract', 'agreement', 'amendment', 'policy', 'terms', 'other'
    )),
    title TEXT NOT NULL,
    parties JSONB NOT NULL, -- Array of party names

    execution_date TEXT,
    effective_date TEXT,
    expiration_date TEXT,

    key_obligations JSONB NOT NULL, -- Array of obligations
    risk_summary TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Financial Tables
CREATE TABLE financial_tables (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    table_id TEXT NOT NULL,
    table_type TEXT NOT NULL CHECK (table_type IN (
        'balance_sheet', 'income_statement', 'cash_flow',
        'schedule', 'footnote', 'other'
    )),
    title TEXT NOT NULL,

    period_start TEXT,
    period_end TEXT,
    currency TEXT DEFAULT 'USD',

    headers JSONB NOT NULL, -- Array of column headers
    rows JSONB NOT NULL, -- Array of row data (cells)
    totals JSONB, -- Summary row
    notes JSONB DEFAULT '[]'::jsonb,

    audited BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Financial Documents Metadata
CREATE TABLE financial_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,

    document_type TEXT NOT NULL CHECK (document_type IN (
        '10-K', '10-Q', 'earnings', 'prospectus', 'other'
    )),
    company_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    filing_date TEXT,

    key_metrics JSONB NOT NULL, -- Key financial metrics
    executive_summary TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Technical Specifications
CREATE TABLE technical_specifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    spec_id TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN (
        'functional', 'non_functional', 'performance',
        'security', 'compliance', 'interface', 'data'
    )),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('P0', 'P1', 'P2', 'P3')),

    acceptance_criteria JSONB NOT NULL, -- Array of criteria
    dependencies JSONB DEFAULT '[]'::jsonb, -- Array of spec IDs
    technical_details JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Technical Documents Metadata
CREATE TABLE technical_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,

    document_type TEXT NOT NULL CHECK (document_type IN (
        'architecture', 'api_spec', 'requirements', 'design', 'other'
    )),
    title TEXT NOT NULL,
    version TEXT NOT NULL,
    last_updated TEXT,

    diagrams_detected INTEGER DEFAULT 0,
    code_snippets JSONB DEFAULT '[]'::jsonb,
    technology_stack JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================
-- REASONING LOGS
-- ============================================

-- Stores reasoning traces and chain-of-thought outputs
CREATE TABLE reasoning_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    reasoning_tier TEXT NOT NULL CHECK (reasoning_tier IN ('system1', 'system2')),
    model_used TEXT NOT NULL,

    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    reasoning_trace TEXT, -- Full chain-of-thought output
    intermediate_steps JSONB, -- Structured reasoning steps

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================
-- VECTOR EMBEDDINGS
-- ============================================

-- Stores embeddings for semantic search
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Source of the embedding (e.g., 'full_document', 'clause_123', 'table_456')
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,

    -- Text content that was embedded
    content TEXT NOT NULL,

    -- Vector embedding (1536 dimensions for bge-large)
    embedding vector(1536) NOT NULL,

    -- Metadata
    model_used TEXT DEFAULT 'bge-large',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ============================================
-- VERIFICATION LOGS
-- ============================================

CREATE TABLE verification_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    verification_type TEXT NOT NULL CHECK (verification_type IN (
        'automatic', 'critic_agent', 'manual'
    )),

    status TEXT NOT NULL CHECK (status IN ('passed', 'failed', 'needs_review')),
    confidence_score NUMERIC(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),

    issues_found JSONB DEFAULT '[]'::jsonb,
    corrections_made JSONB DEFAULT '[]'::jsonb,

    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- ============================================

-- Documents
CREATE INDEX documents_batch_id_idx ON documents(batch_id);
CREATE INDEX documents_status_idx ON documents(status);
CREATE INDEX documents_document_type_idx ON documents(document_type);
CREATE INDEX documents_created_at_idx ON documents(created_at DESC);
CREATE INDEX documents_file_hash_idx ON documents(file_hash);

-- Batches
CREATE INDEX batches_created_at_idx ON batches(created_at DESC);

-- Complexity Markers
CREATE INDEX complexity_markers_document_id_idx ON complexity_markers(document_id);
CREATE INDEX complexity_markers_score_idx ON complexity_markers(complexity_score DESC);

-- Entity tables
CREATE INDEX legal_clauses_document_id_idx ON legal_clauses(document_id);
CREATE INDEX legal_clauses_clause_type_idx ON legal_clauses(clause_type);
CREATE INDEX legal_clauses_risk_level_idx ON legal_clauses(risk_level);

CREATE INDEX financial_tables_document_id_idx ON financial_tables(document_id);
CREATE INDEX financial_tables_table_type_idx ON financial_tables(table_type);

CREATE INDEX technical_specifications_document_id_idx ON technical_specifications(document_id);
CREATE INDEX technical_specifications_priority_idx ON technical_specifications(priority);

-- Reasoning logs
CREATE INDEX reasoning_logs_document_id_idx ON reasoning_logs(document_id);
CREATE INDEX reasoning_logs_created_at_idx ON reasoning_logs(created_at DESC);

-- Embeddings
CREATE INDEX embeddings_document_id_idx ON embeddings(document_id);
CREATE INDEX embeddings_source_type_idx ON embeddings(source_type);

-- Verification logs
CREATE INDEX verification_logs_document_id_idx ON verification_logs(document_id);
CREATE INDEX verification_logs_status_idx ON verification_logs(status);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Function to update batch statistics when documents are updated
CREATE OR REPLACE FUNCTION update_batch_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update batch statistics based on document changes
    UPDATE batches
    SET
        completed_jobs = (
            SELECT COUNT(*) FROM documents
            WHERE batch_id = NEW.batch_id AND status = 'completed'
        ),
        failed_jobs = (
            SELECT COUNT(*) FROM documents
            WHERE batch_id = NEW.batch_id AND status = 'failed'
        ),
        total_cost_usd = (
            SELECT COALESCE(SUM(cost_usd), 0) FROM documents
            WHERE batch_id = NEW.batch_id AND status = 'completed'
        ),
        total_processing_time_ms = (
            SELECT COALESCE(SUM(processing_time_ms), 0) FROM documents
            WHERE batch_id = NEW.batch_id AND status = 'completed'
        )
    WHERE id = NEW.batch_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update batch stats when document status changes
CREATE TRIGGER update_batch_stats_trigger
AFTER UPDATE OF status, cost_usd, processing_time_ms ON documents
FOR EACH ROW
EXECUTE FUNCTION update_batch_stats();

-- Function to calculate complexity score (can be called from app)
CREATE OR REPLACE FUNCTION calculate_complexity_score(marker_id UUID)
RETURNS NUMERIC AS $$
DECLARE
    score NUMERIC := 0;
    marker RECORD;
BEGIN
    SELECT * INTO marker FROM complexity_markers WHERE id = marker_id;

    IF marker.has_nested_tables THEN score := score + 0.15; END IF;
    IF marker.has_multi_column_layout THEN score := score + 0.1; END IF;
    IF marker.has_handwriting THEN score := score + 0.2; END IF;
    IF marker.has_low_quality_scan THEN score := score + 0.15; END IF;
    IF marker.has_ambiguous_language THEN score := score + 0.2; END IF;
    IF marker.has_complex_formulas THEN score := score + 0.1; END IF;
    IF marker.language_is_mixed THEN score := score + 0.05; END IF;
    score := score + LEAST(0.05, marker.page_count::NUMERIC / 200);

    RETURN LEAST(score, 1.0);
END;
$$ LANGUAGE plpgsql;
