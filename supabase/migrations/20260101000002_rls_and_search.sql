-- Deconstruct: Row Level Security and Semantic Search
-- Security policies and vector search functions

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE complexity_markers ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_clauses ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial_tables ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE technical_specifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE technical_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE reasoning_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE verification_logs ENABLE ROW LEVEL SECURITY;

-- For now, allow all authenticated users to read/write
-- In production, you'd want more granular policies

-- Batches policies
CREATE POLICY "Allow all for authenticated users" ON batches
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Documents policies
CREATE POLICY "Allow all for authenticated users" ON documents
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Complexity markers policies
CREATE POLICY "Allow all for authenticated users" ON complexity_markers
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Legal entities policies
CREATE POLICY "Allow all for authenticated users" ON legal_clauses
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow all for authenticated users" ON legal_documents
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Financial entities policies
CREATE POLICY "Allow all for authenticated users" ON financial_tables
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow all for authenticated users" ON financial_documents
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Technical entities policies
CREATE POLICY "Allow all for authenticated users" ON technical_specifications
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow all for authenticated users" ON technical_documents
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Reasoning logs policies
CREATE POLICY "Allow all for authenticated users" ON reasoning_logs
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Embeddings policies
CREATE POLICY "Allow all for authenticated users" ON embeddings
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Verification logs policies
CREATE POLICY "Allow all for authenticated users" ON verification_logs
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- ============================================
-- SEMANTIC SEARCH FUNCTIONS
-- ============================================

-- Function to search for similar content using vector embeddings
CREATE OR REPLACE FUNCTION search_similar_content(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    document_id uuid,
    source_type text,
    source_id text,
    content text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.document_id,
        e.source_type,
        e.source_id,
        e.content,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM embeddings e
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get document by similarity to query
CREATE OR REPLACE FUNCTION find_similar_documents(
    query_embedding vector(1536),
    doc_type text DEFAULT NULL,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    document_id uuid,
    file_name text,
    document_type text,
    avg_similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id as document_id,
        d.file_name,
        d.document_type,
        AVG(1 - (e.embedding <=> query_embedding)) as avg_similarity
    FROM documents d
    JOIN embeddings e ON d.id = e.document_id
    WHERE (doc_type IS NULL OR d.document_type = doc_type)
    GROUP BY d.id, d.file_name, d.document_type
    ORDER BY avg_similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- ANALYTICAL VIEWS
-- ============================================

-- View: Batch Performance Summary
CREATE OR REPLACE VIEW batch_performance AS
SELECT
    b.id,
    b.name,
    b.created_at,
    b.total_jobs,
    b.completed_jobs,
    b.failed_jobs,
    ROUND((b.completed_jobs::NUMERIC / NULLIF(b.total_jobs, 0)) * 100, 2) as completion_rate,
    b.total_cost_usd,
    b.total_processing_time_ms,
    ROUND(b.total_cost_usd / NULLIF(b.completed_jobs, 0), 4) as avg_cost_per_job,
    ROUND(b.total_processing_time_ms / NULLIF(b.completed_jobs, 0), 2) as avg_time_per_job,
    COUNT(CASE WHEN d.reasoning_tier = 'system1' THEN 1 END) as system1_count,
    COUNT(CASE WHEN d.reasoning_tier = 'system2' THEN 1 END) as system2_count
FROM batches b
LEFT JOIN documents d ON b.id = d.batch_id
GROUP BY b.id, b.name, b.created_at, b.total_jobs, b.completed_jobs,
         b.failed_jobs, b.total_cost_usd, b.total_processing_time_ms;

-- View: Document Complexity Distribution
CREATE OR REPLACE VIEW complexity_distribution AS
SELECT
    CASE
        WHEN cm.complexity_score < 0.3 THEN 'Low'
        WHEN cm.complexity_score < 0.6 THEN 'Medium'
        WHEN cm.complexity_score < 0.8 THEN 'High'
        ELSE 'Very High'
    END as complexity_level,
    COUNT(*) as document_count,
    AVG(d.processing_time_ms) as avg_processing_time,
    AVG(d.cost_usd) as avg_cost,
    COUNT(CASE WHEN d.reasoning_tier = 'system1' THEN 1 END) as system1_count,
    COUNT(CASE WHEN d.reasoning_tier = 'system2' THEN 1 END) as system2_count
FROM complexity_markers cm
JOIN documents d ON cm.document_id = d.id
WHERE d.status = 'completed'
GROUP BY complexity_level
ORDER BY
    CASE complexity_level
        WHEN 'Low' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'High' THEN 3
        WHEN 'Very High' THEN 4
    END;

-- View: Model Performance Comparison
CREATE OR REPLACE VIEW model_performance AS
SELECT
    d.model_used,
    d.reasoning_tier,
    COUNT(*) as total_documents,
    AVG(d.processing_time_ms) as avg_processing_time,
    AVG(d.cost_usd) as avg_cost,
    AVG(d.confidence_score) as avg_confidence,
    COUNT(CASE WHEN d.verification_status = 'passed' THEN 1 END)::NUMERIC /
        NULLIF(COUNT(*), 0) * 100 as pass_rate
FROM documents d
WHERE d.status = 'completed'
GROUP BY d.model_used, d.reasoning_tier
ORDER BY d.reasoning_tier, avg_cost;

-- View: Risk Analysis (Legal Documents)
CREATE OR REPLACE VIEW legal_risk_analysis AS
SELECT
    d.id as document_id,
    d.file_name,
    ld.title,
    COUNT(lc.id) as total_clauses,
    COUNT(CASE WHEN lc.risk_level = 'critical' THEN 1 END) as critical_clauses,
    COUNT(CASE WHEN lc.risk_level = 'high' THEN 1 END) as high_risk_clauses,
    COUNT(CASE WHEN lc.risk_level = 'medium' THEN 1 END) as medium_risk_clauses,
    COUNT(CASE WHEN lc.risk_level = 'low' THEN 1 END) as low_risk_clauses,
    CASE
        WHEN COUNT(CASE WHEN lc.risk_level = 'critical' THEN 1 END) > 0 THEN 'Critical'
        WHEN COUNT(CASE WHEN lc.risk_level = 'high' THEN 1 END) > 3 THEN 'High'
        WHEN COUNT(CASE WHEN lc.risk_level = 'medium' THEN 1 END) > 5 THEN 'Medium'
        ELSE 'Low'
    END as overall_risk
FROM documents d
JOIN legal_documents ld ON d.id = ld.document_id
LEFT JOIN legal_clauses lc ON d.id = lc.document_id
GROUP BY d.id, d.file_name, ld.title;

-- ============================================
-- UTILITY FUNCTIONS
-- ============================================

-- Function to get document summary
CREATE OR REPLACE FUNCTION get_document_summary(doc_id uuid)
RETURNS jsonb AS $$
DECLARE
    result jsonb;
BEGIN
    SELECT jsonb_build_object(
        'document_id', d.id,
        'file_name', d.file_name,
        'document_type', d.document_type,
        'status', d.status,
        'reasoning_tier', d.reasoning_tier,
        'model_used', d.model_used,
        'processing_time_ms', d.processing_time_ms,
        'cost_usd', d.cost_usd,
        'confidence_score', d.confidence_score,
        'complexity_markers', jsonb_build_object(
            'complexity_score', cm.complexity_score,
            'page_count', cm.page_count,
            'estimated_entities', cm.estimated_entities
        ),
        'entity_counts', jsonb_build_object(
            'legal_clauses', (SELECT COUNT(*) FROM legal_clauses WHERE document_id = doc_id),
            'financial_tables', (SELECT COUNT(*) FROM financial_tables WHERE document_id = doc_id),
            'technical_specs', (SELECT COUNT(*) FROM technical_specifications WHERE document_id = doc_id)
        ),
        'embeddings_count', (SELECT COUNT(*) FROM embeddings WHERE document_id = doc_id)
    ) INTO result
    FROM documents d
    LEFT JOIN complexity_markers cm ON d.id = cm.document_id
    WHERE d.id = doc_id;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old batches (useful for maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_batches(days_old int DEFAULT 30)
RETURNS TABLE (deleted_batches int, deleted_documents int) AS $$
DECLARE
    batch_count int;
    doc_count int;
BEGIN
    -- Count documents to be deleted
    SELECT COUNT(*) INTO doc_count
    FROM documents d
    JOIN batches b ON d.batch_id = b.id
    WHERE b.created_at < NOW() - (days_old || ' days')::interval;

    -- Count batches to be deleted
    SELECT COUNT(*) INTO batch_count
    FROM batches
    WHERE created_at < NOW() - (days_old || ' days')::interval;

    -- Delete old batches (cascade will delete documents)
    DELETE FROM batches
    WHERE created_at < NOW() - (days_old || ' days')::interval;

    RETURN QUERY SELECT batch_count, doc_count;
END;
$$ LANGUAGE plpgsql;
