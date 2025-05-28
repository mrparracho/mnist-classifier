-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create base tables
CREATE TABLE IF NOT EXISTS model_versions (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_data BYTEA NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction >= 0 AND prediction <= 9),
    true_label INTEGER CHECK (true_label >= 0 AND true_label <= 9),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50) DEFAULT '1.0.0',
    is_correct BOOLEAN,
    feedback_submitted BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS feedback_history (
    id SERIAL PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(id),
    true_label INTEGER NOT NULL CHECK (true_label >= 0 AND true_label <= 9),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_version ON model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_history_prediction_id ON feedback_history(prediction_id);

-- Insert initial model version
INSERT INTO model_versions (version, description)
VALUES ('1.0.0', 'Initial MNIST model version')
ON CONFLICT (version) DO NOTHING;

-- Create maintenance functions
CREATE OR REPLACE FUNCTION cleanup_old_predictions(
    days_to_keep INTEGER,
    model_version VARCHAR(50) DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    query TEXT;
BEGIN
    query := 'DELETE FROM predictions WHERE timestamp < CURRENT_TIMESTAMP - ($1 * INTERVAL ''1 day'')';
    
    IF model_version IS NOT NULL THEN
        query := query || ' AND model_version = $2';
        EXECUTE query INTO deleted_count USING days_to_keep, model_version;
    ELSE
        EXECUTE query INTO deleted_count USING days_to_keep;
    END IF;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION archive_old_predictions(
    days_to_archive INTEGER,
    model_version VARCHAR(50) DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
    query TEXT;
BEGIN
    -- Create archive table if it doesn't exist
    CREATE TABLE IF NOT EXISTS predictions_archive (
        LIKE predictions INCLUDING ALL
    ) INHERITS (predictions);
    
    -- Move old predictions to archive
    query := 'WITH moved_rows AS (
        DELETE FROM ONLY predictions 
        WHERE timestamp < CURRENT_TIMESTAMP - ($1 * INTERVAL ''1 day'')';
    
    IF model_version IS NOT NULL THEN
        query := query || ' AND model_version = $2';
        query := query || ' RETURNING *) INSERT INTO predictions_archive SELECT * FROM moved_rows';
        EXECUTE query INTO archived_count USING days_to_archive, model_version;
    ELSE
        query := query || ' RETURNING *) INSERT INTO predictions_archive SELECT * FROM moved_rows';
        EXECUTE query INTO archived_count USING days_to_archive;
    END IF;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_database_size_info()
RETURNS TABLE (
    table_name TEXT,
    total_size TEXT,
    table_size TEXT,
    index_size TEXT,
    row_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.table_name::TEXT,
        pg_size_pretty(pg_total_relation_size(t.table_name))::TEXT as total_size,
        pg_size_pretty(pg_table_size(t.table_name))::TEXT as table_size,
        pg_size_pretty(pg_indexes_size(t.table_name))::TEXT as index_size,
        (SELECT count(*) FROM predictions)::BIGINT as row_count
    FROM (
        SELECT table_name::regclass as table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    ) t
    ORDER BY pg_total_relation_size(t.table_name) DESC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION analyze_tables()
RETURNS VOID AS $$
BEGIN
    ANALYZE predictions;
    ANALYZE feedback_history;
    ANALYZE model_versions;
END;
$$ LANGUAGE plpgsql;

-- Create model performance view
CREATE OR REPLACE VIEW model_performance AS
WITH digit_stats AS (
    SELECT
        model_version,
        prediction as digit,
        COUNT(*) as total,
        SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct
    FROM predictions
    WHERE true_label IS NOT NULL
    GROUP BY model_version, prediction
),
model_stats AS (
    SELECT
        model_version,
        SUM(total) as total_predictions,
        SUM(correct) as correct_predictions
    FROM digit_stats
    GROUP BY model_version
)
SELECT
    ms.model_version,
    ms.total_predictions,
    ms.correct_predictions,
    ROUND((ms.correct_predictions::numeric / NULLIF(ms.total_predictions, 0))::numeric, 4) as accuracy,
    jsonb_object_agg(
        ds.digit,
        jsonb_build_object(
            'total', ds.total,
            'correct', ds.correct,
            'accuracy', ROUND((ds.correct::numeric / NULLIF(ds.total, 0))::numeric, 4)
        )
    ) as per_digit_stats
FROM model_stats ms
JOIN digit_stats ds ON ms.model_version = ds.model_version
GROUP BY ms.model_version, ms.total_predictions, ms.correct_predictions; 