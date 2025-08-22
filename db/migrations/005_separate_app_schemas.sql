-- Migration: Separate apps into different schemas
-- This migration creates separate schemas for the original MNIST classifier and sequence prediction apps

-- Create separate schemas for each app
CREATE SCHEMA IF NOT EXISTS mnist_classifier;
CREATE SCHEMA IF NOT EXISTS mnist_sequence;

-- Move existing tables to mnist_classifier schema
-- (These are the tables used by the original MNIST classifier app)

-- Move predictions table
CREATE TABLE IF NOT EXISTS mnist_classifier.predictions (
    LIKE predictions INCLUDING ALL
);
INSERT INTO mnist_classifier.predictions SELECT * FROM predictions;
DROP TABLE IF EXISTS predictions;

-- Move models table (but only classifier models)
CREATE TABLE IF NOT EXISTS mnist_classifier.models (
    LIKE models INCLUDING ALL
);
INSERT INTO mnist_classifier.models 
SELECT * FROM models 
WHERE model_type IN ('cnn', 'transformer') 
   OR name IN ('cnn_mnist', 'transformer1_mnist', 'transformer2_mnist');
-- Note: encoder-decoder models will be moved to mnist_sequence schema

-- Move feedback_history table
CREATE TABLE IF NOT EXISTS mnist_classifier.feedback_history (
    LIKE feedback_history INCLUDING ALL
);
INSERT INTO mnist_classifier.feedback_history SELECT * FROM feedback_history;
DROP TABLE IF EXISTS feedback_history;

-- Move model_metrics table
CREATE TABLE IF NOT EXISTS mnist_classifier.model_metrics (
    LIKE model_metrics INCLUDING ALL
);
INSERT INTO mnist_classifier.model_metrics SELECT * FROM model_metrics;
DROP TABLE IF EXISTS model_metrics;

-- Move model_versions table
CREATE TABLE IF NOT EXISTS mnist_classifier.model_versions (
    LIKE model_versions INCLUDING ALL
);
INSERT INTO mnist_classifier.model_versions SELECT * FROM model_versions;
DROP TABLE IF EXISTS model_versions;

-- Create mnist_sequence schema tables
-- Move sequence_predictions to mnist_sequence schema
CREATE TABLE IF NOT EXISTS mnist_sequence.sequence_predictions (
    LIKE sequence_predictions INCLUDING ALL
);
INSERT INTO mnist_sequence.sequence_predictions SELECT * FROM sequence_predictions;
DROP TABLE IF EXISTS sequence_predictions;

-- Create models table for sequence prediction app
CREATE TABLE IF NOT EXISTS mnist_sequence.models (
    name VARCHAR(100) PRIMARY KEY,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    model_type VARCHAR(50) NOT NULL, -- 'encoder_decoder'
    version VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    checkpoint_path VARCHAR(500),
    config JSONB
);

-- Move encoder-decoder models to mnist_sequence schema
INSERT INTO mnist_sequence.models 
SELECT * FROM models 
WHERE model_type = 'encoder_decoder' 
   OR name LIKE 'encoder_decoder%'
ON CONFLICT (name) DO NOTHING;

-- Create model_versions table for sequence app
CREATE TABLE IF NOT EXISTS mnist_sequence.model_versions (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create model_metrics table for sequence app
CREATE TABLE IF NOT EXISTS mnist_sequence.model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) DEFAULT 'encoder_decoder',
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback_history table for sequence app
CREATE TABLE IF NOT EXISTS mnist_sequence.feedback_history (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES mnist_sequence.sequence_predictions(id),
    true_sequence INTEGER[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for mnist_classifier schema
CREATE INDEX IF NOT EXISTS idx_classifier_predictions_model_name ON mnist_classifier.predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_classifier_predictions_timestamp ON mnist_classifier.predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_classifier_feedback_prediction_id ON mnist_classifier.feedback_history(prediction_id);
CREATE INDEX IF NOT EXISTS idx_classifier_model_metrics_model_name ON mnist_classifier.model_metrics(model_name);

-- Create indices for mnist_sequence schema
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_timestamp ON mnist_sequence.sequence_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_model ON mnist_sequence.sequence_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_session ON mnist_sequence.sequence_predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_sequence_feedback_prediction_id ON mnist_sequence.feedback_history(prediction_id);
CREATE INDEX IF NOT EXISTS idx_sequence_model_metrics_model_name ON mnist_sequence.model_metrics(model_name);

-- Create model performance view for mnist_classifier schema
CREATE OR REPLACE VIEW mnist_classifier.model_performance AS
WITH digit_stats AS (
    SELECT
        model_name,
        prediction as digit,
        COUNT(*)::BIGINT as total,
        SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END)::BIGINT as correct
    FROM mnist_classifier.predictions
    WHERE true_label IS NOT NULL
    GROUP BY model_name, prediction
),
model_stats AS (
    SELECT
        model_name,
        SUM(total)::BIGINT as total_predictions,
        SUM(correct)::BIGINT as correct_predictions
    FROM digit_stats
    GROUP BY model_name
)
SELECT
    ms.model_name,
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
JOIN digit_stats ds ON ms.model_name = ds.model_name
GROUP BY ms.model_name, ms.total_predictions, ms.correct_predictions;

-- Create model performance view for mnist_sequence schema
CREATE OR REPLACE VIEW mnist_sequence.model_performance AS
WITH sequence_stats AS (
    SELECT
        model_name,
        grid_size,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions
    FROM mnist_sequence.sequence_predictions
    WHERE true_sequence IS NOT NULL
    GROUP BY model_name, grid_size
)
SELECT
    model_name,
    grid_size,
    total_predictions,
    correct_predictions,
    ROUND((correct_predictions::numeric / NULLIF(total_predictions, 0))::numeric, 4) as accuracy
FROM sequence_stats
ORDER BY model_name, grid_size;

-- Create function to get model statistics for mnist_classifier
CREATE OR REPLACE FUNCTION mnist_classifier.get_model_statistics(target_model_name VARCHAR(100) DEFAULT NULL)
RETURNS TABLE (
    model_name VARCHAR(100),
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy NUMERIC,
    per_digit_stats JSONB
) AS $$
BEGIN
    IF target_model_name IS NULL THEN
        RETURN QUERY
        SELECT * FROM mnist_classifier.model_performance;
    ELSE
        RETURN QUERY
        SELECT * FROM mnist_classifier.model_performance WHERE model_performance.model_name = target_model_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to get available models for mnist_classifier
CREATE OR REPLACE FUNCTION mnist_classifier.get_available_models()
RETURNS TABLE (
    name VARCHAR(100),
    display_name VARCHAR(200),
    description TEXT,
    model_type VARCHAR(50),
    version VARCHAR(50),
    is_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.name,
        m.display_name,
        m.description,
        m.model_type,
        m.version,
        m.is_active
    FROM mnist_classifier.models m
    WHERE m.is_active = TRUE
    ORDER BY m.name;
END;
$$ LANGUAGE plpgsql;

-- Create function to get model statistics for mnist_sequence
CREATE OR REPLACE FUNCTION mnist_sequence.get_model_statistics(target_model_name VARCHAR(100) DEFAULT NULL)
RETURNS TABLE (
    model_name VARCHAR(100),
    grid_size INTEGER,
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy NUMERIC
) AS $$
BEGIN
    IF target_model_name IS NULL THEN
        RETURN QUERY
        SELECT * FROM mnist_sequence.model_performance;
    ELSE
        RETURN QUERY
        SELECT * FROM mnist_sequence.model_performance WHERE model_performance.model_name = target_model_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to get available models for mnist_sequence
CREATE OR REPLACE FUNCTION mnist_sequence.get_available_models()
RETURNS TABLE (
    name VARCHAR(100),
    display_name VARCHAR(200),
    description TEXT,
    model_type VARCHAR(50),
    version VARCHAR(50),
    is_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.name,
        m.display_name,
        m.description,
        m.model_type,
        m.version,
        m.is_active
    FROM mnist_sequence.models m
    WHERE m.is_active = TRUE
    ORDER BY m.name;
END;
$$ LANGUAGE plpgsql;

-- Drop the old shared functions and views
DROP FUNCTION IF EXISTS get_model_statistics(VARCHAR);
DROP FUNCTION IF EXISTS get_available_models();
DROP VIEW IF EXISTS model_performance;

-- Clean up old models table (encoder-decoder models moved to mnist_sequence)
DELETE FROM models WHERE model_type = 'encoder_decoder' OR name LIKE 'encoder_decoder%';
DROP TABLE IF EXISTS models; 