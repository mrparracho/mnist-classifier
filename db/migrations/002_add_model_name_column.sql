-- Migration: Add model_name column to support multiple models
-- This migration adds model_name column to existing tables and creates a models table

-- Add model_name column to predictions table
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_name VARCHAR(100) DEFAULT 'cnn_mnist';

-- Add model_name column to model_metrics table
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS model_name VARCHAR(100) DEFAULT 'cnn_mnist';

-- Add model_name column to feedback_history table
ALTER TABLE feedback_history ADD COLUMN IF NOT EXISTS model_name VARCHAR(100) DEFAULT 'cnn_mnist';

-- Create models table to store model metadata
CREATE TABLE IF NOT EXISTS models (
    name VARCHAR(100) PRIMARY KEY,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    model_type VARCHAR(50) NOT NULL, -- 'cnn', 'transformer', etc.
    version VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    checkpoint_path VARCHAR(500),
    config JSONB
);

-- Create indices for efficient querying with model_name
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_feedback_history_model_name ON feedback_history(model_name);

-- Insert initial model data with updated checkpoint paths
INSERT INTO models (name, display_name, description, model_type, version, checkpoint_path) VALUES
    ('cnn_mnist', 'CNN MNIST', 'Convolutional Neural Network for MNIST digit classification', 'cnn', '1.0.0', 'models/cnn_mnist/checkpoints/cnn_mnist.pt'),
    ('transformer1_mnist', 'Transformer1 MNIST', 'Vision Transformer with encoder layers for MNIST classification', 'transformer', '1.0.0', 'models/transformer1_mnist/checkpoints/transformer1_mnist.pt'),
    ('transformer2_mnist', 'Transformer2 MNIST', 'Vision Transformer with encoder layers and MLP for MNIST classification', 'transformer', '1.0.0', 'models/transformer2_mnist/checkpoints/transformer2_mnist.pt')
ON CONFLICT (name) DO NOTHING;

-- Update existing predictions to use cnn_mnist as default model
UPDATE predictions SET model_name = 'cnn_mnist' WHERE model_name IS NULL;

-- Update existing model_metrics to use cnn_mnist as default model
UPDATE model_metrics SET model_name = 'cnn_mnist' WHERE model_name IS NULL;

-- Update existing feedback_history to use cnn_mnist as default model
UPDATE feedback_history SET model_name = 'cnn_mnist' WHERE model_name IS NULL;

-- Update model_performance view to include model_name
DROP VIEW IF EXISTS model_performance;

CREATE OR REPLACE VIEW model_performance AS
WITH digit_stats AS (
    SELECT
        model_name,
        prediction as digit,
        COUNT(*) as total,
        SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct
    FROM predictions
    WHERE true_label IS NOT NULL
    GROUP BY model_name, prediction
),
model_stats AS (
    SELECT
        model_name,
        SUM(total) as total_predictions,
        SUM(correct) as correct_predictions
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

-- Create function to get model statistics
CREATE OR REPLACE FUNCTION get_model_statistics(target_model_name VARCHAR(100) DEFAULT NULL)
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
        SELECT * FROM model_performance;
    ELSE
        RETURN QUERY
        SELECT * FROM model_performance WHERE model_performance.model_name = target_model_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to get available models
CREATE OR REPLACE FUNCTION get_available_models()
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
    FROM models m
    WHERE m.is_active = TRUE
    ORDER BY m.name;
END;
$$ LANGUAGE plpgsql; 