-- Add model version tracking to predictions table
ALTER TABLE predictions
ADD COLUMN model_version VARCHAR(50) DEFAULT '1.0.0';

-- Add model version tracking to model_metrics table
ALTER TABLE model_metrics
ADD COLUMN model_version VARCHAR(50) DEFAULT '1.0.0';

-- Create model_versions table
CREATE TABLE IF NOT EXISTS model_versions (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Insert initial model version
INSERT INTO model_versions (version, description)
VALUES ('1.0.0', 'Initial MNIST model version');

-- Create index on model_version
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_version ON model_metrics(model_version);

-- Update model_performance view to include model version
CREATE OR REPLACE VIEW model_performance AS
SELECT
    model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0), 4) as accuracy,
    jsonb_object_agg(
        digit,
        jsonb_build_object(
            'total', total,
            'correct', correct,
            'accuracy', ROUND(correct::float / NULLIF(total, 0), 4)
        )
    ) as per_digit_stats
FROM (
    SELECT
        model_version,
        prediction as digit,
        COUNT(*) as total,
        SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct
    FROM predictions
    WHERE true_label IS NOT NULL
    GROUP BY model_version, prediction
) as digit_stats
GROUP BY model_version; 