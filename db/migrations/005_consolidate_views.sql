-- Drop existing views and functions
DROP VIEW IF EXISTS prediction_accuracy;
DROP VIEW IF EXISTS daily_predictions;
DROP VIEW IF EXISTS model_performance;
DROP FUNCTION IF EXISTS calculate_per_digit_stats();

-- Create a single, comprehensive view for model performance
CREATE OR REPLACE VIEW model_performance AS
WITH prediction_stats AS (
    SELECT
        prediction as digit,
        COUNT(*) as total,
        COUNT(CASE WHEN is_correct THEN 1 END) as correct,
        ROUND(
            COUNT(CASE WHEN is_correct THEN 1 END)::float / 
            NULLIF(COUNT(*), 0),
            4
        ) as accuracy
    FROM predictions
    WHERE true_label IS NOT NULL
    GROUP BY prediction
)
SELECT
    (SELECT COUNT(*) FROM predictions) as total_predictions,
    (SELECT COUNT(*) FROM predictions WHERE feedback_submitted) as total_feedback,
    ROUND(
        (SELECT COUNT(*) FROM predictions WHERE is_correct)::float / 
        NULLIF((SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL), 0),
        4
    ) as overall_accuracy,
    jsonb_object_agg(
        digit::text,
        jsonb_build_object(
            'total', total,
            'correct', correct,
            'accuracy', accuracy
        )
    ) as per_digit_stats,
    (
        SELECT jsonb_object_agg(
            prediction_date::text,
            jsonb_build_object(
                'total', total_predictions,
                'with_feedback', predictions_with_feedback,
                'accuracy', daily_accuracy
            )
        )
        FROM (
            SELECT
                DATE(timestamp) as prediction_date,
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN feedback_submitted THEN 1 END) as predictions_with_feedback,
                ROUND(
                    COUNT(CASE WHEN is_correct THEN 1 END)::float / 
                    NULLIF(COUNT(CASE WHEN feedback_submitted THEN 1 END), 0),
                    4
                ) as daily_accuracy
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY prediction_date DESC
            LIMIT 30  -- Keep last 30 days
        ) as daily_stats
    ) as daily_stats
FROM prediction_stats;

-- Create function to clean up old predictions
CREATE OR REPLACE FUNCTION cleanup_old_predictions(days_to_keep INTEGER)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM predictions
    WHERE timestamp < CURRENT_TIMESTAMP - (days_to_keep * INTERVAL '1 day')
    RETURNING COUNT(*) INTO deleted_count;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for predictions table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_predictions_updated_at'
    ) THEN
        CREATE TRIGGER update_predictions_updated_at
        BEFORE UPDATE ON predictions
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$; 