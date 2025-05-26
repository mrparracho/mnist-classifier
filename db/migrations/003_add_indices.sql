-- Create composite index for efficient feedback analysis
CREATE INDEX IF NOT EXISTS idx_predictions_feedback_analysis 
ON predictions(prediction, true_label)
WHERE true_label IS NOT NULL;

-- Create index for efficient time-based statistics
CREATE INDEX IF NOT EXISTS idx_predictions_time_stats 
ON predictions(timestamp, prediction)
WHERE true_label IS NOT NULL;

-- Create index for efficient confidence analysis
CREATE INDEX IF NOT EXISTS idx_predictions_confidence 
ON predictions(prediction, confidence)
WHERE confidence > 0.5;

-- Create materialized view for daily statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats AS
SELECT 
    DATE_TRUNC('day', timestamp) as date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END)::float / COUNT(*), 4) as accuracy,
    AVG(confidence) as avg_confidence
FROM predictions
WHERE true_label IS NOT NULL
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY date DESC;

-- Create index on the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_stats_date 
ON daily_stats(date);

-- Create function to refresh daily statistics
CREATE OR REPLACE FUNCTION refresh_daily_stats()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stats;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Create trigger to refresh daily stats
CREATE TRIGGER refresh_daily_stats_trigger
    AFTER INSERT OR UPDATE ON predictions
    FOR EACH STATEMENT
    EXECUTE FUNCTION refresh_daily_stats(); 