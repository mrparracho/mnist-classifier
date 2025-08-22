-- Add sequence predictions table
CREATE TABLE IF NOT EXISTS sequence_predictions (
    id SERIAL PRIMARY KEY,
    image_data BYTEA NOT NULL,
    predicted_sequence INTEGER[] NOT NULL,
    true_sequence INTEGER[],
    confidence FLOAT NOT NULL,
    grid_size INTEGER NOT NULL,
    model_name VARCHAR(100) DEFAULT 'encoder_decoder',
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_correct BOOLEAN
);

-- Create index on timestamp for faster queries
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_timestamp ON sequence_predictions(timestamp);

-- Create index on model_name for filtering
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_model ON sequence_predictions(model_name);

-- Create index on session_id for session-based queries
CREATE INDEX IF NOT EXISTS idx_sequence_predictions_session ON sequence_predictions(session_id); 