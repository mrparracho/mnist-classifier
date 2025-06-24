-- Add session_id column to link predictions from the same multi-model request
ALTER TABLE predictions ADD COLUMN session_id VARCHAR(36);

-- Add index for faster session-based queries
CREATE INDEX idx_predictions_session_id ON predictions(session_id);

-- Add index for timestamp queries (used in history)
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp DESC); 