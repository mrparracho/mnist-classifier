-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    image_data BYTEA NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction >= 0 AND prediction <= 9),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    true_label INTEGER CHECK (true_label >= 0 AND true_label <= 9),
    feedback_submitted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create index on timestamp for efficient time-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);

-- Create index on prediction for efficient digit-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);

-- Create index on true_label for efficient feedback analysis
CREATE INDEX IF NOT EXISTS idx_predictions_true_label ON predictions(true_label);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 