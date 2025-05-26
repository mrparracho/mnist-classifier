-- Add is_correct column to predictions table
ALTER TABLE predictions
ADD COLUMN is_correct BOOLEAN;

-- Update existing records
UPDATE predictions 
SET is_correct = (prediction = true_label)
WHERE true_label IS NOT NULL;

-- Create an index for faster accuracy queries
CREATE INDEX idx_predictions_is_correct 
ON predictions(is_correct)
WHERE is_correct IS NOT NULL; 