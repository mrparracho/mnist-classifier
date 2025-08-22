-- Migration: Add encoder-decoder model to models table
-- This migration adds the encoder-decoder model for sequence predictions

-- Insert encoder-decoder model into models table
INSERT INTO models (name, display_name, description, model_type, version, is_active, checkpoint_path, config) VALUES
    ('encoder_decoder', 'Encoder-Decoder', 'Encoder-Decoder model for sequence MNIST digit prediction', 'encoder_decoder', '1.0.0', TRUE, 'encoder_decoder/checkpoints/encoder_decoder.pt', 
     '{"image_size": 28, "patch_size": 7, "encoder_embed_dim": 64, "decoder_embed_dim": 64, "num_layers": 4, "num_heads": 8, "dropout": 0.1, "max_seq_len": 20, "normalize_mean": 0.1307, "normalize_std": 0.3081, "grid_checkpoints": {"1": "encoder_decoder/checkpoints/mnist-encoder-decoder-1-varlen.pt", "2": "encoder_decoder/checkpoints/mnist-encoder-decoder-2-varlen.pt", "3": "encoder_decoder/checkpoints/mnist-encoder-decoder-3-varlen.pt", "4": "encoder_decoder/checkpoints/mnist-encoder-decoder-4-varlen.pt"}}')
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    model_type = EXCLUDED.model_type,
    version = EXCLUDED.version,
    is_active = EXCLUDED.is_active,
    checkpoint_path = EXCLUDED.checkpoint_path,
    config = EXCLUDED.config; 