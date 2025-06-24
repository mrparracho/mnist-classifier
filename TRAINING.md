# Training Guide

This guide explains how to train each model in the MNIST classifier system.

## Available Models

1. **CNN MNIST** (`cnn_mnist`) - Convolutional Neural Network
2. **Transformer1 MNIST** (`transformer1_mnist`) - Vision Transformer variant 1
3. **Transformer2 MNIST** (`transformer2_mnist`) - Vision Transformer variant 2

## Training Commands

### Using Makefile (Recommended)

```bash
# Train CNN model
make train-cnn

# Train Transformer1 model
make train-transformer1

# Train Transformer2 model
make train-transformer2

# Train all models sequentially
make train-all
```

### Using Python directly

```bash
# Activate the model virtual environment
source .venv/model/bin/activate

# Train CNN model
python models/cnn_mnist/train.py

# Train Transformer1 model
python models/transformer1_mnist/train.py

# Train Transformer2 model
python models/transformer2_mnist/train.py
```

## Training Parameters

Each training script accepts the following parameters:

```bash
python models/cnn_mnist/train.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.001 \
    --checkpoint-dir ./models/cnn_mnist/checkpoints \
    --data-dir ./models/cnn_mnist/data
```

### Parameters:

- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--checkpoint-dir`: Directory to save model checkpoints
- `--data-dir`: Directory to store/download MNIST dataset
- `--no-cuda`: Disable CUDA training (use CPU only)

## Model Checkpoints

After training, each model saves its checkpoints in its respective directory:

- `models/cnn_mnist/checkpoints/cnn_mnist.pt`
- `models/transformer1_mnist/checkpoints/transformer1_mnist.pt`
- `models/transformer2_mnist/checkpoints/transformer2_mnist.pt`

## Training Output

During training, you'll see:

1. **Progress bars** showing training and validation progress
2. **Loss values** for each epoch
3. **Accuracy metrics** on the test set
4. **Checkpoint saves** when a new best model is found

Example output:
```
2025-06-24 15:30:00 - INFO - Training CNN MNIST model for 10 epochs
2025-06-24 15:30:00 - INFO - Model parameters: 1,199,882
Epoch 1/10 [Train]: 100%|██████████| 938/938 [00:45<00:00, loss: 0.234]
Epoch 1/10 [Test]: 100%|██████████| 157/157 [00:05<00:00, accuracy: 96.45]
2025-06-24 15:30:50 - INFO - Epoch 1/10 - Test Accuracy: 96.45%
2025-06-24 15:30:50 - INFO - New best model saved with accuracy: 96.45%
```

## GPU Training

If you have a CUDA-capable GPU, training will automatically use it. To force CPU training:

```bash
python models/cnn_mnist/train.py --no-cuda
```

## Data Management

- MNIST dataset is automatically downloaded on first run
- Data is stored in `models/{model_name}/data/`
- Each model has its own data directory to avoid conflicts

## Tips for Better Training

1. **Start with fewer epochs** (5-10) to test the setup
2. **Use smaller batch sizes** if you run out of memory
3. **Adjust learning rate** if training is unstable
4. **Monitor validation accuracy** to avoid overfitting
5. **Use GPU** for faster training when available

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size
2. **Slow Training**: Check if GPU is being used
3. **Poor Accuracy**: Try different learning rates or more epochs
4. **Import Errors**: Make sure virtual environment is activated

### Check GPU Availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
``` 