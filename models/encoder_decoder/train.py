#!/usr/bin/env python3
"""
Training script for Encoder-Decoder MNIST model.
"""

import os
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Suppress urllib3 warnings (used by huggingface-hub)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Fix multiprocessing issues on macOS
if os.name == 'posix':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image

# Add the parent directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.encoder_decoder.model import EncoderDecoder

# Get the logger - initialize with basic config, will be enhanced in main()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Hugging Face integration
try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face libraries not available. Install with: pip install huggingface-hub")

def setup_logging(base_dir):
    """
    Set up comprehensive logging to both console and timestamped file.
    
    Args:
        base_dir (str): Base directory for the models (contains encoder_decoder folder)
        
    Returns:
        str: log_file_path
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"encoder_decoder_training_{timestamp}.log"
    log_file_path = os.path.join(logs_dir, log_filename)
    
    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info("ENCODER-DECODER MNIST TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return log_file_path

class MNISTSequenceDataset(Dataset):
    """Dataset that creates sequences from MNIST digits with variable content in fixed grid"""
    def __init__(self, mnist_dataset, max_grid_size=10, max_output_length=None, length_weights=None, static_length=None):
        """
        Initialize the MNIST Sequence Dataset.
        
        Args:
            mnist_dataset: The base MNIST dataset
            max_grid_size: Maximum grid size (e.g., 10 for 10x10 grid)
            max_output_length: Maximum output sequence length (None = grid capacity)
            length_weights: Weights for different sequence lengths
            static_length: If specified, force all sequences to have exactly this many digits
        """
        self.mnist_dataset = mnist_dataset
        self.max_grid_size = max_grid_size
        self.full_grid_size = max_grid_size * 28  # 28 is MNIST image size
        
        # FIXED: Set max_output_length to grid capacity if not specified
        if max_output_length is None:
            self.max_output_length = max_grid_size * max_grid_size  # e.g., 10*10 = 100
        else:
            self.max_output_length = max_output_length
        
        # Length distribution weights - if None, use uniform distribution
        if length_weights is None:
            # Equal probability for each length from 1 to max_output_length
            num_lengths = self.max_output_length
            self.length_weights = [1.0] * num_lengths
        else:
            self.length_weights = length_weights
            
        # Normalize weights to probabilities
        total_weight = sum(self.length_weights)
        self.length_probs = [float(w) / float(total_weight) for w in self.length_weights]
        
        # Calculate the size of the full grid image
        self.single_image_size = 28  # MNIST image size
        self.full_grid_size = self.single_image_size * self.max_grid_size  # e.g., 28*10 = 280
        
        # Define token indices (matching DecoderEmbedding)
        self.start_token = 10  # <start>
        self.finish_token = 11  # <finish>
        self.pad_token = 12     # <pad>
        
        # STATIC LENGTH: Override all length settings if static_length is specified
        if static_length is not None:
            # Validate static_length against grid constraints
            max_possible_digits = max_grid_size * max_grid_size
            if static_length < 1:
                raise ValueError(f"static_length must be >= 1, got {static_length}")
            if static_length > max_possible_digits:
                raise ValueError(f"static_length ({static_length}) cannot exceed grid capacity ({max_possible_digits}) for {max_grid_size}x{max_grid_size} grid")
            
            self.static_length = static_length
            # Override length weights to only allow static_length
            self.length_weights = [1.0]  # Single weight for static length
            logger.info(f"STATIC LENGTH MODE: {static_length} digits per sequence")
        else:
            self.static_length = None
        
    def __len__(self):
        # Allow image reuse across samples - this dramatically increases dataset size
        # For a 3x3 grid, we can create many more samples by reusing images
        # Use a large multiplier to create sufficient training samples
        return len(self.mnist_dataset) * 10  # 10x more samples by reusing images
    
    def __getitem__(self, idx):
        # STATIC LENGTH: Use fixed length if specified, otherwise sample randomly
        if self.static_length is not None:
            num_digits = self.static_length
        else:
            # FIXED: Sample how many actual digits to include with uniform distribution
            sampled_index = int(torch.multinomial(torch.tensor(self.length_probs, dtype=torch.float32), 1).item())
            # sampled_index is 0-based, maps to lengths 1 to max_output_length
            num_digits = sampled_index + 1  # This gives us lengths 1 to max_output_length
        
        # Get actual MNIST digits and their labels - allow reuse across samples
        digit_images = []
        digit_labels = []
        
        # Use modulo to allow image reuse - this creates much more training data
        for i in range(num_digits):
            data_idx = (idx + i * 1000) % len(self.mnist_dataset)  # Spread out image selection
            image, label = self.mnist_dataset[data_idx]
            digit_images.append(image)
            digit_labels.append(label)
        
        # Create full grid image filled with background
        full_grid_image = torch.zeros(1, self.full_grid_size, self.full_grid_size)
        
        # Randomly place the actual digits in the grid
        total_positions = self.max_grid_size * self.max_grid_size
        available_positions = list(range(total_positions))
        
        # Shuffle and select positions for our digits
        import random
        random.shuffle(available_positions)
        digit_positions = available_positions[:num_digits]
        
        # Place actual digits and fill remaining positions
        placed_labels = []  # Track labels in the order they appear spatially
        
        for pos in range(total_positions):
            # Calculate grid coordinates
            grid_row = pos // self.max_grid_size
            grid_col = pos % self.max_grid_size
            
            # Calculate pixel coordinates in the full image
            start_h = grid_row * self.single_image_size
            end_h = start_h + self.single_image_size
            start_w = grid_col * self.single_image_size
            end_w = start_w + self.single_image_size
            
            if pos in digit_positions:
                # Place an actual digit
                digit_idx = digit_positions.index(pos)
                image = digit_images[digit_idx]
                label = digit_labels[digit_idx]
                full_grid_image[:, start_h:end_h, start_w:end_w] = image
                placed_labels.append((pos, label))  # Store position and label
            else:
                # Fill with black/empty background for all unused positions
                # This ensures consistency between static and variable length modes
                # and makes it clear which positions should be empty
                filler_image = torch.zeros(1, self.single_image_size, self.single_image_size)
                full_grid_image[:, start_h:end_h, start_w:end_w] = filler_image
        
        # Sort labels by position to create consistent reading order (left-to-right, top-to-bottom)
        placed_labels.sort(key=lambda x: x[0])  # Sort by position
        sequence_labels = [label for pos, label in placed_labels]
        
        # Create target sequence: [start_token, digit1, digit2, ..., finish_token]
        target_sequence = [self.start_token]  # start token
        target_sequence.extend(sequence_labels)  # add the digit labels in order
        target_sequence.append(self.finish_token)  # finish token
        
        # NO PADDING: Return sequence at natural length
        # This eliminates wasted computation and cleaner learning signal
        target_sequence = torch.tensor(target_sequence, dtype=torch.long)
        
        return full_grid_image, target_sequence

def get_device():
    """
    Get the best available device for training.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        torch.device: The best available device
    """
    # Check for MPS (Apple Silicon) - available in PyTorch 1.12+
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon GPU) for training")
                return device
        except (AttributeError, RuntimeError):
            pass
    
    # Fall back to CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU for training")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")
    
    return device

def get_data_loaders(batch_size=64, data_dir=None, max_grid_size=10, 
                     max_output_length=None, static_length=None):
    """
    Create data loaders for training and testing MNIST dataset.
    
    Args:
        batch_size (int): Batch size for training and testing
        data_dir (str): Directory to store the dataset
        max_grid_size (int): Maximum grid size (e.g., 10 for 10x10 grid)
        max_output_length (int): Maximum output sequence length (None = grid capacity)
        static_length (int): If specified, force all sequences to have exactly this many digits
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Set max_output_length to grid capacity if not specified
    if max_output_length is None:
        max_output_length = max_grid_size * max_grid_size  # e.g., 10*10 = 100
        
    logger.info(f"Data loader: {max_grid_size}x{max_grid_size} grid, {max_output_length} digits max")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load the test data
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Use uniform distribution over all possible sequence lengths
    # For a 10x10 grid, this means uniform probability for lengths 1-100
    max_possible_digits = max_grid_size * max_grid_size  # e.g., 100 for 10x10 grid
    length_weights = [1.0] * max_possible_digits  # Uniform distribution
    
    # Log dataset configuration once
    logger.info(f"Dataset: {max_grid_size}x{max_grid_size} grid, {max_output_length} digits max, uniform distribution")
    
    # Wrap with sequence dataset
    train_sequence_dataset = MNISTSequenceDataset(
        train_dataset, 
        max_grid_size=max_grid_size, 
        max_output_length=max_output_length,
        length_weights=length_weights,
        static_length=static_length
    )
    test_sequence_dataset = MNISTSequenceDataset(
        test_dataset, 
        max_grid_size=max_grid_size, 
        max_output_length=max_output_length,
        length_weights=length_weights,
        static_length=static_length
    )
    
    # Log dataset sizes
    logger.info(f"Dataset sizes: {len(train_sequence_dataset)} train samples, {len(test_sequence_dataset)} test samples")
    logger.info(f"  (with image reuse: {len(train_sequence_dataset) // len(train_dataset)}x more samples)")
    
    # Create data loaders with length-based batching for optimal efficiency
    train_batch_sampler = LengthBasedBatchSampler(
        train_sequence_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        length_tolerance=3  # Allow max 3-token difference within batch
    )
    
    train_loader = DataLoader(
        train_sequence_dataset, 
        batch_sampler=train_batch_sampler,
        num_workers=4,
        collate_fn=collate_variable_length  # Minimal padding within length groups
    )
    
    # For evaluation, use regular batching since we want representative sampling
    test_loader = DataLoader(
        test_sequence_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_variable_length
    )
    
    logger.info(f"Batching: {batch_size} batch size, {len(train_batch_sampler)} train batches, {len(test_loader)} test batches")
    
    # Log static length configuration if active
    if static_length is not None:
        max_possible_digits = max_grid_size * max_grid_size
        logger.info(f"STATIC LENGTH MODE: {static_length} digits per sequence (max {max_possible_digits})")
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device, epoch=None, max_length=50, verbose=True, max_debug_images=3):
    """
    Comprehensive evaluation of the Encoder-Decoder MNIST model.
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to evaluate on
        epoch (int, optional): Current epoch number for logging
        max_length (int): Maximum sequence length for generation
        verbose (bool): Whether to print detailed results
        max_debug_images (int): Maximum number of debug images to save (default: 3)
    """
    model.eval()
    
    # Initialize counters for detailed analysis
    total_samples = 0
    perfect_matches = 0
    length_errors = 0
    content_errors_correct_length = 0
    order_errors_correct_content = 0
    partial_correct_positions = 0
    total_possible_positions = 0
    
    # Token structure analysis
    start_token_correct = 0
    no_start_token = 0
    finish_token_correct = 0
    no_finish_token = 0
    
    # Length distribution tracking
    length_distribution = {}
    
    # Digit confusion matrix
    digit_confusion_matrix = {}
    
    # Debug image counter
    debug_images_saved = 0
    
    def extract_digits_from_sequence(sequence):
        """Extract digits from a sequence, handling start/finish/pad tokens."""
        digits = []
        start_found = False
        for token in sequence:
            token_val = token.item() if hasattr(token, 'item') else token
            if token_val == 10:  # start token
                start_found = True
            elif token_val == 11:  # finish token
                break
            elif start_found and token_val != 12:  # not pad token
                if 0 <= token_val <= 9:  # valid digit
                    digits.append(token_val)
        return digits
    
    def has_proper_tokens(sequence):
        """Check if sequence has proper start and finish tokens."""
        seq_list = sequence.tolist() if hasattr(sequence, 'tolist') else list(sequence)
        has_start = len(seq_list) > 0 and seq_list[0] == 10
        has_finish = 11 in seq_list
        return has_start, has_finish
    
    with torch.no_grad():
        epoch_desc = f"Epoch {epoch} [Test]" if epoch is not None else "Evaluation"
        progress_bar = tqdm(test_loader, desc=epoch_desc)
        
        for batch_idx, (data, target_sequence, target_lengths) in enumerate(progress_bar):
            # Move data to device
            data, target_sequence, target_lengths = data.to(device), target_sequence.to(device), target_lengths.to(device)
            
            # Forward pass - inference mode (no target_sequence)
            predicted_sequence = model(data, max_length=max_length)
            
            # Evaluate each sample in the batch
            for i in range(data.size(0)):
                target_seq = target_sequence[i][:target_lengths[i]]  # Use actual length
                pred_seq = predicted_sequence[i]
                total_samples += 1
                
                # Debug: Print shapes and first batch on first iteration
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"Input sequence shape: {data.shape}")
                    logger.info(f"Target sequence shape: {target_seq.shape}")
                    logger.info(f"Output sequence shape: {pred_seq.shape}")
                    logger.info(f"First input sequence: {data[0]}")
                    logger.info(f"First target sequence: {target_seq[0]}")
                    logger.info(f"First output sequence: {pred_seq[0]}")
                
                # Extract actual digits
                target_digits = extract_digits_from_sequence(target_seq)
                pred_digits = extract_digits_from_sequence(pred_seq)
                
                if verbose and i < 3 and batch_idx == 0:
                    logger.info(f"Example {i}: Target {target_digits}, Predicted {pred_digits}")
                    
                    # Save debug image for visual inspection (limit number of images)
                    input_image = data[i]  # [1, H, W] tensor
                    save_mnist_image_for_debug(
                        input_image, 
                        "mnist_debug", 
                        target_digits, 
                        pred_digits, 
                        epoch if epoch is not None else 0, 
                        batch_idx, 
                        i
                    )
                    debug_images_saved += 1
                    logger.info(f"  Debug image {debug_images_saved}/{max_debug_images} saved")
                
                # Check token structure
                has_start_target, has_finish_target = has_proper_tokens(target_seq)
                has_start_pred, has_finish_pred = has_proper_tokens(pred_seq)
                
                # Update token structure metrics
                if has_start_pred:
                    start_token_correct += 1
                else:
                    no_start_token += 1
                    
                if has_finish_pred:
                    finish_token_correct += 1
                else:
                    no_finish_token += 1
                
                # Track length distribution
                target_len = len(target_digits)
                pred_len = len(pred_digits)
                total_possible_positions += target_len
                
                if target_len not in length_distribution:
                    length_distribution[target_len] = {
                        'count': 0, 
                        'correct_length_predictions': 0,
                        'perfect_matches': 0
                    }
                length_distribution[target_len]['count'] += 1
                
                if target_len == pred_len:
                    length_distribution[target_len]['correct_length_predictions'] += 1
                
                # Detailed analysis
                if len(target_digits) != len(pred_digits):
                    # Length mismatch
                    length_errors += 1
                else:
                    # Same length - check content
                    if target_digits == pred_digits:
                        # Perfect match
                        perfect_matches += 1
                        length_distribution[target_len]['perfect_matches'] += 1
                    else:
                        # Same length but different content
                        content_errors_correct_length += 1
                        
                        # Check if it's just an order issue (same digits, wrong order)
                        if sorted(target_digits) == sorted(pred_digits):
                            order_errors_correct_content += 1
                
                # Calculate partial correctness (position-wise accuracy)
                min_len = min(len(target_digits), len(pred_digits))
                for j in range(min_len):
                    if target_digits[j] == pred_digits[j]:
                        partial_correct_positions += 1
                        
                    # Track digit confusion matrix
                    if j < len(target_digits) and j < len(pred_digits):
                        target_digit = target_digits[j]
                        pred_digit = pred_digits[j]
                        if target_digit not in digit_confusion_matrix:
                            digit_confusion_matrix[target_digit] = {}
                        if pred_digit not in digit_confusion_matrix[target_digit]:
                            digit_confusion_matrix[target_digit][pred_digit] = 0
                        digit_confusion_matrix[target_digit][pred_digit] += 1
            
            # Update progress bar with current metrics
            if total_samples > 0:
                perfect_acc = 100 * perfect_matches / total_samples
                length_acc = 100 * (total_samples - length_errors) / total_samples
                progress_bar.set_postfix({
                    "perfect": f"{perfect_acc:.1f}%", 
                    "length": f"{length_acc:.1f}%"
                })
    
    # Calculate comprehensive metrics
    if total_samples > 0:
        perfect_accuracy = 100 * perfect_matches / total_samples
        length_accuracy = 100 * (total_samples - length_errors) / total_samples
        start_token_accuracy = 100 * start_token_correct / total_samples
        finish_token_accuracy = 100 * finish_token_correct / total_samples
        
        # Content accuracy given correct length
        correct_length_samples = total_samples - length_errors
        content_accuracy_given_correct_length = (
            100 * perfect_matches / max(1, correct_length_samples)
        )
        
        # Position-wise accuracy
        positional_accuracy = (
            100 * partial_correct_positions / max(1, total_possible_positions)
        )
        
        # Error breakdown percentages
        length_error_rate = 100 * length_errors / total_samples
        content_error_rate = 100 * content_errors_correct_length / total_samples
        order_error_rate = 100 * order_errors_correct_content / total_samples
        
        # Compile results dictionary
        results = {
            'total_samples': total_samples,
            'perfect_accuracy': perfect_accuracy,
            'length_accuracy': length_accuracy,
            'positional_accuracy': positional_accuracy,
            'content_accuracy_given_correct_length': content_accuracy_given_correct_length,
            'start_token_accuracy': start_token_accuracy,
            'finish_token_accuracy': finish_token_accuracy,
            'length_error_rate': length_error_rate,
            'content_error_rate': content_error_rate,
            'order_error_rate': order_error_rate,
            'length_errors': length_errors,
            'content_errors_correct_length': content_errors_correct_length,
            'order_errors_correct_content': order_errors_correct_content,
            'partial_correct_positions': partial_correct_positions,
            'total_possible_positions': total_possible_positions,
            'length_distribution': length_distribution,
            'digit_confusion_matrix': digit_confusion_matrix
        }
        
        if verbose:
            # Log comprehensive results
            epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
            logger.info(f"{epoch_str}Comprehensive Evaluation Results:")
            logger.info("="*60)
            
            # Main metrics
            logger.info("MAIN METRICS:")
            logger.info(f"  Perfect Sequence Accuracy: {perfect_accuracy:.2f}%")
            logger.info(f"  Length Accuracy: {length_accuracy:.2f}%")
            logger.info(f"  Positional Accuracy: {positional_accuracy:.2f}%")
            logger.info(f"  Content Accuracy (given correct length): {content_accuracy_given_correct_length:.2f}%")
            
            # Token structure
            logger.info("TOKEN STRUCTURE:")
            logger.info(f"  Start Token Accuracy: {start_token_accuracy:.2f}%")
            logger.info(f"  Finish Token Accuracy: {finish_token_accuracy:.2f}%")
            
            # Error breakdown
            logger.info("ERROR BREAKDOWN:")
            logger.info(f"  Length Errors: {length_errors}/{total_samples} ({length_error_rate:.2f}%)")
            logger.info(f"  Content Errors (correct length): {content_errors_correct_length}/{total_samples} ({content_error_rate:.2f}%)")
            logger.info(f"  Order Errors (correct content): {order_errors_correct_content}/{total_samples} ({order_error_rate:.2f}%)")
            
            # Length distribution
            logger.info("LENGTH DISTRIBUTION:")
            for length in sorted(length_distribution.keys()):
                stats = length_distribution[length]
                length_acc = 100 * stats['correct_length_predictions'] / stats['count']
                perfect_acc_for_length = 100 * stats['perfect_matches'] / stats['count']
                logger.info(f"  Length {length}: {stats['count']} samples, {length_acc:.1f}% correct length, {perfect_acc_for_length:.1f}% perfect")
            
            # Most confused digits
            logger.info("MOST CONFUSED DIGITS (top 5):")
            confusion_pairs = []
            for target_digit, predictions in digit_confusion_matrix.items():
                total_predictions = sum(predictions.values())
                for pred_digit, count in predictions.items():
                    if target_digit != pred_digit:  # Only confusion pairs
                        confusion_rate = 100 * count / total_predictions
                        confusion_pairs.append((target_digit, pred_digit, count, confusion_rate))
            
            # Sort by confusion count and show top 5
            confusion_pairs.sort(key=lambda x: x[2], reverse=True)
            for i, (target, pred, count, rate) in enumerate(confusion_pairs[:5]):
                logger.info(f"  {target}→{pred}: {count} times ({rate:.1f}%)")
            
            logger.info("="*60)
        
        return results
    else:
        # Return empty results if no samples
        return {
            'total_samples': 0,
            'perfect_accuracy': 0.0,
            'length_accuracy': 0.0,
            'positional_accuracy': 0.0,
            'content_accuracy_given_correct_length': 0.0,
            'start_token_accuracy': 0.0,
            'finish_token_accuracy': 0.0,
            'length_error_rate': 0.0,
            'content_error_rate': 0.0,
            'order_error_rate': 0.0,
            'length_errors': 0,
            'content_errors_correct_length': 0,
            'order_errors_correct_content': 0,
            'partial_correct_positions': 0,
            'total_possible_positions': 0,
            'length_distribution': {},
            'digit_confusion_matrix': {}
        }

def calculate_length_aware_loss(output, target, target_lengths):
    """Calculate loss with length-aware approach - FIXED version"""
    # Reshape for cross entropy (use reshape instead of view for better compatibility)
    batch_size, seq_len, vocab_size = output.shape
    output_flat = output.reshape(-1, vocab_size)
    target_flat = target.reshape(-1)
    
    # FIXED: Don't ignore padding tokens - model needs to learn to predict finish tokens
    # instead of continuing to generate padding
    losses = F.cross_entropy(output_flat, target_flat, reduction='none')
    
    # Create mask to only calculate loss on valid positions (before padding)
    # This ensures model learns proper sequence termination
    mask = torch.arange(seq_len, device=target.device).unsqueeze(0) < (target_lengths - 1).unsqueeze(1)
    mask = mask.reshape(-1)
    
    # Apply mask and calculate mean loss only over valid positions
    masked_losses = losses * mask.float()
    valid_positions = mask.sum()
    
    return masked_losses.sum() / max(valid_positions, 1)

def count_digit_length(seq, seq_length):
    """Count digits in a sequence (excluding special tokens) - OPTIMIZED VERSION"""
    # Early termination for common cases
    if seq_length < 3:  # Need at least start + digit + finish
        return 0
    
    length = 0
    start_found = False
    
    # Only iterate up to seq_length (not full tensor length)
    for i in range(seq_length):
        token = seq[i].item()  # Convert to scalar once
        
        if token == 10:  # start token
            start_found = True
        elif token == 11:  # finish token
            break  # Early termination
        elif start_found and 0 <= token <= 9:  # digit token
            length += 1
    
    return length

def train_model(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001, 
                checkpoint_dir="./checkpoints", length_loss_weight=0.5, max_seq_len=102, max_grid_size=10):
    """
    Train the Encoder-Decoder MNIST model.
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to train on (CPU, CUDA, or MPS)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        checkpoint_dir (str): Directory to save model checkpoints
        length_loss_weight (float): Weight for the length prediction loss
        max_seq_len (int): Maximum sequence length supported by the model
        max_grid_size (int): Maximum grid size for the model
        
    Returns:
        float: Best accuracy achieved during training
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=12)  # Ignore pad token
    length_criterion = nn.MSELoss()  # For length prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_length_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (data, target_sequence, target_lengths) in enumerate(progress_bar):
            # Move data to device
            data, target_sequence, target_lengths = data.to(device), target_sequence.to(device), target_lengths.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data, target_sequence[:, :-1])  # Exclude last token for teacher forcing
            
            # Prepare targets (exclude first token - start token)
            target_for_loss = target_sequence[:, 1:]  # Remove start token
            
            # Calculate loss using length-aware approach
            primary_loss = calculate_length_aware_loss(output, target_for_loss, target_lengths)
            
            # Add length prediction loss to improve length accuracy - OPTIMIZED VERSION
            # Count actual digit lengths for length prediction loss
            actual_lengths = []
            for i in range(target_sequence.size(0)):
                length = count_digit_length(target_sequence[i], target_lengths[i])
                actual_lengths.append(length)
            actual_lengths = torch.tensor(actual_lengths, dtype=torch.float32, device=device)
            
            # OPTIMIZED: Vectorized length prediction from finish token positions
            batch_size, seq_len, vocab_size = output.shape
            
            # Get finish token probabilities for all positions at once
            finish_probs = torch.softmax(output, dim=-1)[:, :, 11]  # [batch, seq_len]
            
            # Create position weights for each sequence
            positions = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device).unsqueeze(0)  # [1, seq_len]
            
            # Mask out invalid positions based on target_lengths
            valid_mask = torch.arange(seq_len, device=device).unsqueeze(0) < (target_lengths - 1).unsqueeze(1)  # [batch, seq_len]
            
            # Calculate expected finish positions (vectorized)
            masked_finish_probs = finish_probs * valid_mask.float()
            expected_finish_pos = torch.sum(masked_finish_probs * positions, dim=1)  # [batch]
            
            # Clamp predictions to valid range
            predicted_lengths = torch.clamp(expected_finish_pos, min=0.0, max=float(max_grid_size * max_grid_size))
            
            # Length prediction loss
            length_loss = length_criterion(predicted_lengths, actual_lengths)
            
            # Combined loss with length prediction
            total_loss = primary_loss + length_loss_weight * length_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update running losses
            running_loss += primary_loss.item()
            running_length_loss += length_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": running_loss / (batch_idx + 1),
                "length_loss": running_length_loss / (batch_idx + 1)
            })
        
        # Evaluation phase using the comprehensive evaluation function
        # FIXED: Use max_length that scales dynamically with grid size
        # For 2x2 grid: max_length ~10, for 10x10 grid: max_length ~210
        max_possible_sequence = max_grid_size * max_grid_size + 2  # +2 for start/finish
        max_eval_length = min(max_possible_sequence + 10, max_seq_len)  # Don't exceed model's max_seq_len
        logger.info(f"Evaluation: {max_eval_length} tokens max (model supports {max_seq_len})")
        results = evaluate_model(model, test_loader, device, epoch=epoch+1, max_length=max_eval_length, verbose=True, max_debug_images=3)
        accuracy = results['perfect_accuracy']
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(checkpoint_dir, "encoder_decoder_mnist.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, checkpoint_path)
            logger.info(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"encoder_decoder_mnist_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }, checkpoint_path)
    
    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy

def collate_variable_length(batch):
    """
    Optimized collate function for variable-length sequences.
    Handles batching with minimal padding for efficiency.
    """
    images, sequences = zip(*batch)
    
    # Stack images (they're all the same size)
    images = torch.stack(images, dim=0)
    
    # Get sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Use PyTorch's built-in pad_sequence for efficiency
    sequences = pad_sequence(list(sequences), batch_first=True, padding_value=12)  # 12 is pad_token
    
    return images, sequences, lengths

class LengthBasedBatchSampler:
    """
    Batch sampler that groups sequences of similar lengths together.
    This minimizes padding and maximizes training efficiency.
    """
    def __init__(self, dataset, batch_size, shuffle=True, length_tolerance=2):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Target batch size
            shuffle: Whether to shuffle the data
            length_tolerance: Maximum length difference within a batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length_tolerance = length_tolerance
        
        # Pre-compute sequence lengths for all samples
        self._compute_lengths()
        
    def _compute_lengths(self):
        """Pre-compute sequence lengths for efficient batching"""
        self.lengths = []
        for i in range(len(self.dataset)):
            # Get a sample to compute its length
            _, sequence = self.dataset[i]
            self.lengths.append(len(sequence))
    
    def __iter__(self):
        """Generate batches grouped by similar sequence lengths"""
        # Create indices sorted by length
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Shuffle within length groups for better training diversity
            import random
            random.shuffle(indices)
        
        # Sort by length to group similar lengths together
        indices.sort(key=lambda i: self.lengths[i])
        
        # Create batches of similar lengths
        batches = []
        current_batch = []
        current_length = None
        
        for idx in indices:
            seq_length = self.lengths[idx]
            
            # Start new batch if length difference is too large or batch is full
            if (current_length is not None and 
                abs(seq_length - current_length) > self.length_tolerance) or \
               len(current_batch) >= self.batch_size:
                
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_length = seq_length
            else:
                current_batch.append(idx)
                if current_length is None:
                    current_length = seq_length
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            import random
            random.shuffle(batches)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self):
        """Return approximate number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def save_mnist_image_for_debug(image_tensor, filename, target_digits, predicted_digits, epoch, batch_idx, sample_idx):
    """
    Save MNIST image for debugging - PNG only.
    
    Args:
        image_tensor: Input image tensor [1, H, W]
        filename: Base filename to save
        target_digits: Target digit sequence
        predicted_digits: Predicted digit sequence
        epoch: Current epoch
        batch_idx: Batch index
        sample_idx: Sample index within batch
    """
    try:
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Convert tensor to numpy
        img_np = image_tensor.squeeze().cpu().numpy()
        
        # Save as PNG for easy viewing
        png_filename = f"{filename}_e{epoch}_b{batch_idx}_s{sample_idx}.png"
        png_path = os.path.join(debug_dir, png_filename)
        
        # Normalize to 0-255 range for PNG
        img_normalized = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_normalized, mode='L')  # L mode for grayscale
        pil_image.save(png_path)
        
        logger.info(f"  Debug image saved: {png_path}")
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")

def save_model_to_huggingface(model, repo_name, token=None, commit_message="Add trained encoder-decoder MNIST model", 
                             model_config=None):
    """
    Save the trained model to Hugging Face Hub.
    
    Args:
        model: The trained PyTorch model
        repo_name: Hugging Face repository name (e.g., 'username/model-name')
        token: Hugging Face API token (optional if logged in)
        commit_message: Commit message for the upload
        model_config: Additional model configuration to save
    """
    if not HF_AVAILABLE:
        logger.warning("Hugging Face libraries not available. Skipping HF upload.")
        return False
    
    try:
        # Login to Hugging Face if token provided
        if token:
            login(token=token)
            logger.info("Logged in to Hugging Face Hub")
        
        # Create a temporary directory for model files
        import tempfile
        import json
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model state dict
            model_path = os.path.join(temp_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), model_path)
            
            # Create model configuration
            config = {
                "model_type": "encoder_decoder_mnist",
                "architecture": "Vision Transformer + Transformer Decoder",
                "task": "sequence-to-sequence",
                "dataset": "MNIST",
                "max_seq_len": getattr(model, 'max_seq_len', 102),
                "vocab_size": 13,  # 0-9 digits + start(10) + finish(11) + pad(12)
                "pad_token_id": 12,
                "start_token_id": 10,
                "finish_token_id": 11,
                "created_by": "encoder_decoder_mnist_trainer"
            }
            
            # Add custom config if provided
            if model_config:
                config.update(model_config)
            
            # Save config
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create README
            readme_content = f"""# Encoder-Decoder MNIST Model

This model was trained to recognize and transcribe sequences of MNIST digits arranged in a grid.

## Model Architecture
- **Encoder**: Vision Transformer for image processing
- **Decoder**: Transformer decoder for sequence generation
- **Task**: Sequence-to-sequence digit recognition

## Usage
```python
import torch
from models.encoder_decoder.model import EncoderDecoder

# Load model
model = EncoderDecoder.from_pretrained("{repo_name}")
model.eval()

# Process image and generate sequence
with torch.no_grad():
    sequence = model(image, max_length=50)
```

## Training Details
- **Dataset**: MNIST
- **Grid Size**: Variable (configurable)
- **Sequence Length**: Variable (1 to grid capacity)
- **Vocabulary**: 0-9 digits + special tokens

## Model Configuration
```json
{json.dumps(config, indent=2)}
```
"""
            
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Upload to Hugging Face Hub
            api = HfApi()
            
            # Create repository if it doesn't exist
            try:
                api.create_repo(repo_name, exist_ok=True, private=False)
                logger.info(f"Repository {repo_name} ready for upload")
            except Exception as e:
                logger.warning(f"Could not create repository {repo_name}: {e}")
                return False
            
            # Upload files
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                commit_message=commit_message
            )
            
            logger.info(f"✅ Model successfully uploaded to https://huggingface.co/{repo_name}")
            logger.info(f"   Repository: https://huggingface.co/{repo_name}")
            logger.info(f"   Model files: config.json, pytorch_model.bin, README.md")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to upload model to Hugging Face Hub: {e}")
        return False

def main():
    """Main function to train the Encoder-Decoder MNIST model."""
    # Set up logging first
    base_dir = os.path.dirname(os.path.dirname(__file__))  # models directory
    log_file_path = setup_logging(base_dir)
    
    parser = argparse.ArgumentParser(description="Train Encoder-Decoder MNIST Model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, help="Directory to store dataset")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA training")
    parser.add_argument("--no-mps", action="store_true", default=False, help="Disable MPS training")
    parser.add_argument("--max-grid-size", type=int, default=10, help="Maximum grid size (e.g., 10 for 10x10 grid)")
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size for vision transformer")
    parser.add_argument("--max-output-length", type=int, default=None, help="Maximum output sequence length (None = grid capacity)")
    parser.add_argument("--static-length", type=int, default=None, help="If specified, force all sequences to have exactly this many digits")
    args = parser.parse_args()
    
    # Log command line arguments
    logger.info("COMMAND LINE ARGUMENTS:")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-')}: {value}")
    
    # Highlight static length mode if active
    if args.static_length is not None:
        logger.info(f"STATIC LENGTH MODE: All sequences will have {args.static_length} digits")
    else:
        logger.info(f"VARIABLE LENGTH MODE: Sequences will have 1-{args.max_grid_size * args.max_grid_size} digits")
    
    # Set default paths if not provided
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # Get the best available device
    if args.no_mps:
        # Skip MPS, use CUDA or CPU
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"Using device: {device}")
    else:
        # Use full device selection (MPS > CUDA > CPU)
        device = get_device()
    
    # Set number of CPU threads for CPU training
    if device.type == "cpu":
        torch.set_num_threads(4)  # Use multiple CPU cores
    
    # Calculate image configuration dynamically
    max_grid_size = args.max_grid_size
    single_image_size = 28  # MNIST image size
    full_image_size = single_image_size * max_grid_size  # e.g., 28*10=280 for 10x10
    
    logger.info(f"Image configuration: {max_grid_size}x{max_grid_size} grid, {full_image_size}x{full_image_size} pixels, patch size {args.patch_size}")
    
    # Validate patch size
    if full_image_size % args.patch_size != 0:
        logger.warning(f"Patch size {args.patch_size} does not divide evenly into image size {full_image_size}")
        valid_patch_sizes = [i for i in range(1, full_image_size+1) if full_image_size % i == 0]
        logger.warning(f"Consider using patch sizes that divide evenly: {valid_patch_sizes[:10]}...")  # Show first 10
    
    # Get data loaders with balanced length distribution
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, 
        data_dir=args.data_dir,
        max_grid_size=max_grid_size,  # Pass max grid size parameter
        max_output_length=args.max_output_length,  # Use specified or default to grid capacity
        static_length=args.static_length
    )
    
    # Calculate max sequence length: max_output_length + start + finish tokens
    # If max_output_length is None, it defaults to grid capacity in get_data_loaders
    effective_max_output_length = args.max_output_length if args.max_output_length is not None else (max_grid_size * max_grid_size)
    max_seq_len = effective_max_output_length + 2  # +2 for start and finish tokens
    
    logger.info(f"Model configuration: {effective_max_output_length} digits max, {max_seq_len} tokens max")
    
    # Create the Encoder-Decoder model with dynamic sizing
    model = EncoderDecoder(
        image_size=full_image_size,  # Full grid image size
        patch_size=args.patch_size,
        encoder_embed_dim=128,
        decoder_embed_dim=128,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_seq_len=max_seq_len  # Pass calculated max sequence length
    )
    model = model.to(device)
    
    logger.info(f"Training Encoder-Decoder MNIST model for {args.epochs} epochs")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    best_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        max_seq_len=max_seq_len,  # Pass the calculated max sequence length
        max_grid_size=max_grid_size  # Pass the max_grid_size parameter
    )
    
    # Export model for inference
    # Create filename with grid size and length mode
    length_mode = "fixlen" if args.static_length is not None else "varlen"
    model_filename = f"encoder_decoder_mnist_{max_grid_size}_{length_mode}.pt"
    export_path = os.path.join(args.checkpoint_dir, model_filename)
    torch.save(model.state_dict(), export_path)
    logger.info(f"Model exported for inference to {export_path}")

    # Upload to Hugging Face Hub if environment variables are set
    hf_user = os.environ.get('HUGGING_FACE_USER')
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')
    
    if hf_user and hf_token:
        logger.info("="*60)
        logger.info("UPLOADING MODEL TO HUGGING FACE HUB")
        logger.info("="*60)
        
        # Create repository name with grid size and length mode
        length_mode = "fixlen" if args.static_length is not None else "varlen"
        repo_name = f"{hf_user}/mnist-encoder-decoder-{max_grid_size}-{length_mode}"
        
        # Prepare model configuration
        model_config = {
            "max_grid_size": max_grid_size,
            "patch_size": args.patch_size,
            "encoder_embed_dim": 128,
            "decoder_embed_dim": 128,
            "num_layers": 8,
            "num_heads": 8,
            "dropout": 0.1,
            "static_length": args.static_length,
            "max_output_length": effective_max_output_length,
            "training_epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "best_accuracy": best_accuracy
        }
        
        success = save_model_to_huggingface(
            model=model,
            repo_name=repo_name,
            token=hf_token,
            commit_message="Add trained encoder-decoder MNIST model",
            model_config=model_config
        )
        
        if success:
            logger.info("✅ Hugging Face upload completed successfully!")
        else:
            logger.warning("❌ Hugging Face upload failed. Check logs above for details.")
    else:
        logger.info("Hugging Face upload skipped (HUGGING_FACE_USER and/or HUGGING_FACE_TOKEN not set)")

    # Log PyTorch version and MPS availability
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Log session completion
    logger.info("="*80)
    logger.info("TRAINING SESSION COMPLETED SUCCESSFULLY")
    logger.info(f"Full log saved to: {log_file_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main() 