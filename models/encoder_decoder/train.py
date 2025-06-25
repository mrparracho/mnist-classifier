#!/usr/bin/env python3
"""
Training script for Encoder-Decoder MNIST model.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

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
    def __init__(self, mnist_dataset, max_grid_size=10, max_output_length=None, length_weights=None, 
                 min_digits=1, max_digits=None, fill_strategy='random'):
        self.mnist_dataset = mnist_dataset
        self.max_grid_size = max_grid_size  # e.g., 10 for 10x10 grid
        
        # Set max_output_length to grid capacity if not specified
        if max_output_length is None:
            self.max_output_length = max_grid_size * max_grid_size  # e.g., 10*10 = 100
        else:
            self.max_output_length = max_output_length
            
        self.min_digits = min_digits
        self.max_digits = max_digits if max_digits is not None else min(max_grid_size * max_grid_size, self.max_output_length)
        self.fill_strategy = fill_strategy  # 'random', 'zeros', 'noise'
        
        # Length distribution weights - if None, use uniform distribution
        if length_weights is None:
            # Equal probability for each length from min_digits to max_digits
            num_lengths = self.max_digits - self.min_digits + 1
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
        
        logger.info(f"Dataset configuration:")
        logger.info(f"  Max grid size: {self.max_grid_size}x{self.max_grid_size}")
        logger.info(f"  Full image size: {self.full_grid_size}x{self.full_grid_size}")
        logger.info(f"  Digit range: {self.min_digits}-{self.max_digits} per sample")
        logger.info(f"  Fill strategy: {self.fill_strategy}")
        
    def __len__(self):
        return len(self.mnist_dataset) // self.max_digits  # Ensure we have enough data
    
    def _get_filler_image(self):
        """Get a filler image based on fill strategy"""
        if self.fill_strategy == 'zeros':
            return torch.zeros(1, self.single_image_size, self.single_image_size)
        elif self.fill_strategy == 'noise':
            # Generate small amount of noise (like paper texture)
            return torch.randn(1, self.single_image_size, self.single_image_size) * 0.1
        elif self.fill_strategy == 'random':
            # Use a random MNIST image but don't include its label
            random_idx = torch.randint(0, len(self.mnist_dataset), (1,)).item()
            image, _ = self.mnist_dataset[random_idx]
            return image
        else:
            return torch.zeros(1, self.single_image_size, self.single_image_size)
    
    def __getitem__(self, idx):
        # Sample how many actual digits to include
        num_digits = int(torch.multinomial(torch.tensor(self.length_probs, dtype=torch.float32), 1).item())
        num_digits = self.min_digits + num_digits  # Offset by min_digits
        num_digits = min(num_digits, self.max_digits)  # Ensure we don't exceed max
        
        # Get actual MNIST digits and their labels
        digit_images = []
        digit_labels = []
        
        start_idx = idx * self.max_digits
        for i in range(num_digits):
            data_idx = (start_idx + i) % len(self.mnist_dataset)
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
                # Fill with background
                filler_image = self._get_filler_image()
                full_grid_image[:, start_h:end_h, start_w:end_w] = filler_image
        
        # Sort labels by position to create consistent reading order (left-to-right, top-to-bottom)
        placed_labels.sort(key=lambda x: x[0])  # Sort by position
        sequence_labels = [label for pos, label in placed_labels]
        
        # Create target sequence: [start_token, digit1, digit2, ..., finish_token]
        target_sequence = [self.start_token]  # start token
        target_sequence.extend(sequence_labels)  # add the digit labels in order
        target_sequence.append(self.finish_token)  # finish token
        
        # Pad sequence to max_output_length + 2 (for start and finish tokens)
        max_seq_length = self.max_output_length + 2
        while len(target_sequence) < max_seq_length:
            target_sequence.append(self.pad_token)
        
        # Truncate if too long (shouldn't happen with our logic, but just in case)
        target_sequence = target_sequence[:max_seq_length]
        
        # Convert to tensor
        target_sequence = torch.tensor(target_sequence, dtype=torch.long)
        
        return full_grid_image, target_sequence

def get_device():
    """
    Get the best available device for training.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        torch.device: The best available device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU for training")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")
    
    return device

def get_data_loaders(batch_size=64, data_dir=None, balance_lengths=True, max_grid_size=10, 
                     max_output_length=None):
    """
    Create data loaders for training and testing MNIST dataset.
    
    Args:
        batch_size (int): Batch size for training and testing
        data_dir (str): Directory to store the dataset
        balance_lengths (bool): Whether to use balanced length distribution
        max_grid_size (int): Maximum grid size (e.g., 10 for 10x10 grid)
        max_output_length (int): Maximum output sequence length (None = grid capacity)
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Set max_output_length to grid capacity if not specified
    if max_output_length is None:
        max_output_length = max_grid_size * max_grid_size  # e.g., 10*10 = 100
        
    logger.info(f"Data loader configuration:")
    logger.info(f"  Max grid size: {max_grid_size}x{max_grid_size} = {max_grid_size * max_grid_size} positions")
    logger.info(f"  Max output length: {max_output_length} digits")
    
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
    
    # Configure length weights for better balance
    if balance_lengths:
        # For large grids, focus on smaller sequences initially to avoid overwhelming the model
        if max_output_length > 20:
            # Weight heavily toward smaller sequences for large grids
            length_weights = [3.0, 2.0, 1.5, 1.0, 1.0] + [0.5] * (max_output_length - 5)
            logger.info(f"Using weighted length distribution: favoring shorter sequences for large grid")
        else:
            # Equal probability for smaller grids
            length_weights = [1.0] * min(10, max_output_length)  # Up to 10 equal weights
            logger.info(f"Using balanced length distribution: equal probability for lengths 1-{min(10, max_output_length)}")
    else:
        # Use uniform random (original behavior)
        length_weights = None
        logger.info("Using uniform random length distribution")
    
    # Wrap with sequence dataset
    train_sequence_dataset = MNISTSequenceDataset(
        train_dataset, 
        max_grid_size=max_grid_size, 
        max_output_length=max_output_length,
        length_weights=length_weights
    )
    test_sequence_dataset = MNISTSequenceDataset(
        test_dataset, 
        max_grid_size=max_grid_size, 
        max_output_length=max_output_length,
        length_weights=length_weights
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_sequence_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_sequence_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device, epoch=None, max_length=12, verbose=True):
    """
    Comprehensive evaluation of the Encoder-Decoder MNIST model.
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to evaluate on
        epoch (int, optional): Current epoch number for logging
        max_length (int): Maximum sequence length for generation
        verbose (bool): Whether to print detailed logs
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    
    # Initialize comprehensive metrics
    total_samples = 0
    perfect_matches = 0
    length_errors = 0
    content_errors_correct_length = 0
    order_errors_correct_content = 0
    partial_correct_positions = 0
    total_possible_positions = 0
    no_finish_token = 0
    no_start_token = 0
    start_token_correct = 0
    finish_token_correct = 0
    
    # Length distribution tracking
    length_distribution = {}
    
    # Digit confusion matrix
    digit_confusion_matrix = {}
    
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
        
        for batch_idx, (data, target_sequence) in enumerate(progress_bar):
            # Move data to device
            data, target_sequence = data.to(device), target_sequence.to(device)
            
            # Forward pass - inference mode (no target_sequence)
            predicted_sequence = model(data, max_length=max_length)
            
            # Evaluate each sample in the batch
            for i in range(data.size(0)):
                target_seq = target_sequence[i]
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
                    logger.info(f"Example {i}:")
                    logger.info(f"  Target sequence: {target_seq.tolist()}")
                    logger.info(f"  Predicted sequence: {pred_seq.tolist()}")
                
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

def train_model(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001, 
                checkpoint_dir="./checkpoints", length_loss_weight=0.5, max_seq_len=102):
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
        
    Returns:
        model: Trained model
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
        for batch_idx, (data, target_sequence) in enumerate(progress_bar):
            # Move data to device
            data, target_sequence = data.to(device), target_sequence.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass - model expects target_sequence for training
            # For teacher forcing, use input sequence (all tokens except the last one)
            # Target for loss is the shifted sequence (all tokens except the first one)
            input_sequence = target_sequence[:, :-1]  # Remove last token
            target_for_loss = target_sequence[:, 1:]   # Remove first token (start token)
            
            output = model(data, target_sequence=input_sequence)
            
            # Debug: Print shapes and first batch on first iteration
            if batch_idx == 0 and epoch == 0:
                logger.info(f"Input sequence shape: {input_sequence.shape}")
                logger.info(f"Target for loss shape: {target_for_loss.shape}")
                logger.info(f"Output shape: {output.shape}")
                logger.info(f"First input sequence: {input_sequence[0]}")
                logger.info(f"First target for loss: {target_for_loss[0]}")
                logger.info(f"First output (argmax): {torch.argmax(output[0], dim=-1)}")
            
            # Reshape output and target for loss calculation
            # output: [batch_size, seq_len, vocab_size]
            # target: [batch_size, seq_len]
            output_flat = output.reshape(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
            target_flat = target_for_loss.reshape(-1)  # [batch_size * seq_len]
            
            # Calculate primary loss (next token prediction)
            primary_loss = criterion(output_flat, target_flat)
            
            # Calculate length prediction loss
            # Count actual sequence lengths (excluding start/finish/pad tokens)
            def count_digit_length(seq):
                """Count digits in a sequence (excluding special tokens)"""
                length = 0
                start_found = False
                for token in seq:
                    if token == 10:  # start token
                        start_found = True
                    elif token == 11:  # finish token
                        break
                    elif start_found and 0 <= token <= 9:  # digit token
                        length += 1
                return length
            
            # Get actual lengths for each sequence in the batch
            actual_lengths = []
            for i in range(target_sequence.size(0)):
                length = count_digit_length(target_sequence[i])
                actual_lengths.append(length)
            actual_lengths = torch.tensor(actual_lengths, dtype=torch.float32, device=device)
            
            # Predict lengths from finish token positions in output
            predicted_lengths = []
            for i in range(output.size(0)):
                # Find where the model predicts finish token (token 11)
                finish_probs = torch.softmax(output[i], dim=-1)[:, 11]  # Probability of finish token at each position
                # Use expectation of finish token position as length prediction
                positions = torch.arange(1, output.size(1) + 1, dtype=torch.float32, device=device)
                expected_finish_pos = torch.sum(finish_probs * positions)
                # Subtract 1 because position 1 would mean 0 digits (just start + finish)
                predicted_length = torch.clamp(expected_finish_pos - 1, min=0.0, max=10.0)
                predicted_lengths.append(predicted_length)
            predicted_lengths = torch.stack(predicted_lengths)
            
            # Length prediction loss
            length_loss = length_criterion(predicted_lengths, actual_lengths)
            
            # Combined loss
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
                "len_loss": running_length_loss / (batch_idx + 1)
            })
        
        # Evaluation phase using the comprehensive evaluation function
        # Use the model's configured maximum sequence length
        max_eval_length = max_seq_len  # Use the model's configured maximum sequence length
        results = evaluate_model(model, test_loader, device, epoch=epoch+1, max_length=max_eval_length, verbose=True)
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
    return model


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
    parser.add_argument("--min-digits", type=int, default=1, help="Minimum number of digits per sample")
    parser.add_argument("--max-digits", type=int, default=None, help="Maximum number of digits per sample (None = limited by max_output_length)")
    args = parser.parse_args()
    
    # Log command line arguments
    logger.info("COMMAND LINE ARGUMENTS:")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-')}: {value}")
    logger.info("")
    
    # Set default paths if not provided
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # Get the best available device
    if args.no_mps or not torch.backends.mps.is_available():
        # Fall back to CUDA/CPU logic
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = get_device()
    
    logger.info(f"Using device: {device}")
    
    # Set number of CPU threads for CPU training
    if device.type == "cpu":
        torch.set_num_threads(4)  # Use multiple CPU cores
    
    # Calculate image configuration dynamically
    max_grid_size = args.max_grid_size
    single_image_size = 28  # MNIST image size
    full_image_size = single_image_size * max_grid_size  # e.g., 28*10=280 for 10x10
    
    logger.info(f"Image configuration:")
    logger.info(f"  Max grid size: {max_grid_size}x{max_grid_size}")
    logger.info(f"  Individual cell size: {single_image_size}x{single_image_size}")
    logger.info(f"  Full image size: {full_image_size}x{full_image_size}")
    logger.info(f"  Patch size: {args.patch_size}")
    
    # Validate patch size
    if full_image_size % args.patch_size != 0:
        logger.warning(f"Patch size {args.patch_size} does not divide evenly into image size {full_image_size}")
        valid_patch_sizes = [i for i in range(1, full_image_size+1) if full_image_size % i == 0]
        logger.warning(f"Consider using patch sizes that divide evenly: {valid_patch_sizes[:10]}...")  # Show first 10
    
    # Get data loaders with balanced length distribution
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, 
        data_dir=args.data_dir,
        balance_lengths=True,  # Enable balanced length distribution
        max_grid_size=max_grid_size,  # Pass max grid size parameter
        max_output_length=args.max_output_length  # Use specified or default to grid capacity
    )
    
    # Calculate max sequence length: max_output_length + start + finish tokens
    # If max_output_length is None, it defaults to grid capacity in get_data_loaders
    effective_max_output_length = args.max_output_length if args.max_output_length is not None else (max_grid_size * max_grid_size)
    max_seq_len = effective_max_output_length + 2  # +2 for start and finish tokens
    
    logger.info(f"Model configuration:")
    logger.info(f"  Max output length: {effective_max_output_length} digits")
    logger.info(f"  Max sequence length: {max_seq_len} tokens (including start/finish)")
    
    # Create the Encoder-Decoder model with dynamic sizing
    model = EncoderDecoder(
        image_size=full_image_size,  # Full grid image size
        patch_size=args.patch_size,
        encoder_embed_dim=128,
        decoder_embed_dim=128,
        num_layers=16,
        num_heads=16,
        dropout=0.1,
        max_seq_len=max_seq_len  # Pass calculated max sequence length
    )
    model = model.to(device)
    
    logger.info(f"Training Encoder-Decoder MNIST model for {args.epochs} epochs")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        max_seq_len=max_seq_len  # Pass the calculated max sequence length
    )
    
    # Export model for inference
    export_path = os.path.join(args.checkpoint_dir, "encoder_decoder_mnist.pt")
    torch.save(model.state_dict(), export_path)
    logger.info(f"Model exported for inference to {export_path}")

    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Log session completion
    logger.info("="*80)
    logger.info("TRAINING SESSION COMPLETED SUCCESSFULLY")
    logger.info(f"Full log saved to: {log_file_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main() 