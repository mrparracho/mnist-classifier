import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Transformer2 as Transformer

# Get the model directory path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_data_loaders(batch_size=64, data_dir=None):
    """
    Create data loaders for training and testing MNIST dataset.
    
    Args:
        batch_size (int): Batch size for training and testing
        data_dir (str): Directory to store the dataset
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    if data_dir is None:
        data_dir = os.path.join(MODEL_DIR, "data")
    
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001, 
                checkpoint_dir="./checkpoints"):
    """
    Train the MNIST model.
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to train on (CPU or GPU)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        checkpoint_dir (str): Directory to save model checkpoints
        
    Returns:
        model: Trained model
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": running_loss / (batch_idx + 1)})
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
            for data, target in progress_bar:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Get predictions
                _, predicted = torch.max(output.data, 1)
                
                # Update counters
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Update progress bar
                accuracy = 100 * correct / total
                progress_bar.set_postfix({"accuracy": accuracy})
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {accuracy:.2f}%")
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(checkpoint_dir, "mnist_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, checkpoint_path)
            logger.info(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }, checkpoint_path)
    
    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    return model


def main():
    """Main function to train the MNIST model."""
    parser = argparse.ArgumentParser(description="Train MNIST Model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, help="Directory to store dataset")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA training")
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.data_dir is None:
        args.data_dir = os.path.join(MODEL_DIR, "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(MODEL_DIR, "checkpoints")
    
    # Check for CUDA availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    # get shapes of train_loader and test_loader to feed into model
    train_loader_shape = train_loader.dataset[0][0].shape
    # test_loader_shape = test_loader.dataset[0][0].shape
    # print(f'{train_loader_shape} - train_loader_shape')
    # print(f'{test_loader_shape} - test_loader_shape')
    
    # Create model and move to device
    model = Transformer(
        image_size=train_loader_shape[1],
        patch_size=7,
        embed_dim=128,
        num_layers=3,
        num_classes=10
    )
    model = model.to(device)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=100,
        # learning_rate=args.lr,
        # checkpoint_dir=args.checkpoint_dir
    )
    
    # # Export model for inference
    # export_path = os.path.join(args.checkpoint_dir, "mnist_model.pt")
    # torch.save(model.state_dict(), export_path)
    # logger.info(f"Model exported for inference to {export_path}")


if __name__ == "__main__":
    main()
