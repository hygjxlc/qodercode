"""
Training Script for CAE Stress Prediction MLP
Generated from OpenSpec: cae_stress_prediction_mlp v1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import json
import time
from model import CAEStressPredictionMLP, create_model


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'mse': [],
            'rmse': [],
            'r2': []
        }
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """Calculate regression metrics"""
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_targets, all_preds)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device
) -> MetricsTracker:
    """
    Main training loop
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
    
    Returns:
        MetricsTracker with training history
    """
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    criterion = nn.SmoothL1Loss()
    
    tracker = MetricsTracker()
    best_val_loss = float('inf')
    
    print(f"Starting training for {config['max_epochs']} epochs...")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print("-" * 60)
    
    for epoch in range(config['max_epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Update tracker
        tracker.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics
        })
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch [{epoch+1}/{config['max_epochs']}] "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"MAE: {metrics['mae']:.4f} | "
              f"RMSE: {metrics['rmse']:.4f} | "
              f"R²: {metrics['r2']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, 'best_model.pth')
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
    
    print("-" * 60)
    print("Training completed!")
    
    return tracker


def generate_sample_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sample training data for demonstration"""
    np.random.seed(42)
    
    # Generate random input features
    # [length(mm), width(mm), thickness(mm), E(GPa), nu, load(N)]
    X = np.random.rand(n_samples, 6)
    X[:, 0] = X[:, 0] * 200 + 50      # length: 50-250 mm
    X[:, 1] = X[:, 1] * 100 + 20      # width: 20-120 mm
    X[:, 2] = X[:, 2] * 10 + 2        # thickness: 2-12 mm
    X[:, 3] = X[:, 3] * 100 + 100     # E: 100-200 GPa
    X[:, 4] = X[:, 4] * 0.3 + 0.2     # nu: 0.2-0.5
    X[:, 5] = X[:, 5] * 9000 + 1000   # load: 1000-10000 N
    
    # Generate synthetic stress output (simplified physics-based approximation)
    # stress ≈ load / (width * thickness) * geometry_factor
    stress = (X[:, 5] / (X[:, 1] * X[:, 2])) * (1 + 0.1 * np.random.randn(n_samples))
    stress = stress.reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(stress)


if __name__ == "__main__":
    # Configuration from OpenSpec
    config = {
        'framework': 'pytorch',
        'optimizer': {
            'name': 'Adam',
            'lr': 0.0005,
            'weight_decay': 1e-5
        },
        'loss': {
            'name': 'SmoothL1Loss'
        },
        'metrics': ['MAE', 'MSE', 'RMSE', 'R2Score'],
        'batch_size': 64,
        'max_epochs': 200
    }
    
    # Create model
    model, device = create_model()
    print(f"Model created on device: {device}\n")
    
    # Generate sample data
    print("Generating sample data...")
    X_train, y_train = generate_sample_data(800)
    X_val, y_val = generate_sample_data(200)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Train model
    tracker = train(model, train_loader, val_loader, config, device)
    
    # Save training history
    tracker.save('training_history.json')
    print("\nTraining history saved to 'training_history.json'")
    print("Best model saved to 'best_model.pth'")
