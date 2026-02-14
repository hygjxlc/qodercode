"""
CAE Stress Prediction MLP Model
Generated from OpenSpec: cae_stress_prediction_mlp v1.0.0
Domain: CAE / Structural Mechanics / Simulation Surrogate Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CAEStressPredictionMLP(nn.Module):
    """
    Neural Network for CAE Structural Stress Prediction
    
    Input: 6-dimensional features [length, width, thickness, E, nu, load]
    Output: Maximum von Mises stress (MPa)
    """
    
    def __init__(self, input_dim: int = 6, dropout_rate: float = 0.1):
        super(CAEStressPredictionMLP, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, 256)
        
        # Dense block 1
        self.dense_block_1 = nn.Linear(256, 256)
        self.dropout_1 = nn.Dropout(dropout_rate)
        
        # Dense block 2
        self.dense_block_2 = nn.Linear(256, 128)
        
        # Dense block 3
        self.dense_block_3 = nn.Linear(128, 64)
        
        # Output layer
        self.output_layer = nn.Linear(64, 1)
        
        # Initialize weights with He uniform
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, 6]
               Features: [length(mm), width(mm), thickness(mm), E(GPa), nu, load(N)]
        
        Returns:
            Output tensor of shape [batch_size, 1] - Maximum stress in MPa
        """
        # Input layer
        x = F.silu(self.input_layer(x))
        
        # Dense block 1
        x = F.silu(self.dense_block_1(x))
        x = self.dropout_1(x)
        
        # Dense block 2
        x = F.silu(self.dense_block_2(x))
        
        # Dense block 3
        x = F.silu(self.dense_block_3(x))
        
        # Output layer (linear activation)
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction method for inference"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def create_model(device: Optional[str] = None) -> Tuple[CAEStressPredictionMLP, torch.device]:
    """
    Factory function to create model and device
    
    Args:
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        Tuple of (model, device)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    model = CAEStressPredictionMLP().to(device)
    
    return model, device


if __name__ == "__main__":
    # Test model creation
    model, device = create_model()
    print(f"Model created on device: {device}")
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 6)
    output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Test output: {output.squeeze().tolist()}")
