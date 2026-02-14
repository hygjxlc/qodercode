"""
Prediction Script for CAE Stress Prediction MLP
Generated from OpenSpec: cae_stress_prediction_mlp v1.0.0
"""

import torch
import json
import argparse
from typing import Union, List, Dict
from model import CAEStressPredictionMLP, create_model


def predict_stress(
    model: CAEStressPredictionMLP,
    input_features: Union[List[float], List[List[float]]],
    device: torch.device
) -> Dict:
    """
    Predict stress from input features
    
    Args:
        model: Trained model
        input_features: Single sample [6] or batch [N, 6]
                        [length, width, thickness, E, nu, load]
        device: Device to run on
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Convert to tensor
    if isinstance(input_features[0], (int, float)):
        # Single sample
        input_tensor = torch.FloatTensor([input_features])
    else:
        # Batch
        input_tensor = torch.FloatTensor(input_features)
    
    input_tensor = input_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model.predict(input_tensor)
    
    # Convert to numpy
    stress_values = predictions.cpu().numpy().flatten()
    
    return {
        'stress_mpa': stress_values.tolist(),
        'input_features': input_features,
        'model_version': '1.0.0',
        'model_name': 'cae_stress_prediction_mlp'
    }


def load_model(checkpoint_path: str, device: torch.device) -> CAEStressPredictionMLP:
    """Load trained model from checkpoint"""
    model = CAEStressPredictionMLP().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='CAE Stress Prediction')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, default='result.json', help='Output JSON file path')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Get features
    if 'features' in input_data:
        features = input_data['features']
    else:
        features = input_data
    
    # Create device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model, device)
        print(f"Model loaded from: {args.model}")
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Using untrained model for demonstration")
        model, _ = create_model(device)
    
    # Predict
    results = predict_stress(model, features, device)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPrediction results:")
    print(f"  Input features: {len(features)} sample(s)")
    print(f"  Predicted stress (MPa): {results['stress_mpa']}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    # Example usage
    print("CAE Stress Prediction MLP - Prediction Script")
    print("=" * 50)
    
    # Create example input file
    example_input = {
        "features": [
            [100.0, 50.0, 5.0, 200.0, 0.3, 5000.0],   # Sample 1
            [150.0, 80.0, 8.0, 180.0, 0.25, 8000.0],  # Sample 2
        ],
        "description": "Input features: [length(mm), width(mm), thickness(mm), E(GPa), nu, load(N)]"
    }
    
    with open('example_input.json', 'w') as f:
        json.dump(example_input, f, indent=2)
    
    print("\nExample input file created: example_input.json")
    print("\nTo run prediction:")
    print("  python predict.py --input example_input.json --output result.json")
    print("\nOr use the API server:")
    print("  python api_server.py")
