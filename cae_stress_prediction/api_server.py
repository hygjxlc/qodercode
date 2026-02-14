"""
HTTP API Server for CAE Stress Prediction MLP
Generated from OpenSpec: cae_stress_prediction_mlp v1.0.0
Interfaces: /api/predict/stress, /api/model/openspec
"""

from flask import Flask, request, jsonify
import torch
import json
from typing import Dict, Any
from model import CAEStressPredictionMLP, create_model
from predict import predict_stress, load_model

app = Flask(__name__)

# Global model and device
model = None
device = None

# OpenSpec configuration
OPENSPEC_CONFIG = {
    "openspec": "v1",
    "kind": "NeuralNetwork",
    "name": "cae_stress_prediction_mlp",
    "version": "1.0.0",
    "description": "基于几何、材料、载荷的CAE结构应力预测神经网络",
    "domain": "CAE / 结构力学 / 仿真代理模型",
    "network": {
        "type": "sequential",
        "layers": [
            {"name": "input_layer", "type": "Input", "shape": [None, 6]},
            {"name": "dense_block_1", "type": "Dense", "units": 256, "activation": "silu"},
            {"name": "dropout_1", "type": "Dropout", "rate": 0.1},
            {"name": "dense_block_2", "type": "Dense", "units": 128, "activation": "silu"},
            {"name": "dense_block_3", "type": "Dense", "units": 64, "activation": "silu"},
            {"name": "output_layer", "type": "Dense", "units": 1, "activation": "linear"}
        ]
    },
    "input": {
        "type": "tensor",
        "dtype": "float32",
        "shape": ["batch", 6],
        "features": [
            "长度 (mm)",
            "宽度 (mm)",
            "厚度 (mm)",
            "弹性模量 (GPa)",
            "泊松比",
            "载荷大小 (N)"
        ]
    },
    "output": {
        "type": "tensor",
        "dtype": "float32",
        "shape": ["batch", 1],
        "description": "结构最大等效应力 (MPa)"
    }
}


def init_model(model_path: str = 'best_model.pth'):
    """Initialize model on startup"""
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model(model_path, device)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found, using untrained model")
        model, _ = create_model(device)
    
    print(f"Model ready on device: {device}")


@app.route('/api/model/openspec', methods=['GET'])
def get_model_info():
    """
    GET /api/model/openspec
    Return OpenSpec model configuration
    """
    return jsonify({
        'success': True,
        'data': OPENSPEC_CONFIG
    })


@app.route('/api/predict/stress', methods=['POST'])
def predict():
    """
    POST /api/predict/stress
    Predict stress from input features
    
    Request Body:
        {
            "features": [[length, width, thickness, E, nu, load], ...] or
                       [length, width, thickness, E, nu, load]
        }
    
    Response:
        {
            "success": true,
            "data": {
                "stress_mpa": [value1, value2, ...],
                "input_features": [...],
                "model_version": "1.0.0"
            }
        }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required field: features'
            }), 400
        
        features = data['features']
        
        # Validate input
        if not isinstance(features, list):
            return jsonify({
                'success': False,
                'message': 'features must be a list'
            }), 400
        
        # Check if single sample or batch
        if isinstance(features[0], (int, float)):
            # Single sample - should have 6 features
            if len(features) != 6:
                return jsonify({
                    'success': False,
                    'message': 'Single sample must have 6 features: [length, width, thickness, E, nu, load]'
                }), 400
        else:
            # Batch - each sample should have 6 features
            for i, sample in enumerate(features):
                if len(sample) != 6:
                    return jsonify({
                        'success': False,
                        'message': f'Sample {i} must have 6 features: [length, width, thickness, E, nu, load]'
                    }), 400
        
        # Predict
        results = predict_stress(model, features, device)
        
        return jsonify({
            'success': True,
            'data': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'CAE Stress Prediction MLP API',
        'version': '1.0.0',
        'endpoints': {
            '/api/model/openspec': 'GET - Model configuration',
            '/api/predict/stress': 'POST - Predict stress',
            '/api/health': 'GET - Health check'
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("CAE Stress Prediction MLP - API Server")
    print("=" * 60)
    
    # Initialize model
    init_model()
    
    print("\nStarting API server...")
    print("Endpoints:")
    print("  GET  /api/model/openspec  - Model configuration")
    print("  POST /api/predict/stress  - Predict stress")
    print("  GET  /api/health          - Health check")
    print("\nServer running at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)
