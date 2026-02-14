"""
CAE Stress Dataset Preprocessing - Usage Example
根据 OpenSpec 规范执行完整的数据预处理流程
"""

from data_preprocessing import CAEDataPreprocessor, create_sample_data
import json


def main():
    print("=" * 70)
    print("CAE Stress Dataset Preprocessing")
    print("OpenSpec: cae_stress_dataset_prep v1.0.0")
    print("=" * 70)
    
    # Step 1: Create sample data (for demonstration)
    print("\n[Step 1] Creating sample raw data...")
    raw_data = create_sample_data('./data/raw_cae_data.csv', n_samples=1000)
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Columns: {list(raw_data.columns)}")
    
    # Step 2: Initialize preprocessor
    print("\n[Step 2] Initializing preprocessor...")
    preprocessor = CAEDataPreprocessor()
    
    # Step 3: Process data
    print("\n[Step 3] Running preprocessing pipeline...")
    result = preprocessor.process()
    
    # Step 4: Get report
    print("\n[Step 4] Processing Report:")
    report = preprocessor.get_processing_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Step 5: Verify outputs
    print("\n[Step 5] Verifying outputs...")
    import pandas as pd
    from pathlib import Path
    
    output_files = [
        './data/train_data.csv',
        './data/val_data.csv',
        './data/test_data.csv'
    ]
    
    for file_path in output_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            print(f"  ✓ {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  ✗ {file_path}: Not found")
    
    print("\n" + "=" * 70)
    print("Preprocessing completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check processed data in ./data/ directory")
    print("  2. Run training: python train.py")
    print("  3. Start API server: python api_server.py")


if __name__ == "__main__":
    main()
