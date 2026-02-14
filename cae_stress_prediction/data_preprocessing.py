"""
CAE Stress Dataset Preprocessing
Generated from OpenSpec: cae_stress_dataset_prep v1.0.0
Description: CAE应力预测数据集制作（从CSV/Excel读取→字段处理→清洗→输出训练集）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CAEDataPreprocessor:
    """
    CAE Stress Dataset Preprocessor
    
    Features:
    - Read CSV/Excel/JSON files
    - Field filtering and renaming
    - Missing value handling
    - Outlier detection and handling
    - Categorical encoding
    - Numerical normalization
    - Train/Val/Test split
    - Data validation
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize preprocessor with OpenSpec configuration
        
        Args:
            config_path: Path to OpenSpec YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.label_encoders = {}
        self.processing_stats = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load OpenSpec configuration"""
        if config_path is None:
            # Use default config from OpenSpec
            return self._default_config()
        
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Default OpenSpec configuration"""
        return {
            'data_source': [{
                'name': 'raw_cae_data',
                'type': 'csv',
                'path': './data/raw_cae_data.csv',
                'encoding': 'utf-8',
                'header': True,
                'delimiter': ','
            }],
            'fields': {
                'raw_fields': [
                    {'name': 'length_mm', 'type': 'numeric', 'desc': '结构长度(mm)'},
                    {'name': 'width_mm', 'type': 'numeric', 'desc': '结构宽度(mm)'},
                    {'name': 'thickness_mm', 'type': 'numeric', 'desc': '结构厚度(mm)'},
                    {'name': 'material', 'type': 'categorical', 'desc': '材料类型（钢/铝/铜）'},
                    {'name': 'load_N', 'type': 'numeric', 'desc': '载荷大小(N)'},
                    {'name': 'stress_mpa', 'type': 'numeric', 'desc': '实测应力(MPa)'},
                    {'name': 'test_time', 'type': 'datetime', 'desc': '测试时间（无用字段，需删除）'}
                ],
                'target_fields': [
                    {'name': 'length', 'source': 'length_mm', 'type': 'numeric', 'unit': 'mm'},
                    {'name': 'width', 'source': 'width_mm', 'type': 'numeric', 'unit': 'mm'},
                    {'name': 'thickness', 'source': 'thickness_mm', 'type': 'numeric', 'unit': 'mm'},
                    {'name': 'material_encoded', 'source': 'material', 'type': 'numeric', 'desc': '材料编码（钢=0，铝=1，铜=2）'},
                    {'name': 'load', 'source': 'load_N', 'type': 'numeric', 'unit': 'N'},
                    {'name': 'stress', 'source': 'stress_mpa', 'type': 'numeric', 'unit': 'MPa'}
                ]
            },
            'processing_rules': {
                'field_filter': {
                    'keep': ['length_mm', 'width_mm', 'thickness_mm', 'material', 'load_N', 'stress_mpa'],
                    'drop': ['test_time']
                },
                'missing_values': {
                    'strategy': {
                        'numeric': 'fill_mean',
                        'categorical': 'fill_mode'
                    },
                    'fields': [
                        {'name': 'load_N', 'strategy': 'fill_median'}
                    ]
                },
                'outlier': {
                    'method': 'iqr',
                    'fields': ['stress_mpa'],
                    'action': 'clip'
                },
                'encoding': {
                    'field': 'material',
                    'type': 'label_encoding',
                    'mapping': {'钢': 0, '铝': 1, '铜': 2}
                },
                'normalization': {
                    'fields': ['length_mm', 'width_mm', 'thickness_mm', 'load_N'],
                    'method': 'standard_scaler'
                },
                'rename': [
                    {'from': 'length_mm', 'to': 'length'},
                    {'from': 'width_mm', 'to': 'width'},
                    {'from': 'stress_mpa', 'to': 'stress'}
                ]
            },
            'output': {
                'type': 'csv',
                'path': './data/processed_cae_stress_data.csv',
                'split': {
                    'enable': True,
                    'train_ratio': 0.8,
                    'val_ratio': 0.1,
                    'test_ratio': 0.1,
                    'random_seed': 42,
                    'output_paths': {
                        'train': './data/train_data.csv',
                        'val': './data/val_data.csv',
                        'test': './data/test_data.csv'
                    }
                }
            },
            'validation': [
                {'rule': 'no_missing_values'},
                {'rule': 'stress >= 0'},
                {'rule': 'material_encoded in [0,1,2]'}
            ]
        }
    
    def read_data(self, data_source: Dict = None) -> pd.DataFrame:
        """
        Read data from CSV/Excel/JSON file
        
        Args:
            data_source: Data source configuration
        
        Returns:
            pandas DataFrame
        """
        if data_source is None:
            data_source = self.config['data_source'][0]
        
        file_path = data_source['path']
        file_type = data_source.get('type', 'csv')
        encoding = data_source.get('encoding', 'utf-8')
        
        logger.info(f"Reading data from: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_type == 'csv':
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=data_source.get('delimiter', ','),
                skiprows=data_source.get('skip_rows', 0)
            )
        elif file_type == 'excel':
            df = pd.read_excel(
                file_path,
                sheet_name=data_source.get('sheet_name', 0),
                skiprows=data_source.get('skip_rows', 0)
            )
        elif file_type == 'json':
            df = pd.read_json(file_path, encoding=encoding)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def filter_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Filter fields (keep/drop)"""
        rules = self.config['processing_rules']['field_filter']
        keep_fields = rules.get('keep', [])
        drop_fields = rules.get('drop', [])
        
        logger.info("Step 1: Filtering fields...")
        
        # Keep only specified fields
        if keep_fields:
            available_fields = [f for f in keep_fields if f in df.columns]
            df = df[available_fields]
            logger.info(f"  Kept fields: {available_fields}")
        
        # Drop specified fields
        if drop_fields:
            df = df.drop(columns=[f for f in drop_fields if f in df.columns], errors='ignore')
            logger.info(f"  Dropped fields: {drop_fields}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Handle missing values"""
        rules = self.config['processing_rules']['missing_values']
        default_strategy = rules.get('strategy', {})
        field_strategies = {f['name']: f['strategy'] for f in rules.get('fields', [])}
        
        logger.info("Step 2: Handling missing values...")
        
        missing_stats = {}
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                missing_stats[col] = missing_count
                
                # Determine strategy
                if col in field_strategies:
                    strategy = field_strategies[col]
                elif df[col].dtype in ['int64', 'float64']:
                    strategy = default_strategy.get('numeric', 'fill_mean')
                else:
                    strategy = default_strategy.get('categorical', 'fill_mode')
                
                # Apply strategy
                if strategy == 'fill_mean':
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                elif strategy == 'fill_median':
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                elif strategy == 'fill_mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                    df[col].fillna(fill_value, inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
                
                logger.info(f"  {col}: filled {missing_count} missing values with {strategy}")
        
        self.processing_stats['missing_values'] = missing_stats
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Handle outliers using IQR method"""
        rules = self.config['processing_rules']['outlier']
        method = rules.get('method', 'iqr')
        fields = rules.get('fields', [])
        action = rules.get('action', 'clip')
        
        logger.info("Step 3: Handling outliers...")
        
        outlier_stats = {}
        
        for col in fields:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if action == 'clip':
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == 'drop':
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif action == 'fill':
                    median = df[col].median()
                    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median
                
                outlier_stats[col] = outlier_count
                logger.info(f"  {col}: {outlier_count} outliers handled with {action}")
        
        self.processing_stats['outliers'] = outlier_stats
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Encode categorical features"""
        rules = self.config['processing_rules']['encoding']
        field = rules.get('field')
        encoding_type = rules.get('type', 'label_encoding')
        mapping = rules.get('mapping', {})
        
        logger.info("Step 4: Encoding categorical features...")
        
        if field not in df.columns:
            logger.warning(f"  Field '{field}' not found, skipping encoding")
            return df
        
        if encoding_type == 'label_encoding':
            if mapping:
                # Use provided mapping
                df[f'{field}_encoded'] = df[field].map(mapping)
                logger.info(f"  {field}: label encoded with custom mapping")
            else:
                # Auto encode
                le = LabelEncoder()
                df[f'{field}_encoded'] = le.fit_transform(df[field].astype(str))
                self.label_encoders[field] = le
                logger.info(f"  {field}: auto label encoded")
        
        elif encoding_type == 'one_hot':
            dummies = pd.get_dummies(df[field], prefix=field)
            df = pd.concat([df, dummies], axis=1)
            logger.info(f"  {field}: one-hot encoded")
        
        return df
    
    def normalize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Normalize numeric features"""
        rules = self.config['processing_rules']['normalization']
        fields = rules.get('fields', [])
        method = rules.get('method', 'standard_scaler')
        
        logger.info("Step 5: Normalizing numeric features...")
        
        for col in fields:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Choose scaler
            if method == 'standard_scaler':
                scaler = StandardScaler()
            elif method == 'min_max':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                continue
            
            # Fit and transform
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            logger.info(f"  {col}: normalized with {method}")
        
        return df
    
    def rename_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6: Rename fields"""
        rules = self.config['processing_rules']['rename']
        
        logger.info("Step 6: Renaming fields...")
        
        rename_dict = {r['from']: r['to'] for r in rules if r['from'] in df.columns}
        df = df.rename(columns=rename_dict)
        
        logger.info(f"  Renamed: {rename_dict}")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate processed data
        
        Returns:
            True if validation passes, raises exception otherwise
        """
        rules = self.config.get('validation', [])
        
        logger.info("Validating processed data...")
        
        errors = []
        
        for rule in rules:
            rule_str = rule.get('rule', '')
            
            if rule_str == 'no_missing_values':
                missing = df.isnull().sum().sum()
                if missing > 0:
                    errors.append(f"Validation failed: {missing} missing values found")
            
            elif rule_str == 'stress >= 0':
                if 'stress' in df.columns and (df['stress'] < 0).any():
                    errors.append("Validation failed: negative stress values found")
            
            elif rule_str == 'material_encoded in [0,1,2]':
                if 'material_encoded' in df.columns:
                    invalid = df[~df['material_encoded'].isin([0, 1, 2])]
                    if len(invalid) > 0:
                        errors.append(f"Validation failed: {len(invalid)} invalid material_encoded values")
        
        if errors:
            for error in errors:
                logger.error(error)
            raise ValueError("Data validation failed")
        
        logger.info("Validation passed!")
        return True
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/val/test sets"""
        output_config = self.config['output']
        split_config = output_config.get('split', {})
        
        if not split_config.get('enable', False):
            return {'all': df}
        
        logger.info("Splitting data into train/val/test sets...")
        
        train_ratio = split_config.get('train_ratio', 0.8)
        val_ratio = split_config.get('val_ratio', 0.1)
        test_ratio = split_config.get('test_ratio', 0.1)
        random_seed = split_config.get('random_seed', 42)
        
        # First split: separate test set
        train_val, test = train_test_split(
            df, test_size=test_ratio, random_state=random_seed
        )
        
        # Second split: separate train and val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=random_seed
        )
        
        logger.info(f"  Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
        logger.info(f"  Val: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
        
        return {'train': train, 'val': val, 'test': test}
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files"""
        output_config = self.config['output']
        output_type = output_config.get('type', 'csv')
        
        logger.info("Saving processed data...")
        
        # Create output directory
        output_dir = Path(output_config['path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in data_dict.items():
            if split_name == 'all':
                output_path = output_config['path']
            else:
                split_paths = output_config.get('split', {}).get('output_paths', {})
                output_path = split_paths.get(split_name, f'./data/{split_name}_data.csv')
            
            # Save based on type
            if output_type == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif output_type == 'excel':
                df.to_excel(output_path, index=False)
            elif output_type == 'json':
                df.to_json(output_path, orient='records', force_ascii=False)
            elif output_type == 'npy':
                np.save(output_path, df.values)
            elif output_type == 'pkl':
                df.to_pickle(output_path)
            
            logger.info(f"  Saved {split_name}: {output_path} ({len(df)} rows)")
    
    def process(self, data_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Main processing pipeline
        
        Args:
            data_path: Path to input data file (optional, uses config if not provided)
        
        Returns:
            Dictionary of processed DataFrames
        """
        logger.info("=" * 60)
        logger.info("CAE Stress Dataset Preprocessing Started")
        logger.info("=" * 60)
        
        # 1. Read data
        if data_path:
            df = pd.read_csv(data_path)
        else:
            df = self.read_data()
        
        # 2. Filter fields
        df = self.filter_fields(df)
        
        # 3. Handle missing values
        df = self.handle_missing_values(df)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Encode categorical
        df = self.encode_categorical(df)
        
        # 6. Normalize numeric
        df = self.normalize_numeric(df)
        
        # 7. Rename fields
        df = self.rename_fields(df)
        
        # 8. Validate
        self.validate_data(df)
        
        # 9. Split data
        data_dict = self.split_data(df)
        
        # 10. Save
        self.save_data(data_dict)
        
        logger.info("=" * 60)
        logger.info("Preprocessing Completed Successfully!")
        logger.info("=" * 60)
        
        return data_dict
    
    def get_processing_report(self) -> Dict:
        """Get processing statistics report"""
        return {
            'processing_stats': self.processing_stats,
            'scalers': {k: type(v).__name__ for k, v in self.scalers.items()},
            'encoders': {k: type(v).__name__ for k, v in self.label_encoders.items()}
        }


def create_sample_data(output_path: str = './data/raw_cae_data.csv', n_samples: int = 1000):
    """Create sample CAE data for testing"""
    logger.info(f"Creating sample data: {output_path}")
    
    np.random.seed(42)
    
    data = {
        'length_mm': np.random.uniform(50, 250, n_samples),
        'width_mm': np.random.uniform(20, 120, n_samples),
        'thickness_mm': np.random.uniform(2, 12, n_samples),
        'material': np.random.choice(['钢', '铝', '铜'], n_samples),
        'load_N': np.random.uniform(1000, 10000, n_samples),
        'stress_mpa': np.random.uniform(10, 500, n_samples),
        'test_time': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    }
    
    # Introduce some missing values
    mask = np.random.random(n_samples) < 0.02
    data['load_N'][mask] = np.nan
    
    df = pd.DataFrame(data)
    
    # Create directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Sample data created: {n_samples} samples")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAE Stress Dataset Preprocessing')
    parser.add_argument('--input', type=str, help='Input data file path')
    parser.add_argument('--config', type=str, help='OpenSpec config file path')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
    else:
        # Initialize preprocessor
        preprocessor = CAEDataPreprocessor(config_path=args.config)
        
        # Process data
        result = preprocessor.process(data_path=args.input)
        
        # Print report
        report = preprocessor.get_processing_report()
        print("\nProcessing Report:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
