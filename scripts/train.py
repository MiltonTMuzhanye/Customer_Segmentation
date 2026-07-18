"""Training script for customer segmentation models"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.customer_segmentation.data.ingestion import DataIngestion
from src.customer_segmentation.data.preprocessing import DataPreprocessor
from src.customer_segmentation.data.validation import DataValidator
from src.customer_segmentation.training.trainer import ModelTrainer
from src.customer_segmentation.utils.logger import get_logger
from src.customer_segmentation.utils.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train customer segmentation models')
    parser.add_argument('--config', type=str, default='configs/model.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data', type=str, default='data/raw/Online Retail.xlsx',
                       help='Path to input data file')
    parser.add_argument('--output', type=str, default='artifacts',
                       help='Path to output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger(__name__)
    config = get_config()
    
    logger.info("Starting training pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Load data
        ingestion = DataIngestion()
        df = ingestion.load_excel(args.data)
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate data
        validator = DataValidator()
        data_config = config.get_config('data')
        validator.validate_all(df, data_config.get('validation', {}))
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess(df)
        logger.info(f"Preprocessed data: {len(df_processed)} rows")
        
        # Save processed data
        processed_path = Path('data/processed')
        processed_path.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(processed_path / 'preprocessed_data.csv', index=False)
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.run_training_pipeline(df_processed)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        logger.info(f"Number of clusters: {results['n_clusters']}")
        logger.info(f"Metrics: {results['results'][results['best_model']]}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()