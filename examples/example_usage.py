#!/usr/bin/env python
"""
Example Usage of Robot Dataset Pipeline

This script demonstrates how to use the robot dataset pipeline
programmatically instead of using the command line interface.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import logging
from pathlib import Path

from robot_dataset_pipeline import (
    PipelineConfig,
    RosbagReader,
    ParquetSerializer,
    HFDatasetExporter
)

def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

def main():
    """
    Example usage of the robot dataset pipeline.
    
    This demonstrates:
    1. Loading configuration from YAML
    2. Processing ROS bags individually
    3. Creating Parquet shards
    4. Building Hugging Face dataset
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error("Configuration file not found. Please copy config/example_config.yaml to config/config.yaml")
        return
    
    try:
        cfg = PipelineConfig.load_from_yaml(config_path)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Initialize components
    reader = RosbagReader(cfg)
    serializer = ParquetSerializer(cfg)
    exporter = HFDatasetExporter(cfg)
    
    # Process ROS bags if enabled
    if cfg.read_from_rosbag:
        bag_files = list(Path(cfg.rosbag_folder).glob("*.bag"))
        
        if not bag_files:
            logger.warning(f"No .bag files found in {cfg.rosbag_folder}")
        else:
            logger.info(f"Found {len(bag_files)} ROS bag files to process")
            
            # Process each bag file
            for i, bag_path in enumerate(bag_files):
                try:
                    logger.info(f"Processing bag {i+1}/{len(bag_files)}: {bag_path.name}")
                    
                    # Read trajectory data from bag
                    trajectory = reader.read_one(bag_path)
                    
                    # Write to Parquet shard
                    serializer.write_shard(trajectory)
                    
                    logger.info(f"Successfully processed {bag_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {bag_path}: {e}")
                    continue
    else:
        logger.info("Skipping ROS bag reading (read_from_rosbag=False)")
    
    # Build and export Hugging Face dataset
    try:
        logger.info("Building Hugging Face dataset from Parquet shards...")
        exporter.build_and_save()
        logger.info("Dataset export completed successfully!")
        
        # Print dataset location
        dataset_path = cfg.hf_output_dir / (cfg.repo_id or "dataset")
        logger.info(f"Dataset saved to: {dataset_path}")
        
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")

def example_load_dataset():
    """
    Example of how to load and use the generated dataset.
    """
    try:
        from datasets import load_from_disk
        
        # Load the dataset (adjust path as needed)
        dataset_path = "path/to/your/hf_output_dir/your_dataset_name/train"
        
        if Path(dataset_path).exists():
            dataset = load_from_disk(dataset_path)
            print(f"Dataset loaded: {len(dataset)} samples")
            print(f"Features: {list(dataset.features.keys())}")
            
            # Print first sample info
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"First sample keys: {list(sample.keys())}")
        else:
            print(f"Dataset not found at {dataset_path}")
            
    except ImportError:
        print("datasets library not available. Install with: pip install datasets")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    print("Robot Dataset Pipeline Example")
    print("=" * 40)
    
    # Run the main pipeline
    main()
    
    print("\n" + "=" * 40)
    print("Example: Loading the generated dataset")
    example_load_dataset()
    
    print("\n" + "=" * 40)
    print("Done! Check the output directories for your processed data.")
