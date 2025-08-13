#!/usr/bin/env python
"""
Robot Dataset Pipeline Command Line Interface

This module provides the main command-line interface for the robot dataset pipeline.
It orchestrates the conversion of ROS bag files to Parquet format and then to
Hugging Face datasets.

The pipeline consists of three main steps:
1. Read ROS bag files and extract relevant topics
2. Serialize the data into compressed Parquet shards
3. Export to Hugging Face dataset format with proper splits

Usage:
    convert-dataset

The configuration is read from config/config.yaml by default.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import logging
from pathlib import Path

from omegaconf import OmegaConf

from pydantic import ValidationError
from tqdm import tqdm

from robot_dataset_pipeline.config_model import PipelineConfig
from robot_dataset_pipeline.exporter import HFDatasetExporter
from robot_dataset_pipeline.reader import RosbagReader
from robot_dataset_pipeline.serializer import ParquetSerializer

def setup_logging(level: int = logging.INFO):
    """
    Configure root logger with appropriate formatting.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """
    Main entry point for the dataset conversion pipeline.
    
    This function:
    1. Sets up logging
    2. Loads and validates configuration from YAML
    3. Instantiates pipeline components (reader, serializer, exporter)
    4. Processes ROS bags to Parquet shards (if enabled)
    5. Builds and saves Hugging Face dataset from Parquet shards
    """
    setup_logging()

    # Load and validate config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        plain_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        cfg = PipelineConfig.model_validate(plain_cfg)
    except ValidationError as e:
        logging.error("Configuration validation failed:\n%s", e)
        return

    # Instantiate pipeline components
    reader = RosbagReader(cfg)
    serializer = ParquetSerializer(cfg)
    exporter = HFDatasetExporter(cfg)

    # Step 1: Read each ROS bag and write a Parquet shard
    if cfg.read_from_rosbag:
        bag_paths = list(Path(cfg.rosbag_folder).glob("*.bag"))
        if not bag_paths:
            logging.warning("No .bag files found in %s", cfg.rosbag_folder)
        else:
            logging.info("Found %d ROS bag files to process", len(bag_paths))
            
        for bag_path in tqdm(bag_paths, desc="Processing ROS bags", unit="bag"):
            try:
                traj = reader.read_one(bag_path)
                serializer.write_shard(traj)
            except Exception as e:
                logging.error("Failed to process %s: %s", bag_path, e)
        
    # Step 2: Build Hugging Face Dataset from all Parquet shards
    exporter.build_and_save()


if __name__ == "__main__":
    main()
