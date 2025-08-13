#!/usr/bin/env python
"""
Robot Dataset Pipeline Configuration Model

This module defines the configuration schema for the robot dataset pipeline using Pydantic.
It provides validation and type checking for all configuration parameters used throughout
the pipeline to convert ROS bag files into structured Parquet datasets.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineConfig(BaseModel):
    """
    Configuration model for the robot dataset pipeline.
    
    This class defines all the configuration parameters needed to run the robot dataset
    conversion pipeline, including input/output paths, data processing options, and 
    Hugging Face integration settings.
    
    All paths are automatically converted to Path objects and validated.
    """

    verbose: bool = Field(False, description="Enable verbose logging for debugging")

    model_config = ConfigDict(validate_default=True)

    # === File Paths ===
    rosbag_folder: Path = Field(
        ..., description="Path to the folder containing ROS bag files"
    )
    parquet_folder: Path = Field(
        ..., description="Path to the folder where Parquet shards will be saved"
    )
    hf_output_dir: Path = Field(
        ..., description="Path to the Hugging Face dataset output directory"
    )

    # === Data Configuration ===
    topics: List[str] = Field(
        ..., description="List of ROS topics to extract data from"
    )
    modalities: List[str] = Field(
        ...,
        description="List of modalities in dataset (e.g., 'obs_pose', 'obs_image', 'act_pose')",
    )
    fps: int = Field(..., gt=0, description="Frames per second for data sampling, must be positive")
    batch_size: int = Field(5, gt=0, description="Number of ROS bags per Parquet shard")
    compression: Optional[str] = Field(
        None, description="Parquet compression codec (e.g. 'lz4', 'zstd', 'snappy')"
    )
    use_streaming: bool = Field(
        False, description="Whether to build an IterableDataset for streaming large datasets"
    )
    curtail_time: list = Field(
        default_factory=lambda: [0, 0],
        description="Time to curtail the rosbag, in seconds [start, end]. Used for debugging or processing only part of a bag",
    )

    # === Dataset Splits ===
    splits: dict = Field(
        default_factory=lambda: {
            "train": "0:80",
            "validation": "80:90", 
            "test": "90:100",
        },
        description="Split ratios for train, validation, and test sets in the format 'start:end' (percentages)",
    )

    # === Point Cloud Configuration ===
    num_points: int = Field(
        88,
        gt=0,
        description="Maximum number of points in point clouds, must be positive",
    )
    n_next_obs: int = Field(
        1,
        ge=0,
        description="Number of next observations to include when computing actions",
    )

    # === Relative Coordinate Configuration ===
    # Used for diffusion policy and other applications that benefit from relative coordinates
    use_relative_coordinates: bool = Field(
        False, 
        description="Whether to use relative coordinates instead of absolute coordinates"
    )
    relative_mode: str = Field(
        "initial", 
        description="Mode for relative coordinates: 'initial' (relative to first pose), 'sequential' (relative to previous pose), 'between_hands' (hands relative to each other)"
    )

    # === ROS Bag Processing ===
    read_from_rosbag: bool = Field(
        False, description="If True, read from ROS bags. If False, skip reading ROS bags and use existing Parquet shards"
    )

    # === Hugging Face Integration ===
    push_to_hub: bool = Field(
        False, description="Whether to push the dataset to Hugging Face Hub"
    )
    private: bool = Field(
        False, description="Whether to make the Hugging Face dataset private"
    )
    repo_id: Optional[str] = Field(
        None, description="Hugging Face repository ID to push the dataset (format: username/dataset_name)"
    )
    hf_token: Optional[str] = Field(
        None, description="Hugging Face authentication token for pushing datasets"
    )

    @field_validator("rosbag_folder", "parquet_folder", "hf_output_dir", mode="before")
    @classmethod
    def _validate_paths(cls, v: str) -> Path:
        """
        Validate and convert string paths to Path objects.
        
        This validator ensures that:
        1. String paths are converted to pathlib.Path objects
        2. Parent directories exist (creates them if they don't)
        
        Args:
            v: Path as string or Path object
            
        Returns:
            Path: Validated Path object
            
        Raises:
            ValueError: If parent directory doesn't exist and can't be created
        """
        p = Path(v)
        # Make sure the parent directory exists
        if not p.parent.exists():
            raise ValueError(f"Parent directory does not exist: {p.parent}")
        return p

    @classmethod
    def load_from_yaml(cls, path: Path) -> "PipelineConfig":
        """
        Load configuration from a YAML file and validate it.
        
        This method loads a YAML configuration file and creates a validated
        PipelineConfig instance. All validation rules defined in the model
        will be applied.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            PipelineConfig: Validated configuration instance
            
        Raises:
            ValidationError: If the configuration is invalid
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
