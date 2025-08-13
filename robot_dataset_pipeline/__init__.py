#!/usr/bin/env python
"""
Robot Dataset Pipeline

A modular pipeline for converting robot trajectory data from ROS bags 
into structured and compressed Parquet datasets, formatted for machine 
learning applications.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

__version__ = "0.1.0"
__author__ = "Xiangyu Fu"
__email__ = "xiangyu.fu@tum.de"
__license__ = "MIT"

from .config_model import PipelineConfig
from .reader import RosbagReader  
from .serializer import ParquetSerializer
from .exporter import HFDatasetExporter

__all__ = [
    "PipelineConfig",
    "RosbagReader", 
    "ParquetSerializer",
    "HFDatasetExporter",
]