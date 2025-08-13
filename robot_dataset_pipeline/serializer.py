#!/usr/bin/env python
"""
Parquet Serializer Module

This module provides functionality to serialize individual trajectory dictionaries
into compressed Parquet shards. It handles various data types including:
- Numerical data (poses, transforms)
- Image data (PNG encoded)
- Point clouds with masks
- Tactile/sensor data

The serializer computes statistics for each shard and stores them as metadata
for later aggregation during dataset export.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import io
import json
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image as PILImage

from .config_model import PipelineConfig

logger = logging.getLogger(__name__)


class ParquetSerializer:
    """
    Serializes individual trajectory dictionaries into Parquet shards.
    
    This class converts trajectory data from Python dictionaries to Parquet format
    for efficient storage and later processing. Each trajectory becomes one shard.
    
    Features:
    - Handles heterogeneous data types (images, poses, point clouds)
    - Computes per-shard statistics (mean, std, min, max)
    - Supports various compression codecs
    - Stores metadata for later aggregation
    - Episode indexing for tracking individual demonstrations
    """

    def __init__(self, cfg: PipelineConfig):
        """
        Initialize the Parquet serializer.
        
        Args:
            cfg: Pipeline configuration containing output directory and compression settings
        """
        self.cfg = cfg
        self.shard_idx = 0
        self.cfg.parquet_folder.mkdir(parents=True, exist_ok=True)

    def write_shard(self, traj: dict):
        """
        Convert one trajectory dictionary to a DataFrame and write to Parquet.
        
        This method processes a single trajectory and serializes it to a Parquet file.
        It performs the following steps:
        
        1. Convert trajectory data to a pandas DataFrame
        2. Compute statistical summaries for numerical and image data
        3. Store statistics as metadata in the Parquet file
        4. Write the data with optional compression
        
        The method supports various data types:
        - Numerical arrays (poses, transforms): Converted to nested lists
        - Image data: Statistical analysis of pixel values
        - Point clouds: Analysis of valid points using masks
        - Tactile data: Voxel-based sensor information
        
        Args:
            traj (dict): Dictionary containing trajectory data with:
                - 'timestamp': List of timestamps
                - topic keys: Lists of corresponding data values
                
        Returns:
            None: Data is written to disk as a Parquet file
            
        Raises:
            AssertionError: If input format is invalid
        """
        assert isinstance(traj, dict), "Input must be a dictionary"
        assert "timestamp" in traj, "Trajectory dict must contain 'timestamp' key"

        # Step 1: Convert to DataFrame format
        dataframe_dict = {}

        ts_list = traj["timestamp"]
        dataframe_dict["timestamp"] = ts_list

        # Convert data to DataFrame-compatible format
        for topic, data_list in traj.items():
            if topic == "timestamp":
                continue
            # Convert numpy arrays to nested lists for Parquet compatibility
            cleaned = [
                v.tolist() if isinstance(v, np.ndarray) else v for v in data_list
            ]
            dataframe_dict[topic] = cleaned

        # Add episode index for tracking individual demonstrations
        dataframe_dict["episode_index"] = [self.shard_idx] * len(ts_list)
        dataframe = pd.DataFrame(dataframe_dict)

        # Step 2: Compute statistical summaries for each column
        stats: dict[str, dict] = {}
        for col, series in dataframe.items():
            # Skip numerical columns (handled differently)
            if np.issubdtype(series.dtype, np.number):
                continue

            first = series.iloc[0]
            
            # Handle array-like data (poses, transforms, etc.)
            if isinstance(first, (list, tuple, np.ndarray)):
                mats = []
                for x in series:
                    if x is None:
                        continue
                    arr = np.asarray(x, dtype=float)
                    if arr.ndim == 1:
                        mats.append(arr)
                    # Uncomment for point cloud statistics per point
                    # elif arr.ndim == 2 and arr.shape[1] == 6:  # point cloud
                    #     for i in range(arr.shape[0]):
                    #         mats.append(arr[i])

                if mats:
                    M = np.stack(mats, axis=0)  # shape (N_samples, dim)
                    stats[str(col)] = {
                        "mean": M.mean(axis=0).tolist(),
                        "std": M.std(axis=0).tolist(),
                        "min": M.min(axis=0).tolist(),
                        "max": M.max(axis=0).tolist(),
                    }

            # Handle image data - compute pixel-level statistics
            if col == "obs_image":
                pixs = []
                for cell in series:
                    img_bytes = None
                    if isinstance(cell, dict):
                        img_bytes = cell.get("bytes")
                    elif isinstance(cell, (bytes, bytearray)):
                        img_bytes = cell
                    if not img_bytes:
                        continue
                    with io.BytesIO(img_bytes) as buf:
                        img = PILImage.open(buf).convert("RGB")
                    arr = np.array(img, dtype=float)  # shape (H, W, 3)
                    h, w, c = arr.shape
                    pixs.append(arr.reshape(h * w, c))

                if pixs:
                    all_pix = np.concatenate(pixs, axis=0)  # (total_pixels, 3)
                    stats[str(col)] = {
                        "mean": all_pix.mean(axis=0).tolist(),
                        "std": all_pix.std(axis=0).tolist(),
                        "min": all_pix.min(axis=0).tolist(),
                        "max": all_pix.max(axis=0).tolist(),
                    }
                continue

            # Handle point cloud data with masks
            if col == "obs_point_cloud_skin_contact":
                mask_series = dataframe["mask_point_cloud_skin_contact"]
                valid_points = []
                for pc, mask in zip(series, mask_series):
                    arr = np.asarray(pc, dtype=float)
                    m = np.asarray(mask, dtype=bool)
                    pts = arr[m]  # Only valid points
                    if pts.size:
                        valid_points.append(pts)
                if not valid_points:
                    continue
                all_pts = np.concatenate(valid_points, axis=0)
                stats[str(col)] = {
                    "mean": all_pts.mean(axis=0).tolist(),
                    "std": all_pts.std(axis=0).tolist(),
                    "min": all_pts.min(axis=0).tolist(),
                    "max": all_pts.max(axis=0).tolist(),
                }
                continue

            # Handle tactile map data
            if col == "obs_tactile_map":
                voxels = []
                for cell in series:
                    if cell is None:
                        continue
                    # Different data types based on topic configuration
                    arr = np.asarray(cell, dtype=float) if "/tactile_map_pc2" in self.cfg.topics else np.asarray(cell, dtype=int)
                    voxels.append(arr.reshape(-1))

                if voxels:                                  
                    stacked = np.concatenate(voxels, axis=0)  # (N_voxels_total,)
                    stats[str(col)] = {
                        "mean": stacked.mean(axis=0).tolist(),
                        "std": stacked.std(axis=0).tolist(),
                        "min": stacked.min(axis=0).tolist(),
                        "max": stacked.max(axis=0).tolist(),
                    }
                continue 

        # Step 3: Create PyArrow table with metadata
        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        meta = table.schema.metadata or {}
        meta = {**meta, b"shard_stats": json.dumps(stats).encode("utf8")}
        table = table.replace_schema_metadata(meta)

        # Step 4: Write to Parquet file with compression
        out_file = self.cfg.parquet_folder / f"shard_{self.shard_idx:04d}.parquet"
        if out_file.exists():
            out_file.unlink()
        pq.write_table(
            table,
            str(out_file),
            compression=self.cfg.compression,  # Use configured compression
        )
        self.shard_idx += 1
