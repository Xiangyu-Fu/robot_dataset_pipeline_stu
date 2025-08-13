#!/usr/bin/env python
"""
Hugging Face Dataset Exporter Module

This module provides functionality to aggregate Parquet shards into Hugging Face
datasets and export them for use in machine learning pipelines. It handles:

- Loading and combining multiple Parquet shards
- Computing global statistics across all data
- Creating train/validation/test splits
- Exporting to Hugging Face dataset format
- Optional uploading to Hugging Face Hub

The exporter supports various data modalities and can handle large datasets
through streaming and sharding mechanisms.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import json
import logging

import numpy as np
import pandas as pd

import pyarrow.parquet as pq
import torch
from datasets import Array2D, Array3D, Dataset, Features, Image, Sequence, Value
from huggingface_hub import HfApi
from huggingface_hub import upload_file
from safetensors.torch import save_file

from .config_model import PipelineConfig

logger = logging.getLogger(__name__)


class HFDatasetExporter:
    """
    Aggregates Parquet shards into a Hugging Face Dataset and saves/pushes it.
    
    This class handles the final stage of the pipeline, converting the intermediate
    Parquet shards into a complete Hugging Face dataset. It provides:
    
    - Global statistics computation across all shards
    - Dataset splitting (train/validation/test)
    - Metadata generation for downstream applications
    - Local saving and optional Hub uploading
    - Episode indexing for trajectory-based learning
    """

    def __init__(self, cfg: PipelineConfig):
        """
        Initialize the Hugging Face dataset exporter.
        
        Args:
            cfg: Pipeline configuration containing output paths and Hub settings
        """
        self.cfg = cfg

    def _write_meta(self):
        """
        Write metadata about the dataset to a JSON file.
        This can include things like topics, version, etc.
        """
        meta_dir = self.cfg.hf_output_dir / self.cfg.repo_id / "meta_data"
        meta_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "robot_type": "unknown",
            "topics": self.cfg.topics,
            "fps": self.cfg.fps,
            "video_path": None,
            "features": {
                "obs_image": {
                    "dtype": "image",
                    "shape": [None, None, 3],
                    "names": ["height", "width", "channel"],
                    "fps": 10.0,
                },
                "episode_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "obs_obs_point_cloud_skin_contact": {
                    "dtype": "float64",
                    "shape": [None, 6],
                    "names": ["x", "y", "z", "r", "g", "b"],
                    "description": "Point cloud data with RGB color values",
                },
                "obs_tactile_map": {
                    "dtype": "int32",
                    "shape": self.cfg.map_output_dim,
                    "description": "Tactile map data with 12 channels",
                },
                "obs_pose_left": {
                    "dtype": "float64",
                    "shape": [None, 9],
                    "names": ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32"],
                },
                "obs_pose_right": {
                    "dtype": "float64",
                    "shape": [None, 9],
                    "names": ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32"],
                },
            },
        }

        with open(meta_dir / "info.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _write_stats(self, ds: Dataset):
        """
        Aggregates per-shard statistics from Parquet files and writes global statistics to a JSON file.
        This function scans all Parquet shard files in the configured directory, reads per-shard statistics
        (mean, standard deviation, min, max) stored in the file metadata, and accumulates them to compute
        global statistics across all shards. The computed global statistics for each column are then saved
        as a JSON file in the output meta directory.1
        Parameters:
            ds (Dataset): The dataset object (not directly used in this function, but kept for interface consistency).
        Side Effects:
            - Writes a 'global_stats.json' file containing aggregated statistics to the meta output directory.
            - Logs warnings if no shards are found and info upon successful write.
        """
        parquet_dir = self.cfg.parquet_folder
        shard_files = sorted(parquet_dir.glob("*.parquet"))
        if not shard_files:
            logger.warning("No parquet shards found for stats aggregation.")
            return

        accum = {}
        total_rows = 0

        for shard in shard_files:
            pf = pq.ParquetFile(str(shard))
            md = pf.metadata.metadata or {}

            n = pf.metadata.num_rows
            total_rows += n

            stats = json.loads(md.get(b"shard_stats", b"{}").decode("utf8"))
            # stats example: {col: {"mean": [...], "std":[...], "min":[...], "max":[...]}, ...}

            for col, s in stats.items():
                if col not in accum:
                    # detect if vector or scalar
                    mu = np.array(s["mean"], dtype=float)
                    sigma = np.array(s["std"], dtype=float)
                    mn = np.array(s["min"], dtype=float)
                    mx = np.array(s["max"], dtype=float)
                    # sum of values = mean * n
                    sum_ = mu * n
                    # sum of squares = (sigma^2 + mean^2) * n
                    sumsq = (sigma**2 + mu**2) * n
                    accum[col] = {
                        "sum": sum_,
                        "sumsq": sumsq,
                        "min": mn,
                        "max": mx,
                    }
                else:
                    mu = np.array(s["mean"], dtype=float)
                    sigma = np.array(s["std"], dtype=float)
                    mn = np.array(s["min"], dtype=float)
                    mx = np.array(s["max"], dtype=float)
                    accum[col]["sum"] += mu * n
                    accum[col]["sumsq"] += (sigma**2 + mu**2) * n
                    accum[col]["min"] = np.minimum(accum[col]["min"], mn)
                    accum[col]["max"] = np.maximum(accum[col]["max"], mx)

        global_stats = {}
        for col, a in accum.items():
            sum_ = a["sum"]
            sumsq = a["sumsq"]
            mn = a["min"]
            mx = a["max"]

            mu_global = sum_ / total_rows
            var_global = (sumsq / total_rows) - (mu_global**2)
            sigma_global = np.sqrt(np.maximum(var_global, 0))

            if mu_global.size == 1:
                global_stats[col] = {
                    "mean": mu_global.item(),
                    "std": sigma_global.item(),
                    "min": mn.item(),
                    "max": mx.item(),
                }
            else:
                if "obs_image" in col:
                    global_stats[col] = {
                        "mean": mu_global.reshape(3, 1, 1).tolist(),
                        "std": sigma_global.reshape(3, 1, 1).tolist(),
                        "min": mn.reshape(3, 1, 1).tolist(),
                        "max": mx.reshape(3, 1, 1).tolist(),
                    }
                else:
                    global_stats[col] = {
                        "mean": mu_global.tolist(),
                        "std": sigma_global.tolist(),
                        "min": mn.tolist(),
                        "max": mx.tolist(),
                    }

        meta_dir = self.cfg.hf_output_dir / self.cfg.repo_id / "meta_data"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(global_stats, f, indent=2)
        logger.info("Wrote global stats to %s", meta_dir / "stats.json")

        tensor_dict: dict[str, torch.Tensor] = {}
        for col, stat in global_stats.items():
            for name, values in stat.items():
                key = f"{col}/{name}"
                tensor_dict[key] = torch.tensor(values, dtype=torch.float32)

        safetensors_path = meta_dir / "stats.safetensors"
        save_file(tensor_dict, str(safetensors_path))
        logger.info("Wrote global stats to %s", safetensors_path)

    def build_and_save(self):
        """
        Load Parquet shards, compute metadata & global stats, split train/val/test,
        save locally and/or push to Hugging Face Hub.
        """
        pattern = str(self.cfg.parquet_folder / "*.parquet")
        logger.info("Loading Parquet shards from %s", pattern)


        # import pandas as pd
        # df = pd.read_parquet(str(self.cfg.parquet_folder / "shard_0000.parquet"))
        # print(df.dtypes)
        # print(df.iloc[0])  # sample example to inspect structure

        # 1) Load full dataset
        ds = Dataset.from_parquet(
            pattern,
            features=Features(
                {
                    "timestamp": Value("float64"),
                    "episode_index": Value("int64"),
                    "obs_point_cloud_skin_contact": Array2D((None, 6), dtype="float64"),
                    "mask_point_cloud_skin_contact": Sequence(Value("bool")),
                    "obs_pose_left": Sequence(Value("float64")),
                    "obs_pose_right": Sequence(Value("float64")),
                    "obs_tactile_map": Array2D((None, 6), dtype="float64") if "/tactile_map_pc2" in self.cfg.topics else Array3D(tuple(self.cfg.map_output_dim), dtype="int8"),
                    "act_pose_left": Sequence(Value("float64")),
                    "act_pose_right": Sequence(Value("float64")),
                }
            ),
        )
        logger.info("Built dataset with %d examples", len(ds))

        # 2) Static metadata + global stats
        if self.cfg.hf_output_dir:
            self._write_meta()  # info.json
            self._write_stats(ds)  # global_stats.json

        # 3) Save episode_data_index
        if "episode_index" in ds.column_names:
            ep_ids = np.array(ds["episode_index"])
            unique_ids, first_pos = np.unique(ep_ids, return_index=True)

            order = np.argsort(first_pos)
            unique_ids = unique_ids[order]
            first_pos = first_pos[order]
            last_pos = np.r_[first_pos[1:] - 1, len(ep_ids) - 1]

            epi_dict = {
                "episode_id": torch.tensor(unique_ids, dtype=torch.long),
                "from":       torch.tensor(first_pos,  dtype=torch.long),
                "to":         torch.tensor(last_pos,    dtype=torch.long),
            }

            meta_dir = self.cfg.hf_output_dir / self.cfg.repo_id / "meta_data"
            meta_dir.mkdir(parents=True, exist_ok=True)
            epi_path = meta_dir / "episode_data_index.safetensors"
            save_file(epi_dict, str(epi_path))
            logger.info("Wrote episode_data_index to %s", epi_path)

            if self.cfg.push_to_hub:
                upload_file(
                    path_or_fileobj=str(epi_path),
                    path_in_repo="meta/episode_data_index.safetensors",
                    repo_id=self.cfg.repo_id,
                    repo_type="dataset",
                    revision="main",
                )
                logger.info("Uploaded episode_data_index to Hub")

        # 3) Split according to percentages in cfg.splits
        splits = getattr(self.cfg, "splits", {"train": "0:100"})
        total = len(ds)
        ds_splits = {}
        for name, pct_range in splits.items():
            start_pct, end_pct = map(int, pct_range.split(":"))
            start_idx = int(start_pct / 100 * total)
            end_idx = int(end_pct / 100 * total)
            indices = list(range(start_idx, end_idx))
            ds_splits[name] = ds.select(indices)
            logger.info("Split %s: %d examples", name, len(indices))

        # 4) Save locally
        if self.cfg.hf_output_dir:
            for name, split_ds in ds_splits.items():
                out_dir = self.cfg.hf_output_dir / self.cfg.repo_id / name
                split_ds.save_to_disk(str(out_dir), max_shard_size="100MB")
                logger.info("Saved split '%s' to %s", name, out_dir)
        logger.info(
            "Dataset splits saved to %s/%s", self.cfg.hf_output_dir, self.cfg.repo_id
        )

        # 5) Push to Hub
        if self.cfg.push_to_hub:
            api = HfApi()
            for name, split_ds in ds_splits.items():
                split_ds.push_to_hub(
                    repo_id=self.cfg.repo_id,
                    split=name,
                    private=self.cfg.private,
                    max_shard_size="100MB",
                    revision="main",
                )
                logger.info("Pushed split '%s' to Hub", name)

            # 6) Upload meta/ folder
            meta_dir = self.cfg.hf_output_dir / self.cfg.repo_id / "meta_data"
            api.upload_folder(
                folder_path=str(meta_dir),
                path_in_repo="meta",
                repo_id=self.cfg.repo_id,
                repo_type="dataset",
                revision="main",
            )
            logger.info("Uploaded metadata directory to Hub/meta_data")
