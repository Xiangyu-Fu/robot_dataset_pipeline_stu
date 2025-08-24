#!/usr/bin/env python
"""
ROS Bag Reader Module

This module provides functionality to read ROS bag files and extract sensor data
as Python dictionaries. It handles various sensor modalities including:
- Images (camera feeds)  
- Point clouds
- Pose/transform data
- Tactile information

The reader supports:
- TF (transform) buffer management for pose extraction
- Point cloud filtering and downsampling
- Relative coordinate transformations
- Data alignment and temporal synchronization

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm

from .config_model import PipelineConfig
from .common.transformbuffer import TransformBuffer
from .common.utils import transform_poses
from .common.relative_transform import (
    compute_relative_poses_to_initial,
    compute_relative_poses_sequential, 
    compute_relative_poses_between_hands
)

logger = logging.getLogger(__name__)


class RosbagReader:
    """
    Reads ROS bag files and extracts sensor data as Python dictionaries.
    
    This class handles the reading and processing of ROS bag files, extracting
    various sensor modalities including images, point clouds, and pose data.
    It supports:
    
    - Transform (TF) buffer management for coordinate frame transformations
    - Point cloud downsampling to a fixed number of points
    - Image encoding to PNG format for storage
    - Relative coordinate transformations for diffusion policy applications
    - Temporal alignment and synchronization of different data streams
    
    The reader processes topics specified in the configuration and outputs
    a structured dictionary suitable for further processing by the serializer.
    """

    def __init__(self, cfg: PipelineConfig):
        """
        Initialize the ROS bag reader.
        
        Args:
            cfg: Pipeline configuration containing topics, modalities, and processing options
        """
        self.cfg = cfg
        self.bridge = CvBridge()  # For converting ROS image messages to OpenCV format

    def _tf_cache_path(self, bag_path: Path) -> Path:
        """
        Define where to save/load the TF buffer data for caching.
        
        Args:
            bag_path: Path to the ROS bag file
            
        Returns:
            Path: Path where TF buffer cache should be stored
        """
        return bag_path.parent / "tf_buffer.pkl"

    def fix_num_points_with_mask(self, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fix the number of points in a point cloud to a constant value using random sampling.
        
        This method ensures all point clouds have exactly the same number of points,
        which is required for batching in machine learning applications.
        
        Args:
            pc: Point cloud array of shape (M, C) where M is number of points, C is feature dimension
            
        Returns:
            tuple: (data, mask) where:
                - data: Point cloud array of shape (N, C) with exactly N points
                - mask: Boolean mask of shape (N,) indicating which points are valid
        """
        M, C = pc.shape
        N = self.cfg.num_points
        if M >= N:
            # More points than needed: randomly sample N points
            idx = np.random.choice(M, N, replace=False)
            data = pc[idx]
            mask = np.ones(N, dtype=bool)
        else:
            # Fewer points than needed: pad with zeros and update mask
            pad = np.zeros((N - M, C), dtype=pc.dtype)
            data = np.vstack([pc, pad])
            mask = np.concatenate([np.ones(M, dtype=bool), np.zeros(N - M, dtype=bool)])
        return data, mask

    def read_one(self, bag_path: Path) -> Dict[str, Any]:
        """
        Read a single ROS bag and return dict of topic -> list of (timestamp, data).
        
        This method processes a single ROS bag file and extracts data from the specified
        topics. It handles:
        
        1. TF (transform) data for pose estimation
        2. Image data (converted to PNG bytes for storage)
        3. Point cloud data (downsampled and masked)
        4. Temporal curtailment for debugging
        5. Data alignment and transformation
        6. Optional relative coordinate transformation
        
        Args:
            bag_path: Path to the ROS bag file to process
            
        Returns:
            Dict containing extracted and processed sensor data, with keys:
            - 'timestamp': List of aligned timestamps
            - modality keys (e.g., 'obs_image', 'obs_pose'): Lists of data
        """
        self.tf_buffer = TransformBuffer()

        tqdm.write(f"Reading rosbag: {bag_path}")
        bag = rosbag.Bag(str(bag_path), skip_index=False)
        observations = {t: [] for t in self.cfg.modalities}
        
        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        curtail_time = self.cfg.curtail_time  # [start, end] time to curtail the bag for debugging

        # Debugging options - can be used to process only a subset of the bag
        debug_bag = False
        debug_time = 2
        if debug_bag:
            tqdm.write("\033[93m⚠️ WARNING: Debug mode is enabled! Only a subset of the bag will be processed.\033[0m")

        # First pass: Build TF buffer with all transform data
        for topic, msg, _ in bag.read_messages(topics=["/tf", "/tf_static"]):
            for transform in msg.transforms:
                self.tf_buffer.add_transform(transform)

        if self.cfg.verbose:
            logger.info("Available topics: %s", bag.get_type_and_topic_info().topics.keys())
            logger.info("Reading topics: %s", self.cfg.topics)
            logger.info("Observations: %s", observations.keys())

        # Second pass: Extract data from specified topics
        for topic, msg, t in bag.read_messages(topics=self.cfg.topics):
            ts = t.to_sec()

            # Apply temporal curtailment if specified (for debugging)
            if not debug_bag and (ts < bag_start_time + curtail_time[0] or ts > bag_end_time - curtail_time[1]):
                continue

            # Process TF messages and extract pose data
            elif "tf" in topic:
                for transform in msg.transforms:
                    self.tf_buffer.add_transform(transform)
                    # Example: Extract gripper pose when specific frame is detected
                    if "6115" in transform.child_frame_id:
                        pose_left = self.tf_buffer.get_transform_9d(
                            "world", "gripper_link"
                        )
                        observations["obs_pose"].append((ts, pose_left))

            # Process camera image data
            if topic == "/camera/color/image_raw":
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                success, encoded_img = cv2.imencode(".png", img)
                if not success:
                    logger.error("Failed to encode image at timestamp %s", ts)
                    continue
                png_bytes = encoded_img.tobytes()
                observations["obs_image"].append((ts, png_bytes))

            # Exit early in debug mode
            if debug_bag is True and t.to_sec() > bag_start_time + debug_time:
                break
            
        # Align, transform, and downsample the extracted data
        obs_transformed = transform_poses(
            observations, fps=self.cfg.fps, n_next_obs=self.cfg.n_next_obs
        )

        # Apply relative coordinate transformation if enabled
        if self.cfg.use_relative_coordinates:
            obs_transformed = self._apply_relative_coordinates(obs_transformed)

        return obs_transformed

    def _apply_relative_coordinates(self, observations: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Apply relative coordinate transformation to pose observations and actions.
        
        This method converts absolute pose coordinates to relative coordinates based on
        the specified mode in the configuration. This is particularly useful for
        diffusion policy training where relative coordinates can improve learning.
        
        Supported modes:
        - "initial": All poses relative to the first pose in the sequence
        - "sequential": Each pose relative to the previous pose
        - "between_hands": For dual-arm robots, hands relative to each other
        
        Args:
            observations: Dictionary containing pose data with absolute coordinates
            
        Returns:
            Dict with the same structure but poses converted to relative coordinates
        """
        obs_result = observations.copy()
        
        # Find all pose-related keys (observations and actions)
        pose_keys = [k for k in observations.keys() if k.startswith(('obs_pose_', 'act_pose_'))]
        
        if self.cfg.relative_mode == "initial":
            # Convert to coordinates relative to initial pose
            for key in pose_keys:
                if observations[key] and len(observations[key]) > 0:
                    # Filter out None values
                    valid_poses = [pose for pose in observations[key] if pose is not None]
                    if valid_poses:
                        relative_poses = compute_relative_poses_to_initial(valid_poses, pose_format="9d")
                        obs_result[key] = relative_poses
                        
        elif self.cfg.relative_mode == "sequential":
            # Convert to sequential relative coordinates (current relative to previous)
            for key in pose_keys:
                if observations[key] and len(observations[key]) > 0:
                    valid_poses = [pose for pose in observations[key] if pose is not None]
                    if valid_poses:
                        relative_poses = compute_relative_poses_sequential(valid_poses, pose_format="9d")
                        obs_result[key] = relative_poses
                        
        elif self.cfg.relative_mode == "between_hands":
            # Convert to coordinates relative between hands (for dual-arm robots)
            left_obs_key = "obs_pose_left"
            right_obs_key = "obs_pose_right"
            left_act_key = "act_pose_left" 
            right_act_key = "act_pose_right"
            
            # Process observations
            if (left_obs_key in observations and right_obs_key in observations and
                observations[left_obs_key] and observations[right_obs_key]):
                
                left_poses = [pose for pose in observations[left_obs_key] if pose is not None]
                right_poses = [pose for pose in observations[right_obs_key] if pose is not None]
                
                if len(left_poses) == len(right_poses) and len(left_poses) > 0:
                    left_rel, right_rel = compute_relative_poses_between_hands(
                        left_poses, right_poses, pose_format="9d"
                    )
                    obs_result[left_obs_key] = left_rel
                    obs_result[right_obs_key] = right_rel
            
            # Process actions
            if (left_act_key in observations and right_act_key in observations and
                observations[left_act_key] and observations[right_act_key]):
                
                left_acts = [pose for pose in observations[left_act_key] if pose is not None]
                right_acts = [pose for pose in observations[right_act_key] if pose is not None]
                
                if len(left_acts) == len(right_acts) and len(left_acts) > 0:
                    left_rel, right_rel = compute_relative_poses_between_hands(
                        left_acts, right_acts, pose_format="9d"
                    )
                    obs_result[left_act_key] = left_rel
                    obs_result[right_act_key] = right_rel
        
        if self.cfg.verbose:
            logger.info(f"Applied relative coordinates with mode: {self.cfg.relative_mode}")
        
        return obs_result