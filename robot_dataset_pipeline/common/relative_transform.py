#!/usr/bin/env python
"""
Relative Transform Module

This module provides functions for computing relative coordinate transformations
from absolute poses. This is particularly useful for diffusion policy applications
where relative coordinates can improve learning performance.

Supported transformation modes:
- Initial: All poses relative to the first pose
- Sequential: Each pose relative to the previous pose  
- Between hands: For dual-arm robots, hands relative to each other

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

import numpy as np
from typing import List, Tuple
from .utils import get_7d_array_pose_to_H, get_invert_H, get_R_to_q


def compute_relative_poses_to_initial(
    pose_sequence: List[np.ndarray], 
    pose_format: str = "9d"
) -> List[np.ndarray]:
    """
    Convert absolute poses to relative poses with respect to the initial pose.
    
    This function converts a sequence of absolute poses to poses relative to
    the first pose in the sequence. This is useful for tasks where the starting
    position varies but the relative motion pattern is consistent.
    
    Args:
        pose_sequence: List of poses in 9D format [x, y, z, r11, r21, r31, r12, r22, r32]
                      or 7D format [x, y, z, qx, qy, qz, qw]
        pose_format: "9d" (rotation matrix) or "7d" (quaternion)
    
    Returns:
        List of relative poses in the same format as input
    """
    if len(pose_sequence) == 0:
        return []
    
    # Convert first pose to transformation matrix as reference
    if pose_format == "9d":
        initial_pose_9d = pose_sequence[0]
        initial_H = pose_9d_to_H(initial_pose_9d)
    else:  # 7d format
        initial_pose_7d = pose_sequence[0]  
        initial_H = get_7d_array_pose_to_H(initial_pose_7d)
    
    initial_H_inv = get_invert_H(initial_H)
    
    relative_poses = []
    for pose in pose_sequence:
        if pose_format == "9d":
            current_H = pose_9d_to_H(pose)
            # Compute relative transformation: T_rel = T_init^-1 * T_current
            relative_H = initial_H_inv @ current_H
            relative_pose = H_to_pose_9d(relative_H)
        else:  # 7d format
            current_H = get_7d_array_pose_to_H(pose)
            relative_H = initial_H_inv @ current_H
            relative_pose = H_to_pose_7d(relative_H)
            
        relative_poses.append(relative_pose)
    
    return relative_poses


def compute_relative_poses_sequential(
    pose_sequence: List[np.ndarray], 
    pose_format: str = "9d"
) -> List[np.ndarray]:
    """
    Convert absolute poses to sequential relative poses (current relative to previous).
    
    Args:
        pose_sequence: List of poses in 9D or 7D format
        pose_format: "9d" or "7d"
    
    Returns:
        List of relative poses, first pose is identity
    """
    if len(pose_sequence) == 0:
        return []
    
    relative_poses = []
    
    # First pose is identity (no relative change)
    if pose_format == "9d":
        identity_9d = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        relative_poses.append(identity_9d)
    else:  # 7d format
        identity_7d = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)  # [0,0,0] + unit quaternion
        relative_poses.append(identity_7d)
    
    # Compute relative transformations for subsequent poses
    for i in range(1, len(pose_sequence)):
        if pose_format == "9d":
            prev_H = pose_9d_to_H(pose_sequence[i-1])
            curr_H = pose_9d_to_H(pose_sequence[i])
            # Relative transformation: T_rel = T_prev^-1 * T_curr
            relative_H = get_invert_H(prev_H) @ curr_H
            relative_pose = H_to_pose_9d(relative_H)
        else:  # 7d format
            prev_H = get_7d_array_pose_to_H(pose_sequence[i-1])
            curr_H = get_7d_array_pose_to_H(pose_sequence[i])
            relative_H = get_invert_H(prev_H) @ curr_H
            relative_pose = H_to_pose_7d(relative_H)
            
        relative_poses.append(relative_pose)
    
    return relative_poses


def compute_relative_poses_between_hands(
    left_poses: List[np.ndarray], 
    right_poses: List[np.ndarray],
    pose_format: str = "9d"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert absolute poses to relative poses between left and right hands.
    Left hand poses become relative to right hand, right hand poses relative to left.
    
    Args:
        left_poses: List of left hand poses
        right_poses: List of right hand poses  
        pose_format: "9d" or "7d"
    
    Returns:
        Tuple of (left_relative_to_right, right_relative_to_left)
    """
    if len(left_poses) != len(right_poses):
        raise ValueError("Left and right pose sequences must have same length")
    
    left_rel_to_right = []
    right_rel_to_left = []
    
    for left_pose, right_pose in zip(left_poses, right_poses):
        if pose_format == "9d":
            left_H = pose_9d_to_H(left_pose)
            right_H = pose_9d_to_H(right_pose)
            
            # Left relative to right: T_left_rel = T_right^-1 * T_left
            left_rel_H = get_invert_H(right_H) @ left_H  
            left_rel_pose = H_to_pose_9d(left_rel_H)
            
            # Right relative to left: T_right_rel = T_left^-1 * T_right
            right_rel_H = get_invert_H(left_H) @ right_H
            right_rel_pose = H_to_pose_9d(right_rel_H)
            
        else:  # 7d format
            left_H = get_7d_array_pose_to_H(left_pose)
            right_H = get_7d_array_pose_to_H(right_pose)
            
            left_rel_H = get_invert_H(right_H) @ left_H
            left_rel_pose = H_to_pose_7d(left_rel_H)
            
            right_rel_H = get_invert_H(left_H) @ right_H  
            right_rel_pose = H_to_pose_7d(right_rel_H)
        
        left_rel_to_right.append(left_rel_pose)
        right_rel_to_left.append(right_rel_pose)
    
    return left_rel_to_right, right_rel_to_left


def pose_9d_to_H(pose_9d: np.ndarray) -> np.ndarray:
    """Convert 9D pose [x, y, z, r11, r21, r31, r12, r22, r32] to 4x4 transform matrix."""
    H = np.eye(4, dtype=np.float32)
    # Translation
    H[:3, 3] = pose_9d[:3]
    
    # Rotation matrix reconstruction from first two columns
    col1 = pose_9d[3:6]  # [r11, r21, r31]
    col2 = pose_9d[6:9]  # [r12, r22, r32]
    
    # Normalize first column
    col1_norm = np.linalg.norm(col1)
    if col1_norm > 1e-8:
        col1 = col1 / col1_norm
    else:
        col1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Make col2 orthogonal to col1 using Gram-Schmidt
    col2_proj = col2 - np.dot(col2, col1) * col1
    col2_norm = np.linalg.norm(col2_proj)
    if col2_norm > 1e-8:
        col2 = col2_proj / col2_norm
    else:
        # If col2 is parallel to col1, create a perpendicular vector
        if abs(col1[0]) < 0.9:
            col2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            col2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        col2 = col2 - np.dot(col2, col1) * col1
        col2 = col2 / np.linalg.norm(col2)
    
    # Third column from cross product (ensures right-handed coordinate system)
    col3 = np.cross(col1, col2)
    
    H[:3, :3] = np.column_stack([col1, col2, col3])
    return H


def H_to_pose_9d(H: np.ndarray) -> np.ndarray:
    """4x4 → 9D (x, y, z, r11, r21, r31, r12, r22, r32)"""
    t = H[:3, 3]
    R = H[:3, :3]
    rot6 = R[:, :2].flatten(order="F") 
    return np.concatenate([t, rot6]).astype(np.float32)



def H_to_pose_7d(H: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform matrix to 7D pose [x, y, z, qx, qy, qz, qw]."""
    translation = H[:3, 3]
    quaternion = get_R_to_q(H[:3, :3])
    return np.concatenate([translation, quaternion]).astype(np.float32)
