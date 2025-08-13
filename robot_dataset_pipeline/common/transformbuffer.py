#!/usr/bin/env python
"""
Transform Buffer Module

This module provides a transform buffer for managing coordinate frame
transformations in robotics applications. It stores and retrieves 
transformation matrices between different reference frames.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

from collections import deque

import numpy as np

from .utils import get_invert_H, get_R_to_q, get_tf_msg_to_H


class TransformBuffer:
    """
    A buffer storing transform matrices between frames and allowing lookup of
    arbitrary frame-to-frame transforms via graph search.
    
    This class maintains a graph of coordinate frame transformations and
    provides methods to query transformations between any two frames.
    """

    def __init__(self, zero_frame: str = "world"):
        """
        Initialize the transform buffer.
        
        Args:
            zero_frame: The reference frame for the transformation graph
        """
        # Maps (parent_frame, child_frame) -> 4x4 numpy transformation matrix
        self.tf_buffer: dict[tuple[str, str], np.ndarray] = {}

    def add_transform(self, transform_msg) -> None:
        """
        Add a TransformStamped message into the buffer.

        transform_msg: geometry_msgs.msg.TransformStamped
        """
        parent = transform_msg.header.frame_id
        child = transform_msg.child_frame_id
        H = get_tf_msg_to_H(transform_msg.transform)
        self.tf_buffer[(parent, child)] = H

    def get_transform(self, parent_frame: str, child_frame: str) -> np.ndarray:
        """
        Compute the transform from parent_frame -> child_frame by BFS over the
        stored tf graph, chaining and inverting as needed.
        Returns 4x4 identity if no path found.
        """
        # Short-circuit same-frame
        if parent_frame == child_frame:
            return np.eye(4, dtype=np.float32)

        # BFS queue: (current_frame, accumulated_transform)
        queue = deque([(parent_frame, np.eye(4, dtype=np.float32))])
        visited = {parent_frame}

        while queue:
            curr_frame, curr_H = queue.popleft()
            # Explore neighbors: direct children
            for (p, c), H in self.tf_buffer.items():
                if p == curr_frame and c not in visited:
                    new_H = curr_H @ H
                    if c == child_frame:
                        return new_H
                    visited.add(c)
                    queue.append((c, new_H))
            # Explore inverse edges: direct parents
            for (p, c), H in self.tf_buffer.items():
                if c == curr_frame and p not in visited:
                    H_inv = get_invert_H(H)
                    new_H = curr_H @ H_inv
                    if p == child_frame:
                        return new_H
                    visited.add(p)
                    queue.append((p, new_H))

        # No path found
        raise ValueError(
            f"No transform found from '{parent_frame}' to '{child_frame}'. "
            "Ensure the frames are connected in the TF graph."
        )

    def get_transform_7d(self, parent_frame: str, child_frame: str) -> np.ndarray:
        """
        Compute the transform from parent_frame -> child_frame as a 7D vector
        [x, y, z, qx, qy, qz, qw] where (x, y, z) is the translation and (qx, qy, qz, qw)
        is the quaternion rotation.
        input: parent_frame, child_frame
        returns: [x, y, z, qx, qy, qz, qw]
        """
        H = self.get_transform(parent_frame, child_frame)
        translation = H[:3, 3]
        quaternion = get_R_to_q(H[:3, :3])
        return np.concatenate([translation, quaternion])
    
    def get_transform_9d(self, parent_frame: str, child_frame: str) -> np.ndarray:
        """
        Compute the transform from parent_frame -> child_frame as a 9D vector
        [x, y, z, r11, r21, r31, r12, r22, r32] where (x, y, z) is the translation 
        and the next 6 values are the first two columns of the rotation matrix R.
        input: parent_frame, child_frame
        returns: [x, y, z, r11, r21, r31, r12, r22, r32]
        """
        H = self.get_transform(parent_frame, child_frame)
        translation = H[:3, 3]
        rotation_matrix = H[:3, :3]
        # Take the first two columns of the rotation matrix and flatten them
        first_two_columns = rotation_matrix[:, :2].flatten()
        return np.concatenate([translation, first_two_columns])
    
    def list_frames(self) -> set[str]:
        """
        List all frames in the buffer.
        Returns a set of frame names.
        """
        frames = set()
        for parent, child in self.tf_buffer.keys():
            frames.add(parent)
            frames.add(child)
        return frames