#!/usr/bin/env python
"""
Common Utility Functions

This module provides utility functions for the robot dataset pipeline,
including point cloud processing, coordinate transformations, and 
data visualization helpers.

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def decode_pointcloud(msg) -> np.ndarray:
    """
    Convert sensor_msgs/PointCloud2 message to Nx6 numpy array (x,y,z,r,g,b).
    """
    points = []
    for x, y, z, rgb in pc2.read_points(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=False
    ):
        # Decode float RGB into bytes
        rgb_uint32 = int(np.frombuffer(np.float32(rgb).tobytes(), dtype=np.uint32)[0])
        r = (rgb_uint32 >> 16) & 0xFF
        g = (rgb_uint32 >> 8) & 0xFF
        b = rgb_uint32 & 0xFF
        points.append([x, y, z, r, g, b])
    return np.array(points, dtype=np.float32)

def downsample_data(data: list, target_fps: int, original_fps: int) -> list:
    """
    Downsample a list of samples from original_fps to target_fps.
    """
    if original_fps <= target_fps:
        return data
    ratio = original_fps / target_fps
    indices = np.arange(0, len(data), step=ratio).astype(int)
    return [data[i] for i in indices]

# TODO: Check here
def transform_poses(
    observations: Dict[str, List[Tuple[float, Any]]],
    fps: int,
    transform_fns: Dict[str, callable] = None,
    n_next_obs: int = 1,
) -> Dict[str, List[Any]]:
    """
    Align all topic streams to a common set of timestamps at given fps,
    and apply optional geometric transform functions.

    Args:
        observations:
            dict where key is topic name, value is list of (timestamp, data) tuples.
        fps:
            desired frequency (Hz) for alignment.
        transform_fns:
            optional dict mapping topic -> function(data) -> transformed data.

    Returns:
        aligned: dict with keys:
          - "timestamp": list of uniformly spaced timestamps
          - for each topic in observations: list of data aligned to those timestamps
    """
    # 1) collect global start/end
    all_ts = [ts for seq in observations.values() for ts, _ in seq]
    t_start, t_end = min(all_ts), max(all_ts)
    dt = 1.0 / fps

    # generate uniform timestamps (inclusive of start, exclusive of end)
    uniform_ts = np.arange(t_start, t_end, dt)

    aligned: Dict[str, List[Any]] = {}
    aligned["timestamp"] = uniform_ts.tolist()

    # prepare transform functions dict
    transform_fns = transform_fns or {}

    # for each topic, build a fast lookup: separate lists
    for topic, seq in observations.items():
        # if empty, fill with None
        if len(seq) == 0:
            aligned[topic] = [None] * len(uniform_ts)
            continue

        ts_list, data_list = zip(*seq)
        ts_arr = np.array(ts_list)

        aligned_data: List[Any] = []
        for t in uniform_ts:
            # find the index of the closest timestamp
            idx = int(np.abs(ts_arr - t).argmin())
            value = data_list[idx]

            # apply geometric transform if provided
            if topic in transform_fns:
                value = transform_fns[topic](value)

            aligned_data.append(value)

        aligned[topic] = aligned_data

    N = len(uniform_ts)
    for act_topic in [k for k in aligned if k.startswith("act_")]:
        obs_topic = "obs_" + act_topic[len("act_") :]
        if obs_topic not in aligned:
            continue
        obs_seq = aligned[obs_topic]
        shifted = obs_seq[n_next_obs:] # + [None] * n_next_obs
        aligned[act_topic] = shifted
        N = len(shifted)

    # cut obs to fit the action length
    for obs_topic in [k for k in aligned if not k.startswith("act_")]:
        if obs_topic not in aligned:
            continue
        obs_seq = aligned[obs_topic]
        if len(obs_seq) > N:
            aligned[obs_topic] = obs_seq[:N]

    return aligned

def get_tf_msg_to_H(transform) -> np.ndarray:
    """Convert a geometry_msgs/Transform message to a 4x4 transformation matrix."""
    tx, ty, tz = (
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    )
    qx, qy, qz, qw = (
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )

    # Convert quaternion to rotation matrix
    rotation_matrix = get_quat_to_R(qx, qy, qz, qw)

    # Build the transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [tx, ty, tz]

    return T

def get_tf_msg_from_H(H: np.ndarray, parent_frame: str, child_frame: str) -> Any:
    """
    Convert a 4x4 transformation matrix to a geometry_msgs/TransformStamped message.
    """
    from geometry_msgs.msg import TransformStamped, Transform

    transform = Transform()
    transform.translation.x = H[0, 3]
    transform.translation.y = H[1, 3]
    transform.translation.z = H[2, 3]

    # Convert rotation matrix to quaternion
    qx, qy, qz, qw = get_R_to_q(H[:3, :3])
    transform.rotation.x = qx
    transform.rotation.y = qy
    transform.rotation.z = qz
    transform.rotation.w = qw

    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform = transform

    return tf_msg

def get_7d_array_pose_to_H(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 7D pose array [x, y, z, qx, qy, qz, qw] to a 4x4 transformation matrix.
    """
    x, y, z, qx, qy, qz, qw = pose
    T = np.eye(4)
    T[:3, :3] = get_quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T

def get_quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )

def get_invert_H(T):
    """
    Compute the inverse of a 4x4 transformation matrix.
    """
    assert T.shape == (4, 4)
    R = T[:3, :3]  # Rotation part
    t = T[:3, 3]  # Translation part

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def get_R_to_q(matrix) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].

    Args:
        matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: A quaternion (qx, qy, qz, qw).
    """
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return qx, qy, qz, qw

def get_wrench_msg_to_array(msg):
    """
    Convert a geometry_msgs/Wrench message to a numpy array.
    Returns an array of shape (6,) with forces and torques.
    """
    return np.array(
        [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ],
        dtype=np.float32,
    )

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """ """
    # numerator = 2*(qw*qz + qx*qy)
    # denominator = 1 - 2*(qy*qy + qz*qz)
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def get_table_corners(T_rd: np.ndarray, T_lu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the four corners of a table given two diagonal corners in homogeneous coordinates.
    Args:
        T_rd: 4x4 numpy array for the right-down corner (T_rd).
        T_lu: 4x4 numpy array for the left-up corner (T_lu).
    Returns:
        Tuple of four 4x4 numpy arrays representing the corners:
    """
    p_rd = T_rd[:3, 3]
    p_lu = T_lu[:3, 3]
    diag = p_lu - p_rd
    x_rd = T_rd[:3, 0]
    y_rd = T_lu[:3, 1]
    dx = np.dot(diag, x_rd) * x_rd
    dy = np.dot(diag, y_rd) * y_rd

    p_ru = p_rd + dy
    p_ld = p_rd + dx

    def make_T(p, R):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    R_avg = (T_rd[:3, :3] + T_lu[:3, :3]) / 2

    U, _, Vt = np.linalg.svd(R_avg)
    R_ortho = U @ Vt

    T_ru = make_T(p_ru, R_ortho)
    T_ld = make_T(p_ld, R_ortho)

    t_mid = 0.5 * (p_rd + p_lu)
    T_mid = make_T(t_mid, R_ortho)

    return T_mid, T_ru, T_ld

# == Map Visualization Utilities ==
def debug_scatter(grid, res, origin, save: bool = False,):
    """
    grid: (H,W,D) uint8     voxel grid
    res:  voxel size (meters)
    origin: (x_min, y_min, z_min)  world coordinate of voxel (0,0,0)
    """
    idx = np.argwhere(grid > 0)  # only visualize non-UNKNOWN voxels
    if len(idx) == 0:
        print("grid empty"); return
    xyz = idx[:, [2, 1, 0]] * res + origin  # convert k,y,z → x,y,z
    colors = np.zeros((len(idx), 3))
    colors[grid[tuple(idx.T)] == 1] = (0.7, 0.7, 0.7)  # FREE → gray
    colors[grid[tuple(idx.T)] == 2] = (0.2, 0.4, 1.0)  # OCCUPIED → blue
    colors[grid[tuple(idx.T)] == 3] = (1.0, 0.0, 0.0)  # CONTACT → red

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=4, c=colors)

    # # Set manual axis limits
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.2, 0.8)
    # ax.set_zlim(-0.2, 0.8)

    # # Force equal aspect ratio
    # ax.set_box_aspect([1.0, 1.0, 0.4])  # Z is smaller range

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    plt.show()

    if save:
        vis_dir = Path("output/")
        vis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(vis_dir / "occupancy_map.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def debug_voxels(grid: np.ndarray,           # (H, W, D) uint8
                 res: float,                 # voxel size (m)
                 origin: tuple,              # (x_min, y_min, z_min)
                 save: bool = False,
                 save_dir: Path = Path("output/"),
                 fname: str = "occupancy_voxels.png"):
    """
    Voxel state encoding: 0=UNKNOWN, 1=FREE, 2=OCCUPIED, 3=CONTACT
    """
    # ------------------------ basic check ------------------------
    H, W, D = grid.shape
    filled = grid > 0  # (H,W,D) boolean mask of non-empty voxels
    if not filled.any():
        print("grid empty"); return

    # ------------------------ facecolors ------------------------
    # RGBA color array (H,W,D,4)
    colors = np.zeros((*grid.shape, 4), dtype=float)
    colors[grid == 1] = (0.7, 0.7, 0.7, 0.25)    # FREE → semi-transparent gray
    colors[grid == 2] = (0.2, 0.4, 1.0, 0.75)    # OCCUPIED → blue
    colors[grid == 3] = (1.0, 0.0, 0.0, 1.00)    # CONTACT → opaque red

    # matplotlib.voxels expects shape (X, Y, Z) = (D, W, H)
    filled   = np.transpose(filled,  (2, 1, 0))     # (D, W, H)
    facecols = np.transpose(colors,  (2, 1, 0, 3))  # (D, W, H, 4)

    # ------------------------ voxel grid corners ------------------------
    # Compute voxel corner coordinates in world space (meters)
    x = np.arange(D + 1) * res + origin[0]
    y = np.arange(W + 1) * res + origin[1]
    z = np.arange(H + 1) * res + origin[2]
    Xc, Yc, Zc = np.meshgrid(x, y, z, indexing="ij")  # shape: (D+1, W+1, H+1)

    # ------------------------ draw ------------------------
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.voxels(Xc, Yc, Zc,
              filled,
              facecolors=facecols,
              edgecolor="k",         # black grid lines to highlight voxel boundaries
              linewidth=0.1)

    ax.set_box_aspect([D * res, W * res, H * res])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=25, azim=35)  # initial view angle

    # ------------------------ save & show ------------------------
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
