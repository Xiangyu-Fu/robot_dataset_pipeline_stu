
# Robot Dataset Pipeline

<!--
Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
-->

A modular pipeline for converting ROS bag files into ML-ready datasets with Parquet shards and Hugging Face Dataset integration.

```
ROS Bags → Parquet Shards → Hugging Face Dataset
```

## Installation

> **Why RoboStack?** This pipeline uses ROS packages (`rosbag`, `cv_bridge`, `tf2_ros`) to read and process ROS bag files. These packages are not available on PyPI. ROS Noetic officially only supports Ubuntu 20.04, so **if you're using any other OS (Ubuntu 22.04+, macOS, Windows)**, RoboStack is the only way to install ROS packages via conda/mamba without building from source.

### With RoboStack (Required for ROS bag processing)

```bash
# Create environment with Python 3.10
conda create -n ros_env python=3.10
conda activate ros_env

# Install ROS packages via RoboStack
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda install ros-noetic-rosbag ros-noetic-cv-bridge ros-noetic-tf2-ros

# Install pipeline
git clone https://github.com/Xiangyu-Fu/robot_dataset_pipeline_stu.git
cd robot_dataset_pipeline_stu
pip install -e .
```

## Configuration

Key settings in `config/config.yaml`:

```yaml
# Paths
rosbag_folder: "/path/to/rosbags"
parquet_folder: "/path/to/output/parquet"
hf_output_dir: "/path/to/output/hf_dataset"

# Processing
fps: 10                          # Sampling rate
compression: "zstd"              # Compression codec
topics: ["/camera/image", "/tf"] # ROS topics to extract
modalities: ["obs_image", "obs_pose"]

# Dataset splits
splits:
  train: "0:80"
  validation: "80:90"
  test: "90:100"

# Optional: Relative coordinates for diffusion policies
use_relative_coordinates: true
relative_mode: "sequential"      # "initial", "sequential", "between_hands"
```

## Output

The pipeline generates:
- **Parquet shards**: Compressed trajectory data with metadata
- **HF Dataset**: Train/val/test splits with automatic indexing
- **Statistics**: Mean, std, min, max for normalization

## Project Structure

```
robot_dataset_pipeline/
├── config/
│   ├── config.yaml              # Main configuration
│   └── example_config.yaml
├── robot_dataset_pipeline/
│   ├── config_model.py          # Configuration schema
│   ├── convert_cli.py           # CLI entry point
│   ├── reader.py                # ROS bag reader
│   ├── serializer.py            # Parquet serializer
│   ├── exporter.py              # HF dataset exporter
│   └── common/                  # Utilities
└── pyproject.toml
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This pipeline requires ROS packages (`rosbag`, `cv_bridge`, `tf2_ros`) which are only available through RoboStack, not PyPI.