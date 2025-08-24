
# Robot Dataset Pipeline

<!--
Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
-->

A modular data pipeline for converting robot trajectory data (e.g. ROS bags) into structured and compressed Parquet datasets, optionally formatted for [Hugging Face Datasets](https://huggingface.co/docs/datasets/).

This pipeline is designed for robotics researchers and students who need to convert ROS bag files into machine learning-ready datasets. It supports trajectory serialization, transformation, feature extraction, and dataset export with proper train/validation/test splits.

```
ROS Bag (.bag) files → Parquet shards → Hugging Face Dataset
```

## ✨ Features

- **Multiple Data Modalities**: Support for images, point clouds, poses, and sensor data
- **Efficient Storage**: Compressed Parquet format with configurable codecs
- **ML-Ready**: Direct integration with Hugging Face Datasets
- **Flexible Configuration**: YAML-based configuration system
- **Relative Coordinates**: Support for diffusion policy applications
- **Statistics Computation**: Automatic dataset statistics for normalization
- **Error Handling**: Robust processing with detailed logging

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Xiangyu-Fu/robot_dataset_pipeline_stu.git
cd robot_dataset_pipeline

# Install in editable mode
pip install -e .
```

### 2. Configuration

Copy and modify the example configuration:

```bash
cp config/config.yaml config/my_config.yaml
# Edit the paths and settings according to your setup
```

### 3. Run the Pipeline

```bash
convert-dataset
```

The pipeline will:
1. Read ROS bag files from the configured directory
2. Extract specified topics and convert to Parquet shards  
3. Build a Hugging Face dataset with train/validation/test splits
4. Optionally upload to Hugging Face Hub
## Environment Setup

### Option 1: Standard Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv robot_pipeline_env
source robot_pipeline_env/bin/activate  # On Windows: robot_pipeline_env\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: ROS Noetic with RoboStack (conda)

If you need ROS Noetic for bag processing:

```bash
# Install mamba (faster than conda)
conda install mamba -c conda-forge

# Create ROS environment
mamba create -n ros_env
mamba activate ros_env

# Configure channels
conda config --env --add channels conda-forge
conda config --env --remove channels defaults
conda config --env --add channels robostack-noetic

# Install ROS Noetic
mamba install ros-noetic-desktop

# Install build tools
mamba install compilers cmake pkg-config make ninja \
    colcon-common-extensions catkin_tools rosdep

# Install the pipeline
pip install -e .
```

### Pre-commit Hooks (Optional)

For development and code quality:

```bash
pip install pre-commit
pre-commit install
```

## 🛠️ Usage Examples

### Basic Usage

```bash
# Make sure your config.yaml is properly configured
convert-dataset
```

### Python API Usage

```python
from robot_dataset_pipeline.config_model import PipelineConfig
from robot_dataset_pipeline.reader import RosbagReader
from robot_dataset_pipeline.serializer import ParquetSerializer

# Load configuration
cfg = PipelineConfig.load_from_yaml("config/config.yaml")

# Process a single bag
reader = RosbagReader(cfg)
trajectory = reader.read_one("/path/to/your/bag.bag")

# Serialize to Parquet
serializer = ParquetSerializer(cfg)
serializer.write_shard(trajectory)
```

## 🎯 Supported Data Types

| Modality | Description | Example Topics |
|----------|-------------|----------------|
| **Images** | Camera feeds (RGB, depth) | `/camera/color/image_raw` |
| **Poses** | Robot poses, transforms | `/tf`, joint states |
| **Point Clouds** | 3D sensor data | `/velodyne_points`, `/realsense/points` |
| **Tactile** | Touch sensor data | `/tactile_map`, `/skin_contact` |
| **Actions** | Robot commands | Joint commands, gripper states |

## 🔬 Advanced Features

### Relative Coordinate Transformations

For diffusion policy training, convert absolute poses to relative coordinates:

```yaml
use_relative_coordinates: true
relative_mode: "sequential"  # Options: "initial", "sequential", "between_hands"
```

### Statistics Computation

The pipeline automatically computes dataset statistics (mean, std, min, max) for:
- Numerical data normalization
- Image preprocessing
- Point cloud analysis

### Error Handling

- Robust bag processing with error logging
- Graceful handling of corrupted or incomplete data
- Detailed logging for debugging

## 📊 Output Format

The pipeline generates:

1. **Parquet Shards**: Compressed intermediate files with metadata
2. **Hugging Face Dataset**: ML-ready format with train/val/test splits
3. **Statistics Files**: Global dataset statistics for normalization
4. **Episode Index**: Mapping for trajectory-based learning

## 🤝 Contributing

This pipeline is designed for educational and research purposes. When contributing:

1. Follow the existing code structure and documentation style
2. Add type hints and docstrings for new functions
3. Update configuration schema for new parameters
4. Test with sample ROS bag data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📁 Project Structure

```text
robot_dataset_pipeline/
├── config/                       ← Configuration files
│   └── config.yaml              ← Main configuration template
├── robot_dataset_pipeline/      ← Main source code
│   ├── __init__.py
│   ├── config_model.py          ← Pydantic configuration schema
│   ├── convert_cli.py           ← CLI entrypoint
│   ├── exporter.py              ← Exports data to Hugging Face datasets
│   ├── reader.py                ← Reads ROS bag files
│   ├── serializer.py            ← Converts trajectory to Parquet
│   └── common/                  ← Utility modules
│       ├── transformbuffer.py   ← TF transform handling
│       ├── utils.py             ← Helper utilities
│       └── relative_transform.py ← Relative coordinate transformations
├── docs/                        ← Documentation (if available)
├── examples/                    ← Example scripts and notebooks
├── README.md
├── LICENSE
└── setup.py
```

## ⚙️ Configuration

The pipeline is configured via a YAML file (`config/config.yaml`). Here are the key configuration sections:

### Basic Settings

```yaml
verbose: false                    # Enable detailed logging
read_from_rosbag: true           # Process ROS bags (false = use existing Parquet)
fps: 10                          # Sampling frequency (Hz)
compression: "zstd"              # Compression codec ("zstd", "lz4", "snappy")
```

### File Paths

```yaml
rosbag_folder: "/path/to/rosbags"         # Input: ROS bag directory
parquet_folder: "/path/to/parquet"        # Output: Parquet shards directory  
hf_output_dir: "/path/to/hf_datasets"     # Output: Hugging Face dataset directory
```

### Data Configuration

```yaml
topics:                           # ROS topics to extract
  - "/camera/color/image_raw"
  - "/tf"  
  - "/joint_states"

modalities:                       # Expected data modalities
  - "obs_image"
  - "obs_pose"
  - "act_pose"

num_points: 512                   # Max points in point clouds
```

### Dataset Splits

```yaml
splits:
  train: "0:80"                   # 80% for training
  validation: "80:90"             # 10% for validation  
  test: "90:100"                  # 10% for testing
```

### Relative Coordinates (Optional)

For diffusion policy and other applications:

```yaml
use_relative_coordinates: true
relative_mode: "sequential"       # "initial", "sequential", "between_hands"
```

### Hugging Face Integration

```yaml
push_to_hub: false               # Upload to HF Hub
private: true                    # Make dataset private
repo_id: "username/dataset_name" # HF repository ID
```

