# Robot Dataset Pipeline - Student Guide

<!--
Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
-->

This guide will help you understand and use the Robot Dataset Pipeline to convert your ROS bag files into machine learning-ready datasets.

## Overview

The Robot Dataset Pipeline is a tool that converts robot data from ROS bags into structured datasets that can be used for machine learning research. It's particularly useful for:

- Imitation learning and behavior cloning
- Diffusion policy training
- Multi-modal robot learning
- Dataset standardization and sharing

## Quick Start for Students

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Xiangyu-Fu/robot_dataset_pipeline_stu.git
cd robot_dataset_pipeline

# Create a virtual environment (Or Conda env)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### 2. Prepare Your Configuration

```bash
# Copy the example configuration
cp config/example_config.yaml config/config.yaml

# Edit the configuration file
nano config/config.yaml  # or use your preferred editor
```

**Important**: You must modify these paths in `config.yaml`:
- `rosbag_folder`: Path to your ROS bag files
- `parquet_folder`: Where to save intermediate files
- `hf_output_dir`: Where to save the final dataset
- `topics`: ROS topics that contain your robot's sensor data

### 3. Run the Pipeline

```bash
convert-dataset
```

The pipeline will:
1. Read all `.bag` files from your specified folder
2. Extract the configured topics
3. Create compressed Parquet files
4. Build a Hugging Face dataset with train/val/test splits

## Understanding the Configuration

### Essential Settings

```yaml
# File paths - CHANGE THESE
rosbag_folder: "/path/to/your/bags"
parquet_folder: "/path/to/output/parquet" 
hf_output_dir: "/path/to/output/dataset"

# Topics - MODIFY FOR YOUR ROBOT
topics:
  - "/tf"  # Always include for pose data
  - "/camera/color/image_raw"  # Your camera topic
  - "/joint_states"  # Your robot's joint states
  
# Modalities - WHAT DATA YOU WANT
modalities:
  - "obs_pose"    # Robot poses
  - "obs_image"   # Camera images
  - "act_pose"    # Action commands
```

### Common Robot Configurations

#### For Arm Manipulation:
```yaml
topics:
  - "/tf"
  - "/joint_states"
  - "/camera/color/image_raw"
  - "/gripper/command"
  - "/end_effector_pose"
```

#### For Mobile Robots:
```yaml
topics:
  - "/tf"
  - "/cmd_vel"
  - "/scan"
  - "/odom"
  - "/camera/image_raw"
```

## Understanding the Output

After running the pipeline, you'll have:

### 1. Parquet Shards
- Location: `parquet_folder/shard_XXXX.parquet`
- Compressed intermediate files
- One file per ROS bag (episode)

### 2. Hugging Face Dataset
- Location: `hf_output_dir/your_dataset_name/`
- Contains `train/`, `validation/`, `test/` folders
- Ready for machine learning frameworks

### 3. Metadata Files
- `meta_data/stats.json`: Dataset statistics for normalization
- `meta_data/info.json`: Dataset information
- `meta_data/episode_data_index.safetensors`: Episode boundaries

## Common Use Cases

### 1. Imitation Learning
Configure to extract:
- Observations: Camera images, robot poses
- Actions: Joint commands, gripper states

### 2. Diffusion Policy
Enable relative coordinates:
```yaml
use_relative_coordinates: true
relative_mode: "sequential"
```

### 3. Multi-modal Learning
Include multiple sensor types:
```yaml
modalities:
  - "obs_image"
  - "obs_point_cloud"
  - "obs_pose"
  - "obs_tactile"
```

## Troubleshooting

### Issue: "No .bag files found"
- Check your `rosbag_folder` path
- Ensure ROS bag files have `.bag` extension

### Issue: "Topic not found"
- Use `rosbag info your_file.bag` to see available topics
- Update your `topics` list in the configuration

### Issue: "Memory error"
- Reduce `batch_size` in configuration
- Use `compression: "zstd"` for better compression
- Process fewer bags at once

### Issue: "Transform lookup failed"
- Ensure `/tf` topic is included
- Check that your robot publishes transform data

## Best Practices

1. **Start Small**: Test with 1-2 bag files first
2. **Check Topics**: Use `rosbag info` to verify topic names
3. **Monitor Resources**: Large datasets need sufficient disk space
4. **Version Control**: Keep your configuration files in git
5. **Documentation**: Document your dataset's specifics

## Advanced Features

### Relative Coordinates
For diffusion policy and other applications:
```yaml
use_relative_coordinates: true
relative_mode: "sequential"  # or "initial", "between_hands"
```

### Custom Statistics
The pipeline automatically computes:
- Mean and standard deviation for normalization
- Min/max values for clipping
- Point cloud statistics
- Image statistics

### Streaming Support
For very large datasets:
```yaml
use_streaming: true
```

## Extending the Pipeline

You can extend the pipeline by:

1. **Adding new data types**: Modify `reader.py`
2. **Custom preprocessing**: Edit `serializer.py`  
3. **New export formats**: Extend `exporter.py`
4. **Additional topics**: Update configuration schema

## Getting Help

1. Check the error logs for specific issues
2. Verify your ROS bag files are not corrupted
3. Test with the provided example configuration
4. Consult the main README for detailed API documentation

## Example Workflow

```bash
# 1. Record some data with your robot
rosbag record -a  # Record all topics

# 2. Check what's in your bag
rosbag info your_recording.bag

# 3. Configure the pipeline
cp config/example_config.yaml config/config.yaml
# Edit config.yaml with your paths and topics

# 4. Run the conversion
convert-dataset

# 5. Use your dataset
python -c "
from datasets import load_from_disk
ds = load_from_disk('path/to/your/hf_output_dir/dataset_name/train')
print(ds[0])  # First sample
"
```

Good luck with your robotics research! 🤖

---
*This pipeline is designed for educational and research purposes at TUM's Institute of Cognitive Systems.*
