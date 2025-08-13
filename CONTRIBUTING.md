# Contributing to Robot Dataset Pipeline

<!--
Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
-->

Thank you for your interest in contributing to the Robot Dataset Pipeline! This project is designed for educational and research purposes, and we welcome contributions from students and researchers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic knowledge of ROS and robotics
- Familiarity with Python and data processing

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/robot_dataset_pipeline_stu.git
   cd robot_dataset_pipeline
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e .
   pip install pre-commit pytest black flake8
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Development Guidelines

### Code Style

- Use **Black** for code formatting
- Follow **PEP 8** guidelines
- Add **type hints** for all functions
- Write **comprehensive docstrings**

Example:
```python
def process_trajectory(
    trajectory: Dict[str, Any], 
    fps: int = 10
) -> Dict[str, List[Any]]:
    """
    Process a single trajectory with temporal alignment.
    
    Args:
        trajectory: Raw trajectory data from ROS bag
        fps: Target sampling frequency in Hz
        
    Returns:
        Processed trajectory with aligned timestamps
        
    Raises:
        ValueError: If trajectory format is invalid
    """
    # Implementation here
    pass
```

### Project Structure

When adding new features, follow this structure:

```
robot_dataset_pipeline/
├── config_model.py          # Configuration schema
├── reader.py                # Data reading (ROS bags, etc.)
├── serializer.py            # Data serialization (Parquet)
├── exporter.py              # Dataset export (Hugging Face)
├── convert_cli.py           # Command line interface
└── common/                  # Utility modules
    ├── utils.py             # General utilities
    ├── transformbuffer.py   # Transform handling
    └── relative_transform.py # Coordinate transformations
```

## Types of Contributions

### 1. Bug Fixes
- Fix data processing errors
- Improve error handling
- Resolve configuration issues

### 2. New Features
- Support for new sensor types
- Additional export formats
- New coordinate transformation modes
- Enhanced preprocessing options

### 3. Documentation
- Improve code comments
- Add usage examples
- Update configuration guides
- Write tutorials

### 4. Testing
- Add unit tests
- Create integration tests
- Test with different robot data

## How to Contribute

### 1. Choose an Issue
- Check existing [issues](https://github.com/Xiangyu-Fu/robot_dataset_pipeline_stu/issues)
- Look for `good first issue` labels
- Or propose new features

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes
- Write clean, documented code
- Follow the coding standards
- Add tests if applicable

### 4. Test Your Changes
```bash
# Run existing tests
python -m pytest tests/

# Test with sample data
python examples/example_usage.py

# Check code style
black robot_dataset_pipeline/
flake8 robot_dataset_pipeline/
```

### 5. Commit Changes
```bash
git add .
git commit -m "feat: add support for tactile sensor data"
# or
git commit -m "fix: handle empty point clouds gracefully"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code refactoring

### 6. Submit Pull Request
- Push to your fork
- Create a pull request
- Describe your changes clearly
- Link any related issues

## Specific Contribution Areas

### Adding New Sensor Support

To add support for a new sensor type:

1. **Update Configuration Schema** (`config_model.py`)
   ```python
   # Add new modality to the configuration
   modalities: List[str] = Field(
       ...,
       description="List of modalities including 'obs_lidar'"
   )
   ```

2. **Extend Reader** (`reader.py`)
   ```python
   def read_one(self, bag_path: Path) -> Dict[str, Any]:
       # Add handling for your new topic
       if topic == "/velodyne_points":
           lidar_data = self.process_lidar(msg)
           observations["obs_lidar"].append((ts, lidar_data))
   ```

3. **Update Serializer** (`serializer.py`)
   ```python
   def write_shard(self, traj: dict):
       # Add statistics computation for new data type
       if col == "obs_lidar":
           # Compute statistics for LiDAR data
           pass
   ```

4. **Update Exporter** (`exporter.py`)
   ```python
   features = Features({
       # Add feature definition for new modality
       "obs_lidar": Array2D((None, 4), dtype="float32"),
   })
   ```

### Adding New Coordinate Transformations

To add a new relative coordinate mode:

1. **Extend relative_transform.py**
   ```python
   def compute_relative_poses_custom(
       pose_sequence: List[np.ndarray],
       pose_format: str = "9d"
   ) -> List[np.ndarray]:
       """Your custom transformation logic."""
       pass
   ```

2. **Update Reader**
   ```python
   elif self.cfg.relative_mode == "custom":
       relative_poses = compute_relative_poses_custom(valid_poses, pose_format="9d")
   ```

3. **Update Configuration**
   ```python
   relative_mode: str = Field(
       "initial",
       description="Mode: 'initial', 'sequential', 'between_hands', 'custom'"
   )
   ```

## Testing Guidelines

### Writing Tests

Create tests in the `tests/` directory:

```python
import pytest
from robot_dataset_pipeline import PipelineConfig

def test_config_loading():
    """Test configuration loading from YAML."""
    config = PipelineConfig.load_from_yaml("tests/fixtures/test_config.yaml")
    assert config.fps == 10
    assert "obs_image" in config.modalities
```

### Test Data

- Use small, synthetic test data
- Don't commit large ROS bag files
- Create minimal examples for testing

## Documentation Guidelines

### Code Documentation

- Every class and function needs a docstring
- Include parameter types and descriptions
- Provide usage examples

### User Documentation

- Update README.md for new features
- Add configuration examples
- Create tutorials for complex features

## Review Process

1. **Automated Checks**
   - Code style (Black, Flake8)
   - Tests must pass
   - No linting errors

2. **Manual Review**
   - Code quality assessment
   - Documentation review
   - Testing with real data

3. **Approval**
   - Maintainer approval required
   - Address review comments
   - Final testing

## Getting Help

- **Questions**: Open a GitHub issue with `question` label
- **Discussions**: Use GitHub Discussions for ideas
- **Email**: Contact xiangyu.fu@tum.de for research collaboration

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and improve
- Maintain professional communication

## Recognition

Contributors will be acknowledged in:
- Repository contributors list
- Release notes for significant contributions
- Academic papers when appropriate

Thank you for contributing to robotics research and education! 🤖

---

**Institute of Cognitive Systems, Technical University of Munich (TUM)**
