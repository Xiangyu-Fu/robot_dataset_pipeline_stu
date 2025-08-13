#!/usr/bin/env python
"""
Setup script for Robot Dataset Pipeline

Copyright (c) 2025 Xiangyu Fu, Institute of Cognitive Systems, TUM
Licensed under the MIT License
"""

from setuptools import find_packages, setup

setup(
    name="robot_dataset_pipeline",
    version="0.1.0",
    author="Xiangyu Fu",
    author_email="xiangyu.fu@tum.de",
    description="A modular pipeline for converting ROS bags into Parquet shards and Hugging Face Datasets",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xiangyu-Fu/robot_dataset_pipeline_stu",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "PyYAML>=5.4",
        "pydantic>=2.0",
        "tqdm>=4.0",
        
        # Data processing
        "pandas>=1.2",
        "pyarrow>=6.0",
        "numpy>=1.25.2",
        
        # Machine learning frameworks
        "torch>=2.2.1", 
        "torchvision>=0.20.1",
        "datasets>=2.0",
        "safetensors>=0.3.0",
        
        # Configuration management
        "omegaconf>=2.3.0",
        
        # Computer vision and robotics
        "opencv-python>=4.9.0",
        "matplotlib>=3.7.1",
        "scipy>=1.15.1",
    ],
    entry_points={
        "console_scripts": [
            "convert-dataset = robot_dataset_pipeline.convert_cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="robotics dataset ros machine-learning parquet huggingface",
)
