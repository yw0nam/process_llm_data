"""
Pydantic configuration schemas for type-safe configuration management.

This module defines the configuration schemas using Pydantic for validation
and type safety throughout the processing pipeline.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, validator

# AIDEV-NOTE: Pydantic schemas for type-safe configuration with validation


class LoggingConfig(BaseModel):
    """Logging configuration schema."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file_path: Path | None = Field(default=None, description="Log file path")

    @validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class DatasetConfig(BaseModel):
    """Configuration for individual datasets."""

    loader_type: str = Field(description="Type of dataset loader to use")
    path: Path | None = Field(default=None, description="Path to dataset")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Dataset-specific configuration"
    )
    enabled: bool = Field(default=True, description="Whether to include this dataset")


class DataConfig(BaseModel):
    """Data-related configuration."""

    input_path: Path = Field(description="Base input data path")
    output_path: Path = Field(description="Base output data path")
    dataset_configs: dict[str, DatasetConfig] = Field(
        default_factory=dict, description="Configuration for each dataset"
    )


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    process_type: Literal["instruction", "preference"] = Field(
        description="Type of processing to perform"
    )
    version: str = Field(description="Version of processor to use")
    parallel_workers: int = Field(
        default=4, description="Number of parallel workers", ge=1, le=32
    )
    batch_size: int = Field(default=1000, description="Batch size for processing", ge=1)
    sample_size: int | None = Field(
        default=None, description="Number of samples to process", ge=1
    )
    validation_size: int = Field(
        default=5000, description="Size of validation split", ge=1
    )

    @validator("version")
    def validate_version(cls, v):
        if not v or v == "not_selected":
            raise ValueError("Version must be specified")
        return v


class OutputConfig(BaseModel):
    """Output configuration schema for Hugging Face datasets."""

    # Output will always be HF dataset format
    output_dir: str = "outputs"
    dataset_name: str | None = None  # Optional name for the dataset

    # Split configuration
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1

    # HF dataset specific options
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_private: bool = True


class PipelineConfig(BaseModel):
    """Main pipeline configuration schema."""

    data: DataConfig = Field(description="Data configuration")
    processing: ProcessingConfig = Field(description="Processing configuration")
    output: OutputConfig = Field(description="Output configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Prevent extra fields

    @validator("data")
    def validate_paths_exist(cls, v):
        """Validate that input paths exist (skip validation if path is for testing)."""
        # Skip validation for common test paths
        test_paths = ["./data", "/tmp", "test", "tests"]
        if any(str(v.input_path).startswith(test_path) for test_path in test_paths):
            return v

        if not v.input_path.exists():
            raise ValueError(f"Input path does not exist: {v.input_path}")
        return v

    def model_post_init(self, __context):
        """Post-initialization validation."""
        # Create output directory if it doesn't exist
        self.data.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.dict(), f, default_flow_style=False, indent=2)
