"""Configuration schemas for dataset processing flow.

AIDEV-NOTE: Pydantic schemas for the complete dataset processing workflow.
Supports YAML-based dataset definitions with custom processing functions.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    name: str = Field(..., description="Unique name for the dataset")
    type: str = Field(..., description="Dataset type (huggingface, local_files, etc.)")
    sample_size: Optional[int] = Field(
        None, description="Number of samples to use (None for all)"
    )
    processing_function: str = Field(
        ..., description="Name of the processing function to use"
    )

    # Dataset-specific parameters
    dataset_id: Optional[str] = Field(None, description="HuggingFace dataset ID")
    file_path: Optional[Union[str, Path]] = Field(
        None, description="Path to local file"
    )
    split: Optional[str] = Field("train", description="Dataset split to use")
    streaming: Optional[bool] = Field(False, description="Whether to use streaming")

    # Additional parameters for dataset loading
    extra_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional parameters"
    )

    @validator("sample_size")
    def validate_sample_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("sample_size must be positive")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for the processing pipeline."""

    output_format: str = Field(
        "instruction", description="Output format type (instruction, preference)"
    )
    merge_strategy: str = Field("concatenate", description="How to merge datasets")
    shuffle: bool = Field(True, description="Whether to shuffle the final dataset")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    path: Union[str, Path] = Field(..., description="Output directory path")
    name: str = Field(..., description="Output dataset name")
    format: str = Field("huggingface", description="Output format (huggingface)")
    save_method: str = Field("save_to_disk", description="HuggingFace save method")

    # Metadata
    description: Optional[str] = Field(None, description="Dataset description")
    version: Optional[str] = Field(None, description="Dataset version")
    tags: Optional[List[str]] = Field(default_factory=list, description="Dataset tags")


class VersionConfig(BaseModel):
    """Complete version configuration for dataset processing."""

    version: str = Field(..., description="Version identifier")
    description: Optional[str] = Field(None, description="Version description")

    datasets: List[DatasetConfig] = Field(
        ..., description="List of datasets to process"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Processing configuration"
    )
    output: OutputConfig = Field(..., description="Output configuration")

    @validator("datasets")
    def validate_datasets(cls, v):
        if not v:
            raise ValueError("At least one dataset must be specified")

        # Check for duplicate dataset names
        names = [dataset.name for dataset in v]
        if len(names) != len(set(names)):
            raise ValueError("Dataset names must be unique")

        return v
