#!/usr/bin/env python3
"""Integration test for Phase 1 foundation validation.

AIDEV-NOTE: Integration test for Phase 1 foundation implementation.
Validates the complete foundation architecture is working properly.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
def test_imports():
    """Test that all modules can be imported."""
    # Test core imports
    # Test config imports
    from src.config.schemas import (
        DataConfig,
        OutputConfig,
        PipelineConfig,
        ProcessingConfig,
    )
    from src.core.exceptions import ConfigurationError, DatasetError, ProcessingError
    from src.core.pipeline import DatasetProcessingPipeline
    from src.core.processor import BaseProcessor

    # Test dataset imports
    from src.dataset.base import BaseDataset

    # Test utils imports
    from src.utils import (
        DatasetSaver,
        ParallelProcessor,
        format_instruction_data,
        setup_logging,
        validate_instruction_format,
    )

    assert True  # If we get here, all imports worked


@pytest.mark.integration
def test_version_yaml_configurations():
    """Test version-specific YAML configuration files validation."""
    import yaml
    from src.config.version_schemas import VersionConfig
    from pathlib import Path

    # Find all version YAML files
    version_config_dir = Path("src/config/versions")
    yaml_files = list(version_config_dir.glob("*.yaml"))

    assert len(yaml_files) > 0, "No version YAML files found"

    for yaml_file in yaml_files:
        print(f"Testing {yaml_file.name}")

        # Load and parse YAML
        with open(yaml_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Test 1: Check if description exists
        assert "description" in config_data, f"Missing description in {yaml_file.name}"
        assert (
            config_data["description"] is not None
        ), f"Description is None in {yaml_file.name}"
        assert isinstance(
            config_data["description"], str
        ), f"Description is not a string in {yaml_file.name}"
        assert (
            len(config_data["description"].strip()) > 0
        ), f"Description is empty in {yaml_file.name}"

        # Validate using Pydantic schema
        config = VersionConfig(**config_data)

        # Test 2: Check version field
        assert config.version is not None, f"Version is None in {yaml_file.name}"
        assert isinstance(
            config.version, str
        ), f"Version is not a string in {yaml_file.name}"

        # Test 3: Check datasets configuration
        assert len(config.datasets) > 0, f"No datasets defined in {yaml_file.name}"

        for dataset in config.datasets:
            # Test dataset name
            assert dataset.name is not None, f"Dataset name is None in {yaml_file.name}"
            assert (
                len(dataset.name.strip()) > 0
            ), f"Dataset name is empty in {yaml_file.name}"

            # Test dataset type
            assert dataset.type in [
                "huggingface",
                "local_files",
            ], f"Invalid dataset type '{dataset.type}' in {yaml_file.name}"

            # Test HuggingFace dataset specific fields
            if dataset.type == "huggingface":
                assert (
                    dataset.dataset_id is not None
                ), f"Missing dataset_id for HuggingFace dataset '{dataset.name}' in {yaml_file.name}"
                assert isinstance(
                    dataset.dataset_id, str
                ), f"dataset_id is not a string for '{dataset.name}' in {yaml_file.name}"
                assert (
                    len(dataset.dataset_id.strip()) > 0
                ), f"dataset_id is empty for '{dataset.name}' in {yaml_file.name}"

                # Check split field
                if dataset.split is not None:
                    assert dataset.split in [
                        "train",
                        "validation",
                        "test",
                        "all",
                    ], f"Invalid split '{dataset.split}' for '{dataset.name}' in {yaml_file.name}"

            # Test local files dataset specific fields
            if dataset.type == "local_files":
                assert (
                    dataset.file_path is not None
                ), f"Missing file_path for local dataset '{dataset.name}' in {yaml_file.name}"
                assert isinstance(
                    dataset.file_path, str
                ), f"file_path is not a string for '{dataset.name}' in {yaml_file.name}"

            # Test processing function
            assert (
                dataset.processing_function is not None
            ), f"Missing processing_function for '{dataset.name}' in {yaml_file.name}"
            assert isinstance(
                dataset.processing_function, str
            ), f"processing_function is not a string for '{dataset.name}' in {yaml_file.name}"

            # Test sample size if specified
            if dataset.sample_size is not None:
                assert isinstance(
                    dataset.sample_size, int
                ), f"sample_size is not an integer for '{dataset.name}' in {yaml_file.name}"
                assert (
                    dataset.sample_size > 0
                ), f"sample_size must be positive for '{dataset.name}' in {yaml_file.name}"

        # Test 4: Check processing configuration
        assert config.processing.output_format in [
            "instruction",
            "preference",
        ], f"Invalid output_format '{config.processing.output_format}' in {yaml_file.name}"
        assert config.processing.merge_strategy in [
            "concatenate",
            "interleave",
        ], f"Invalid merge_strategy '{config.processing.merge_strategy}' in {yaml_file.name}"
        assert isinstance(
            config.processing.shuffle, bool
        ), f"shuffle is not a boolean in {yaml_file.name}"

        if config.processing.seed is not None:
            assert isinstance(
                config.processing.seed, int
            ), f"seed is not an integer in {yaml_file.name}"

        # Test 5: Check output configuration
        assert (
            config.output.path is not None
        ), f"Missing output path in {yaml_file.name}"
        assert (
            config.output.name is not None
        ), f"Missing output name in {yaml_file.name}"
        assert config.output.format in [
            "huggingface",
            "parquet",
        ], f"Invalid output format '{config.output.format}' in {yaml_file.name}"

        # Test output metadata
        if config.output.description is not None:
            assert isinstance(
                config.output.description, str
            ), f"Output description is not a string in {yaml_file.name}"

        if config.output.version is not None:
            assert isinstance(
                config.output.version, str
            ), f"Output version is not a string in {yaml_file.name}"

        if config.output.tags is not None:
            assert isinstance(
                config.output.tags, list
            ), f"Output tags is not a list in {yaml_file.name}"
            for tag in config.output.tags:
                assert isinstance(tag, str), f"Tag is not a string in {yaml_file.name}"

        print(f"✓ {yaml_file.name} passed all validation tests")


@pytest.mark.integration
def test_utilities():
    """Test utility functions."""
    from src.utils.formatting import (
        format_instruction_data,
        validate_instruction_format,
    )
    from src.utils.io import DatasetSaver, ensure_directory
    from src.utils.parallel import ParallelProcessor

    # Test formatting
    sample_data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "source": "test",
    }

    formatted = format_instruction_data(sample_data)
    assert validate_instruction_format(formatted)

    # Test parallel processor initialization
    processor = ParallelProcessor(n_workers=2)
    assert processor.n_workers == 2

    # Test directory creation
    test_dir = ensure_directory("test_output")
    assert test_dir.exists()


@pytest.mark.integration
def test_dataset_saver():
    """Test dataset saver with sample data."""
    import pandas as pd
    from src.utils.io import DatasetSaver, load_hf_dataset

    # Create sample data
    sample_df = pd.DataFrame(
        {
            "messages": ['[{"role": "user", "content": "test"}]'] * 10,
            "source": ["test_source"] * 10,
        }
    )

    # Test dataset saver
    saver = DatasetSaver("test_output", "sample_dataset")
    saved_path = saver.save_with_splits(
        sample_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    assert Path(saved_path).exists()

    # Verify the saved dataset
    loaded_dataset = load_hf_dataset(saved_path)
    expected_splits = {"train", "validation", "test"}
    assert set(loaded_dataset.keys()) == expected_splits


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Clean up test files after all tests."""
    yield

    import shutil

    if Path("test_output").exists():
        shutil.rmtree("test_output")
