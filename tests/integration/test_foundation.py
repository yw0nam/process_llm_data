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
def test_configuration():
    """Test configuration loading."""
    from src.config.schemas import PipelineConfig

    # Test loading development config
    config = PipelineConfig.from_yaml("src/config/environments/development.yaml")
    assert config.data.input_path is not None

    # Test configuration validation
    assert config.processing.process_type in ["instruction", "preference"]
    assert (
        config.output.train_split_ratio
        + config.output.val_split_ratio
        + config.output.test_split_ratio
        == 1.0
    )

    # Test other environments (skip production for testing)
    PipelineConfig.from_yaml("src/config/environments/testing.yaml")


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
