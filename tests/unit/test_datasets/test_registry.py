"""Tests for dataset registry and loaders.

AIDEV-NOTE: Unit tests for the dataset registry system and loader plugins.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from src.dataset.base import BaseDataset
from src.dataset.loaders.huggingface import HuggingFaceDataset
from src.dataset.loaders.local_files import LocalFilesDataset
from src.dataset.registry import DatasetRegistry


class TestDatasetRegistry:
    """Test cases for DatasetRegistry."""

    def test_register_and_check_registration(self):
        """Test registering a dataset loader."""

        # Create a mock loader class
        @DatasetRegistry.register("test_loader")
        class TestLoader(BaseDataset):
            def __init__(self, config):
                super().__init__(config)

            def load(self):
                return pd.DataFrame()

            def validate(self, data):
                return True

        # Test that it's registered
        assert DatasetRegistry.is_registered("test_loader")

    def test_get_nonexistent_dataset(self):
        """Test creating a non-existent dataset raises ValueError."""
        with pytest.raises(
            ValueError, match="Dataset 'nonexistent_loader' not registered"
        ):
            DatasetRegistry.create_dataset("nonexistent_loader", {})

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = DatasetRegistry.list_datasets()
        assert "huggingface" in datasets
        assert "local_files" in datasets

    def test_create_dataset(self):
        """Test creating a dataset instance through the registry."""
        config = {"dataset_id": "test/dataset", "split": "train"}

        dataset = DatasetRegistry.create_dataset("huggingface", config)
        assert isinstance(dataset, HuggingFaceDataset)
        assert dataset.dataset_id == "test/dataset"
        assert dataset.split == "train"

    def test_duplicate_registration(self):
        """Test that duplicate registration overwrites existing loader."""

        @DatasetRegistry.register("duplicate_test")
        class FirstLoader(BaseDataset):
            def __init__(self, config):
                super().__init__(config)

            def load(self):
                return pd.DataFrame({"first": [1]})

            def validate(self, data):
                return True

        @DatasetRegistry.register("duplicate_test")
        class SecondLoader(BaseDataset):
            def __init__(self, config):
                super().__init__(config)

            def load(self):
                return pd.DataFrame({"second": [2]})

            def validate(self, data):
                return True

        # Create instance and check it's the second loader
        instance = DatasetRegistry.create_dataset("duplicate_test", {})
        data = instance.load()
        assert "second" in data.columns
        assert "first" not in data.columns


class TestProcessingFunctions:
    """Test cases for processing function registration and usage."""

    def test_register_processing_function(self):
        """Test registering a processing function."""

        @DatasetRegistry.register_processing_function("test_processing")
        def test_processor(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            df_copy["processed"] = True
            return df_copy

        assert DatasetRegistry.has_processing_function("test_processing")
        func = DatasetRegistry.get_processing_function("test_processing")
        assert func is not None
        assert func.__name__ == "test_processor"

    def test_process_dataset_with_custom_function(self):
        """Test processing dataset with custom processing function."""

        @DatasetRegistry.register_processing_function("custom_process_test")
        def custom_processor(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            df_copy["custom_processed"] = True
            df_copy["row_count"] = len(df_copy)
            return df_copy

        # Create test data
        test_data = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})

        # Process the data
        processed_data = DatasetRegistry.process_dataset(
            "custom_process_test", test_data
        )

        assert "custom_processed" in processed_data.columns
        assert "row_count" in processed_data.columns
        assert processed_data["custom_processed"].all()
        assert processed_data["row_count"].iloc[0] == 2

    def test_process_dataset_without_custom_function(self):
        """Test processing dataset without custom processing function."""
        test_data = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})

        # Process data for non-existent processing function
        processed_data = DatasetRegistry.process_dataset(
            "no_custom_function", test_data
        )

        # Should return original data unchanged
        pd.testing.assert_frame_equal(processed_data, test_data)

    def test_process_dataset_skip_custom_processing(self):
        """Test processing dataset with custom processing disabled."""

        @DatasetRegistry.register_processing_function("skip_test")
        def skip_processor(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            df_copy["should_not_appear"] = True
            return df_copy

        test_data = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})

        # Process with custom processing disabled
        processed_data = DatasetRegistry.process_dataset(
            "skip_test", test_data, use_custom_processing=False
        )

        # Should return original data unchanged
        pd.testing.assert_frame_equal(processed_data, test_data)
        assert "should_not_appear" not in processed_data.columns

    def test_list_processing_functions(self):
        """Test listing all processing functions."""

        @DatasetRegistry.register_processing_function("list_test_1")
        def processor_1(df):
            return df

        @DatasetRegistry.register_processing_function("list_test_2")
        def processor_2(df):
            return df

        processing_functions = DatasetRegistry.list_processing_functions()
        assert "list_test_1" in processing_functions
        assert "list_test_2" in processing_functions

    def test_get_dataset_info_with_processing(self):
        """Test getting dataset info includes processing function information."""

        @DatasetRegistry.register("info_with_processing")
        class TestDataset(BaseDataset):
            def __init__(self, config):
                super().__init__(config)

            def load(self):
                return pd.DataFrame()

            def validate(self, data):
                return True

        @DatasetRegistry.register_processing_function("info_with_processing")
        def test_processing_func(df):
            return df

        info = DatasetRegistry.get_dataset_info("info_with_processing")

        assert info["has_processing_function"] is True
        assert info["processing_function_name"] == "test_processing_func"

    def test_get_dataset_info_without_processing(self):
        """Test getting dataset info for dataset without processing function."""

        @DatasetRegistry.register("info_without_processing")
        class TestDataset(BaseDataset):
            def __init__(self, config):
                super().__init__(config)

            def load(self):
                return pd.DataFrame()

            def validate(self, data):
                return True

        info = DatasetRegistry.get_dataset_info("info_without_processing")

        assert info["has_processing_function"] is False
        assert info["processing_function_name"] is None

    def test_unregister_processing_function(self):
        """Test unregistering a processing function."""

        @DatasetRegistry.register_processing_function("unregister_test")
        def test_processor(df):
            return df

        assert DatasetRegistry.has_processing_function("unregister_test")

        DatasetRegistry.unregister_processing_function("unregister_test")

        assert not DatasetRegistry.has_processing_function("unregister_test")
        assert DatasetRegistry.get_processing_function("unregister_test") is None

    def test_duplicate_processing_function_registration(self):
        """Test that duplicate processing function registration overwrites."""

        @DatasetRegistry.register_processing_function("duplicate_processing")
        def first_processor(df):
            df_copy = df.copy()
            df_copy["version"] = "first"
            return df_copy

        @DatasetRegistry.register_processing_function("duplicate_processing")
        def second_processor(df):
            df_copy = df.copy()
            df_copy["version"] = "second"
            return df_copy

        test_data = pd.DataFrame({"data": [1, 2, 3]})
        processed = DatasetRegistry.process_dataset("duplicate_processing", test_data)

        assert processed["version"].iloc[0] == "second"

    def test_processing_function_error_handling(self):
        """Test error handling in processing functions."""

        @DatasetRegistry.register_processing_function("error_test")
        def error_processor(df):
            raise ValueError("Intentional processing error")

        test_data = pd.DataFrame({"data": [1, 2, 3]})

        with pytest.raises(ValueError, match="Intentional processing error"):
            DatasetRegistry.process_dataset("error_test", test_data)


class TestHuggingFaceDataset:
    """Test cases for HuggingFaceDataset."""

    def test_init_success(self):
        """Test successful initialization."""
        config = {"dataset_id": "test/dataset", "split": "train", "streaming": False}

        dataset = HuggingFaceDataset(config)
        assert dataset.dataset_id == "test/dataset"
        assert dataset.split == "train"
        assert dataset.streaming is False

    def test_init_missing_dataset_id(self):
        """Test initialization fails without dataset_id."""
        config = {"split": "train"}

        with pytest.raises(ValueError, match="dataset_id is required"):
            HuggingFaceDataset(config)

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = {"dataset_id": "test/dataset"}

        dataset = HuggingFaceDataset(config)
        assert dataset.split == "train"
        assert dataset.streaming is False
        assert dataset.trust_remote_code is False

    @patch("src.dataset.loaders.huggingface.load_dataset")
    def test_load_success(self, mock_load_dataset):
        """Test successful dataset loading."""
        # Mock the HuggingFace dataset
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = pd.DataFrame(
            {"text": ["Hello world", "How are you?"], "label": [0, 1]}
        )
        mock_load_dataset.return_value = mock_dataset

        config = {"dataset_id": "test/dataset"}
        dataset = HuggingFaceDataset(config)

        df = dataset.load()

        assert len(df) == 2
        assert "text" in df.columns
        assert "label" in df.columns
        mock_load_dataset.assert_called_once_with(
            "test/dataset",
            split="train",
            streaming=False,
            token=None,
            trust_remote_code=False,
        )

    @patch("src.dataset.loaders.huggingface.load_dataset")
    def test_load_streaming(self, mock_load_dataset):
        """Test loading with streaming enabled."""
        # Mock streaming dataset
        mock_dataset = [
            {"text": "Hello world", "label": 0},
            {"text": "How are you?", "label": 1},
        ]
        mock_load_dataset.return_value = iter(mock_dataset)

        config = {"dataset_id": "test/dataset", "streaming": True}
        dataset = HuggingFaceDataset(config)

        df = dataset.load()

        assert len(df) == 2
        assert "text" in df.columns
        assert "label" in df.columns

    def test_validate_success(self):
        """Test successful dataset validation."""
        config = {"dataset_id": "test/dataset"}
        dataset = HuggingFaceDataset(config)

        df = pd.DataFrame({"text": ["Hello world", "How are you?"], "label": [0, 1]})

        assert dataset.validate(df) is True

    def test_validate_empty_dataset(self):
        """Test validation fails for empty dataset."""
        config = {"dataset_id": "test/dataset"}
        dataset = HuggingFaceDataset(config)

        df = pd.DataFrame()

        assert dataset.validate(df) is False

    def test_source_name(self):
        """Test source name property."""
        config = {"dataset_id": "test/dataset"}
        dataset = HuggingFaceDataset(config)

        assert dataset.source_name == "huggingface:test/dataset"

    def test_get_info(self):
        """Test getting dataset info."""
        config = {
            "dataset_id": "test/dataset",
            "split": "validation",
            "streaming": True,
        }
        dataset = HuggingFaceDataset(config)

        info = dataset.get_info()

        assert info["dataset_id"] == "test/dataset"
        assert info["split"] == "validation"
        assert info["streaming"] is True
        assert info["source_type"] == "huggingface"


class TestLocalFilesDataset:
    """Test cases for LocalFilesDataset."""

    def test_init_success(self, tmp_path):
        """Test successful initialization."""
        # Create a test CSV file
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\nval1,val2\n")

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        assert dataset.file_path == test_file
        assert dataset.format == "csv"
        assert dataset.encoding == "utf-8"

    def test_init_missing_file_path(self):
        """Test initialization fails without file_path."""
        config = {"format": "csv"}

        with pytest.raises(ValueError, match="file_path is required"):
            LocalFilesDataset(config)

    def test_init_nonexistent_file(self):
        """Test initialization fails with non-existent file."""
        config = {"file_path": "/nonexistent/file.csv"}

        with pytest.raises(FileNotFoundError):
            LocalFilesDataset(config)

    def test_format_detection(self, tmp_path):
        """Test automatic format detection."""
        test_cases = [
            ("test.csv", "csv"),
            ("test.json", "json"),
            ("test.jsonl", "jsonl"),
            ("test.parquet", "parquet"),
            ("test.xlsx", "excel"),
            ("test.tsv", "tsv"),
        ]

        for filename, expected_format in test_cases:
            test_file = tmp_path / filename
            test_file.write_text("dummy content")

            config = {"file_path": str(test_file)}
            dataset = LocalFilesDataset(config)

            assert dataset.format == expected_format

    def test_unsupported_format(self, tmp_path):
        """Test unsupported file format raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("dummy content")

        config = {"file_path": str(test_file)}

        with pytest.raises(ValueError, match="Unsupported file format"):
            LocalFilesDataset(config)

    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\nval1,val2\nval3,val4\n")

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        df = dataset.load()

        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]
        assert df.iloc[0]["col1"] == "val1"
        assert df.iloc[0]["col2"] == "val2"

    def test_load_json(self, tmp_path):
        """Test loading JSON file."""
        test_file = tmp_path / "test.json"
        test_file.write_text(
            '[{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}]'
        )

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        df = dataset.load()

        assert len(df) == 2
        assert "col1" in df.columns
        assert "col2" in df.columns

    def test_load_jsonl(self, tmp_path):
        """Test loading JSONL file."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"col1": "val1", "col2": "val2"}\n{"col1": "val3", "col2": "val4"}\n'
        )

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        df = dataset.load()

        assert len(df) == 2
        assert "col1" in df.columns
        assert "col2" in df.columns

    def test_validate_success(self, tmp_path):
        """Test successful dataset validation."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\nval1,val2\n")

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        df = pd.DataFrame({"col1": ["val1", "val2"], "col2": ["val3", "val4"]})

        assert dataset.validate(df) is True

    def test_validate_empty_dataset(self, tmp_path):
        """Test validation fails for empty dataset."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n")

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        df = pd.DataFrame()

        assert dataset.validate(df) is False

    def test_source_name(self, tmp_path):
        """Test source name property."""
        test_file = tmp_path / "test_dataset.csv"
        test_file.write_text("col1,col2\nval1,val2\n")

        config = {"file_path": str(test_file)}
        dataset = LocalFilesDataset(config)

        assert dataset.source_name == "local_files:test_dataset.csv"

    def test_get_info(self, tmp_path):
        """Test getting dataset info."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\nval1,val2\n")

        config = {"file_path": str(test_file), "encoding": "utf-8", "format": "csv"}
        dataset = LocalFilesDataset(config)

        info = dataset.get_info()

        assert info["file_path"] == str(test_file)
        assert info["format"] == "csv"
        assert info["encoding"] == "utf-8"
        assert info["source_type"] == "local_files"
        assert "file_size" in info
        assert "file_modified" in info
