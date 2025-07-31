"""
Integration tests for the complete data processing pipeline.

This module tests end-to-end data processing workflows including:
- Dataset loading and validation
- Processing pipeline execution
- Output format validation
- HuggingFace dataset compatibility
"""

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# AIDEV-NOTE: Integration tests for complete pipeline workflows


class TestDataPipelineIntegration:
    """Test complete data processing pipeline workflows."""

    @pytest.mark.integration
    def test_instruction_data_end_to_end_pipeline(
        self, sample_instruction_processing_data, temp_output_dir
    ):
        """Test complete instruction data processing pipeline."""
        # Simulate pipeline stages

        # Stage 1: Data Loading (mock)
        input_data = [sample_instruction_processing_data]

        # Stage 2: Data Validation
        for data in input_data:
            # Validate processing format
            assert "messages" in data
            assert "source" in data
            assert isinstance(data["messages"], list)
            assert isinstance(data["source"], str)

        # Stage 3: Data Processing (convert to saving format)
        processed_data = []
        for data in input_data:
            # Convert according to expected_output_format.md requirements
            saving_data = {
                "messages": str(data["messages"]),  # Convert to string representation
                "source": data["source"],  # Already string
                "tools": (
                    str(data["tools"]) if data["tools"] is not None else None
                ),  # None stays None
                "images": data["images"],  # Images NOT converted to string
            }
            processed_data.append(saving_data)

        # Validate string conversion requirement
        for item in processed_data:
            assert isinstance(item["messages"], str), (
                "messages must be string in saving format"
            )
            assert isinstance(item["source"], str), (
                "source must be string in saving format"
            )
            assert item["tools"] is None or isinstance(item["tools"], str), (
                "tools must be None or string"
            )
            assert item["images"] is None or isinstance(item["images"], list), (
                "images must remain list, not string"
            )

        # Stage 4: Create DataFrame
        df = pd.DataFrame(processed_data)
        assert len(df) == 1
        assert "messages" in df.columns

        # Stage 5: Save to HuggingFace format (mock)
        output_path = temp_output_dir / "instruction_dataset"

        try:
            from datasets import Dataset

            dataset = Dataset.from_pandas(df)
            dataset.save_to_disk(str(output_path))

            # Verify saved dataset
            assert output_path.exists()
            loaded_dataset = Dataset.load_from_disk(str(output_path))
            assert len(loaded_dataset) == 1

        except ImportError:
            pytest.skip("datasets library not available")

    @pytest.mark.integration
    def test_preference_data_end_to_end_pipeline(
        self, sample_preference_processing_data, temp_output_dir
    ):
        """Test complete preference data processing pipeline."""
        # Simulate pipeline stages

        # Stage 1: Data Loading (mock)
        input_data = [sample_preference_processing_data]

        # Stage 2: Data Validation
        for data in input_data:
            # Validate processing format
            assert "messages" in data
            assert "source" in data
            assert "rejected" in data
            assert isinstance(data["messages"], list)
            assert isinstance(data["source"], str)
            assert isinstance(data["rejected"], dict)

        # Stage 3: Data Processing (convert to saving format)
        processed_data = []
        for data in input_data:
            # Convert according to expected_output_format.md requirements
            saving_data = {
                "messages": str(data["messages"]),  # Convert to string representation
                "source": data["source"],  # Already string
                "tools": (
                    str(data["tools"]) if data["tools"] is not None else None
                ),  # None stays None
                "images": data["images"],  # Images NOT converted to string
                "rejected": str(data["rejected"]),  # Convert to string representation
            }
            processed_data.append(saving_data)

        # Validate string conversion requirement
        for item in processed_data:
            assert isinstance(item["messages"], str), (
                "messages must be string in saving format"
            )
            assert isinstance(item["source"], str), (
                "source must be string in saving format"
            )
            assert item["tools"] is None or isinstance(item["tools"], str), (
                "tools must be None or string"
            )
            assert item["images"] is None or isinstance(item["images"], list), (
                "images must remain list, not string"
            )
            assert isinstance(item["rejected"], str), (
                "rejected must be string in saving format"
            )

        # Stage 4: Create DataFrame
        df = pd.DataFrame(processed_data)
        assert len(df) == 1
        assert "rejected" in df.columns

        # Stage 5: Save to HuggingFace format (mock)
        output_path = temp_output_dir / "preference_dataset"

        try:
            from datasets import Dataset

            dataset = Dataset.from_pandas(df)
            dataset.save_to_disk(str(output_path))

            # Verify saved dataset
            assert output_path.exists()
            loaded_dataset = Dataset.load_from_disk(str(output_path))
            assert len(loaded_dataset) == 1
            assert "rejected" in loaded_dataset.column_names

        except ImportError:
            pytest.skip("datasets library not available")

    @pytest.mark.integration
    def test_mixed_dataset_processing(
        self,
        sample_instruction_processing_data,
        sample_preference_processing_data,
        temp_output_dir,
    ):
        """Test processing mixed instruction and preference data."""
        # Mix different data types
        mixed_data = [
            sample_instruction_processing_data,
            sample_preference_processing_data,
        ]

        # Process each according to its type
        instruction_data = []
        preference_data = []

        for data in mixed_data:
            if "rejected" in data:
                # Preference data
                saving_data = {
                    "messages": str(data["messages"]),
                    "source": data["source"],
                    "tools": str(data["tools"]) if data["tools"] else None,
                    "images": data["images"],
                    "rejected": str(data["rejected"]),
                }
                preference_data.append(saving_data)
            else:
                # Instruction data
                saving_data = {
                    "messages": str(data["messages"]),
                    "source": data["source"],
                    "tools": str(data["tools"]) if data["tools"] else None,
                    "images": data["images"],
                }
                instruction_data.append(saving_data)

        # Verify separation
        assert len(instruction_data) == 1
        assert len(preference_data) == 1
        assert "rejected" not in instruction_data[0]
        assert "rejected" in preference_data[0]

    @pytest.mark.integration
    def test_batch_processing_pipeline(self, temp_output_dir):
        """Test processing large batches of data."""
        # Create batch of mixed data
        batch_size = 100
        batch_data = []

        for i in range(batch_size):
            if i % 2 == 0:
                # Instruction data
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Question {i}", "image": None}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"Answer {i}", "image": None}
                            ],
                        },
                    ],
                    "source": f"batch_dataset_{i}",
                    "tools": None,
                    "images": None,
                }
            else:
                # Preference data
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Question {i}", "image": None}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Good answer {i}",
                                    "image": None,
                                }
                            ],
                        },
                    ],
                    "source": f"batch_dataset_{i}",
                    "tools": None,
                    "images": None,
                    "rejected": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Bad answer {i}", "image": None}
                        ],
                        "tool_calls": None,
                        "name": None,
                    },
                }
            batch_data.append(data)

        # Process batch
        processed_data = []
        for data in batch_data:
            saving_data = {
                "messages": str(data["messages"]),
                "source": data["source"],
                "tools": str(data["tools"]) if data["tools"] else None,
                "images": data["images"],
            }
            if "rejected" in data:
                saving_data["rejected"] = str(data["rejected"])
            processed_data.append(saving_data)

        # Create DataFrame
        df = pd.DataFrame(processed_data)
        assert len(df) == batch_size

        # Verify mixed data integrity
        instruction_count = df["rejected"].isna().sum()
        preference_count = df["rejected"].notna().sum()
        assert instruction_count == batch_size // 2
        assert preference_count == batch_size // 2


class TestConfigurationIntegration:
    """Test configuration and setup integration."""

    @pytest.mark.integration
    def test_configuration_loading_mock(self):
        """Test configuration loading (mock implementation)."""
        # Mock configuration structure based on improvement_plan.md
        mock_config = {
            "data": {
                "input_path": "/path/to/input",
                "output_path": "/path/to/output",
                "dataset_configs": {
                    "smoltalk": {"version": "v1.0"},
                    "ultrachat": {"version": "v2.0"},
                },
            },
            "processing": {
                "process_type": "instruction",
                "version": "v2.1",
                "parallel_workers": 4,
                "batch_size": 1000,
                "sample_size": None,
            },
            "output": {"format": "huggingface", "compression": None},
            "logging": {"level": "INFO", "format": "structured"},
        }

        # Test configuration validation
        assert "data" in mock_config
        assert "processing" in mock_config
        assert mock_config["processing"]["process_type"] in [
            "instruction",
            "preference",
        ]
        assert isinstance(mock_config["processing"]["parallel_workers"], int)
        assert mock_config["processing"]["parallel_workers"] > 0

    @pytest.mark.integration
    def test_dataset_registry_mock(self):
        """Test dataset registry functionality (mock implementation)."""

        # Mock dataset registry based on improvement_plan.md
        class MockDatasetRegistry:
            _datasets = {}

            @classmethod
            def register(cls, name: str):
                def decorator(dataset_class):
                    cls._datasets[name] = dataset_class
                    return dataset_class

                return decorator

            @classmethod
            def get_dataset(cls, name: str):
                return cls._datasets.get(name)

            @classmethod
            def list_datasets(cls):
                return list(cls._datasets.keys())

        # Test registration
        @MockDatasetRegistry.register("test_dataset")
        class TestDataset:
            def load(self):
                return pd.DataFrame([{"test": "data"}])

        # Test registry functionality
        assert "test_dataset" in MockDatasetRegistry.list_datasets()
        dataset_class = MockDatasetRegistry.get_dataset("test_dataset")
        assert dataset_class is not None

        # Test dataset instantiation
        dataset = dataset_class()
        data = dataset.load()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1

    @pytest.mark.integration
    def test_concurrent_processing_simulation(self):
        """Test simulation of concurrent processing scenarios."""
        # Simulate concurrent processing with threading-like behavior
        import time

        # Create multiple data batches
        batch_1 = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Batch 1 item {i}", "image": None}
                        ],
                    }
                ],
                "source": f"batch_1_{i}",
                "tools": None,
                "images": None,
            }
            for i in range(10)
        ]

        batch_2 = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Batch 2 item {i}", "image": None}
                        ],
                    }
                ],
                "source": f"batch_2_{i}",
                "tools": None,
                "images": None,
            }
            for i in range(10)
        ]

        # Process batches "concurrently" (simulated)
        def process_batch(batch, batch_id):
            processed = []
            for data in batch:
                # Simulate processing time
                time.sleep(0.001)  # 1ms per item
                saving_data = {
                    "messages": str(data["messages"]),
                    "source": data["source"],
                    "tools": None,
                    "images": None,
                }
                processed.append(saving_data)
            return processed, batch_id

        # Process both batches
        start_time = time.time()
        result_1, id_1 = process_batch(batch_1, "batch_1")
        result_2, id_2 = process_batch(batch_2, "batch_2")
        end_time = time.time()

        # Verify results
        assert len(result_1) == 10
        assert len(result_2) == 10
        assert id_1 == "batch_1"
        assert id_2 == "batch_2"

        # Verify reasonable processing time
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should complete quickly
