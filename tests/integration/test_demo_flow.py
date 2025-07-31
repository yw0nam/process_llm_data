"""Test demo for the complete dataset processing flow.

AIDEV-NOTE: Integration test that demonstrates the complete flow with mock data.
"""

import ast
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from datasets import load_from_disk

from src.core.pipeline import DatasetProcessingPipeline
from src.dataset.registry import DatasetRegistry


class TestDemoFlow:
    """Integration test for the complete processing flow."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_datasets(self, temp_dir):
        """Create mock datasets for testing."""
        # Create mock Alpaca-style data
        alpaca_data = [
            {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris.",
            },
            {
                "instruction": "Translate the following to Spanish",
                "input": "Hello, how are you?",
                "output": "Hola, ¿cómo estás?",
            },
            {
                "instruction": "Write a short poem about nature",
                "input": "",
                "output": "Trees sway in the gentle breeze,\nFlowers bloom with graceful ease,\nNature's beauty never cease.",
            },
        ]

        alpaca_file = temp_dir / "alpaca_mock.jsonl"
        with open(alpaca_file, "w") as f:
            for item in alpaca_data:
                f.write(json.dumps(item) + "\n")

        # Create mock SmolTalk-style data
        smoltalk_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello there!"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                    {"role": "user", "content": "What's the weather like?"},
                    {
                        "role": "assistant",
                        "content": "I don't have access to current weather data, but I'd be happy to help you find a weather service or discuss weather in general!",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you help me with math?"},
                    {
                        "role": "assistant",
                        "content": "Of course! I'd be happy to help you with math. What specific topic or problem would you like assistance with?",
                    },
                ]
            },
        ]

        smoltalk_file = temp_dir / "smoltalk_mock.jsonl"
        with open(smoltalk_file, "w") as f:
            for item in smoltalk_data:
                f.write(json.dumps(item) + "\n")

        return alpaca_file, smoltalk_file

    @pytest.fixture
    def mock_config(self, temp_dir, mock_datasets):
        """Create a mock configuration file."""
        alpaca_file, smoltalk_file = mock_datasets
        
        config_content = f"""
version: "1.0.0"
description: "Demo processing with mock datasets"

datasets:
  - name: "mock_alpaca"
    type: "local_files"
    file_path: "{alpaca_file}"
    sample_size: 2
    processing_function: "alpaca_to_messages"
    extra_params:
      encoding: "utf-8"

  - name: "mock_smoltalk"
    type: "local_files"
    file_path: "{smoltalk_file}"
    sample_size: 1
    processing_function: "smoltalk_to_messages"
    extra_params:
      encoding: "utf-8"

processing:
  output_format: "instruction"
  merge_strategy: "concatenate"
  shuffle: true
  seed: 42

output:
  path: "{temp_dir}/output"
  name: "demo_dataset"
  format: "huggingface"
  save_method: "save_to_disk"
  description: "Demo processed dataset"
  version: "1.0.0"
  tags: ["demo", "test"]
"""

        config_file = temp_dir / "demo_config.yaml"
        with open(config_file, "w") as f:
            f.write(config_content)

        return config_file

    def test_complete_flow(self, temp_dir, mock_config):
        """Test the complete dataset processing flow."""
        # Test that processing functions are available
        available_functions = DatasetRegistry.list_processing_functions()
        assert "alpaca_to_messages" in available_functions
        assert "smoltalk_to_messages" in available_functions

        # Run the pipeline
        pipeline = DatasetProcessingPipeline(mock_config)
        pipeline.run()

        # Verify output was created
        output_path = temp_dir / "output" / "demo_dataset"
        assert output_path.exists(), "Output dataset directory should exist"

        # Load and verify the processed dataset
        dataset = load_from_disk(str(output_path))
        
        # Basic structure checks
        assert len(dataset) > 0, "Dataset should contain processed data"
        assert "messages" in dataset.column_names, "Dataset should have messages column"
        assert "source" in dataset.column_names, "Dataset should have source column"
        assert "tools" in dataset.column_names, "Dataset should have tools column"
        assert "images" in dataset.column_names, "Dataset should have images column"

        # Test first row structure
        first_row = dataset[0]
        
        # Test that messages field is a string (as per specification)
        assert isinstance(first_row["messages"], str), "Messages should be string format"
        
        # Test that we can parse messages back using ast.literal_eval
        parsed_messages = ast.literal_eval(first_row["messages"])
        assert isinstance(parsed_messages, list), "Parsed messages should be a list"
        assert len(parsed_messages) > 0, "Messages should contain conversation turns"
        
        # Test message structure
        first_message = parsed_messages[0]
        assert isinstance(first_message, dict), "Each message should be a dict"
        assert "role" in first_message, "Message should have role field"
        assert "content" in first_message, "Message should have content field"
        
        # Test content structure
        content = first_message["content"]
        assert isinstance(content, list), "Content should be a list"
        if len(content) > 0:
            content_item = content[0]
            assert isinstance(content_item, dict), "Content item should be a dict"
            assert "type" in content_item, "Content item should have type field"
            assert content_item["type"] in ["text", "image"], "Content type should be text or image"

        # Test source field
        assert isinstance(first_row["source"], str), "Source should be string"
        assert first_row["source"] in ["alpaca_gpt4", "smoltalk"], "Source should be from known datasets"

        # Test tools field (should be None or string)
        tools = first_row["tools"]
        assert tools is None or isinstance(tools, str), "Tools should be None or string"

        # Test images field (should be None in this demo)
        images = first_row["images"]
        assert images is None, "Images should be None in this demo"

    def test_str_format_compatibility(self, temp_dir, mock_config):
        """Test that str() format is compatible with ast.literal_eval()."""
        # Run the pipeline
        pipeline = DatasetProcessingPipeline(mock_config)
        pipeline.run()

        # Load the dataset
        output_path = temp_dir / "output" / "demo_dataset"
        dataset = load_from_disk(str(output_path))
        
        # Test all rows for str/ast.literal_eval compatibility
        for i, row in enumerate(dataset):
            # Test messages field
            messages_str = row["messages"]
            assert isinstance(messages_str, str), f"Row {i}: messages should be string"
            
            # Should be able to parse back
            parsed_messages = ast.literal_eval(messages_str)
            assert isinstance(parsed_messages, list), f"Row {i}: parsed messages should be list"
            
            # Test tools field if not None
            if row["tools"] is not None:
                tools_str = row["tools"]
                assert isinstance(tools_str, str), f"Row {i}: tools should be string when not None"
                parsed_tools = ast.literal_eval(tools_str)
                # Tools should parse back to original type

    def test_pipeline_summary(self, mock_config):
        """Test that pipeline summary provides correct information."""
        pipeline = DatasetProcessingPipeline(mock_config)
        summary = pipeline.get_summary()
        
        assert summary["version"] == "1.0.0"
        assert summary["description"] == "Demo processing with mock datasets"
        assert summary["num_datasets"] == 2
        assert len(summary["datasets"]) == 2
        
        # Check dataset details
        dataset_names = [d["name"] for d in summary["datasets"]]
        assert "mock_alpaca" in dataset_names
        assert "mock_smoltalk" in dataset_names


def test_demo_as_standalone():
    """Test the demo flow as a standalone function (for backwards compatibility)."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create the same demo flow as before
        # Create mock Alpaca-style data
        alpaca_data = [
            {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris.",
            },
        ]

        alpaca_file = temp_dir / "alpaca_mock.jsonl"
        with open(alpaca_file, "w") as f:
            for item in alpaca_data:
                f.write(json.dumps(item) + "\n")

        # Create mock SmolTalk-style data
        smoltalk_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello there!"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                ]
            },
        ]

        smoltalk_file = temp_dir / "smoltalk_mock.jsonl"
        with open(smoltalk_file, "w") as f:
            for item in smoltalk_data:
                f.write(json.dumps(item) + "\n")

        # Create config
        config_content = f"""
version: "1.0.0"
description: "Demo processing with mock datasets"

datasets:
  - name: "mock_alpaca"
    type: "local_files"
    file_path: "{alpaca_file}"
    sample_size: 1
    processing_function: "alpaca_to_messages"

  - name: "mock_smoltalk"
    type: "local_files"
    file_path: "{smoltalk_file}"
    sample_size: 1
    processing_function: "smoltalk_to_messages"

processing:
  output_format: "instruction"
  merge_strategy: "concatenate"
  shuffle: false
  seed: 42

output:
  path: "{temp_dir}/output"
  name: "demo_dataset"
  format: "huggingface"
  save_method: "save_to_disk"
  description: "Demo processed dataset"
  version: "1.0.0"
"""

        config_file = temp_dir / "demo_config.yaml"
        with open(config_file, "w") as f:
            f.write(config_content)

        # Run pipeline
        pipeline = DatasetProcessingPipeline(config_file)
        pipeline.run()

        # Verify output
        output_path = temp_dir / "output" / "demo_dataset"
        assert output_path.exists()
        
        # Load and check basic structure
        dataset = load_from_disk(str(output_path))
        assert len(dataset) == 2  # One from each dataset
        
        # Check that ast.literal_eval works on the output
        first_row = dataset[0]
        parsed_messages = ast.literal_eval(first_row["messages"])
        assert isinstance(parsed_messages, list)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run the standalone demo when executed directly
    test_demo_as_standalone()
    print("✅ Demo test completed successfully!")
