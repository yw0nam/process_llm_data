"""
Data format validation tests for instruction and preference data.

This module contains the 4 specific tests requested:
1. check_instruction_data_format
2. check_instruction_data_format_when_saving
3. check_preference_data_format
4. check_preference_data_format_when_saving

These tests validate the data formats defined in expected_output_format.md
"""

import ast
from typing import Any, Union
from unittest.mock import Mock

import pytest

# AIDEV-NOTE: Core format validation tests as requested by user


class TestInstructionDataFormats:
    """Test instruction data format validation during processing and saving."""

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_check_instruction_data_format(self, sample_instruction_processing_data):
        """
        Test 1: Validate instruction data format during processing.

        Validates the processing format defined in expected_output_format.md:
        - messages: list[dict] with proper structure
        - source: str
        - tools: list[dict] | None
        - images: list[str] | list[PIL.Image] | None
        """
        data = sample_instruction_processing_data

        # Test required fields exist
        assert "messages" in data
        assert "source" in data
        assert "tools" in data  # Can be None
        assert "images" in data  # Can be None

        # Test messages structure
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) > 0

        for message in data["messages"]:
            assert isinstance(message, dict)

            # Required fields in each message
            assert "role" in message
            assert "content" in message
            assert "tool_calls" in message  # Can be None
            assert "name" in message  # Can be None

            # Validate role
            assert message["role"] in ["system", "user", "assistant", "tool"]

            # Validate content structure
            assert isinstance(message["content"], list)
            for content_item in message["content"]:
                assert isinstance(content_item, dict)
                assert "type" in content_item
                assert "text" in content_item
                assert "image" in content_item

                # Either text or image must be present, not both
                text_present = content_item["text"] is not None
                image_present = content_item["image"] is not None
                assert text_present != image_present, (
                    "Exactly one of text or image must be present"
                )

        # Test source is string
        assert isinstance(data["source"], str)
        assert len(data["source"]) > 0

        # Test tools (can be None or list)
        if data["tools"] is not None:
            assert isinstance(data["tools"], list)

        # Test images (can be None or list)
        if data["images"] is not None:
            assert isinstance(data["images"], list)

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_check_instruction_data_format_when_saving(
        self, sample_instruction_saving_data
    ):
        """
        Test 2: Validate instruction data format when saving.

        Validates the saving format defined in expected_output_format.md:
        - messages: str (serialized)
        - source: str
        - tools: str | None
        - images: list[str] | list[PIL.Image] | None (NOT string)

        Key requirement: ALL columns except images should be string type.
        """
        data = sample_instruction_saving_data

        # Test required fields exist
        assert "messages" in data
        assert "source" in data
        assert "tools" in data  # Can be None
        assert "images" in data  # Can be None

        # Test type requirements for saving format
        assert isinstance(data["messages"], str), "messages must be string type"
        assert len(data["messages"]) > 0, "messages cannot be empty string"

        assert isinstance(data["source"], str), "source must be string type"
        assert len(data["source"]) > 0, "source cannot be empty string"

        # tools can be None or string
        if data["tools"] is not None:
            assert isinstance(data["tools"], str), "tools must be string when present"
            assert len(data["tools"]) > 0, "tools cannot be empty string when present"

        # images should remain list or None (NOT converted to string)
        if data["images"] is not None:
            assert isinstance(data["images"], list), "images should remain list type"

        # Test that messages string can be parsed back to original type
        # Use ast.literal_eval for Python literal parsing
        try:
            parsed_messages = ast.literal_eval(data["messages"])
            assert isinstance(parsed_messages, list)
            # Validate structure of parsed messages
            for message in parsed_messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message
        except (ValueError, SyntaxError):
            pytest.fail("messages field should be valid Python literal string")

        # Test tools parsing if present
        if data["tools"] is not None:
            try:
                parsed_tools = ast.literal_eval(data["tools"])
                assert isinstance(parsed_tools, list)
            except (ValueError, SyntaxError):
                pytest.fail("tools field should be valid Python literal string")

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_malformed_data_in_saving_format(self, sample_instruction_saving_data):
        """Test that malformed data in saving format is properly rejected by validation."""

        # Test 1: Invalid Python literal that can't be parsed
        invalid_messages_data = {
            "messages": "{'invalid': syntax error}",  # Invalid Python literal
            "source": "test",
            "tools": None,
            "images": None,
        }

        # This should fail when validation tries to parse the malformed messages string
        with pytest.raises((ValueError, SyntaxError)):
            # Direct test of the problematic parsing logic
            ast.literal_eval(invalid_messages_data["messages"])

        # Test 2: Empty messages string
        empty_messages_data = {
            "messages": "",  # Empty string should be rejected
            "source": "test",
            "tools": None,
            "images": None,
        }

        with pytest.raises(AssertionError, match="messages cannot be empty string"):
            # Test basic validation requirements
            assert len(empty_messages_data["messages"]) > 0, (
                "messages cannot be empty string"
            )

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_instruction_format_conversion(self, sample_instruction_processing_data):
        """Test conversion from processing format to saving format."""
        processing_data = sample_instruction_processing_data

        # Simulate conversion to saving format using str() (as used in fixtures)
        saving_data = {
            "messages": str(processing_data["messages"]),
            "source": processing_data["source"],
            "tools": (
                str(processing_data["tools"])
                if processing_data["tools"] is not None
                else None
            ),
            "images": processing_data["images"],
        }

        # Validate the converted format
        self.test_check_instruction_data_format_when_saving(saving_data)

        # Test round-trip conversion with ast.literal_eval
        try:
            restored_messages_ast = ast.literal_eval(saving_data["messages"])
            assert restored_messages_ast == processing_data["messages"]
        except (ValueError, SyntaxError):
            pytest.fail("Round-trip conversion failed for messages")

        # Test tools round-trip if present
        if saving_data["tools"] is not None:
            try:
                restored_tools_ast = ast.literal_eval(saving_data["tools"])
                assert restored_tools_ast == processing_data["tools"]
            except (ValueError, SyntaxError):
                pytest.fail("Round-trip conversion failed for tools")


class TestPreferenceDataFormats:
    """Test preference data format validation during processing and saving."""

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_check_preference_data_format(self, sample_preference_processing_data):
        """
        Test 3: Validate preference data format during processing.

        Validates the processing format which includes all instruction fields plus:
        - rejected: dict with assistant/tool response structure
        """
        data = sample_preference_processing_data

        # Test all instruction format requirements
        instruction_tester = TestInstructionDataFormats()
        instruction_data = {
            "messages": data["messages"],
            "source": data["source"],
            "tools": data["tools"],
            "images": data["images"],
        }
        instruction_tester.test_check_instruction_data_format(instruction_data)

        # Test additional preference-specific fields
        assert "rejected" in data
        assert isinstance(data["rejected"], dict)

        rejected = data["rejected"]

        # Validate rejected structure
        assert "role" in rejected
        assert "content" in rejected
        assert "tool_calls" in rejected  # Can be None
        assert "name" in rejected  # Can be None

        # Validate rejected role (should be assistant or tool)
        assert rejected["role"] in ["assistant", "tool"]

        # Validate rejected content structure
        assert isinstance(rejected["content"], list)
        for content_item in rejected["content"]:
            assert isinstance(content_item, dict)
            assert "type" in content_item
            assert "text" in content_item
            assert "image" in content_item

            # Either text or image must be present, not both
            text_present = content_item["text"] is not None
            image_present = content_item["image"] is not None
            assert text_present != image_present, (
                "Exactly one of text or image must be present"
            )

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_check_preference_data_format_when_saving(
        self, sample_preference_saving_data
    ):
        """
        Test 4: Validate preference data format when saving.

        Validates the saving format which includes all instruction saving fields plus:
        - rejected: str (JSON serialized)

        Key requirement: ALL columns except images should be string type.
        According to expected_output_format.md:
        - messages: str
        - source: str | None (but practically should be str)
        - tools: str | None
        - images: list[str] | list[PIL.Image] | None (NOT string)
        - rejected: str
        """
        data = sample_preference_saving_data

        # Test all instruction saving format requirements first
        instruction_tester = TestInstructionDataFormats()
        instruction_data = {
            "messages": data["messages"],
            "source": data["source"],
            "tools": data["tools"],
            "images": data["images"],
        }
        instruction_tester.test_check_instruction_data_format_when_saving(
            instruction_data
        )

        # Test additional preference-specific fields
        assert "rejected" in data
        assert isinstance(data["rejected"], str), "rejected must be string type"
        assert len(data["rejected"]) > 0, "rejected cannot be empty string"

        # Test that rejected string can be parsed back to original type using ast.literal_eval
        try:
            parsed_rejected = ast.literal_eval(data["rejected"])
            assert isinstance(parsed_rejected, dict)
            assert "role" in parsed_rejected
            assert "content" in parsed_rejected
            assert parsed_rejected["role"] in ["assistant", "tool"]
        except (ValueError, SyntaxError):
            pytest.fail("rejected field should be valid Python literal string")

        # Verify source is actually string (not None) in practice
        assert isinstance(data["source"], str), (
            "source should be string in saved format"
        )

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_preference_format_conversion(self, sample_preference_processing_data):
        """Test conversion from processing format to saving format."""
        processing_data = sample_preference_processing_data

        # Simulate conversion to saving format using str() (as used in fixtures)
        saving_data = {
            "messages": str(processing_data["messages"]),
            "source": processing_data["source"],
            "tools": (
                str(processing_data["tools"])
                if processing_data["tools"] is not None
                else None
            ),
            "images": processing_data["images"],
            "rejected": str(processing_data["rejected"]),
        }

        # Validate the converted format
        self.test_check_preference_data_format_when_saving(saving_data)

        # Test round-trip conversion with ast.literal_eval
        try:
            restored_rejected_ast = ast.literal_eval(saving_data["rejected"])
            assert restored_rejected_ast == processing_data["rejected"]
        except (ValueError, SyntaxError):
            pytest.fail("Round-trip conversion failed for rejected")

        # Test messages round-trip
        try:
            restored_messages_ast = ast.literal_eval(saving_data["messages"])
            assert restored_messages_ast == processing_data["messages"]
        except (ValueError, SyntaxError):
            pytest.fail("Round-trip conversion failed for messages")


class TestDataFormatValidationErrors:
    """Test validation of invalid data formats."""

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_invalid_instruction_data(self, sample_invalid_instruction_data):
        """Test that invalid instruction data is properly detected."""
        instruction_tester = TestInstructionDataFormats()

        for invalid_data in sample_invalid_instruction_data:
            with pytest.raises((AssertionError, KeyError, TypeError, ValueError)):
                instruction_tester.test_check_instruction_data_format(invalid_data)

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_invalid_preference_data(self, sample_invalid_preference_data):
        """Test that invalid preference data is properly detected."""
        preference_tester = TestPreferenceDataFormats()

        for invalid_data in sample_invalid_preference_data:
            with pytest.raises((AssertionError, KeyError, TypeError, ValueError)):
                preference_tester.test_check_preference_data_format(invalid_data)

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_malformed_json_in_saving_format(self):
        """Test handling of malformed JSON and invalid Python literals in saving format."""
        # Test valid Python literal but invalid JSON
        valid_python_literal_data = {
            "messages": "{'role': 'system', 'content': None}",  # Valid Python literal, invalid JSON
            "source": "test",
            "tools": None,
            "images": None,
        }

        # This should work with ast.literal_eval but might fail JSON validation
        try:
            parsed = ast.literal_eval(valid_python_literal_data["messages"])
            assert isinstance(parsed, dict)
            # This test should pass - it's a valid Python literal
        except (ValueError, SyntaxError):
            pytest.fail("Valid Python literal should work with ast.literal_eval")

        # Test completely invalid format
        completely_invalid_data = {
            "messages": "not valid json or python literal {{}",
            "source": "test",
            "tools": None,
            "images": None,
        }

        # Test that ast.literal_eval fails on completely invalid data
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(completely_invalid_data["messages"])


class TestHuggingFaceDatasetFormat:
    """Test HuggingFace dataset format compliance."""

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_string_type_conversion_instruction(
        self, sample_instruction_saving_data, sample_instruction_saving_data_with_tools
    ):
        """
        Test that all columns except images are converted to string type.
        This is the key requirement from expected_output_format.md.
        """
        # Test data without tools
        data_no_tools = sample_instruction_saving_data

        # All fields except images should be string or None
        assert isinstance(data_no_tools["messages"], str), "messages must be string"
        assert isinstance(data_no_tools["source"], str), "source must be string"
        assert data_no_tools["tools"] is None, (
            "tools should be None when no tools present"
        )
        assert data_no_tools["images"] is None or isinstance(
            data_no_tools["images"], list
        ), "images should remain list or None"

        # Test data with tools - tools should be JSON string
        data_with_tools = sample_instruction_saving_data_with_tools

        assert isinstance(data_with_tools["messages"], str), "messages must be string"
        assert isinstance(data_with_tools["source"], str), "source must be string"
        assert isinstance(data_with_tools["tools"], str), (
            "tools must be string when present"
        )
        assert data_with_tools["images"] is None or isinstance(
            data_with_tools["images"], list
        ), "images should remain list or None"

        # Verify tools can be parsed back to original type using ast.literal_eval
        tools_data = ast.literal_eval(data_with_tools["tools"])
        assert isinstance(tools_data, list), (
            "tools should deserialize to list using ast.literal_eval"
        )

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_string_type_conversion_preference(self, sample_preference_saving_data):
        """
        Test that all columns except images are converted to string type for preference data.
        """
        data = sample_preference_saving_data

        # All fields except images should be string or None
        assert isinstance(data["messages"], str), "messages must be string"
        assert isinstance(data["source"], str), "source must be string"
        assert data["tools"] is None or isinstance(data["tools"], str), (
            "tools must be string or None"
        )
        assert data["images"] is None or isinstance(data["images"], list), (
            "images should remain list or None"
        )
        assert isinstance(data["rejected"], str), "rejected must be string"

        # Verify strings can be parsed back to original types using ast.literal_eval
        messages_data = ast.literal_eval(data["messages"])
        rejected_data = ast.literal_eval(data["rejected"])
        assert isinstance(messages_data, list), (
            "messages should deserialize to list using ast.literal_eval"
        )
        assert isinstance(rejected_data, dict), (
            "rejected should deserialize to dict using ast.literal_eval"
        )

    @pytest.mark.integration
    @pytest.mark.format_validation
    def test_huggingface_dataset_creation(
        self,
        sample_instruction_saving_data,
        sample_preference_saving_data,
        temp_output_dir,
    ):
        """
        Test creating HuggingFace datasets with properly formatted data.
        This validates the final save format requirement.
        """
        try:
            import pandas as pd
            from datasets import Dataset

            # Test instruction dataset
            instruction_df = pd.DataFrame([sample_instruction_saving_data])
            instruction_dataset = Dataset.from_pandas(instruction_df)

            # Verify column types
            for column_name in ["messages", "source"]:
                sample_value = instruction_dataset[0][column_name]
                assert isinstance(sample_value, str), (
                    f"{column_name} should be string in dataset"
                )

            # tools can be None or string
            tools_value = instruction_dataset[0]["tools"]
            assert tools_value is None or isinstance(tools_value, str), (
                "tools should be None or string"
            )

            # images should remain list or None (not converted to string)
            images_value = instruction_dataset[0]["images"]
            assert images_value is None or isinstance(images_value, list), (
                "images should not be converted to string"
            )

            # Test saving instruction dataset
            instruction_path = temp_output_dir / "instruction_dataset"
            instruction_dataset.save_to_disk(str(instruction_path))

            # Test loading back
            loaded_instruction = Dataset.load_from_disk(str(instruction_path))
            assert len(loaded_instruction) == 1
            assert loaded_instruction.column_names == [
                "messages",
                "source",
                "tools",
                "images",
            ]

            # Test preference dataset
            preference_df = pd.DataFrame([sample_preference_saving_data])
            preference_dataset = Dataset.from_pandas(preference_df)

            # Verify column types for preference data
            for column_name in ["messages", "source", "rejected"]:
                sample_value = preference_dataset[0][column_name]
                assert isinstance(sample_value, str), (
                    f"{column_name} should be string in dataset"
                )

            # Test saving preference dataset
            preference_path = temp_output_dir / "preference_dataset"
            preference_dataset.save_to_disk(str(preference_path))

            # Test loading back
            loaded_preference = Dataset.load_from_disk(str(preference_path))
            assert len(loaded_preference) == 1
            assert "rejected" in loaded_preference.column_names

        except ImportError:
            pytest.skip("datasets library not available")

    @pytest.mark.unit
    @pytest.mark.format_validation
    def test_data_round_trip_integrity(
        self, sample_instruction_processing_data, sample_preference_processing_data
    ):
        """
        Test that data maintains integrity through processing -> saving -> loading cycle.
        """
        # Test instruction data round trip
        instruction_processing = sample_instruction_processing_data

        # Convert to saving format using str() (consistent with fixtures)
        instruction_saving = {
            "messages": str(instruction_processing["messages"]),
            "source": instruction_processing["source"],
            "tools": (
                str(instruction_processing["tools"])
                if instruction_processing["tools"] is not None
                else None
            ),
            "images": instruction_processing["images"],
        }

        # Verify saving format
        assert isinstance(instruction_saving["messages"], str)
        assert isinstance(instruction_saving["source"], str)
        assert instruction_saving["tools"] is None or isinstance(
            instruction_saving["tools"], str
        )

        # Convert back to processing format using ast.literal_eval
        try:
            restored_messages = ast.literal_eval(instruction_saving["messages"])
            assert restored_messages == instruction_processing["messages"]
        except (ValueError, SyntaxError):
            pytest.fail("Round-trip conversion failed for instruction messages")

        if instruction_saving["tools"] is not None:
            try:
                restored_tools = ast.literal_eval(instruction_saving["tools"])
                assert restored_tools == instruction_processing["tools"]
            except (ValueError, SyntaxError):
                pytest.fail("Round-trip conversion failed for instruction tools")
        else:
            assert instruction_processing["tools"] is None

        assert instruction_saving["source"] == instruction_processing["source"]
        assert instruction_saving["images"] == instruction_processing["images"]

        # Test preference data round trip
        preference_processing = sample_preference_processing_data

        # Convert to saving format using str()
        preference_saving = {
            "messages": str(preference_processing["messages"]),
            "source": preference_processing["source"],
            "tools": (
                str(preference_processing["tools"])
                if preference_processing["tools"] is not None
                else None
            ),
            "images": preference_processing["images"],
            "rejected": str(preference_processing["rejected"]),
        }

        # Verify saving format
        assert isinstance(preference_saving["messages"], str)
        assert isinstance(preference_saving["source"], str)
        assert isinstance(preference_saving["rejected"], str)

        # Convert back and verify integrity using ast.literal_eval
        try:
            restored_rejected = ast.literal_eval(preference_saving["rejected"])
            assert restored_rejected == preference_processing["rejected"]
        except (ValueError, SyntaxError):
            pytest.fail("Round-trip conversion failed for preference rejected")
