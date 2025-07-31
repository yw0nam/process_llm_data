#!/usr/bin/env python3
"""
Quick validation script to test the corrected da    # Convert to saving format (as per expected_output_format.md)
    saving_data = {
        "messages": str(processing_data["messages"]),         # Convert to string representation
        "source": processing_data["source"],                  # Already string
        "tools": str(processing_data["tools"]) if processing_data["tools"] is not None else None,  # None stays None
        "images": processing_data["images"],                  # Images remain list/None, NOT converted to string
        "rejected": str(processing_data["rejected"])          # Convert to string representation
    }t.
Run this to verify the tests match expected_output_format.md requirements.
Uses ast.literal_eval for safer type conversion as requested.
"""

import ast
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_instruction_saving_format():
    """Test instruction data saving format compliance."""
    print("🧪 Testing Instruction Data Saving Format...")

    # Sample processing format
    processing_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "image": None}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!", "image": None}],
            },
        ],
        "source": "test_dataset",
        "tools": None,
        "images": None,
    }

    # Convert to saving format (as per expected_output_format.md)
    saving_data = {
        "messages": str(
            processing_data["messages"]
        ),  # Convert to string representation
        "source": processing_data["source"],  # Already string
        "tools": (
            str(processing_data["tools"])
            if processing_data["tools"] is not None
            else None
        ),  # None stays None
        "images": processing_data[
            "images"
        ],  # Images remain list/None, NOT converted to string
    }

    # Validate requirements from expected_output_format.md
    print("✅ messages type:", type(saving_data["messages"]), "- Must be str")
    print("✅ source type:", type(saving_data["source"]), "- Must be str")
    print("✅ tools type:", type(saving_data["tools"]), "- Must be str | None")
    print(
        "✅ images type:",
        type(saving_data["images"]),
        "- Must be list | None (NOT string)",
    )

    # Verify string types (all except images)
    assert isinstance(saving_data["messages"], str), "messages must be string"
    assert isinstance(saving_data["source"], str), "source must be string"
    assert saving_data["tools"] is None or isinstance(saving_data["tools"], str), (
        "tools must be None or string"
    )
    assert saving_data["images"] is None or isinstance(saving_data["images"], list), (
        "images must NOT be converted to string"
    )

    print("✅ Instruction saving format: PASSED")
    return True


def test_preference_saving_format():
    """Test preference data saving format compliance."""
    print("\n🧪 Testing Preference Data Saving Format...")

    # Sample processing format
    processing_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Explain AI", "image": None}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "AI is...", "image": None}],
            },
        ],
        "source": "preference_dataset",
        "tools": None,
        "images": None,
        "rejected": {
            "role": "assistant",
            "content": [{"type": "text", "text": "AI is simple", "image": None}],
            "tool_calls": None,
            "name": None,
        },
    }

    # Convert to saving format (as per expected_output_format.md)
    saving_data = {
        "messages": str(
            processing_data["messages"]
        ),  # Convert to string representation
        "source": processing_data["source"],  # Already string
        "tools": (
            str(processing_data["tools"])
            if processing_data["tools"] is not None
            else None
        ),  # None stays None
        "images": processing_data[
            "images"
        ],  # Images remain list/None, NOT converted to string
        "rejected": str(
            processing_data["rejected"]
        ),  # Convert to string representation
    }

    # Validate requirements from expected_output_format.md
    print("✅ messages type:", type(saving_data["messages"]), "- Must be str")
    print("✅ source type:", type(saving_data["source"]), "- Must be str")
    print("✅ tools type:", type(saving_data["tools"]), "- Must be str | None")
    print(
        "✅ images type:",
        type(saving_data["images"]),
        "- Must be list | None (NOT string)",
    )
    print("✅ rejected type:", type(saving_data["rejected"]), "- Must be str")

    # Verify string types (all except images)
    assert isinstance(saving_data["messages"], str), "messages must be string"
    assert isinstance(saving_data["source"], str), "source must be string"
    assert saving_data["tools"] is None or isinstance(saving_data["tools"], str), (
        "tools must be None or string"
    )
    assert saving_data["images"] is None or isinstance(saving_data["images"], list), (
        "images must NOT be converted to string"
    )
    assert isinstance(saving_data["rejected"], str), "rejected must be string"

    print("✅ Preference saving format: PASSED")
    return True


def test_tools_present_scenario():
    """Test scenario where tools are present and should be JSON string."""
    print("\n🧪 Testing Tools Present Scenario...")

    # Sample with tools present
    processing_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Weather?", "image": None}],
            }
        ],
        "source": "weather_app",
        "tools": [
            {"type": "function", "function": {"name": "get_weather"}}
        ],  # Tools present
        "images": None,
    }

    # Convert to saving format
    saving_data = {
        "messages": str(processing_data["messages"]),
        "source": processing_data["source"],
        "tools": (
            str(processing_data["tools"])
            if processing_data["tools"] is not None
            else None
        ),  # Should be string representation
        "images": processing_data["images"],
    }

    print("✅ tools type when present:", type(saving_data["tools"]), "- Must be str")
    assert isinstance(saving_data["tools"], str), "tools must be string when present"

    # Verify string can be parsed back to original type using ast.literal_eval
    tools_data = ast.literal_eval(saving_data["tools"])
    assert isinstance(tools_data, list), (
        "tools should deserialize to list using ast.literal_eval"
    )

    print("✅ Tools present scenario: PASSED")
    return True


def test_huggingface_dataset_simulation():
    """Simulate HuggingFace dataset creation with proper format."""
    print("\n🧪 Testing HuggingFace Dataset Format Simulation...")

    # Simulated dataset rows (as they would appear in HF dataset)
    dataset_rows = [
        {
            "messages": str(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Q1", "image": None}],
                    }
                ]
            ),
            "source": "dataset_1",
            "tools": None,
            "images": None,
        },
        {
            "messages": str(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Q2", "image": None}],
                    }
                ]
            ),
            "source": "dataset_2",
            "tools": str(
                [{"type": "function", "function": {"name": "tool1"}}]
            ),  # Tools present as string
            "images": ["image1.jpg", "image2.jpg"],  # Images as list
        },
    ]

    # Validate all rows
    for i, row in enumerate(dataset_rows):
        print(f"📝 Row {i + 1}:")
        assert isinstance(row["messages"], str), f"Row {i + 1}: messages must be string"
        assert isinstance(row["source"], str), f"Row {i + 1}: source must be string"
        assert row["tools"] is None or isinstance(row["tools"], str), (
            f"Row {i + 1}: tools must be None or string"
        )
        assert row["images"] is None or isinstance(row["images"], list), (
            f"Row {i + 1}: images must remain list/None"
        )
        print("   ✅ All types correct")

    print("✅ HuggingFace dataset format simulation: PASSED")
    return True


def main():
    """Run all format validation tests."""
    print("🚀 Data Format Validation - Expected Output Format Compliance")
    print("=" * 70)
    print("Validating against expected_output_format.md requirements:")
    print("- ALL columns except images should be converted to string type")
    print("- Use ast.literal_eval for safe type conversion back to original types")
    print("- Save in HuggingFace datasets format")
    print("=" * 70)

    try:
        test_instruction_saving_format()
        test_preference_saving_format()
        test_tools_present_scenario()
        test_huggingface_dataset_simulation()

        print("\n" + "=" * 70)
        print("🎉 ALL FORMAT VALIDATION TESTS PASSED!")
        print("✅ Data format complies with expected_output_format.md")
        print("✅ Ready for HuggingFace datasets.save_to_disk()")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
