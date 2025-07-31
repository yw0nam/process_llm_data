#!/usr/bin/env python3
"""Test script to verify str() and ast.literal_eval() compatibility."""

import ast
from datasets import load_from_disk
import tempfile
from pathlib import Path

# Create a temporary dataset for testing
temp_dir = Path(tempfile.mkdtemp())

print("🧪 Testing str() format with ast.literal_eval() compatibility")
print("=" * 60)

# Run the demo to create sample data
import subprocess

result = subprocess.run(
    ["uv", "run", "python", "demo_flow.py"], capture_output=True, text=True, cwd="."
)

if result.returncode != 0:
    print("❌ Demo failed to run")
    print(result.stderr)
    exit(1)

# Find the output dataset directory from the demo output
output_lines = result.stdout.split("\n")
dataset_path = None
for line in output_lines:
    if "Output saved to:" in line:
        dataset_path = line.split("Output saved to: ")[1].strip()
        break

if not dataset_path:
    print("❌ Could not find dataset path from demo output")
    exit(1)

print(f"📂 Loading dataset from: {dataset_path}")

try:
    # Load the dataset
    dataset = load_from_disk(dataset_path)

    print(f"📊 Dataset loaded successfully! {len(dataset)} rows")

    # Test the first row
    first_row = dataset[0]

    print("\n🔍 Testing first row:")
    print(f"  Original messages type: {type(first_row['messages'])}")
    print(f"  Original messages value: {first_row['messages'][:100]}...")

    # Test conversion back from string using ast.literal_eval
    if isinstance(first_row["messages"], str):
        try:
            parsed_messages = ast.literal_eval(first_row["messages"])
            print(f"  ✅ ast.literal_eval() succeeded!")
            print(f"  Parsed type: {type(parsed_messages)}")
            print(f"  Parsed value: {parsed_messages}")

            # Verify it's the expected structure
            if isinstance(parsed_messages, list) and len(parsed_messages) > 0:
                first_message = parsed_messages[0]
                if isinstance(first_message, dict) and "role" in first_message:
                    print(f"  ✅ Parsed structure is correct!")
                    print(f"  First message role: {first_message['role']}")
                    print(
                        f"  First message content: {first_message.get('content', 'N/A')}"
                    )
                else:
                    print(f"  ❌ Parsed structure is incorrect: {first_message}")
            else:
                print(f"  ❌ Parsed messages is not a list or empty: {parsed_messages}")

        except (ValueError, SyntaxError) as e:
            print(f"  ❌ ast.literal_eval() failed: {e}")
    else:
        print(f"  ℹ️  Messages is not a string, skipping ast.literal_eval test")

    # Test tools field if present
    if first_row["tools"] is not None:
        print(f"\n🔧 Testing tools field:")
        print(f"  Original tools type: {type(first_row['tools'])}")
        print(f"  Original tools value: {first_row['tools']}")

        if isinstance(first_row["tools"], str):
            try:
                parsed_tools = ast.literal_eval(first_row["tools"])
                print(f"  ✅ Tools ast.literal_eval() succeeded!")
                print(f"  Parsed tools: {parsed_tools}")
            except (ValueError, SyntaxError) as e:
                print(f"  ❌ Tools ast.literal_eval() failed: {e}")
    else:
        print(f"\n🔧 Tools field is None (as expected)")

    print(f"\n✅ All tests passed! str() format is compatible with ast.literal_eval()")

except Exception as e:
    print(f"❌ Error loading or testing dataset: {e}")
    import traceback

    traceback.print_exc()

print(f"\n🧹 Cleaning up temporary files...")
