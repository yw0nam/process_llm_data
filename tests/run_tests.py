"""
Test runner script for the LLM data processing pipeline.

This script provides an easy way to run all tests with proper configuration
and reporting during the TDD development process.
"""

import subprocess
import sys
from pathlib import Path

# AIDEV-NOTE: Test runner supporting TDD workflow during restructuring


def run_format_tests():
    """Run the specific format validation tests requested by user."""
    print("🧪 Running Format Validation Tests...")
    print("=" * 50)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_datasets/test_formats.py",
        "-v",
        "--tb=short",
        "-m",
        "format_validation",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_unit_tests():
    """Run all unit tests."""
    print("\n🔬 Running Unit Tests...")
    print("=" * 50)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-m",
        "unit",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m",
        "integration",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_all_tests():
    """Run all tests with coverage reporting."""
    print("\n🚀 Running All Tests...")
    print("=" * 50)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--durations=10",
    ]

    # Add coverage if available
    try:
        import coverage

        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    except ImportError:
        print("📝 Coverage not available (install with: pip install pytest-cov)")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_specific_tests():
    """Run the 4 specific tests requested by the user."""
    print("🎯 Running Specific Requested Tests...")
    print("=" * 50)

    specific_tests = [
        "tests/unit/test_datasets/test_formats.py::TestInstructionDataFormats::test_check_instruction_data_format",
        "tests/unit/test_datasets/test_formats.py::TestInstructionDataFormats::test_check_instruction_data_format_when_saving",
        "tests/unit/test_datasets/test_formats.py::TestPreferenceDataFormats::test_check_preference_data_format",
        "tests/unit/test_datasets/test_formats.py::TestPreferenceDataFormats::test_check_preference_data_format_when_saving",
    ]

    for test in specific_tests:
        print(f"\n▶️  Running: {test.split('::')[-1]}")
        cmd = [sys.executable, "-m", "pytest", test, "-v", "--tb=short"]

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"❌ Test failed: {test}")
            return False
        else:
            print(f"✅ Test passed: {test}")

    return True


def main():
    """Main test runner function."""
    print("🧪 LLM Data Processing Pipeline - Test Suite")
    print("=" * 60)
    print("Tests for TDD implementation of data format validation")
    print("Based on expected_output_format.md specifications")
    print("=" * 60)

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "format":
            success = run_format_tests()
        elif test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_integration_tests()
        elif test_type == "specific":
            success = run_specific_tests()
        elif test_type == "all":
            success = run_all_tests()
        else:
            print(f"❌ Unknown test type: {test_type}")
            print("Available options: format, unit, integration, specific, all")
            return 1
    else:
        print("🎯 Running specific requested tests by default...")
        success = run_specific_tests()

        if success:
            print("\n✅ All requested tests passed!")
            print("\n📋 Test Summary:")
            print("1. ✅ check_instruction_data_format")
            print("2. ✅ check_instruction_data_format_when_saving")
            print("3. ✅ check_preference_data_format")
            print("4. ✅ check_preference_data_format_when_saving")
        else:
            print("\n❌ Some tests failed. Check output above for details.")

    if success:
        print("\n🎉 Test suite completed successfully!")
        return 0
    else:
        print("\n💥 Test suite failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
