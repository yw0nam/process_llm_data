# Test Suite Documentation

## Overview

This test suite provides comprehensive testing support for the LLM data processing pipeline. The tests validate data formats specified in `expected_format.md` and support the modular architecture with dataset registry, processing functions, and YAML-driven configuration.

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── run_tests.py                   # Test runner script utilities
├── fixtures/
│   └── sample_data.py            # Realistic sample data for testing
├── unit/
│   └── test_datasets/
│       ├── test_formats.py       # Core format validation tests
│       └── test_registry.py      # Dataset registry and processing function tests
└── integration/
    ├── test_data_pipeline.py     # End-to-end pipeline tests
    ├── test_demo_flow.py         # Demo flow integration tests
    └── test_foundation.py        # Foundation architecture tests
```

## Core Test Coverage

### Format Validation Tests (`test_formats.py`)
**Status**: ✅ COMPLETE - All format validation tests passing

- **Instruction data format validation**: Processing and saving formats
- **Preference data format validation**: Processing and saving formats with rejected responses  
- **String serialization compatibility**: Uses `str()` and `ast.literal_eval()` as specified
- **HuggingFace dataset format compliance**: Validates final output format
- **Error handling**: Malformed data and edge cases

### Dataset Registry Tests (`test_registry.py`)
**Status**: ✅ COMPLETE - Full registry functionality tested

- **Dataset loader registration**: HuggingFace and local files loaders
- **Processing function registration**: Custom processing functions for different dataset types
- **Dataset creation and loading**: End-to-end dataset loading workflow
- **Processing function execution**: Applies registered processing functions to datasets
- **Error handling**: Missing functions, duplicate registrations, processing errors

### Integration Tests
**Status**: ✅ COMPLETE - End-to-end workflows validated

- **Complete flow testing** (`test_demo_flow.py`): Full YAML-driven pipeline
- **Data pipeline integration** (`test_data_pipeline.py`): Multi-dataset processing
- **Foundation architecture** (`test_foundation.py`): Core component integration
- **Format compatibility**: str() serialization and ast.literal_eval() parsing

## Architecture Overview

### Current Implementation Status ✅

**YAML-Driven Pipeline**: Complete flow implemented
- ✅ Version YAML configuration with dataset definitions
- ✅ Dataset registry with plugin architecture  
- ✅ Processing function registry for custom transformations
- ✅ Format converters ensuring strict output schema compliance
- ✅ Pipeline orchestration (load → process → merge → save)
- ✅ HuggingFace dataset output with metadata

**Data Format Compliance**: All specifications met
- ✅ Uses `str()` for serialization and `ast.literal_eval()` for parsing
- ✅ Expected schema: `messages`, `source`, `tools`, `images` (all strings except images)
- ✅ Full validation for both instruction and preference data formats
- ✅ Round-trip compatibility verified through comprehensive testing

**Test Coverage**: Comprehensive validation
- ✅ **65 tests passing** covering all core functionality
- ✅ Unit tests for format validation and registry functionality
- ✅ Integration tests for end-to-end pipeline workflows
- ✅ Demo flow tests validating complete YAML-driven processing

## Running Tests

### Run All Tests (Recommended)
```bash
uv run pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Format validation tests
uv run pytest tests/unit/test_datasets/test_formats.py -v

# Dataset registry tests  
uv run pytest tests/unit/test_datasets/test_registry.py -v

# Integration tests
uv run pytest tests/integration/ -v

# Demo flow tests
uv run pytest tests/test_demo_flow.py -v
```

### Run with Coverage
```bash
uv run pytest tests/ --cov=src --cov-report=html -v
```

### Run the Demo Flow Manually
```bash
uv run python demo.py
```

## Key Features Validated

### YAML-Driven Configuration
- **Version files**: Define datasets, processing functions, and output settings
- **Dataset registration**: Automatic loading based on type (huggingface, local_files)
- **Processing functions**: Custom transformations registered by name
- **Output configuration**: HuggingFace dataset format with metadata

### Format Converters  
- **Alpaca to messages**: Instruction/input/output → conversation format
- **SmolTalk to messages**: Multi-turn conversations → standardized format
- **Custom processing**: Extensible registry for new dataset types
- **Strict schema**: All outputs conform to expected format specification

### Pipeline Flow
1. **Load YAML configuration** defining datasets and processing
2. **Register datasets** using appropriate loaders (HF, local files)
3. **Apply processing functions** to convert raw data to expected format
4. **Merge datasets** according to specified strategy (concatenate, etc.)
5. **Save as HuggingFace dataset** with proper metadata and validation

### String Serialization  
- **Uses str() instead of json.dumps()**: As required by specification
- **Compatible with ast.literal_eval()**: Verified through testing
- **Handles complex data structures**: Lists, dicts, nested objects
- **Maintains data integrity**: Round-trip conversion validated

## Current Status

✅ **COMPLETE**: All core functionality implemented and tested
- **65 tests passing** with comprehensive coverage
- **YAML-driven pipeline** fully functional  
- **Dataset registry** with processing functions
- **Format compliance** with expected specifications
- **String serialization** using str() and ast.literal_eval()
- **Demo flow** validating end-to-end functionality

The test suite validates that the implementation meets all requirements for YAML-driven, processing-function-based dataset processing with strict output schema compliance.
