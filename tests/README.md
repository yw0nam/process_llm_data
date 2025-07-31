# Test Suite Documentation

## Overview

This test suite provides comprehensive Test-Driven Development (TDD) support for the LLM data processing pipeline restructuring. The tests validate data formats specified in `expected_output_format.md` and support the improvement plan outlined in `improvement_plan.md`.

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── run_tests.py                   # Test runner script  
├── fixtures/
│   └── sample_data.py            # Realistic sample data for testing
├── unit/
│   └── test_datasets/
│       ├── test_formats.py       # Core format validation tests (4 requested)
└── integration/
    └── test_data_pipeline.py     # End-to-end pipeline tests
```

## The 4 Requested Tests

### 1. `test_check_instruction_data_format`
**Location**: `tests/unit/test_datasets/test_formats.py::TestInstructionDataFormats::test_check_instruction_data_format`

**Purpose**: Validates instruction data format during processing phase.

**Validates**:
- `messages`: list[dict] with proper structure
- `source`: str (required)
- `tools`: list[dict] | None
- `images`: list[str] | list[PIL.Image] | None
- Message role validation (system, user, assistant, tool)
- Content structure with type/text/image fields
- Mutual exclusivity of text and image

### 2. `test_check_instruction_data_format_when_saving`
**Location**: `tests/unit/test_datasets/test_formats.py::TestInstructionDataFormats::test_check_instruction_data_format_when_saving`

**Purpose**: Validates instruction data format when saving to HuggingFace datasets.

**Validates**:
- `messages`: str (JSON serialized)
- `source`: str (required)
- `tools`: str | None (JSON serialized)
- `images`: list[str] | list[PIL.Image] | None
- JSON string validity and round-trip conversion

### 3. `test_check_preference_data_format`
**Location**: `tests/unit/test_datasets/test_formats.py::TestPreferenceDataFormats::test_check_preference_data_format`

**Purpose**: Validates preference data format during processing phase.

**Validates**:
- All instruction format requirements
- `rejected`: dict with assistant/tool response structure
- Rejected role validation (assistant, tool only)
- Rejected content structure matching message format

### 4. `test_check_preference_data_format_when_saving`
**Location**: `tests/unit/test_datasets/test_formats.py::TestPreferenceDataFormats::test_check_preference_data_format_when_saving`

**Purpose**: Validates preference data format when saving to HuggingFace datasets.

**Validates**:
- All instruction saving format requirements
- `rejected`: str (JSON serialized)
- JSON string validity for rejected field

## Additional Valuable Tests

### Data Schema Validation (`test_validators.py`)
- **Message role validation**: Ensures only valid roles are accepted
- **Content type validation**: Validates text/image content structure
- **Mutual exclusivity**: Ensures text and image are mutually exclusive
- **Empty data handling**: Tests edge cases with empty/minimal data

### Data Transformation Tests
- **JSON serialization/deserialization**: Round-trip testing
- **Special characters**: Unicode and special character handling
- **Large data serialization**: Performance with large datasets

### Dataset Integration Tests
- **DataFrame creation**: pandas DataFrame compatibility
- **HuggingFace compatibility**: Dataset format validation
- **Batch processing**: Consistent format across batches

### Error Handling Tests
- **JSON decode errors**: Malformed JSON handling
- **Missing required fields**: Field validation
- **Type validation**: Wrong data type detection

### Performance Tests
- **Large dataset processing**: Memory and speed testing
- **Memory efficiency**: Resource usage validation
- **Concurrent processing**: Thread safety simulation

### Integration Tests (`test_data_pipeline.py` & `test_foundation.py`)
- **End-to-end instruction pipeline**: Complete workflow testing
- **End-to-end preference pipeline**: Complete workflow with rejected data
- **Foundation integration**: Phase 1 architecture validation
- **Mixed dataset processing**: Handling different data types
- **Batch processing pipeline**: Large-scale processing
- **Configuration integration**: Mock configuration testing
- **Dataset registry**: Plugin system testing
- **Error recovery**: Pipeline resilience testing

## Running Tests

### Quick Start - Run the 4 Requested Tests
```bash
cd /home/spow12/codes/2024_upper/NLP/process_llm_data
uv run python tests/run_tests.py specific
```

### Run All Format Validation Tests
```bash
uv run python tests/run_tests.py format
```

### Run Unit Tests
```bash
uv run python tests/run_tests.py unit
```

### Run Integration Tests
```bash
uv run python tests/run_tests.py integration
```

### Run All Tests
```bash
uv run python tests/run_tests.py all
```

### Manual pytest Execution
```bash
# Run specific test
uv run  pytest tests/unit/test_datasets/test_formats.py::TestInstructionDataFormats::test_check_instruction_data_format -v

# Run with coverage
uv run  pytest tests/ --cov=src --cov-report=html

# Run only format validation tests
uv run  pytest tests/ -m format_validation -v
```

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.format_validation`: Data format validation tests
- `@pytest.mark.slow`: Tests that take longer to run

## Dependencies

### Required for Basic Testing
```bash
uv add pytest
```

### Optional for Enhanced Testing
```bash
uv add pytest-cov coverage  # Coverage reporting
uv add datasets pandas       # For HuggingFace integration tests
```

## TDD Workflow

1. **Red Phase**: Tests fail initially (no implementation)
2. **Green Phase**: Implement minimal code to pass tests
3. **Refactor Phase**: Improve code while keeping tests passing

### Development Workflow
1. Run the 4 specific tests: `uv run python tests/run_tests.py specific`
2. Implement data validators to make tests pass
3. Add more comprehensive tests as needed
4. Refactor and optimize while maintaining test coverage

## Test Data

### Sample Data Sources
- **Realistic conversations**: Educational Q&A, coding assistance, productivity advice
- **Tool usage examples**: Function calling scenarios
- **Image-based conversations**: Vision model interactions
- **Invalid data samples**: For negative testing

### Data Characteristics
- Follows exact format specifications from `expected_output_format.md`
- Includes edge cases and error conditions
- Supports both instruction and preference data types
- Scalable for performance testing

## Troubleshooting

### Common Issues

1. **Import errors**: Install missing dependencies
2. **Path issues**: Run tests from project root directory
3. **Fixture not found**: Check `conftest.py` is properly loaded
4. **JSON errors**: Verify sample data format in fixtures

### Debugging
```bash
# Run with verbose output and stop on first failure
uv run pytest tests/ -v -x --tb=long

# Run specific test with debugging
uv run pytest tests/unit/test_datasets/test_formats.py -v -s --tb=long
```

## Contributing

When adding new tests:

1. Follow the naming conventions
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Add documentation for complex test scenarios
5. Update this README if adding new test categories

## Integration with Improvement Plan

These tests support the improvement plan phases:

- **Phase 1 (Foundation)**: Core format validation tests
- **Phase 2 (Data Layer)**: Dataset registry and validation tests  
- **Phase 3 (Processor Modernization)**: Processing pipeline tests
- **Phase 4 (Testing & Quality)**: Comprehensive test coverage
- **Phase 5 (Documentation & CLI)**: Integration and performance tests

The test suite ensures that the restructuring maintains data integrity and format compliance throughout the development process.
