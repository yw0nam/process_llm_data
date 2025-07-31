# 📖 PROJECT_BIBLE.md: LLM Data Processing Repository
*Last updated 2025-07-29*

> **Purpose** – This document is the onboarding manual for every AI pair programmer and human developer who contributes to this repository. It defines the coding standards, architecture, and core principles to ensure consistent and high-quality development during the repository restructuring process.

---

## 1. Project Overview

The LLM Data Processing Repository is a comprehensive toolkit for preprocessing datasets for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Its primary goal is to provide a modular, scalable, and maintainable pipeline for processing various LLM training datasets.

- **Project Name**: `process_llm_data`
- **Repository**: [https://github.com/yw0nam/process_llm_data](https://github.com/yw0nam/process_llm_data)
- **Current Status**: Major restructuring in progress (see `improvement_plan.md`)

---

## 2. Core Rules & Prohibitions

| #: | AI *may* do                                                            | AI *must NOT* do                                                                    |
|---|------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| G-0 | Whenever unsure about something that's related to the project, ask the developer for clarification before making changes.    |  ❌ Write changes or use tools when you are not sure about something project specific, or if you don't have context for a particular feature/decision. |
| G-1 | Generate code **only inside** relevant source directories (e.g., `src/processors/`, `src/datasets/`, `src/utils/`) or explicitly pointed files. Follow the new structure defined in `improvement_plan.md`.    | ❌ Touch existing files in `inst/`, `preference/`, or `tools/` during restructuring without explicit permission. |
| G-2 | Add/update **`AIDEV-NOTE:` anchor comments** near non-trivial edited code. | ❌ Delete or mangle existing `AIDEV-` comments.                                     |
| G-3 | Follow lint/style configs (`pyproject.toml`) Use the project's configured linter, if available, instead of manually re-formatting code. | ❌ Re-format code to any other style.                                               |
| G-4 | For changes >300 LOC or >3 files, **ask for confirmation**.            | ❌ Refactor large modules without human guidance.                                     |
| G-5 | Stay within the current task context. Inform the dev if it'd be better to start afresh.                                  | ❌ Continue work from a prior prompt after "new task" – start a fresh session.      |

---

## 3. Coding Standards & Conventions

**Golden Rule**: All code must pass the `lint.sh` script checks before being committed. AI assistants must adhere to these rules when generating code.

- **Formatter**: `black` with a line-length of 88.
- **Import Sorter**: `isort` using the `black` profile.
- **Linter**: `ruff` for linting and formatting checks (to be added during Phase 5).
- **Naming**:
    - `snake_case` for functions and variables.
    - `PascalCase` for classes.
    - `SCREAMING_SNAKE_CASE` for constants.
- **Error Handling**: Use typed, hierarchical exceptions and proper logging.
- **Documentation**: Use Google-style docstrings for all public functions and classes.
- **Testing**: Test files should be located in the new `tests/` directory structure.

### Data Processing Specific Standards:

- **Dataset Loaders**: Must inherit from `BaseDataset` and be registered in the dataset registry.
- **Processors**: Must inherit from `BaseProcessor` and follow the factory pattern.
- **Configuration**: Use Pydantic schemas for type-safe configuration.
- **Logging**: Use structured logging with appropriate log levels.

### Test code:

- Start with the Core: Write tests for the most critical parts of the data processing pipeline first.
- Expand Incrementally: Once the core components are stable, gradually extend test suite to cover dataset loaders and utility functions.
- Use Coverage as a Guide: Aim for 90% test coverage for core components as specified in the improvement plan.

- 1. Core Data Processing Logic
    - What it is: The code that handles dataset loading, processing, and output generation.
    - Why it's first: Bugs in data processing can corrupt training datasets and affect model performance.
- 2. Dataset Integration Points
    - What it is: Dataset loaders that interact with external data sources (HuggingFace, local files, APIs).
    - Why it's important: These are common points of failure and should use mocking for external dependencies.
- 3. Configuration and Pipeline Logic
    - What it is: Configuration validation, pipeline orchestration, and data flow management.
    - Why it's important: Complex configuration logic needs thorough testing with various input combinations.

### Linting & Formatting Script
The following script (`lint.sh`) is used to enforce code quality. It formats, sorts imports, and lints the codebase automatically.

```bash
#!/bin/bash
# lint.sh

# 1. Black: Check and apply code formatting
echo "Running black..."
uv run black src tests
uv run black --check src tests

# 2. isort: Check and apply import sorting
echo "Running isort..."
uv run isort src tests
uv run isort --check src tests

# 3. Ruff: Lint, format, and apply automatic fixes
echo "Running ruff..."
uv run ruff check src tests --fix
uv run ruff format src tests
uv run ruff check src tests
```

### Error Handling Pattern
Catch specific, typed exceptions instead of generic `Exception`. Use proper logging for data processing errors.

```python
from src.core.exceptions import DataValidationError, ProcessingError

async def process_dataset(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Process data
        return processed_data
    except KeyError as e:
        # Re-raise with a typed, project-specific exception
        raise DataValidationError(f"Missing required field: {e}") from e
    except Exception as e:
        # Log unexpected errors and re-raise as ProcessingError
        logger.error(f"Unexpected error in dataset processing: {e}")
        raise ProcessingError(f"Dataset processing failed: {e}") from e
```

---


## 4. Technology Stack

| Category      | Technology                 | Description                                                                                                                              |
| :------------ | :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Language** | Python 3.10+               | The core programming language (currently using 3.10, considering upgrade to 3.11+).                                                   |
|               | Pydantic                   | Data validation and settings management (to be added in restructuring).                                                                 |
|               | Click                      | CLI framework (to be added for enhanced CLI).                                                                                           |
| **Data Processing**| pandas                | Primary data manipulation library.                                                                                                       |
|               | datasets (HuggingFace)     | For loading and processing HuggingFace datasets.                                                                                        |
|               | scikit-learn               | For data splitting and sampling operations.                                                                                              |
| **Tooling** | pip                        | Package manager (current, considering migration to `uv` or `poetry`).                                                                   |
| **Dependencies**| `pyproject.toml`           | See `[project.dependencies]` for main dependencies and `[project.optional-dependencies].dev` for development tools like `ruff`, `black`, and `isort`. |


---

## 5. Architecture & Directory Structure

### Current Structure (Legacy)
| Directory         | Description                                                                          |
| :---------------- | :----------------------------------------------------------------------------------- |
| `main.py`         | The application entrypoint using for configuration management.                |
| `basemodel.py`    | Base preprocessing class with common functionality.                                  |
| `configs/`        | configuration files for different processing versions.                        |
| `inst/`           | Instruction dataset processors organized by version (ver_1, ver_2).                 |
| `preference/`     | Preference dataset processors for DPO training.                                     |
| `tools/`          | Utility functions for data processing and I/O operations.                           |

### Target Structure (After Restructuring)
| Directory         | Description                                                                          |
| :---------------- | :----------------------------------------------------------------------------------- |
| `src/`            | The root for all core application source code.                                       |
| `src/main.py`     | Enhanced CLI entrypoint with Click framework.                                       |
| `src/core/`       | Core pipeline components, base classes, and exceptions.                             |
| `src/config/`     | Pydantic-based configuration schemas and environment configs.                       |
| `src/datasets/`   | Dataset registry, loaders, and validation schemas.                                  |
| `src/processors/` | Instruction and preference processors following factory pattern.                    |
| `src/utils/`      | Reusable utility functions for I/O, formatting, and parallel processing.           |
| `tests/`          | Comprehensive pytest test suite with fixtures and integration tests.               |

### Migration Strategy
- **Phase 1**: Maintain backward compatibility while introducing new structure
- **Phase 2-3**: Gradually migrate existing code to new architecture
- **Phase 4-5**: Complete migration and remove legacy structure

---
## 6. Setup & Execution

### Current Setup
Use standard Python package management and execution:

```bash
# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run data processing (current)
python main.py instruction=ver_2.1 main.version=ver_2.1 main.process_type=instruction

# Run tests (to be implemented)
pytest tests/ -v
```

### Target Setup (After Restructuring)
Enhanced setup with modern tooling:

```bash
# Using pip with pyproject.toml (Phase 5)
uv sync

# Enhanced CLI interface
uv run python -m src.main process --config configs/instruction/v2.yaml --output data/output/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run linting and formatting
uv run ./lint.sh
```

---

## 7. Anchor Comments

Add specially formatted comments throughout the codebase where appropriate. This creates a trail of inline knowledge that can be easily searched for by both humans and AI.

**Guidelines**:
- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` (all-caps prefix) for comments aimed at AI and developers.
- Keep them concise (≤ 120 chars).
- **Important:** Before scanning files, always first try to **locate existing anchors** `AIDEV-*` in relevant subdirectories.
- **Update relevant anchors** when modifying associated code.
- **Do not remove `AIDEV-NOTE`s** without explicit human instruction.
- Make sure to add relevant anchor comments, whenever a file or piece of code is:
  * too long, or
  * too complex, or
  * very important, or
  * confusing, or
  * could have a bug unrelated to the task you are currently working on.
**Example**:
```python
# AIDEV-NOTE: This dataset loader uses registry pattern for dynamic loading
@DatasetRegistry.register("smoltalk")
class SmolTalkDataset(BaseDataset):
    async def load(self) -> pd.DataFrame:
        # Implementation here
        pass
```

---

## 8. Commit Discipline

-  **Granular commits**: One logical change per commit.
- **Clear Commit Messages**: Explain the *why* behind the change, not just the *what*.
- **Tag AI-generated commits**: e.g., `feat: add dataset registry pattern [AI]` or `refactor: migrate ver_1_2 to new structure [AI]`.
- **Use conventional commits**: Follow conventional commit format for better tracking.
- **Review AI-generated code**: Never merge code you don't understand, especially during restructuring.
- **Phase-based branching**: Use separate branches for each improvement phase (e.g., `phase-1-foundation`, `phase-2-data-layer`).

### Restructuring-Specific Commit Guidelines
- **Backward compatibility**: Always ensure existing functionality works before committing.
- **Migration commits**: Clearly mark commits that migrate code from old to new structure.
- **Documentation updates**: Update relevant documentation with each structural change.
- **Test additions**: Include tests with any new functionality or refactored code.

---

## 9. Versioning Conventions

This project follows Semantic Versioning (SemVer: `MAJOR.MINOR.PATCH`), as specified in the `pyproject.toml` file.

### Project Versioning
- **MAJOR** version update: For incompatible API changes or major architectural restructuring.
- **MINOR** version update: For adding new datasets, processors, or features in a backward-compatible manner.
- **PATCH** version update: For backward-compatible bug fixes and minor improvements.

### Component Versioning  
- **Processor Versions**: Use semantic naming (v1, v2, etc.) with clear migration paths.
- **Dataset Versions**: Track dataset schema changes and processing logic updates.
- **Configuration Versions**: Maintain compatibility across configuration schema changes.

### Current Version Status
- **Project**: Currently in major restructuring (v1.0.0 → v2.0.0)
- **Processors**: Legacy versions (ver_1, ver_2) being migrated to new structure
- **Datasets**: Maintaining compatibility during migration

---

## 10. AI Assistant Workflow: Step-by-Step Methodology

When responding to user instructions during the repository restructuring, the AI assistant should follow this process:

1.  **Consult Project Documents**: First consult this `rule.md` and `improvement_plan.md` to understand current phase and context.
2.  **Identify Current Phase**: Determine which improvement phase the request falls under (Foundation, Data Layer, etc.).
3.  **Check Phase Dependencies**: Ensure previous phase deliverables are complete before proceeding.
4.  **Clarify Scope**: If any part of the request is unclear or crosses phase boundaries, ask for clarification.
5.  **Plan Within Phase**: Break down the request into steps that align with current phase objectives.
6.  **Maintain Compatibility**: Ensure all changes maintain backward compatibility during transition.
7.  **Execute & Document**: Implement changes with proper AIDEV-NOTE comments and documentation updates.
8.  **Validate Changes**: Test that existing functionality still works after modifications.
9.  **Update Progress**: Mark phase deliverables as complete when applicable.
10. **Request Review**: Ask for validation before moving to next phase or making major structural changes.

### Phase-Specific Guidelines
- **Phase 1**: Focus on foundation without breaking existing code
- **Phase 2**: Implement new patterns while maintaining old interfaces  
- **Phase 3**: Begin deprecating old patterns with clear migration paths
- **Phase 4**: Add comprehensive testing for new architecture
- **Phase 5**: Complete migration and remove legacy code