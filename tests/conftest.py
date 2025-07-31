"""
Pytest configuration and fixtures for LLM data processing tests.

This module provides common fixtures and test configurations used across
the test suite, including sample data, temporary directories, and mock
objects for external dependencies.
"""

import ast
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# AIDEV-NOTE: Fixtures for testing data formats defined in expected_output_format.md


@pytest.fixture
def sample_instruction_processing_data() -> dict[str, Any]:
    """Sample instruction data in processing format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of France?",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The capital of France is Paris.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "test_dataset",
        "tools": None,
        "images": None,
    }


@pytest.fixture
def sample_instruction_saving_data() -> dict[str, Any]:
    """Sample instruction data in saving format (string representations)."""
    return {
        "messages": str(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the capital of France?",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The capital of France is Paris.",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
            ]
        ),
        "source": "test_dataset",
        "tools": None,  # Should remain None, not converted to string when None
        "images": None,
    }


@pytest.fixture
def sample_preference_processing_data() -> dict[str, Any]:
    """Sample preference data in processing format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain quantum computing.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Quantum computing is a revolutionary technology...",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "test_preference_dataset",
        "tools": None,
        "images": None,
        "rejected": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Quantum computing is just regular computing but faster.",
                    "image": None,
                }
            ],
            "tool_calls": None,
            "name": None,
        },
    }


@pytest.fixture
def sample_preference_saving_data() -> dict[str, Any]:
    """Sample preference data in saving format (string representations)."""
    return {
        "messages": str(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Explain quantum computing.",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Quantum computing is a revolutionary technology...",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
            ]
        ),
        "source": "test_preference_dataset",
        "tools": None,  # Should remain None when no tools present
        "images": None,
        "rejected": str(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Quantum computing is just regular computing but faster.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            }
        ),
    }


@pytest.fixture
def sample_instruction_saving_data_with_tools() -> dict[str, Any]:
    """Sample instruction data in saving format with tools converted to string representation."""
    return {
        "messages": str(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather like?",
                            "image": None,
                        }
                    ],
                    "tool_calls": None,
                    "name": None,
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll check the weather for you.",
                            "image": None,
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                    "name": None,
                },
            ]
        ),
        "source": "weather_assistant",
        "tools": str(
            [
                {  # When tools exist, they should be string representation
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
        "images": None,
    }


@pytest.fixture
def sample_invalid_instruction_data() -> list[dict[str, Any]]:
    """Sample invalid instruction data for negative testing."""
    return [
        # Missing required fields
        {
            "messages": [],
            # Missing source
            "tools": None,
            "images": None,
        },
        # Invalid role
        {
            "messages": [
                {
                    "role": "invalid_role",
                    "content": [{"type": "text", "text": "test", "image": None}],
                    "tool_calls": None,
                    "name": None,
                }
            ],
            "source": "test",
            "tools": None,
            "images": None,
        },
        # Invalid content structure
        {
            "messages": [
                {
                    "role": "user",
                    "content": "should be list not string",
                    "tool_calls": None,
                    "name": None,
                }
            ],
            "source": "test",
            "tools": None,
            "images": None,
        },
        # Both text and image present (should be mutually exclusive)
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "test", "image": "base64_image_data"}
                    ],
                    "tool_calls": None,
                    "name": None,
                }
            ],
            "source": "test",
            "tools": None,
            "images": None,
        },
    ]


@pytest.fixture
def sample_invalid_preference_data() -> list[dict[str, Any]]:
    """Sample invalid preference data for negative testing."""
    return [
        # Missing rejected field
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test", "image": None}],
                    "tool_calls": None,
                    "name": None,
                }
            ],
            "source": "test",
            "tools": None,
            "images": None,
            # Missing rejected field
        },
        # Invalid rejected role
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test", "image": None}],
                    "tool_calls": None,
                    "name": None,
                }
            ],
            "source": "test",
            "tools": None,
            "images": None,
            "rejected": {
                "role": "user",  # Should be assistant or tool
                "content": [{"type": "text", "text": "rejected", "image": None}],
                "tool_calls": None,
                "name": None,
            },
        },
    ]


@pytest.fixture
def temp_output_dir():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_datasets_df() -> pd.DataFrame:
    """Sample DataFrame for testing dataset operations."""
    return pd.DataFrame(
        [
            {
                "messages": str(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "test1", "image": None}
                            ],
                        }
                    ]
                ),
                "source": "dataset1",
                "tools": None,
                "images": None,
            },
            {
                "messages": str(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "test2", "image": None}
                            ],
                        }
                    ]
                ),
                "source": "dataset2",
                "tools": None,
                "images": None,
            },
        ]
    )


# AIDEV-NOTE: Configuration for pytest runs during restructuring
def pytest_configure(config):
    """Configure pytest for the data processing pipeline tests."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "format_validation: mark test as data format validation"
    )
