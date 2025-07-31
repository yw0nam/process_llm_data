"""
Sample data fixtures for testing the LLM data processing pipeline.

This module contains realistic sample data that matches the expected
formats defined in expected_output_format.md
"""

import ast
from typing import Any

# AIDEV-NOTE: Realistic sample data for comprehensive testing


# Instruction data samples
SAMPLE_INSTRUCTION_CONVERSATIONS = [
    {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful AI assistant that provides accurate and informative responses.",
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
                        "text": "Can you explain the difference between machine learning and deep learning?",
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
                        "text": "Machine learning is a broader field that includes various algorithms for pattern recognition and prediction. Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to learn complex patterns in data. Deep learning is particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "educational_qa",
        "tools": None,
        "images": None,
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a Python function to calculate the factorial of a number.",
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
                        "text": 'Here\'s a Python function to calculate the factorial of a number:\n\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError("Factorial is not defined for negative numbers")\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Example usage:\nprint(factorial(5))  # Output: 120\n```\n\nThis function uses recursion to calculate the factorial. For better performance with large numbers, you could also use an iterative approach or the math.factorial() function.',
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "coding_assistant",
        "tools": None,
        "images": None,
    },
]

# Preference data samples
SAMPLE_PREFERENCE_CONVERSATIONS = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "How can I improve my productivity while working from home?",
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
                        "text": "Here are some effective strategies to improve productivity while working from home:\n\n1. **Create a dedicated workspace**: Set up a specific area for work to help maintain focus and separation between work and personal life.\n\n2. **Establish a routine**: Start your day at the same time and follow a consistent schedule.\n\n3. **Minimize distractions**: Turn off unnecessary notifications and create boundaries with household members.\n\n4. **Take regular breaks**: Use techniques like the Pomodoro Technique (25 minutes work, 5 minute break).\n\n5. **Stay connected**: Regular communication with colleagues helps maintain collaboration and motivation.\n\n6. **Set clear goals**: Define daily and weekly objectives to stay focused and track progress.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "productivity_advice",
        "tools": None,
        "images": None,
        "rejected": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Just work harder and avoid distractions. Set up a workspace and stick to a schedule.",
                    "image": None,
                }
            ],
            "tool_calls": None,
            "name": None,
        },
    },
    {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that provides accurate information about science and technology.",
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
                        "text": "What are the main benefits of renewable energy?",
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
                        "text": "Renewable energy offers several significant benefits:\n\n**Environmental Benefits:**\n- Reduces greenhouse gas emissions and air pollution\n- Helps combat climate change\n- Minimizes environmental degradation from resource extraction\n\n**Economic Benefits:**\n- Decreasing costs as technology improves\n- Job creation in green technology sectors\n- Reduced dependence on imported fossil fuels\n- Long-term price stability\n\n**Energy Security:**\n- Diversifies energy supply\n- Reduces vulnerability to supply disruptions\n- Provides energy independence\n\n**Health Benefits:**\n- Improves air quality\n- Reduces health costs associated with pollution\n\n**Sustainability:**\n- Inexhaustible energy sources (sun, wind, water)\n- Sustainable for future generations",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "science_qa",
        "tools": None,
        "images": None,
        "rejected": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Renewable energy is good for the environment and can save money. It includes solar and wind power.",
                    "image": None,
                }
            ],
            "tool_calls": None,
            "name": None,
        },
    },
]

# Tool usage examples
SAMPLE_TOOL_CONVERSATIONS = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather like in New York today?",
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
                        "text": "I'll check the current weather in New York for you.",
                        "image": None,
                    }
                ],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "New York, NY"}',
                        },
                    }
                ],
                "name": None,
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "text",
                        "text": '{"temperature": 72, "condition": "sunny", "humidity": 65, "wind_speed": 8}',
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": "get_weather",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The weather in New York today is sunny with a temperature of 72°F. The humidity is at 65% and there's a light wind at 8 mph. It's a beautiful day!",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "weather_assistant",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state/country",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        "images": None,
    }
]

# Image-based conversations
SAMPLE_IMAGE_CONVERSATIONS = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "text": None,
                        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 pixel PNG
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
                        "text": "What do you see in this image?",
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
                        "text": "I can see a very small image that appears to be a single pixel. It's essentially a minimal test image, often used for testing image processing capabilities. The image is too small to contain meaningful visual content.",
                        "image": None,
                    }
                ],
                "tool_calls": None,
                "name": None,
            },
        ],
        "source": "vision_assistant",
        "tools": None,
        "images": [
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ],
    }
]

# Invalid data samples for negative testing
INVALID_INSTRUCTION_SAMPLES = [
    # Missing required field
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "test", "image": None}],
            }
        ],
        # Missing "source" field
        "tools": None,
        "images": None,
    },
    # Invalid role
    {
        "messages": [
            {
                "role": "invalid_role",
                "content": [{"type": "text", "text": "test", "image": None}],
            }
        ],
        "source": "test",
        "tools": None,
        "images": None,
    },
    # Invalid content format
    {
        "messages": [{"role": "user", "content": "should be list, not string"}],
        "source": "test",
        "tools": None,
        "images": None,
    },
    # Both text and image present
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Should not have both",
                        "image": "base64_image_data",
                    }
                ],
            }
        ],
        "source": "test",
        "tools": None,
        "images": None,
    },
]

INVALID_PREFERENCE_SAMPLES = [
    # Missing rejected field
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "test", "image": None}],
            }
        ],
        "source": "test",
        "tools": None,
        "images": None,
        # Missing "rejected" field
    },
    # Invalid rejected role
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "test", "image": None}],
            }
        ],
        "source": "test",
        "tools": None,
        "images": None,
        "rejected": {
            "role": "user",  # Should be "assistant" or "tool"
            "content": [{"type": "text", "text": "bad response", "image": None}],
        },
    },
]


def get_sample_instruction_data() -> list[dict[str, Any]]:
    """Get all sample instruction data."""
    return (
        SAMPLE_INSTRUCTION_CONVERSATIONS
        + SAMPLE_TOOL_CONVERSATIONS
        + SAMPLE_IMAGE_CONVERSATIONS
    )


def get_sample_preference_data() -> list[dict[str, Any]]:
    """Get all sample preference data."""
    return SAMPLE_PREFERENCE_CONVERSATIONS


def get_invalid_instruction_data() -> list[dict[str, Any]]:
    """Get invalid instruction data for negative testing."""
    return INVALID_INSTRUCTION_SAMPLES


def get_invalid_preference_data() -> list[dict[str, Any]]:
    """Get invalid preference data for negative testing."""
    return INVALID_PREFERENCE_SAMPLES


def convert_to_saving_format(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert processing format to saving format (string representations).

    Key requirement from expected_output_format.md:
    - ALL columns except images should be converted to string type
    - tools should be None when no tools present, string representation when present
    - Use string representations that can be safely evaluated with ast.literal_eval
    """
    saving_data = {
        "messages": str(
            data["messages"]
        ),  # Convert to string representation (can be parsed with ast.literal_eval)
        "source": data["source"],  # Already string
        "tools": (
            str(data["tools"]) if data["tools"] is not None else None
        ),  # None stays None, list becomes string
        "images": data[
            "images"
        ],  # Images remain as list or None, NOT converted to string
    }

    # Add rejected field for preference data
    if "rejected" in data:
        saving_data["rejected"] = str(
            data["rejected"]
        )  # Convert to string representation

    return saving_data


def create_large_dataset(
    size: int, data_type: str = "instruction"
) -> list[dict[str, Any]]:
    """Create a large dataset for performance testing."""
    if data_type == "instruction":
        base_samples = SAMPLE_INSTRUCTION_CONVERSATIONS
    elif data_type == "preference":
        base_samples = SAMPLE_PREFERENCE_CONVERSATIONS
    else:
        raise ValueError("data_type must be 'instruction' or 'preference'")

    large_dataset = []
    for i in range(size):
        # Cycle through base samples and modify them
        base_sample = base_samples[i % len(base_samples)].copy()

        # Modify to make unique
        for message in base_sample["messages"]:
            if message["content"][0]["text"]:
                message["content"][0]["text"] = (
                    f"[Sample {i}] " + message["content"][0]["text"]
                )

        base_sample["source"] = f"{base_sample['source']}_sample_{i}"
        large_dataset.append(base_sample)

    return large_dataset
