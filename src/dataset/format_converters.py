"""Format conversion functions for standardized dataset processing.

AIDEV-NOTE: Processing functions that convert various dataset formats to the expected
instruction/preference format with messages, source, tools, and images fields.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.dataset.registry import DatasetRegistry

logger = logging.getLogger(__name__)


def create_message_content(
    text: Optional[str] = None, image: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create standardized message content.

    Args:
        text: Text content
        image: Image content (URL or base64)

    Returns:
        List of content dictionaries
    """
    if not text and not image:
        raise ValueError("Either text or image must be provided")

    if text and image:
        raise ValueError("Only one of text or image can be provided")

    content = []
    if text:
        content.append({"type": "text", "text": text, "image": None})
    elif image:
        content.append({"type": "image", "text": None, "image": image})

    return content


def standardize_to_instruction_format(
    messages: List[Dict[str, Any]],
    source: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert to standardized instruction format.

    Args:
        messages: List of message dictionaries
        source: Source dataset name
        tools: Optional tools list
        images: Optional images list

    Returns:
        Standardized format dictionary
    """
    # Convert messages to string for output format
    messages_str = str(messages)

    result = {
        "messages": messages_str,
        "source": source,
        "tools": str(tools) if tools else None,
        "images": images,
    }

    return result


@DatasetRegistry.register_processing_function("alpaca_to_messages")
def process_alpaca_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca-style dataset to standardized message format.

    Expected input columns: instruction, input, output
    """
    logger.info("Converting Alpaca dataset to message format")

    results = []

    for _, row in df.iterrows():
        messages = []

        # Add system message if we have input context
        if pd.notna(row.get("input")) and row.get("input", "").strip():
            # Combine instruction and input
            user_content = f"{row['instruction']}\n\n{row['input']}"
        else:
            user_content = row["instruction"]

        # User message
        messages.append(
            {
                "role": "user",
                "content": create_message_content(text=user_content),
                "tool_calls": None,
                "name": None,
            }
        )

        # Assistant message
        messages.append(
            {
                "role": "assistant",
                "content": create_message_content(text=row["output"]),
                "tool_calls": None,
                "name": None,
            }
        )

        # Convert to standard format
        result = standardize_to_instruction_format(
            messages=messages, source="alpaca_gpt4"
        )

        results.append(result)

    logger.info(f"Converted {len(results)} Alpaca samples to message format")
    return pd.DataFrame(results)


@DatasetRegistry.register_processing_function("smoltalk_to_messages")
def process_smoltalk_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SmolTalk dataset to standardized message format.

    Expected input columns: messages (list of conversation turns)
    """
    logger.info("Converting SmolTalk dataset to message format")

    results = []

    for _, row in df.iterrows():
        if "messages" not in row or not row["messages"]:
            continue

        messages = []

        for msg in row["messages"]:
            # Convert SmolTalk message to standard format
            content = create_message_content(text=msg.get("content", ""))

            message = {
                "role": msg.get("role", "user"),
                "content": content,
                "tool_calls": None,
                "name": msg.get("name"),
            }
            messages.append(message)

        # Convert to standard format
        result = standardize_to_instruction_format(
            messages=messages, source="smoltalk_data"
        )

        results.append(result)

    logger.info(f"Converted {len(results)} SmolTalk samples to message format")
    return pd.DataFrame(results)


@DatasetRegistry.register_processing_function("custom_to_messages")
def process_custom_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert custom JSONL dataset to standardized message format.

    Flexible processing for custom datasets - adapt as needed.
    """
    logger.info("Converting custom dataset to message format")

    results = []

    for _, row in df.iterrows():
        messages = []

        # Handle different possible column names
        if "question" in row and "answer" in row:
            # Question-answer format
            messages.append(
                {
                    "role": "user",
                    "content": create_message_content(text=row["question"]),
                    "tool_calls": None,
                    "name": None,
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": create_message_content(text=row["answer"]),
                    "tool_calls": None,
                    "name": None,
                }
            )

        elif "prompt" in row and "response" in row:
            # Prompt-response format
            messages.append(
                {
                    "role": "user",
                    "content": create_message_content(text=row["prompt"]),
                    "tool_calls": None,
                    "name": None,
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": create_message_content(text=row["response"]),
                    "tool_calls": None,
                    "name": None,
                }
            )

        elif "messages" in row:
            # Already in messages format, just standardize
            for msg in row["messages"]:
                content = create_message_content(text=msg.get("content", ""))
                message = {
                    "role": msg.get("role", "user"),
                    "content": content,
                    "tool_calls": msg.get("tool_calls"),
                    "name": msg.get("name"),
                }
                messages.append(message)

        if messages:  # Only add if we successfully parsed messages
            result = standardize_to_instruction_format(
                messages=messages, source="custom_instructions"
            )
            results.append(result)

    logger.info(f"Converted {len(results)} custom samples to message format")
    return pd.DataFrame(results)


@DatasetRegistry.register_processing_function("sharegpt_to_messages")
def process_sharegpt_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ShareGPT-style dataset to standardized message format.

    Expected input columns: conversations (list of conversation turns)
    """
    logger.info("Converting ShareGPT dataset to message format")

    results = []

    for _, row in df.iterrows():
        if "conversations" not in row or not row["conversations"]:
            continue

        messages = []

        for conv in row["conversations"]:
            # Map ShareGPT roles to standard roles
            role_mapping = {"human": "user", "gpt": "assistant", "system": "system"}

            role = role_mapping.get(conv.get("from", ""), conv.get("from", "user"))
            content = create_message_content(text=conv.get("value", ""))

            message = {
                "role": role,
                "content": content,
                "tool_calls": None,
                "name": None,
            }
            messages.append(message)

        # Convert to standard format
        result = standardize_to_instruction_format(
            messages=messages, source="sharegpt_data"
        )

        results.append(result)

    logger.info(f"Converted {len(results)} ShareGPT samples to message format")
    return pd.DataFrame(results)


@DatasetRegistry.register_processing_function("openai_to_messages")
def process_openai_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert OpenAI-style dataset to standardized message format.

    Expected input: Already in messages format, just need to standardize content.
    """
    logger.info("Converting OpenAI dataset to message format")

    results = []

    for _, row in df.iterrows():
        if "messages" not in row or not row["messages"]:
            continue

        messages = []

        for msg in row["messages"]:
            # Standardize content format
            content_text = msg.get("content", "")
            if isinstance(content_text, list):
                # Handle complex content (text + images)
                content = []
                for item in content_text:
                    if item.get("type") == "text":
                        content.append(
                            {"type": "text", "text": item.get("text"), "image": None}
                        )
                    elif item.get("type") == "image_url":
                        content.append(
                            {
                                "type": "image",
                                "text": None,
                                "image": item.get("image_url", {}).get("url"),
                            }
                        )
            else:
                # Simple text content
                content = create_message_content(text=content_text)

            message = {
                "role": msg.get("role", "user"),
                "content": content,
                "tool_calls": msg.get("tool_calls"),
                "name": msg.get("name"),
            }
            messages.append(message)

        # Convert to standard format
        result = standardize_to_instruction_format(
            messages=messages, source="openai_data", tools=row.get("tools")
        )

        results.append(result)

    logger.info(f"Converted {len(results)} OpenAI samples to message format")
    return pd.DataFrame(results)
