"""Chat template formatting utilities.

AIDEV-NOTE: Utilities for formatting data according to expected output formats.
Handles conversion between different message formats and ensures consistency.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def format_message_content(
    content: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Format message content to expected format.

    Args:
        content: Raw content (string or list of content items)

    Returns:
        Formatted content list
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content, "image": None}]
    elif isinstance(content, list):
        formatted_content = []
        for item in content:
            if isinstance(item, dict):
                # Ensure required fields exist
                formatted_item = {
                    "type": item.get("type", "text"),
                    "text": item.get("text"),
                    "image": item.get("image"),
                }
                # Ensure text or image is present, not both
                if formatted_item["text"] and formatted_item["image"]:
                    logger.warning("Both text and image present, keeping text only")
                    formatted_item["image"] = None
                elif not formatted_item["text"] and not formatted_item["image"]:
                    logger.warning("Neither text nor image present, setting empty text")
                    formatted_item["text"] = ""

                formatted_content.append(formatted_item)
            else:
                # Handle non-dict items
                formatted_content.append(
                    {"type": "text", "text": str(item), "image": None}
                )
        return formatted_content
    else:
        # Handle other types by converting to string
        return [{"type": "text", "text": str(content), "image": None}]


def format_message(message: dict[str, Any]) -> dict[str, Any]:
    """
    Format a single message to expected format.

    Args:
        message: Raw message dictionary

    Returns:
        Formatted message
    """
    formatted_message = {
        "role": message.get("role", "user"),
        "content": format_message_content(message.get("content", "")),
        "tool_calls": message.get("tool_calls"),
        "name": message.get("name"),
    }

    # Validate role
    valid_roles = ["system", "user", "assistant", "tool"]
    if formatted_message["role"] not in valid_roles:
        logger.warning(
            f"Invalid role '{formatted_message['role']}', defaulting to 'user'"
        )
        formatted_message["role"] = "user"

    return formatted_message


def format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format a list of messages to expected format.

    Args:
        messages: List of raw message dictionaries

    Returns:
        List of formatted messages
    """
    return [format_message(msg) for msg in messages]


def format_instruction_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Format instruction data to expected output format.

    Args:
        data: Raw instruction data

    Returns:
        Formatted instruction data
    """
    # Format messages
    messages = format_messages(data.get("messages", []))

    # Convert to strings for output format
    formatted_data = {
        "messages": json.dumps(messages, ensure_ascii=False),
        "source": str(data.get("source", "")),
        "tools": json.dumps(data.get("tools")) if data.get("tools") else None,
        "images": data.get("images"),  # Keep as is for now
    }

    return formatted_data


def format_preference_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Format preference data to expected output format.

    Args:
        data: Raw preference data

    Returns:
        Formatted preference data
    """
    # Format messages and rejected response
    messages = format_messages(data.get("messages", []))
    rejected = data.get("rejected", {})

    # Format rejected response
    if rejected:
        formatted_rejected = {
            "role": rejected.get("role", "assistant"),
            "content": format_message_content(rejected.get("content", "")),
            "tool_calls": rejected.get("tool_calls"),
            "name": rejected.get("name"),
        }
    else:
        formatted_rejected = {}

    # Convert to strings for output format
    formatted_data = {
        "messages": json.dumps(messages, ensure_ascii=False),
        "source": str(data.get("source", "")) if data.get("source") else None,
        "tools": json.dumps(data.get("tools")) if data.get("tools") else None,
        "images": data.get("images"),  # Keep as is for now
        "rejected": json.dumps(formatted_rejected, ensure_ascii=False),
    }

    return formatted_data


def validate_instruction_format(data: dict[str, Any]) -> bool:
    """
    Validate that instruction data matches expected format.

    Args:
        data: Data to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["messages", "source"]

    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False

    # Check if messages is valid JSON string
    try:
        messages = (
            json.loads(data["messages"])
            if isinstance(data["messages"], str)
            else data["messages"]
        )
        if not isinstance(messages, list):
            logger.error("Messages must be a list")
            return False
    except json.JSONDecodeError:
        logger.error("Messages is not valid JSON")
        return False

    return True


def validate_preference_format(data: dict[str, Any]) -> bool:
    """
    Validate that preference data matches expected format.

    Args:
        data: Data to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["messages", "rejected"]

    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False

    # Validate messages and rejected
    try:
        messages = (
            json.loads(data["messages"])
            if isinstance(data["messages"], str)
            else data["messages"]
        )
        rejected = (
            json.loads(data["rejected"])
            if isinstance(data["rejected"], str)
            else data["rejected"]
        )

        if not isinstance(messages, list):
            logger.error("Messages must be a list")
            return False

        if not isinstance(rejected, dict):
            logger.error("Rejected must be a dict")
            return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in data: {e}")
        return False

    return True
