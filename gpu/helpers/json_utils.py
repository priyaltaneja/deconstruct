"""
JSON Parsing Utilities
Functions for extracting and parsing JSON from LLM responses.
"""

import json
import re
from typing import Dict, Any, Optional


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON from model response that may include reasoning or markdown.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find standalone JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find the largest JSON-like structure
    # This handles nested objects better
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    start_idx = None

    # Last resort: try to parse the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"No valid JSON found in response: {str(e)[:100]}")


def safe_json_loads(text: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely parse JSON with a default fallback.

    Args:
        text: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    if default is None:
        default = {}

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def clean_json_string(text: str) -> str:
    """
    Clean a JSON string for parsing.

    Args:
        text: Raw JSON string

    Returns:
        Cleaned JSON string
    """
    # Remove common issues
    text = text.strip()

    # Remove markdown code block markers
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Remove BOM
    if text.startswith('\ufeff'):
        text = text[1:]

    # Remove trailing commas (invalid JSON but common in LLM output)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    return text


def validate_json_schema(data: Dict, required_fields: list) -> tuple:
    """
    Validate that JSON contains required fields.

    Args:
        data: Parsed JSON data
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [field for field in required_fields if field not in data]
    return len(missing) == 0, missing
