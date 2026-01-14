"""
This module provides functionality for extracting code blocks from Markdown text.

It supports extracting code blocks with or without language specifications,
and handles both fenced code blocks and plain text code. Additionally, it provides
JSON parsing capabilities with optional repair functionality for malformed JSON.

The module offers two main functions:
- extract_code: Extracts code blocks from Markdown text with optional language filtering
- parse_json: Parses JSON strings with optional automatic repair for malformed JSON

Dependencies:
- json: Standard library for JSON parsing
- json_repair: Third-party library for repairing malformed JSON
- markdown_it: Markdown parser for extracting code blocks
"""
import json
from typing import Optional, Any

import json_repair
import markdown_it
from markdown_it.tree import SyntaxTreeNode


def extract_code(text: str, language: Optional[str] = None) -> str:
    """
    Extract code blocks from Markdown text.

    This function handles two scenarios:
    1. Plain code without fenced code block markers
    2. Code wrapped in fenced code blocks (```)

    :param text: The input Markdown text to parse.
    :type text: str
    :param language: Optional language type to filter code blocks (e.g., 'python', 'javascript').
                    If None, extracts code blocks of any language.
    :type language: Optional[str]

    :return: The extracted code content as a string.
    :rtype: str

    :raises ValueError: If no code blocks are found in the response.
    :raises ValueError: If multiple code blocks are found when a unique block is expected.

    Example::
        >>> text = "```python\\nprint('hello')\\n```"
        >>> extract_code(text, 'python')
        "print('hello')\\n"

        >>> text = "print('hello')"
        >>> extract_code(text)
        "print('hello')"
    """
    # Case 1: Plain code (without fenced code block markers)
    if not text.strip().startswith('```'):
        return text.strip()

    # Case 2: Code wrapped in fenced code blocks
    md = markdown_it.MarkdownIt()
    tokens = md.parse(text)
    root = SyntaxTreeNode(tokens)

    codes = []
    for node in root.walk():
        if node.type == 'fence':  # Fenced code block type
            if language is None or node.info == language:
                codes.append(node.content)

    if not codes:
        if language:
            raise ValueError(f'No {language} code found in response.')
        else:
            raise ValueError(f'No code found in response.')
    elif len(codes) > 1:
        if language:
            raise ValueError(f'Non-unique {language} code blocks found in response.')
        else:
            raise ValueError(f'Non-unique code blocks found in response.')
    else:
        return codes[0]


def parse_json(text: str, with_repair: bool = True) -> Any:
    """
    Parse JSON text with optional repair functionality.

    This function can parse JSON strings and optionally attempt to repair
    malformed JSON using the json_repair library. This is useful when dealing
    with potentially corrupted or incomplete JSON data.

    :param text: The JSON text string to parse.
    :type text: str
    :param with_repair: If True, attempts to repair malformed JSON before parsing.
                       If False, uses standard JSON parsing which may fail on malformed input.
    :type with_repair: bool

    :return: The parsed JSON object (can be dict, list, or any valid JSON type).
    :rtype: Any

    :raises json.JSONDecodeError: If with_repair is False and the JSON is malformed.
    :raises Exception: If with_repair is True but the JSON cannot be repaired.

    Example::
        >>> parse_json('{"key": "value"}')
        {'key': 'value'}

        >>> parse_json('{"key": "value"', with_repair=True)  # Missing closing brace
        {'key': 'value'}

        >>> parse_json('{"key": "value"', with_repair=False)  # Will raise JSONDecodeError
        Traceback (most recent call last):
        ...
        json.JSONDecodeError: ...
    """
    if with_repair:
        return json_repair.loads(text)
    else:
        return json.loads(text)
