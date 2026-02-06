"""
Code extraction and JSON parsing utilities for Markdown content.

This module provides comprehensive functionality for extracting code blocks from
Markdown text and parsing JSON data with optional repair capabilities. It serves
as a utility module for processing LLM responses that contain code snippets or
JSON data embedded in Markdown format.

The module contains the following main components:

* :func:`extract_code` - Extract code blocks from Markdown text with language filtering
* :func:`parse_json` - Parse JSON strings with automatic repair for malformed data

Key Features:
    - Automatic detection of plain code vs. fenced code blocks
    - Language-specific code block filtering
    - Support for multiple Markdown code block formats
    - JSON parsing with automatic repair for malformed input
    - Comprehensive error handling and validation

Supported Code Block Formats:
    - Plain text code (without fencing)
    - Fenced code blocks with triple backticks (```)
    - Language-tagged code blocks (e.g., ```python, ```javascript)

.. note::
   This module requires the `markdown-it-py` library for Markdown parsing
   and `json-repair` for handling malformed JSON data.

.. warning::
   When extracting code blocks, if multiple blocks match the criteria,
   a ValueError will be raised to prevent ambiguity.

Dependencies:
    - json: Standard library for JSON parsing
    - json_repair: Third-party library for repairing malformed JSON
    - markdown_it: Markdown parser for extracting code blocks
    - typing: Type hint support

Example::

    >>> from hbllmutils.response.code import extract_code, parse_json
    >>> 
    >>> # Extract Python code from Markdown
    >>> markdown_text = '''
    ... Here's some Python code:
    ... ```python
    ... def hello():
    ...     print("Hello, World!")
    ... ```
    ... '''
    >>> code = extract_code(markdown_text, language='python')
    >>> print(code)
    def hello():
        print("Hello, World!")
    <BLANKLINE>
    >>> 
    >>> # Parse JSON with automatic repair
    >>> malformed_json = '{"key": "value", "missing": '
    >>> data = parse_json(malformed_json, with_repair=True)
    >>> print(data)
    {'key': 'value', 'missing': None}

"""

import json
from typing import Optional, Any

import json_repair
import markdown_it
from markdown_it.tree import SyntaxTreeNode


def extract_code(text: str, language: Optional[str] = None) -> str:
    """
    Extract code blocks from Markdown text with optional language filtering.

    This function intelligently handles two distinct scenarios for code extraction:
    
    1. **Plain Code**: Text that is not wrapped in fenced code blocks. The function
       returns the trimmed text as-is, treating the entire input as code.
       
    2. **Fenced Code Blocks**: Text containing one or more code blocks delimited by
       triple backticks (```). The function parses the Markdown structure and extracts
       code blocks, optionally filtering by programming language.

    The function uses the markdown-it parser to accurately identify and extract code
    blocks while preserving their content exactly as written, including whitespace
    and formatting.

    :param text: The input Markdown text to parse for code blocks
    :type text: str
    :param language: Optional programming language identifier to filter code blocks
                    (e.g., 'python', 'javascript', 'java', 'cpp'). If None, extracts
                    code blocks regardless of language tag. Case-sensitive matching.
    :type language: str, optional
    
    :return: The extracted code content as a string, with original formatting preserved.
            For fenced blocks, returns the content between the opening and closing
            fence markers, excluding the markers themselves.
    :rtype: str
    
    :raises ValueError: If no code blocks matching the criteria are found in the text.
                       Error message indicates whether a specific language was requested.
    :raises ValueError: If multiple code blocks matching the criteria are found,
                       preventing ambiguous extraction. Error message indicates whether
                       a specific language was requested.
    
    .. note::
       The function preserves all whitespace, indentation, and line breaks within
       the extracted code block. Trailing whitespace is preserved for fenced blocks
       but stripped for plain text.
    
    .. warning::
       When multiple code blocks exist in the Markdown text, you must specify a
       language parameter to disambiguate, or ensure only one code block is present.
    
    Example::
    
        >>> # Extract plain code (no fencing)
        >>> plain_code = "print('hello world')"
        >>> extract_code(plain_code)
        "print('hello world')"
        
        >>> # Extract from single fenced block
        >>> markdown = '''```python
        ... def greet(name):
        ...     return f"Hello, {name}!"
        ... ```'''
        >>> extract_code(markdown)
        'def greet(name):\\n    return f"Hello, {name}!"\\n'
        
        >>> # Extract with language filtering
        >>> multi_lang = '''
        ... ```python
        ... print("Python code")
        ... ```
        ... ```javascript
        ... console.log("JavaScript code");
        ... ```
        ... '''
        >>> extract_code(multi_lang, language='python')
        'print("Python code")\\n'
        
        >>> # Error: No code blocks found
        >>> extract_code("Just plain text with no code")
        Traceback (most recent call last):
            ...
        ValueError: No code found in response.
        
        >>> # Error: Multiple code blocks without language filter
        >>> extract_code(multi_lang)
        Traceback (most recent call last):
            ...
        ValueError: Non-unique code blocks found in response.
    
    """
    # Case 1: Plain code (without fenced code block markers)
    # If the text doesn't start with triple backticks, treat entire text as code
    if not text.strip().startswith('```'):
        return text.strip()

    # Case 2: Code wrapped in fenced code blocks
    # Initialize markdown-it parser for processing Markdown syntax
    md = markdown_it.MarkdownIt()
    tokens = md.parse(text)
    root = SyntaxTreeNode(tokens)

    # Collect all code blocks that match the language filter (if specified)
    codes = []
    for node in root.walk():
        if node.type == 'fence':  # Fenced code block type
            # node.info contains the language identifier (e.g., 'python', 'javascript')
            if language is None or node.info == language:
                codes.append(node.content)

    # Validate that exactly one code block was found
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
    Parse JSON text with optional automatic repair for malformed input.

    This function provides robust JSON parsing capabilities with two modes of operation:
    
    1. **Standard Parsing** (with_repair=False): Uses Python's built-in json.loads()
       for strict JSON parsing that follows RFC 8259 specifications.
       
    2. **Repair Mode** (with_repair=True): Uses the json-repair library to automatically
       fix common JSON formatting issues before parsing, making it ideal for handling
       LLM-generated JSON that may be incomplete or malformed.

    The repair functionality can handle various JSON issues including:
    - Missing closing brackets, braces, or quotes
    - Trailing commas
    - Single quotes instead of double quotes
    - Unquoted keys
    - Comments in JSON
    - Truncated JSON strings

    :param text: The JSON text string to parse. Can be a complete or partial JSON
                structure depending on the with_repair parameter.
    :type text: str
    :param with_repair: If True, attempts to automatically repair malformed JSON
                       before parsing using the json-repair library. If False, uses
                       standard JSON parsing which requires valid JSON syntax.
    :type with_repair: bool
    
    :return: The parsed JSON object. Return type depends on the JSON structure:
            - dict for JSON objects
            - list for JSON arrays
            - str, int, float, bool, or None for JSON primitives
    :rtype: Any
    
    :raises json.JSONDecodeError: If with_repair is False and the input contains
                                 invalid JSON syntax. Includes details about the
                                 error location and nature.
    :raises Exception: If with_repair is True but the JSON is too malformed to
                      repair automatically. The json-repair library will raise
                      an appropriate exception describing the issue.
    
    .. note::
       When with_repair is True, the function may make assumptions about the
       intended structure of malformed JSON. Review the output to ensure it
       matches expectations.
    
    .. warning::
       The repair functionality is heuristic-based and may not always produce
       the intended result for severely malformed JSON. For production use with
       critical data, consider validating the repaired output.
    
    Example::
    
        >>> # Parse valid JSON
        >>> parse_json('{"name": "Alice", "age": 30}')
        {'name': 'Alice', 'age': 30}
        
        >>> # Parse JSON array
        >>> parse_json('[1, 2, 3, 4, 5]')
        [1, 2, 3, 4, 5]
        
        >>> # Parse with automatic repair (missing closing brace)
        >>> malformed = '{"name": "Bob", "age": 25'
        >>> parse_json(malformed, with_repair=True)
        {'name': 'Bob', 'age': 25}
        
        >>> # Parse with automatic repair (trailing comma)
        >>> malformed = '{"items": [1, 2, 3,]}'
        >>> parse_json(malformed, with_repair=True)
        {'items': [1, 2, 3]}
        
        >>> # Parse with automatic repair (single quotes)
        >>> malformed = "{'key': 'value'}"
        >>> parse_json(malformed, with_repair=True)
        {'key': 'value'}
        
        >>> # Standard parsing fails on malformed JSON
        >>> parse_json('{"key": "value"', with_repair=False)
        Traceback (most recent call last):
            ...
        json.JSONDecodeError: Expecting ',' delimiter: line 1 column 16 (char 15)
        
        >>> # Parse JSON primitives
        >>> parse_json('42')
        42
        >>> parse_json('true')
        True
        >>> parse_json('null')
        None
        >>> parse_json('"hello"')
        'hello'
    
    """
    if with_repair:
        # Use json-repair library to fix malformed JSON before parsing
        return json_repair.loads(text)
    else:
        # Use standard JSON parsing (strict mode)
        return json.loads(text)
