"""
This module provides functionality for extracting code blocks from Markdown text.

It supports extracting code blocks with or without language specifications,
and handles both fenced code blocks and plain text code.
"""

from typing import Optional

import markdown_it
from markdown_it.tree import SyntaxTreeNode


def extract_code_from_markdown(text: str, language: Optional[str] = None) -> str:
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
        >>> extract_code_from_markdown(text, 'python')
        "print('hello')\\n"

        >>> text = "print('hello')"
        >>> extract_code_from_markdown(text)
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
