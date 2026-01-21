"""
This module provides utilities for logging LLM (Large Language Model) history and complex data structures.

It includes functions to truncate and format complex nested data structures (dicts, lists, strings)
for logging purposes, preventing log files from becoming too large when dealing with verbose LLM outputs.

Example usage::
    >>> llm_history = [
    ...     {"role": "system", "content": "You are a helpful assistant"},
    ...     {"role": "user", "content": "Hello" * 1000},
    ... ]
    >>> print(log_pformat(llm_history))
    # Output will show truncated version of the data
"""

import shutil
from pprint import pformat
from typing import Any, Optional


def truncate_dict(
        obj: Any,
        max_string_len: int = 250,
        max_list_items: int = 4,
        max_dict_keys: int = 5,
        current_depth: int = 0
) -> Any:
    """
    Recursively truncate complex data structures for logging purposes.

    This function traverses nested data structures (dicts, lists, tuples, strings) and
    truncates them according to specified limits to prevent excessive log output.

    :param obj: The object to truncate. Can be any type including nested structures.
    :type obj: Any
    :param max_string_len: Maximum length for string values before truncation.
    :type max_string_len: int
    :param max_list_items: Maximum number of items to keep in lists/tuples.
    :type max_list_items: int
    :param max_dict_keys: Maximum number of keys to keep in dictionaries.
    :type max_dict_keys: int
    :param current_depth: Current recursion depth (used internally).
    :type current_depth: int

    :return: Truncated version of the input object.
    :rtype: Any

    Example::
        >>> truncate_dict("a" * 300, max_string_len=10)
        'aaaaaaaaaa...<truncated, total 300 chars>'
        >>> truncate_dict([1, 2, 3, 4, 5], max_list_items=3)
        [1, 2, 3, '...<2 more items>']
    """
    if isinstance(obj, str):
        if len(obj) > max_string_len:
            return obj[:max_string_len] + f"...<truncated, total {len(obj)} chars>"
        return obj

    elif isinstance(obj, (list, tuple)):
        if len(obj) > max_list_items:
            truncated = [
                truncate_dict(item, max_string_len, max_list_items,
                              max_dict_keys, current_depth + 1)
                for item in obj[:max_list_items]
            ]
            truncated.append(f"...<{len(obj) - max_list_items} more items>")
            return truncated
        else:
            return [
                truncate_dict(item, max_string_len, max_list_items,
                              max_dict_keys, current_depth + 1)
                for item in obj
            ]

    elif isinstance(obj, dict):
        if len(obj) > max_dict_keys:
            keys = list(obj.keys())[:max_dict_keys]
            result = {}
            for key in keys:
                result[key] = truncate_dict(
                    obj[key], max_string_len, max_list_items,
                    max_dict_keys, current_depth + 1
                )
            result[f"<truncated>"] = f"{len(obj) - max_dict_keys} more keys"
            return result
        else:
            return {
                key: truncate_dict(
                    value, max_string_len, max_list_items,
                    max_dict_keys, current_depth + 1
                )
                for key, value in obj.items()
            }

    else:
        return obj


def log_pformat(
        obj: Any,
        max_string_len: int = 250,
        max_list_items: int = 4,
        max_dict_keys: int = 5,
        width: Optional[int] = None,
        **kwargs
) -> str:
    """
    Generate a concise formatted string representation for logging purposes.

    This function truncates complex data structures and formats them using pprint.pformat,
    making them suitable for logging without overwhelming log files with verbose output.
    Particularly useful for logging LLM conversation histories and API responses.

    :param obj: The object to format for logging.
    :type obj: Any
    :param max_string_len: Maximum length for string values before truncation. Defaults to 250.
    :type max_string_len: int
    :param max_list_items: Maximum number of items to display in lists/tuples. Defaults to 4.
    :type max_list_items: int
    :param max_dict_keys: Maximum number of keys to display in dictionaries. Defaults to 5.
    :type max_dict_keys: int
    :param width: Output width for formatting. If None, uses terminal width. Defaults to None.
    :type width: Optional[int]
    :param kwargs: Additional keyword arguments to pass to pformat.
    :type kwargs: Any

    :return: A formatted string representation of the truncated object.
    :rtype: str

    Example::
        >>> llm_history = [
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello" * 1000},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> print(log_pformat(llm_history, max_string_len=50))
        [{'content': 'You are a helpful assistant', 'role': 'system'},
         {'content': 'HelloHelloHelloHelloHelloHelloHelloHelloHelloH...<truncated, total 5000 chars>',
          'role': 'user'},
         {'content': 'Hi there!', 'role': 'assistant'}]
    """
    truncated = truncate_dict(
        obj=obj,
        max_string_len=max_string_len,
        max_list_items=max_list_items,
        max_dict_keys=max_dict_keys,
    )
    width = width or shutil.get_terminal_size()[0]
    return pformat(truncated, width=width, **kwargs)
