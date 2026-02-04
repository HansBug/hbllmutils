"""
Data structure truncation utilities for logging purposes.

This module provides utilities for truncating and formatting complex nested data 
structures (dictionaries, lists, strings) to make them suitable for logging. It is 
particularly useful for preventing log files from becoming excessively large when 
dealing with verbose outputs from Large Language Models (LLMs) or other systems 
that generate extensive data structures.

The module contains the following main components:

* :func:`truncate_dict` - Recursively truncate nested data structures
* :func:`log_pformat` - Format truncated data for logging output

.. note::
   This module is designed to handle arbitrarily nested data structures and will
   recursively process all levels of nesting while applying truncation rules.

.. warning::
   Very deeply nested structures may still cause performance issues. Consider
   limiting the depth of structures before processing if performance is critical.

Example::

    >>> from hbllmutils.utils.truncate import log_pformat, truncate_dict
    >>> 
    >>> # Example with LLM conversation history
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
    >>> 
    >>> # Example with large dictionary
    >>> large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(20)}
    >>> truncated = truncate_dict(large_dict, max_dict_keys=3, max_string_len=30)
    >>> print(truncated)
    {'key_0': 'value_0value_0value_0value_0va...<truncated, total 700 chars>',
     'key_1': 'value_1value_1value_1value_1va...<truncated, total 700 chars>',
     'key_2': 'value_2value_2value_2value_2va...<truncated, total 700 chars>',
     '<truncated>': '17 more keys'}

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

    This function traverses nested data structures (dictionaries, lists, tuples, 
    strings) and truncates them according to specified limits to prevent excessive 
    log output. It handles arbitrary nesting depth and preserves the structure 
    while reducing the size of the data.

    The function applies different truncation strategies based on the data type:
    
    * **Strings**: Truncated to max_string_len characters with ellipsis and total length
    * **Lists/Tuples**: Limited to max_list_items elements with count of remaining items
    * **Dictionaries**: Limited to max_dict_keys keys with count of remaining keys
    * **Other types**: Returned unchanged

    :param obj: The object to truncate. Can be any type including nested structures
                such as lists of dictionaries, dictionaries of lists, etc.
    :type obj: Any
    :param max_string_len: Maximum length for string values before truncation.
                          Strings longer than this will be cut and marked with ellipsis.
                          Defaults to 250.
    :type max_string_len: int
    :param max_list_items: Maximum number of items to keep in lists or tuples.
                          Additional items will be replaced with a summary message.
                          Defaults to 4.
    :type max_list_items: int
    :param max_dict_keys: Maximum number of keys to keep in dictionaries.
                         Additional keys will be replaced with a summary message.
                         Defaults to 5.
    :type max_dict_keys: int
    :param current_depth: Current recursion depth, used internally for tracking
                         nesting level. Should not be set by users. Defaults to 0.
    :type current_depth: int

    :return: Truncated version of the input object with the same structure but
            reduced content according to the specified limits.
    :rtype: Any

    .. note::
       The function preserves the original data types (list remains list, dict 
       remains dict) but may add string markers to indicate truncation.

    .. warning::
       This function modifies the structure by adding truncation markers. The
       returned object is not suitable for further processing, only for display.

    Example::

        >>> # Truncate a long string
        >>> truncate_dict("a" * 300, max_string_len=10)
        'aaaaaaaaaa...<truncated, total 300 chars>'
        
        >>> # Truncate a list
        >>> truncate_dict([1, 2, 3, 4, 5], max_list_items=3)
        [1, 2, 3, '...<2 more items>']
        
        >>> # Truncate a nested structure
        >>> data = {
        ...     "messages": [
        ...         {"role": "user", "content": "x" * 500},
        ...         {"role": "assistant", "content": "y" * 500}
        ...     ]
        ... }
        >>> result = truncate_dict(data, max_string_len=20, max_list_items=1)
        >>> print(result)
        {'messages': [{'role': 'user', 'content': 'xxxxxxxxxxxxxxxxxxxx...<truncated, total 500 chars>'},
                      '...<1 more items>']}
        
        >>> # Truncate a large dictionary
        >>> large_dict = {f"key{i}": f"value{i}" for i in range(10)}
        >>> truncate_dict(large_dict, max_dict_keys=3)
        {'key0': 'value0', 'key1': 'value1', 'key2': 'value2', '<truncated>': '7 more keys'}

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

    This function combines truncation and pretty-printing to create log-friendly
    string representations of complex data structures. It first truncates the data
    using :func:`truncate_dict` and then formats it using Python's :func:`pprint.pformat`
    for readable output. This is particularly useful for logging LLM conversation 
    histories, API responses, and other verbose data structures.

    The function automatically detects terminal width for optimal formatting unless
    a specific width is provided. All truncation parameters can be customized to
    balance between detail and brevity in log output.

    :param obj: The object to format for logging. Can be any Python object including
                nested structures like lists of dictionaries or dictionaries of lists.
    :type obj: Any
    :param max_string_len: Maximum length for string values before truncation. Strings
                          exceeding this length will be cut with an ellipsis and total
                          length indicator. Defaults to 250.
    :type max_string_len: int
    :param max_list_items: Maximum number of items to display in lists or tuples.
                          Additional items will be summarized with a count message.
                          Defaults to 4.
    :type max_list_items: int
    :param max_dict_keys: Maximum number of keys to display in dictionaries.
                         Additional keys will be summarized with a count message.
                         Defaults to 5.
    :type max_dict_keys: int
    :param width: Output width for formatting in characters. If None, automatically
                 detects terminal width using :func:`shutil.get_terminal_size`.
                 Defaults to None.
    :type width: Optional[int]
    :param kwargs: Additional keyword arguments passed directly to :func:`pprint.pformat`.
                  Common options include indent, depth, compact, sort_dicts, and
                  underscore_numbers.
    :type kwargs: Any

    :return: A formatted string representation of the truncated object, suitable for
            logging or console output with proper indentation and line breaks.
    :rtype: str

    .. note::
       The function uses terminal width detection to ensure output fits within the
       console. This may not work correctly in all environments (e.g., when output
       is redirected to a file).

    .. warning::
       The returned string is for display purposes only. Do not attempt to parse
       or deserialize it back into the original data structure.

    Example::

        >>> from hbllmutils.utils.truncate import log_pformat
        >>> 
        >>> # Format LLM conversation history
        >>> llm_history = [
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello" * 1000},
        ...     {"role": "assistant", "content": "Hi there! How can I help you?"}
        ... ]
        >>> print(log_pformat(llm_history, max_string_len=50))
        [{'content': 'You are a helpful assistant', 'role': 'system'},
         {'content': 'HelloHelloHelloHelloHelloHelloHelloHelloHelloH...<truncated, total 5000 chars>',
          'role': 'user'},
         {'content': 'Hi there! How can I help you?', 'role': 'assistant'}]
        
        >>> # Format API response with custom width
        >>> api_response = {
        ...     "status": "success",
        ...     "data": {
        ...         "items": [{"id": i, "name": f"Item {i}"} for i in range(10)],
        ...         "metadata": {"total": 10, "page": 1}
        ...     }
        ... }
        >>> print(log_pformat(api_response, max_list_items=2, width=60))
        {'data': {'items': [{'id': 0, 'name': 'Item 0'},
                            {'id': 1, 'name': 'Item 1'},
                            '...<8 more items>'],
                  'metadata': {'page': 1, 'total': 10}},
         'status': 'success'}
        
        >>> # Use with custom pformat options
        >>> nested_data = {"level1": {"level2": {"level3": {"level4": "deep"}}}}
        >>> print(log_pformat(nested_data, depth=2, compact=True))
        {'level1': {'level2': {...}}}

    """
    truncated = truncate_dict(
        obj=obj,
        max_string_len=max_string_len,
        max_list_items=max_list_items,
        max_dict_keys=max_dict_keys,
    )
    width = width or shutil.get_terminal_size()[0]
    return pformat(truncated, width=width, **kwargs)
