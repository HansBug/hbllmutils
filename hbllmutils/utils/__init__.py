"""
Utility helpers for the :mod:`hbllmutils.utils` package.

This module serves as a public entry point for frequently used utility functions
within the ``hbllmutils`` package. It re-exports logging helpers, data truncation
utilities, and hashable conversion tools to provide a concise and consistent
import interface for users.

The module contains the following main components:

* :func:`get_global_logger` - Access the global logger instance for consistent logging
* :func:`log_pformat` - Format and truncate complex data structures for logging
* :func:`truncate_dict` - Recursively truncate complex data structures
* :func:`obj_hashable` - Convert mutable objects to hashable representations

.. note::
   All exported utilities are designed to be non-invasive and safe to use across
   the package without unintended side effects.

.. note::
   The logging utilities integrate with Python's standard logging hierarchy and
   follow the "hbllmutils" logger namespace.

Example::

    >>> from hbllmutils.utils import get_global_logger, log_pformat, truncate_dict, obj_hashable
    >>>
    >>> # Initialize the global logger
    >>> logger = get_global_logger()
    >>> logger.info("Starting application")
    INFO:hbllmutils:Starting application
    >>>
    >>> # Log complex data structures in a compact form
    >>> data = {"key": "value" * 1000}
    >>> logger.debug(log_pformat(data))
    DEBUG:hbllmutils:{'key': 'valuevaluevaluevalue...<truncated, total 5000 chars>'}
    >>>
    >>> # Truncate nested structures for display
    >>> nested = {
    ...     "messages": [
    ...         {"role": "user", "content": "x" * 500},
    ...         {"role": "assistant", "content": "y" * 500}
    ...     ]
    ... }
    >>> truncated = truncate_dict(nested, max_string_len=50, max_list_items=2)
    >>> print(truncated)
    {'messages': [{'role': 'user', 'content': 'xxxxxxxxxx...<truncated, total 500 chars>'},
                  {'role': 'assistant', 'content': 'yyyyyyyyyy...<truncated, total 500 chars>'}]}
    >>>
    >>> # Convert mutable structures to hashable for caching or deduplication
    >>> config = {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 100}
    >>> cache_key = obj_hashable(config)
    >>> cache = {cache_key: 'cached_result'}
    >>> print(cache[cache_key])
    cached_result

"""

from .hashable import obj_hashable
from .logging import get_global_logger
from .truncate import log_pformat, truncate_dict
