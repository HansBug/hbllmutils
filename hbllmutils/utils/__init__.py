"""
Utility module for hbllmutils package.

This module provides utility functions and classes for the hbllmutils package,
including logging utilities, data truncation helpers, and hashable conversion tools.
It serves as a central point for importing commonly used utility components that
facilitate debugging, logging, data representation, and data structure manipulation
throughout the package.

The module contains the following main components:

* :func:`get_global_logger` - Access the global logger instance for consistent logging
* :func:`log_pformat` - Format and truncate complex data structures for logging
* :func:`truncate_dict` - Recursively truncate complex data structures
* :func:`obj_hashable` - Convert mutable objects to hashable representations

These utilities are particularly useful when working with large language model (LLM)
conversation histories, API responses, and other verbose data structures that need
to be logged or displayed in a readable format without overwhelming the output. The
hashable conversion utility is especially valuable for caching and deduplication
operations where dictionary or list keys need to be used as cache keys.

.. note::
   All functions in this module are designed to be non-invasive and can be used
   throughout the hbllmutils package without side effects.

.. note::
   The logging utilities automatically detect terminal width for optimal formatting
   and follow Python's standard logging hierarchy for consistent behavior.

Example::

    >>> from hbllmutils.utils import get_global_logger, log_pformat, truncate_dict, obj_hashable
    >>> 
    >>> # Set up logging
    >>> logger = get_global_logger()
    >>> logger.info("Starting application")
    INFO:hbllmutils:Starting application
    >>> 
    >>> # Log complex data structures
    >>> data = {"key": "value" * 1000}
    >>> logger.debug(log_pformat(data))
    DEBUG:hbllmutils:{'key': 'valuevaluevaluevalue...<truncated, total 5000 chars>'}
    >>> 
    >>> # Truncate nested structures
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
    
    >>> # Convert mutable structures to hashable for caching
    >>> config = {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 100}
    >>> cache_key = obj_hashable(config)
    >>> cache = {cache_key: 'cached_result'}
    >>> print(cache[cache_key])
    'cached_result'

"""

from .hashable import obj_hashable
from .logging import get_global_logger
from .truncate import log_pformat, truncate_dict
