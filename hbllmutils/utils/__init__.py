"""
Utility module for hbllmutils package.

This module provides utility functions and classes for the hbllmutils package,
including logging utilities and data truncation helpers. It serves as a central
point for importing commonly used utility components that facilitate debugging,
logging, and data representation throughout the package.

The module contains the following main components:

* :func:`get_global_logger` - Access the global logger instance for consistent logging
* :func:`log_pformat` - Format and truncate complex data structures for logging
* :func:`truncate_dict` - Recursively truncate complex data structures

These utilities are particularly useful when working with large language model (LLM)
conversation histories, API responses, and other verbose data structures that need
to be logged or displayed in a readable format without overwhelming the output.

.. note::
   All functions in this module are designed to be non-invasive and can be used
   throughout the hbllmutils package without side effects.

Example::

    >>> from hbllmutils.utils import get_global_logger, log_pformat
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
    >>> from hbllmutils.utils import truncate_dict
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

"""

from .logging import get_global_logger
from .truncate import log_pformat, truncate_dict
