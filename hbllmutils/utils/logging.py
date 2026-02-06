"""
Global logger configuration and management utilities.

This module provides centralized logging functionality for the hbllmutils package.
It offers a simple interface to access and manage a consistent logger instance
that can be used across all components of the application, ensuring uniform
logging behavior and configuration throughout the package.

The module contains the following main components:

* :func:`get_global_logger` - Retrieves the global logger instance for the package

.. note::
   The logger returned by this module follows Python's standard logging hierarchy,
   allowing for flexible configuration through the standard logging module.

.. seealso::
   :mod:`logging` - Python's built-in logging module for configuration options

Example::

    >>> from hbllmutils.utils.logging import get_global_logger
    >>> logger = get_global_logger()
    >>> logger.info('Application started')
    INFO:hbllmutils:Application started
    >>> 
    >>> # Configure logging level for the entire package
    >>> import logging
    >>> logger.setLevel(logging.DEBUG)
    >>> logger.debug('Debug information')
    DEBUG:hbllmutils:Debug information

"""

import logging


def get_global_logger() -> logging.Logger:
    """
    Get the global logger instance for the hbllmutils package.

    This function returns a logger instance with the name 'hbllmutils' that can be
    used throughout the package for consistent logging. The logger follows Python's
    standard logging hierarchy, allowing for centralized configuration.

    The returned logger can be configured using standard logging module methods,
    and all child loggers will inherit the configuration. This ensures consistent
    logging behavior across the entire package.

    :return: The global logger instance for hbllmutils
    :rtype: logging.Logger

    .. note::
       This function creates the logger if it doesn't exist, or returns the
       existing instance if it has already been created. Multiple calls to this
       function will return the same logger object.

    .. seealso::
       :func:`logging.getLogger` - Python's standard logger retrieval function

    Example::

        >>> from hbllmutils.utils.logging import get_global_logger
        >>> logger = get_global_logger()
        >>> logger.info('This is an info message')
        INFO:hbllmutils:This is an info message
        >>> logger.warning('This is a warning')
        WARNING:hbllmutils:This is a warning
        >>> 
        >>> # Configure the logger
        >>> import logging
        >>> logger.setLevel(logging.DEBUG)
        >>> handler = logging.StreamHandler()
        >>> formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        >>> handler.setFormatter(formatter)
        >>> logger.addHandler(handler)
        >>> logger.debug('Detailed debug information')
        2024-01-01 12:00:00,000 - hbllmutils - DEBUG - Detailed debug information

    """
    return logging.getLogger('hbllmutils')
