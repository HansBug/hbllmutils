"""
Module for managing global logger configuration.

This module provides utilities for accessing and managing a centralized logger
instance for the hbllmutils package. It ensures consistent logging across all
components of the application.
"""

import logging


def get_global_logger() -> logging.Logger:
    """
    Get the global logger instance for the hbllmutils package.

    This function returns a logger instance with the name 'hbllmutils' that can be
    used throughout the package for consistent logging. The logger follows Python's
    standard logging hierarchy, allowing for centralized configuration.

    :return: The global logger instance for hbllmutils.
    :rtype: logging.Logger

    Example::
        >>> logger = get_global_logger()
        >>> logger.info('This is an info message')
        INFO:hbllmutils:This is an info message
        >>> logger.warning('This is a warning')
        WARNING:hbllmutils:This is a warning
    """
    return logging.getLogger('hbllmutils')
