"""
Management module for Language Learning Models (LLM) configuration.

This module provides utilities for managing LLM configurations, including
loading configuration from YAML files and retrieving model-specific parameters.

The module exports the main configuration class :class:`LLMConfig` which handles
configuration file parsing and model parameter retrieval with support for
default and fallback configurations.

Example::
    >>> from hbllmutils.manage import LLMConfig
    >>> config = LLMConfig.open('config.yaml')
    >>> params = config.get_model_params('gpt-4')
"""

from .config import LLMConfig
