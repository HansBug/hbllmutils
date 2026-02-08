"""
LLM configuration management package.

This module exposes the public API for managing Language Learning Model (LLM)
configurations. It re-exports :class:`LLMConfig` from
:mod:`hbllmutils.manage.config` to provide a concise import path for consumers.

The main component is:

* :class:`LLMConfig` - Loads configuration from YAML files or directories and
  resolves model-specific parameters with default and fallback support.

Example::

    >>> from hbllmutils.manage import LLMConfig
    >>> config = LLMConfig.open("config.yaml")
    >>> params = config.get_model_params("gpt-4")
    >>> print(params["model_name"])
    gpt-4

.. note::
   This module is a lightweight re-export layer. The implementation resides in
   :mod:`hbllmutils.manage.config`.

"""

from .config import LLMConfig
