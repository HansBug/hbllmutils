"""
Jinja2 template utilities for prompt rendering and environment configuration.

This package module provides a comprehensive set of tools for working with Jinja2
templates, including automatic text decoding, environment configuration with Python
builtins, and a flexible prompt template system. It serves as the main entry point
for the :mod:`hbllmutils.template` package, exposing key functionality for template
rendering, matching utilities, and quick rendering helpers.

The module contains the following main components:

* :func:`auto_decode` - Automatic text decoding with support for various encodings
* :func:`add_builtins_to_env` - Mount Python built-in functions to Jinja2 environments
* :func:`add_settings_for_env` - Add custom settings and filters to Jinja2 environments
* :func:`create_env` - Create fully configured Jinja2 environments
* :class:`PromptTemplate` - Flexible template class for rendering prompts
* :class:`QuickPromptTemplate` - Enhanced template with custom environment preprocessing
* :func:`quick_render` - Convenience function for quick template file rendering
* :class:`BaseMatcher` - Base class for file pattern matching with type extraction
* :class:`BaseMatcherPair` - Base class for grouping related matchers

.. note::
   This package requires Jinja2 and provides enhanced functionality beyond standard
   Jinja2 templates, including Python builtin integration and automatic encoding
   detection.

.. warning::
   When using strict_undefined mode (default), ensure all template variables are
   provided during rendering to avoid :exc:`jinja2.UndefinedError` exceptions.

Example::

    >>> from hbllmutils.template import PromptTemplate, auto_decode, create_env
    >>> 
    >>> # Create and render a simple template
    >>> template = PromptTemplate("Hello {{ name }}!")
    >>> result = template.render(name="World")
    >>> print(result)
    Hello World!
    >>> 
    >>> # Auto decode bytes with unknown encoding
    >>> text = auto_decode(b'\\xe4\\xb8\\xad\\xe6\\x96\\x87')
    >>> print(text)
    中文
    >>> 
    >>> # Create enhanced Jinja2 environment
    >>> env = create_env()
    >>> template = env.from_string("{{ items | len }}")
    >>> template.render(items=[1, 2, 3])
    '3'
    >>> 
    >>> # Quick render from file
    >>> from hbllmutils.template import quick_render
    >>> result = quick_render("template.txt", name="Alice", age=30)

"""

from .decode import auto_decode
from .env import add_builtins_to_env, add_settings_for_env, create_env
from .matcher import BaseMatcher
from .matcher_pair import BaseMatcherPair
from .quick import QuickPromptTemplate, quick_render
from .render import PromptTemplate

__all__ = [
    "auto_decode",
    "add_builtins_to_env",
    "add_settings_for_env",
    "create_env",
    "PromptTemplate",
    "QuickPromptTemplate",
    "quick_render",
    "BaseMatcher",
    "BaseMatcherPair",
]
