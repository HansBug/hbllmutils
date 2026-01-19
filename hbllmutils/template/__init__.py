"""
Jinja2 template utilities for prompt rendering and environment configuration.

This module provides a comprehensive set of tools for working with Jinja2 templates,
including automatic text decoding, environment configuration with Python builtins,
and a flexible prompt template system. It serves as the main entry point for the
template package, exposing key functionality for template rendering and processing.

The module exports:
- auto_decode: Automatic text decoding with support for various encodings
- Environment configuration utilities: Functions to enhance Jinja2 environments
- PromptTemplate: A flexible template class for rendering prompts
- BaseMatcher: Base class for creating custom file matchers with pattern matching capabilities
- BaseMatcherPair: Base class for defining and working with matcher pairs

Example::
    >>> from hbllmutils.template import PromptTemplate, auto_decode
    >>> # Create a template
    >>> template = PromptTemplate("Hello {{ name }}!")
    >>> # Render with variables
    >>> result = template.render(name="World")
    >>> print(result)
    Hello World!
    >>> # Auto decode text
    >>> text = auto_decode(b'\\xe4\\xb8\\xad\\xe6\\x96\\x87')
    >>> print(text)
    中文
"""

from .decode import auto_decode
from .env import add_builtins_to_env, add_settings_for_env, create_env
from .matcher import BaseMatcher
from .matcher_pair import BaseMatcherPair
from .render import PromptTemplate
