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
"""

from .decode import auto_decode
from .env import add_builtins_to_env, add_settings_for_env, create_env
from .render import PromptTemplate
