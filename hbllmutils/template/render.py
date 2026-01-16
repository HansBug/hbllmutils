"""
Jinja2-based prompt template module for rendering text templates.

This module provides a flexible prompt template system built on top of Jinja2,
allowing users to create, load, and render text templates with variable substitution.
It supports loading templates from files with automatic encoding detection and
provides hooks for environment customization.
"""

import pathlib

import jinja2

from .decode import auto_decode
from .env import create_env


class PromptTemplate:
    """
    A template class for rendering prompts using Jinja2 templating engine.

    This class wraps Jinja2 functionality to provide a simple interface for
    creating and rendering text templates with variable substitution.

    :param template_text: The Jinja2 template string to use for rendering.
    :type template_text: str

    Example::
        >>> template = PromptTemplate("Hello, {{ name }}!")
        >>> template.render(name="World")
        'Hello, World!'
    """

    def __init__(self, template_text: str):
        """
        Initialize a PromptTemplate with the given template text.

        :param template_text: The Jinja2 template string.
        :type template_text: str
        """
        env = create_env()
        env = self._preprocess_env(env)
        self._template = env.from_string(template_text)

    def _preprocess_env(self, env: jinja2.Environment) -> jinja2.Environment:
        """
        Preprocess the Jinja2 environment before creating the template.

        This method can be overridden in subclasses to customize the Jinja2
        environment, such as adding custom filters, tests, or globals.

        :param env: The Jinja2 environment to preprocess.
        :type env: jinja2.Environment

        :return: The preprocessed Jinja2 environment.
        :rtype: jinja2.Environment
        """
        return env

    def render(self, **kwargs) -> str:
        """
        Render the template with the provided keyword arguments.

        :param kwargs: Variable names and their values to substitute in the template.

        :return: The rendered template string.
        :rtype: str

        Example::
            >>> template = PromptTemplate("Hello, {{ name }}! You are {{ age }} years old.")
            >>> template.render(name="Alice", age=30)
            'Hello, Alice! You are 30 years old.'
        """
        return self._template.render(**kwargs)

    @classmethod
    def from_file(cls, template_file):
        """
        Create a PromptTemplate instance from a template file.

        This method reads a template file with automatic encoding detection
        and creates a PromptTemplate instance from its content.

        :param template_file: Path to the template file (string or Path object).
        :type template_file: str or pathlib.Path

        :return: A new PromptTemplate instance created from the file content.
        :rtype: PromptTemplate

        Example::
            >>> template = PromptTemplate.from_file("templates/greeting.txt")
            >>> template.render(name="Bob")
            'Hello, Bob!'
        """
        return cls(template_text=auto_decode(pathlib.Path(template_file).read_bytes()))
