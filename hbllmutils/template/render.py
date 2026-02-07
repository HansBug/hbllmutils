"""
Jinja2-based prompt template rendering module.

This module provides a flexible and powerful prompt template system built on top
of the Jinja2 templating engine. It enables users to create, load, and render
text templates with variable substitution, supporting both string-based templates
and file-based templates with automatic encoding detection.

The module contains the following main components:

* :class:`PromptTemplate` - Main template class for rendering prompts with Jinja2

Key Features:

* Variable substitution using Jinja2 syntax
* Automatic encoding detection for template files
* Customizable Jinja2 environment preprocessing
* Strict undefined variable handling (configurable)
* Support for loading templates from files or strings

.. note::
   This module requires the Jinja2 templating engine and uses automatic
   encoding detection for file-based templates.

Example::

    >>> from hbllmutils.template.render import PromptTemplate
    >>> 
    >>> # Create template from string
    >>> template = PromptTemplate("Hello, {{ name }}!")
    >>> result = template.render(name="World")
    >>> print(result)
    Hello, World!
    >>> 
    >>> # Create template from file
    >>> template = PromptTemplate.from_file("templates/greeting.txt")
    >>> result = template.render(name="Alice", greeting="Hi")
    >>> print(result)

"""

import pathlib

import jinja2

from .decode import auto_decode
from .env import create_env


class PromptTemplate:
    """
    A template class for rendering prompts using the Jinja2 templating engine.

    This class provides a high-level interface for working with Jinja2 templates,
    wrapping the template creation and rendering process. It supports variable
    substitution, custom environment preprocessing, and strict undefined variable
    handling to catch template errors early.

    The class can be instantiated directly with a template string or created from
    a file using the :meth:`from_file` class method. The Jinja2 environment can
    be customized by overriding the :meth:`_preprocess_env` method in subclasses.

    :param template_text: The Jinja2 template string to use for rendering
    :type template_text: str
    :param strict_undefined: Whether to raise errors on undefined variables.
                            When True, uses StrictUndefined to catch template errors.
                            Defaults to True.
    :type strict_undefined: bool, optional

    :ivar _template: The compiled Jinja2 template object
    :vartype _template: jinja2.Template

    .. note::
       By default, this class uses strict undefined variable handling, which means
       that attempting to render a template with missing variables will raise an
       error. Set strict_undefined=False to allow undefined variables.

    .. warning::
       When strict_undefined is False, undefined variables will be silently ignored
       in the rendered output, which may lead to unexpected results.

    Example::

        >>> # Basic usage with string template
        >>> template = PromptTemplate("Hello, {{ name }}!")
        >>> result = template.render(name="World")
        >>> print(result)
        Hello, World!
        
        >>> # Template with multiple variables
        >>> template = PromptTemplate(
        ...     "{{ greeting }}, {{ name }}! You are {{ age }} years old."
        ... )
        >>> result = template.render(greeting="Hi", name="Alice", age=30)
        >>> print(result)
        Hi, Alice! You are 30 years old.
        
        >>> # Using non-strict mode
        >>> template = PromptTemplate(
        ...     "Hello, {{ name }}!",
        ...     strict_undefined=False
        ... )
        >>> result = template.render()  # Missing 'name' variable
        >>> print(result)
        Hello, !

    """

    def __init__(self, template_text: str, strict_undefined: bool = True):
        """
        Initialize a PromptTemplate with the given template text.

        Creates a new Jinja2 environment, preprocesses it using the
        :meth:`_preprocess_env` hook, and compiles the template string.

        :param template_text: The Jinja2 template string containing variables
                             in {{ variable }} format
        :type template_text: str
        :param strict_undefined: Whether to raise errors on undefined variables.
                                When True, accessing undefined variables will raise
                                jinja2.UndefinedError. Defaults to True.
        :type strict_undefined: bool, optional

        :raises jinja2.TemplateSyntaxError: If the template string contains
                                           invalid Jinja2 syntax

        Example::

            >>> template = PromptTemplate("Hello, {{ name }}!")
            >>> print(template._template)
            <Template memory:...>
            
            >>> # Template with control structures
            >>> template = PromptTemplate('''
            ... {% for item in items %}
            ... - {{ item }}
            ... {% endfor %}
            ... ''')

        """
        env = create_env(strict_undefined=strict_undefined)
        env = self._preprocess_env(env)
        self._template = env.from_string(template_text)

    def _preprocess_env(self, env: jinja2.Environment) -> jinja2.Environment:
        """
        Preprocess the Jinja2 environment before creating the template.

        This hook method allows subclasses to customize the Jinja2 environment
        by adding custom filters, tests, globals, or modifying environment
        settings before the template is compiled. The default implementation
        returns the environment unchanged.

        :param env: The Jinja2 environment to preprocess
        :type env: jinja2.Environment

        :return: The preprocessed Jinja2 environment, potentially with custom
                filters, tests, or globals added
        :rtype: jinja2.Environment

        .. note::
           This method is designed to be overridden in subclasses. The base
           implementation simply returns the environment unchanged.

        Example::

            >>> class CustomTemplate(PromptTemplate):
            ...     def _preprocess_env(self, env):
            ...         # Add a custom filter
            ...         env.filters['reverse'] = lambda x: x[::-1]
            ...         # Add a custom global function
            ...         env.globals['get_timestamp'] = lambda: "2024-01-01"
            ...         return env
            >>> 
            >>> template = CustomTemplate("{{ name | reverse }}")
            >>> result = template.render(name="Alice")
            >>> print(result)
            ecilA

        """
        return env

    def render(self, **kwargs) -> str:
        """
        Render the template with the provided keyword arguments.

        Substitutes all template variables with the provided values and returns
        the rendered string. Variables should be passed as keyword arguments
        matching the variable names in the template.

        :param kwargs: Variable names and their values to substitute in the template.
                      Keys should match the variable names used in the template
                      (e.g., {{ name }} expects a 'name' keyword argument).

        :return: The rendered template string with all variables substituted
        :rtype: str

        :raises jinja2.UndefinedError: If strict_undefined is True and a required
                                      variable is missing from kwargs
        :raises jinja2.TemplateError: If an error occurs during template rendering

        Example::

            >>> template = PromptTemplate("Hello, {{ name }}!")
            >>> result = template.render(name="Alice")
            >>> print(result)
            Hello, Alice!
            
            >>> # Multiple variables
            >>> template = PromptTemplate(
            ...     "{{ greeting }}, {{ name }}! You are {{ age }} years old."
            ... )
            >>> result = template.render(greeting="Hi", name="Bob", age=25)
            >>> print(result)
            Hi, Bob! You are 25 years old.
            
            >>> # With control structures
            >>> template = PromptTemplate('''
            ... Items:
            ... {% for item in items %}
            ... - {{ item }}
            ... {% endfor %}
            ... ''')
            >>> result = template.render(items=["apple", "banana", "cherry"])
            >>> print(result)
            Items:
            - apple
            - banana
            - cherry

        """
        return self._template.render(**kwargs)

    @classmethod
    def from_file(cls, template_file, **params):
        """
        Create a PromptTemplate instance from a template file.

        Reads a template file with automatic encoding detection and creates
        a PromptTemplate instance from its content. This method uses the
        :func:`auto_decode` function to handle various text encodings
        automatically, making it suitable for templates in different languages
        and encoding formats.

        :param template_file: Path to the template file. Can be a string path
                             or a pathlib.Path object.
        :type template_file: str or pathlib.Path
        :param params: Additional keyword arguments to pass to the PromptTemplate
                      constructor (e.g., strict_undefined=False)

        :return: A new PromptTemplate instance created from the file content
        :rtype: PromptTemplate

        :raises FileNotFoundError: If the template file does not exist
        :raises UnicodeDecodeError: If the file cannot be decoded with any
                                   supported encoding
        :raises jinja2.TemplateSyntaxError: If the file contains invalid
                                           Jinja2 syntax

        .. note::
           This method automatically detects the file encoding using chardet
           and tries multiple encoding strategies to successfully read the file.

        Example::

            >>> # Load template from file
            >>> template = PromptTemplate.from_file("templates/greeting.txt")
            >>> result = template.render(name="Bob")
            >>> print(result)
            Hello, Bob!
            
            >>> # Load with custom parameters
            >>> template = PromptTemplate.from_file(
            ...     "templates/greeting.txt",
            ...     strict_undefined=False
            ... )
            >>> result = template.render()  # Missing variables allowed
            
            >>> # Using pathlib.Path
            >>> from pathlib import Path
            >>> template_path = Path("templates") / "greeting.txt"
            >>> template = PromptTemplate.from_file(template_path)

        """
        return cls(template_text=auto_decode(pathlib.Path(template_file).read_bytes()), **params)
