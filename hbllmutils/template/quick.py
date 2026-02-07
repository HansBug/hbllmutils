"""
Quick template rendering utilities for simplified Jinja2 template operations.

This module provides a streamlined interface for creating and rendering Jinja2
templates with minimal configuration. It extends the base PromptTemplate class
to support custom environment preprocessing and offers a convenient function
for one-off template rendering operations.

The module contains the following main components:

* :class:`QuickPromptTemplate` - Enhanced template class with environment preprocessing
* :func:`quick_render` - Convenience function for quick template file rendering

.. note::
   This module is designed for rapid prototyping and simple template rendering
   scenarios. For more complex template management, consider using the base
   :class:`~hbllmutils.template.render.PromptTemplate` class directly.

Example::

    >>> from hbllmutils.template.quick import QuickPromptTemplate, quick_render
    >>> 
    >>> # Using QuickPromptTemplate with custom environment preprocessing
    >>> def add_filters(env):
    ...     env.filters['uppercase'] = str.upper
    ...     return env
    >>> 
    >>> template = QuickPromptTemplate(
    ...     "Hello, {{ name|uppercase }}!",
    ...     fn_env_preprocess=add_filters
    ... )
    >>> print(template.render(name="world"))
    Hello, WORLD!
    >>> 
    >>> # Quick rendering from a file
    >>> result = quick_render(
    ...     "template.txt",
    ...     name="Alice",
    ...     age=30
    ... )

"""

from typing import Optional, Callable

import jinja2

from .render import PromptTemplate


class QuickPromptTemplate(PromptTemplate):
    """
    Enhanced prompt template with support for custom environment preprocessing.

    This class extends :class:`~hbllmutils.template.render.PromptTemplate` to allow
    custom preprocessing of the Jinja2 environment before template creation. This
    enables adding custom filters, tests, globals, or other environment modifications
    in a clean, reusable manner.

    :param template_text: The Jinja2 template string to use for rendering
    :type template_text: str
    :param strict_undefined: Whether to raise errors on undefined variables, defaults to True
    :type strict_undefined: bool, optional
    :param fn_env_preprocess: Optional function to preprocess the Jinja2 environment
                              before template creation. The function should accept a
                              :class:`jinja2.Environment` and return a modified
                              :class:`jinja2.Environment`
    :type fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]], optional

    :ivar _fn_env_preprocess: Stored environment preprocessing function
    :vartype _fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]]

    .. note::
       The environment preprocessing function is called during initialization,
       before the template is compiled from the template text.

    Example::

        >>> import jinja2
        >>> from hbllmutils.template.quick import QuickPromptTemplate
        >>> 
        >>> # Define a custom environment preprocessor
        >>> def add_custom_filters(env):
        ...     env.filters['reverse'] = lambda x: x[::-1]
        ...     env.filters['double'] = lambda x: x * 2
        ...     return env
        >>> 
        >>> # Create template with custom environment
        >>> template = QuickPromptTemplate(
        ...     "{{ text|reverse }} and {{ number|double }}",
        ...     fn_env_preprocess=add_custom_filters
        ... )
        >>> 
        >>> # Render with custom filters applied
        >>> result = template.render(text="hello", number=5)
        >>> print(result)
        olleh and 10

    """

    def __init__(self, template_text: str, strict_undefined: bool = True,
                 fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]] = None):
        """
        Initialize a QuickPromptTemplate with optional environment preprocessing.

        :param template_text: The Jinja2 template string to use for rendering
        :type template_text: str
        :param strict_undefined: Whether to raise errors on undefined variables, defaults to True
        :type strict_undefined: bool, optional
        :param fn_env_preprocess: Optional function to preprocess the Jinja2 environment.
                                  Should accept and return a :class:`jinja2.Environment` object
        :type fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]], optional

        Example::

            >>> template = QuickPromptTemplate(
            ...     "Hello, {{ name }}!",
            ...     strict_undefined=True
            ... )
            >>> print(template.render(name="World"))
            Hello, World!

        """
        self._fn_env_preprocess = fn_env_preprocess
        super().__init__(template_text, strict_undefined)

    def _preprocess_env(self, env: jinja2.Environment) -> jinja2.Environment:
        """
        Preprocess the Jinja2 environment before template compilation.

        This method applies the custom environment preprocessor function if one
        was provided during initialization. Otherwise, it returns the environment
        unchanged. This allows for flexible customization of the Jinja2 environment
        including adding filters, tests, globals, or modifying other environment
        settings.

        :param env: The Jinja2 environment to preprocess
        :type env: jinja2.Environment
        :return: The preprocessed Jinja2 environment, potentially modified by
                 the preprocessing function
        :rtype: jinja2.Environment

        .. note::
           This method is called automatically during initialization and should
           not typically be called directly by users.

        Example::

            >>> import jinja2
            >>> from hbllmutils.template.quick import QuickPromptTemplate
            >>> 
            >>> class CustomTemplate(QuickPromptTemplate):
            ...     def _preprocess_env(self, env):
            ...         env = super()._preprocess_env(env)
            ...         env.globals['version'] = '1.0.0'
            ...         return env

        """
        if self._fn_env_preprocess is not None:
            return self._fn_env_preprocess(env)
        return env


def quick_render(template_file: str, strict_undefined: bool = True,
                 fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]] = None,
                 **params) -> str:
    """
    Quickly render a template file with the provided parameters.

    This convenience function provides a one-line solution for loading a template
    from a file and rendering it with the given parameters. It creates a
    :class:`QuickPromptTemplate` instance from the file and immediately renders
    it with the provided keyword arguments.

    :param template_file: Path to the template file to render (string or Path object)
    :type template_file: str
    :param strict_undefined: Whether to raise errors on undefined variables, defaults to True
    :type strict_undefined: bool, optional
    :param fn_env_preprocess: Optional function to preprocess the Jinja2 environment
                              before rendering. Should accept and return a
                              :class:`jinja2.Environment` object
    :type fn_env_preprocess: Optional[Callable[[jinja2.Environment], jinja2.Environment]], optional
    :param params: Variable names and their values to substitute in the template
    :return: The rendered template string
    :rtype: str
    :raises FileNotFoundError: If the template file does not exist
    :raises jinja2.UndefinedError: If strict_undefined is True and an undefined
                                   variable is referenced in the template

    .. note::
       This function automatically detects the file encoding when reading the
       template file, making it suitable for templates in various encodings.

    .. warning::
       For repeated rendering of the same template, consider creating a
       :class:`QuickPromptTemplate` instance once and calling its
       :meth:`~QuickPromptTemplate.render` method multiple times for better
       performance.

    Example::

        >>> from hbllmutils.template.quick import quick_render
        >>> 
        >>> # Simple template rendering
        >>> result = quick_render(
        ...     "greeting.txt",
        ...     name="Alice",
        ...     age=30
        ... )
        >>> 
        >>> # With custom environment preprocessing
        >>> def add_filters(env):
        ...     env.filters['shout'] = lambda x: x.upper() + "!"
        ...     return env
        >>> 
        >>> result = quick_render(
        ...     "message.txt",
        ...     fn_env_preprocess=add_filters,
        ...     text="hello world"
        ... )

    """
    template = QuickPromptTemplate.from_file(
        template_file=template_file,
        strict_undefined=strict_undefined,
        fn_env_preprocess=fn_env_preprocess,
    )
    return template.render(**params)
