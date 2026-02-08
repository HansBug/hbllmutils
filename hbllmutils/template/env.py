"""
Jinja2 environment enhancement utilities.

This module provides helper functions for configuring :class:`jinja2.Environment`
instances with Python built-ins, environment variables, and custom text helpers.
The goal is to make template authoring more expressive by exposing commonly
needed functionality as filters, tests, and globals.

The module contains the following main components:

* :func:`add_builtins_to_env` - Mount Python built-ins as filters, tests, and globals
* :func:`add_settings_for_env` - Add built-ins, text helpers, and environment variables
* :func:`create_env` - Create a fully configured Jinja2 environment

Example::

    >>> import jinja2
    >>> from hbllmutils.template.env import create_env
    >>> env = create_env()
    >>> template = env.from_string("{{ 3 | ordinalize }} and {{ 'word' | plural }}")
    >>> template.render()
    '3rd and words'

.. note::
   The environment exposes all current OS environment variables as globals.
   Use this carefully when rendering templates with untrusted input.

"""

import builtins
import inspect
import os
import pathlib
import textwrap
from typing import Union

import jinja2
from hbutils.string import plural_word, ordinalize, titleize
from jinja2 import StrictUndefined, Undefined


def add_builtins_to_env(env: jinja2.Environment) -> jinja2.Environment:
    """
    Mount Python built-in functions to a Jinja2 environment.

    This function inspects Python's built-in namespace and mounts functions
    into a Jinja2 environment as filters, tests, and globals. Filters and tests
    are added when no conflicting entries already exist.

    The mounting strategy follows these heuristics:

    * Filters: any callable built-in is added as a filter if no name conflict exists.
    * Tests: callables with an ``is`` prefix, plus common boolean functions
      (``all``, ``any``, ``callable``, ``hasattr``) are added as tests.
    * Globals: all built-in callables are exposed as globals if no conflict exists.

    In addition, a few convenience filter aliases are always ensured:

    * ``str``, ``set``, ``dict``, ``keys``, ``values``, ``enumerate``,
      ``reversed``, ``filter``

    :param env: A Jinja2 environment instance to be enhanced.
    :type env: jinja2.Environment
    :return: The enhanced environment (same instance as input).
    :rtype: jinja2.Environment

    Example::

        >>> import jinja2
        >>> env = add_builtins_to_env(jinja2.Environment())
        >>> env.from_string("{{ items | len }}").render(items=[1, 2, 3])
        '3'
        >>> env.from_string("{{ value is none }}").render(value=None)
        'True'

    .. note::
       Built-in names that conflict with existing filters or tests are not
       overridden, preserving any user-defined behavior.
    """
    # Existing built-in filters, tests and global functions in Jinja2
    existing_filters = set(env.filters.keys())
    existing_tests = set(env.tests.keys())
    existing_globals = set(env.globals.keys())

    # Get all Python built-in functions
    builtin_items = [
        (name, obj)
        for name, obj in inspect.getmembers(builtins)
        if not name.startswith("_")
    ]

    # Categorize functions for appropriate mounting positions
    for name, func in builtin_items:
        # Skip non-callable objects
        if not callable(func):
            continue

        # Determine if the function is suitable as a filter
        is_filter_candidate = inspect.isfunction(func) or inspect.isbuiltin(func)

        # Determine if the function is suitable as a tester
        is_test_candidate = (
                name.startswith("is") or name in ("all", "any", "callable", "hasattr")
        )

        # Mount as a filter (if suitable and no conflict)
        filter_name = name
        if is_filter_candidate and filter_name not in existing_filters:
            env.filters[filter_name] = func
        env.filters["str"] = str
        env.filters["set"] = set
        env.filters["dict"] = dict
        env.filters["keys"] = lambda x: x.keys()
        env.filters["values"] = lambda x: x.values()
        env.filters["enumerate"] = enumerate
        env.filters["reversed"] = reversed
        env.filters["filter"] = lambda x, y: filter(y, x)

        # Mount as a tester (if suitable and no conflict)
        test_name = name
        if name.startswith("is"):
            # For functions starting with 'is', the prefix can be removed as the tester name
            test_name = name[2:].lower()
        if is_test_candidate and test_name not in existing_tests:
            env.tests[test_name] = func

        # Mount as a global function (if no conflict)
        if name not in existing_globals:
            env.globals[name] = func

    return env


def _read_file_text(path: Union[str, os.PathLike]) -> str:
    """
    Read the entire contents of a file path as text.

    This helper is used to provide a Jinja2 filter and global callable
    that can read text content in templates.

    :param path: File system path to read.
    :type path: str or os.PathLike
    :return: File contents as a string.
    :rtype: str
    """
    return pathlib.Path(path).read_text()


def add_settings_for_env(env: jinja2.Environment) -> jinja2.Environment:
    """
    Add additional settings and helper functions to a Jinja2 environment.

    This function enhances a Jinja2 environment by:

    #. Adding Python built-in functions via :func:`add_builtins_to_env`.
    #. Adding text processing filters and globals:

       * ``indent`` - Indent text using :func:`textwrap.indent`
       * ``plural`` - Pluralize a word with its count using
         :func:`hbutils.string.plural_word`
       * ``ordinalize`` - Convert numbers to ordinal form using
         :func:`hbutils.string.ordinalize`
       * ``titleize`` - Convert text to title case using
         :func:`hbutils.string.titleize`
       * ``read_file_text`` - Read text content from a file path

    #. Adding all current environment variables as global variables, allowing
       template authors to reference them directly.

    :param env: The Jinja2 environment to enhance.
    :type env: jinja2.Environment
    :return: The enhanced environment (same instance as input).
    :rtype: jinja2.Environment

    Example::

        >>> import jinja2
        >>> env = add_settings_for_env(jinja2.Environment())
        >>> env.from_string("{{ 'word' | plural }}").render()
        'words'
        >>> env.from_string("{{ 3 | ordinalize }}").render()
        '3rd'

    .. warning::
       Environment variables are added as globals. If templates are rendered
       from untrusted sources, avoid exposing sensitive environment variables.
    """
    env = add_builtins_to_env(env)
    env.globals["indent"] = env.filters["indent"] = textwrap.indent
    env.globals["plural_word"] = env.filters["plural"] = plural_word
    env.globals["ordinalize"] = env.filters["ordinalize"] = ordinalize
    env.globals["titleize"] = env.filters["titleize"] = titleize
    env.globals["read_file_text"] = env.filters["read_file_text"] = _read_file_text
    for key, value in os.environ.items():
        if key not in env.globals:
            env.globals[key] = value
    return env


def create_env(strict_undefined: bool = True) -> jinja2.Environment:
    """
    Create a new Jinja2 environment with enhanced settings.

    This is a convenience function that builds a :class:`jinja2.Environment`
    instance and applies :func:`add_settings_for_env`. It optionally configures
    the undefined-variable behavior using :class:`jinja2.StrictUndefined`.

    :param strict_undefined: If ``True`` (default), use
        :class:`jinja2.StrictUndefined` to raise errors for undefined variables.
        If ``False``, use the default :class:`jinja2.Undefined` behavior.
    :type strict_undefined: bool
    :return: A fully configured Jinja2 environment with all enhancements.
    :rtype: jinja2.Environment

    Example::

        >>> env = create_env()
        >>> env.from_string("{{ 'hello' | upper }}").render()
        'HELLO'
        >>> env.from_string("{{ 3 | ordinalize }}").render()
        '3rd'

    .. note::
       Use ``strict_undefined=False`` if you prefer Jinja2's default behavior,
       where undefined variables render as empty strings.
    """
    env = jinja2.Environment(
        undefined=StrictUndefined if strict_undefined else Undefined
    )
    env = add_settings_for_env(env)
    return env
