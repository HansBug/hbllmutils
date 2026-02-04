"""
Command-line interface dispatch module for hbllmutils.

This module provides the main CLI entry point and version information display
functionality for the hbllmutils package. It sets up the command-line interface
using Click framework and handles version display with author information.

The module contains the following main components:

* :func:`print_version` - Callback function to display version information
* :func:`hbllmutils` - Main CLI group entry point

Example::

    >>> # Command line usage
    >>> # hbllmutils --version
    >>> # Hbllmutils, version 0.3.1.
    >>> # Developed by HansBug (hansbug@buaa.edu.cn).

"""

import click
from click.core import Context, Option

from .base import CONTEXT_SETTINGS
from ..config.meta import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__, __DESCRIPTION__

_raw_authors = [item.strip() for item in __AUTHOR__.split(',') if item.strip()]
_raw_emails = [item.strip() for item in __AUTHOR_EMAIL__.split(',')]
if len(_raw_emails) < len(_raw_authors):  # pragma: no cover
    _raw_emails += [None] * (len(_raw_authors) - len(_raw_emails))
elif len(_raw_emails) > len(_raw_authors):  # pragma: no cover
    _raw_emails[len(_raw_authors) - 1] = tuple(_raw_emails[len(_raw_authors) - 1:])
    del _raw_emails[len(_raw_authors):]

_author_tuples = [
    (author, tuple([item for item in (email if isinstance(email, tuple) else ((email,) if email else ())) if item]))
    for author, email in zip(_raw_authors, _raw_emails)
]
_authors = [
    author if not emails else '{author} ({emails})'.format(author=author, emails=', '.join(emails))
    for author, emails in _author_tuples
]


# noinspection PyUnusedLocal
def print_version(ctx: Context, param: Option, value: bool) -> None:
    """
    Display version information and exit the CLI application.

    This callback function is triggered when the version flag is provided on the
    command line. It prints the application title, version number, and developer
    information, then exits the application gracefully.

    :param ctx: Click context object containing execution state and configuration
    :type ctx: Context
    :param param: Metadata for the current parameter being processed (version option)
    :type param: Option
    :param value: Boolean value indicating whether the version flag was provided
    :type value: bool
    :return: None - function exits the application after printing version info
    :rtype: None

    .. note::
       This function is designed to be used as a Click callback and should not
       be called directly in normal code flow.

    .. note::
       The function respects Click's resilient parsing mode and will not execute
       during completion or validation phases.

    Example::

        >>> # This function is automatically called when using CLI
        >>> # $ hbllmutils --version
        >>> # Hbllmutils, version 0.3.1.
        >>> # Developed by HansBug (hansbug@buaa.edu.cn).

    """
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover
    click.echo('{title}, version {version}.'.format(title=__TITLE__.capitalize(), version=__VERSION__))
    if _authors:
        click.echo('Developed by {authors}.'.format(authors=', '.join(_authors)))
    ctx.exit()


@click.group(context_settings=CONTEXT_SETTINGS, help=__DESCRIPTION__)
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Show hbllmutils's version information.")
def hbllmutils():
    """
    Main CLI group entry point for hbllmutils command-line interface.

    This function serves as the primary command group for the hbllmutils CLI
    application. It provides the foundation for all subcommands and handles
    global options such as version display and help information.

    The function is decorated with Click's group decorator to enable command
    grouping functionality, allowing subcommands to be registered and executed
    under the main hbllmutils command.

    :return: None - serves as a command group container
    :rtype: None

    .. note::
       This function acts as a command group container and does not perform
       any operations directly. Actual functionality is provided by subcommands
       registered to this group.

    .. note::
       Global options like ``--version`` and ``--help`` are available at this
       level and apply to the entire CLI application.

    Example::

        >>> # Display help information
        >>> # $ hbllmutils --help
        >>> # Usage: hbllmutils [OPTIONS] COMMAND [ARGS]...
        >>> #
        >>> # A Python utility library for streamlined Large Language Model
        >>> # interactions with unified API and conversation management.
        >>> #
        >>> # Options:
        >>> #   -v, --version  Show hbllmutils's version information.
        >>> #   -h, --help     Show this message and exit.

    """
    pass  # pragma: no cover
