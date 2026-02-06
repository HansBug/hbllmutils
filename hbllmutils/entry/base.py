"""
Command-line interface base utilities for Click-based applications.

This module provides foundational utilities for building robust command-line
interfaces using the Click framework. It includes custom exception classes,
error handling decorators, and parameter parsing utilities that enhance the
standard Click functionality with improved error reporting and user feedback.

The module contains the following main components:

* :class:`ClickWarningException` - Custom exception for displaying warnings in yellow
* :class:`ClickErrorException` - Custom exception for displaying errors in red
* :class:`KeyboardInterrupted` - Exception for handling keyboard interruptions
* :func:`command_wrap` - Decorator for wrapping Click commands with error handling
* :func:`print_exception` - Utility for formatted exception printing
* :func:`parse_key_value_params` - Parser for key=value parameter strings

.. note::
   This module is designed to work with Click 7.0+ and provides enhanced
   error handling and user feedback mechanisms for CLI applications.

Example::

    >>> import click
    >>> from hbllmutils.entry.base import command_wrap, parse_key_value_params
    >>> 
    >>> @click.command()
    >>> @click.option('--param', multiple=True, help='Key=value parameters')
    >>> @command_wrap()
    >>> def my_command(param):
    ...     params = dict(parse_key_value_params(p) for p in param)
    ...     click.echo(f"Parameters: {params}")
    >>> 
    >>> # Parse key-value parameters
    >>> key, value = parse_key_value_params("threshold=0.5")
    >>> print(f"{key}: {value} (type: {type(value).__name__})")
    threshold: 0.5 (type: float)

"""

import builtins
import itertools
import os
import sys
import traceback
from functools import wraps, partial
from typing import Optional, IO, Callable, Tuple, Union

import click
from click.exceptions import ClickException

from ..utils import get_global_logger

CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


class ClickWarningException(ClickException):
    """
    Custom exception class for displaying warning messages in yellow color.

    This exception class extends Click's base ClickException to provide
    warning-level feedback to users with yellow-colored output, making it
    visually distinct from errors while still being handled by Click's
    exception system.

    :param message: The warning message to display to the user
    :type message: str

    .. note::
       The warning message is always displayed to stderr with yellow color,
       regardless of the terminal's color support settings.

    Example::

        >>> from hbllmutils.entry.base import ClickWarningException
        >>> raise ClickWarningException("This is a warning message")
        # Output in yellow: Error: This is a warning message

    """

    def show(self, file: Optional[IO] = None) -> None:
        """
        Display the warning message in yellow color to stderr.

        This method overrides the base ClickException.show() to provide
        custom colored output for warning messages. The message is always
        written to stderr to maintain consistency with standard error
        reporting conventions.

        :param file: File stream to write the output to (currently unused,
                    always writes to stderr)
        :type file: Optional[IO]

        .. note::
           The file parameter is kept for API compatibility but is not used.
           Output always goes to sys.stderr.

        """
        click.secho(self.format_message(), fg='yellow', file=sys.stderr)


class ClickErrorException(ClickException):
    """
    Custom exception class for displaying error messages in red color.

    This exception class extends Click's base ClickException to provide
    error-level feedback to users with red-colored output, making critical
    issues immediately visible in the terminal output.

    :param message: The error message to display to the user
    :type message: str

    .. note::
       The error message is always displayed to stderr with red color,
       providing clear visual distinction from warnings and normal output.

    Example::

        >>> from hbllmutils.entry.base import ClickErrorException
        >>> raise ClickErrorException("Critical error occurred")
        # Output in red: Error: Critical error occurred

    """

    def show(self, file: Optional[IO] = None) -> None:
        """
        Display the error message in red color to stderr.

        This method overrides the base ClickException.show() to provide
        custom colored output for error messages. The message is always
        written to stderr following standard error reporting conventions.

        :param file: File stream to write the output to (currently unused,
                    always writes to stderr)
        :type file: Optional[IO]

        .. note::
           The file parameter is kept for API compatibility but is not used.
           Output always goes to sys.stderr.

        """
        click.secho(self.format_message(), fg='red', file=sys.stderr)


def print_exception(err: BaseException, print: Optional[Callable] = None):
    """
    Print formatted exception information including full traceback.

    This utility function provides detailed exception reporting by printing
    the complete traceback and exception details in a readable format. It
    supports custom print functions for flexible output handling.

    :param err: The exception object to print information about
    :type err: BaseException
    :param print: Custom print function for output. If None, uses built-in print
    :type print: Optional[Callable]

    .. note::
       The traceback is formatted with line breaks preserved for readability,
       and exception arguments are displayed according to their count.

    Example::

        >>> from hbllmutils.entry.base import print_exception
        >>> try:
        ...     1 / 0
        ... except ZeroDivisionError as e:
        ...     print_exception(e)
        Traceback (most recent call last):
          File "<stdin>", line 2, in <module>
        ZeroDivisionError: division by zero
        >>> 
        >>> # With custom print function
        >>> import functools
        >>> import click
        >>> try:
        ...     raise ValueError("Custom error", 123)
        ... except ValueError as e:
        ...     print_exception(e, functools.partial(click.secho, fg='red'))
        # Output in red with traceback

    """
    print = print or builtins.print

    lines = list(itertools.chain(*map(
        lambda x: x.splitlines(keepends=False),
        traceback.format_tb(err.__traceback__)
    )))

    if lines:
        print('Traceback (most recent call last):')
        print(os.linesep.join(lines))

    if len(err.args) == 0:
        print(f'{type(err).__name__}')
    elif len(err.args) == 1:
        print(f'{type(err).__name__}: {err.args[0]}')
    else:
        print(f'{type(err).__name__}: {err.args}')


class KeyboardInterrupted(ClickWarningException):
    """
    Exception class for handling keyboard interruption (Ctrl+C) events.

    This exception provides a user-friendly way to handle KeyboardInterrupt
    exceptions in Click-based CLI applications, converting them into
    warning-level exceptions with appropriate exit codes and messages.

    :param msg: Custom interruption message. If None, defaults to 'Interrupted.'
    :type msg: Optional[str]

    :ivar exit_code: Exit code returned when this exception is raised (0x7 = 7)
    :vartype exit_code: int

    .. note::
       The exit code 0x7 (7) is used to distinguish keyboard interruptions
       from other types of errors in the application.

    Example::

        >>> from hbllmutils.entry.base import KeyboardInterrupted
        >>> raise KeyboardInterrupted()
        # Output in yellow: Error: Interrupted.
        # Exit code: 7
        >>> 
        >>> raise KeyboardInterrupted("User cancelled operation")
        # Output in yellow: Error: User cancelled operation
        # Exit code: 7

    """
    exit_code = 0x7

    def __init__(self, msg=None):
        """
        Initialize the keyboard interruption exception.

        :param msg: Custom message to display. Defaults to 'Interrupted.' if None
        :type msg: Optional[str]

        """
        ClickWarningException.__init__(self, msg or 'Interrupted.')


def command_wrap():
    """
    Decorator factory for wrapping Click commands with comprehensive error handling.

    This decorator provides a standardized error handling mechanism for Click
    commands, catching and appropriately handling various exception types
    including Click exceptions, keyboard interrupts, and unexpected errors.

    The decorator performs the following error handling:

    * Passes through ClickException instances unchanged
    * Converts KeyboardInterrupt to KeyboardInterrupted exception
    * Catches unexpected exceptions, displays detailed error information,
      and exits with code 1

    :return: Decorator function that wraps Click command functions
    :rtype: Callable

    .. warning::
       This decorator should be applied after Click decorators to ensure
       proper exception handling within the Click context.

    Example::

        >>> import click
        >>> from hbllmutils.entry.base import command_wrap
        >>> 
        >>> @click.command()
        >>> @click.option('--value', type=int, required=True)
        >>> @command_wrap()
        >>> def process_value(value):
        ...     result = 100 / value
        ...     click.echo(f"Result: {result}")
        >>> 
        >>> # Handles keyboard interruption gracefully
        >>> # Handles unexpected errors with detailed traceback
        >>> # Passes through Click exceptions normally

    """

    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ClickException:
                raise
            except KeyboardInterrupt:
                raise KeyboardInterrupted
            except BaseException as err:
                click.secho('Unexpected error found when running pyfcstm!', fg='red', file=sys.stderr)
                print_exception(err, partial(click.secho, fg='red', file=sys.stderr))
                click.get_current_context().exit(1)

        return _new_func

    return _decorator


def parse_key_value_params(param: str) -> Tuple[str, Union[str, int, float]]:
    """
    Parse a parameter string in key=value format and return typed key-value pair.

    This function parses command-line parameter strings in the format 'key=value'
    and automatically converts the value to the most appropriate type (int, float,
    or str). The type inference is performed in order: int -> float -> str.

    :param param: Parameter string in format 'key=value'
    :type param: str
    :return: Tuple containing the key (str) and typed value (int, float, or str)
    :rtype: Tuple[str, Union[str, int, float]]
    :raises ValueError: If parameter format is invalid (missing '=' separator)

    .. note::
       Type conversion priority: integer -> float -> string. If a value can be
       parsed as an integer, it will be returned as int. If not, it tries float,
       and finally defaults to string if both fail.

    Example::

        >>> from hbllmutils.entry.base import parse_key_value_params
        >>> 
        >>> # Integer value
        >>> key, value = parse_key_value_params("max_iter=100")
        >>> print(f"{key}: {value} (type: {type(value).__name__})")
        max_iter: 100 (type: int)
        >>> 
        >>> # Float value
        >>> key, value = parse_key_value_params("threshold=0.75")
        >>> print(f"{key}: {value} (type: {type(value).__name__})")
        threshold: 0.75 (type: float)
        >>> 
        >>> # String value
        >>> key, value = parse_key_value_params("model=bert-base")
        >>> print(f"{key}: {value} (type: {type(value).__name__})")
        model: bert-base (type: str)
        >>> 
        >>> # Value with equals sign
        >>> key, value = parse_key_value_params("equation=x=y+1")
        >>> print(f"{key}: {value}")
        equation: x=y+1
        >>> 
        >>> # Invalid format
        >>> try:
        ...     parse_key_value_params("invalid_param")
        ... except ValueError as e:
        ...     print(e)
        Invalid parameter format: 'invalid_param'. Expected format: key=value

    """
    if '=' not in param:
        get_global_logger().error(f'Invalid parameter format: {param!r}. Expected format: key=value')
        raise ValueError(f'Invalid parameter format: {param!r}. Expected format: key=value')
    key, value = param.split('=', 1)
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return key, value
