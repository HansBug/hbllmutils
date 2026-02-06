"""
Command-line interface entry point for hbllmutils package.

This module serves as the main CLI entry point for the hbllmutils package,
providing a unified command-line interface by composing various subcommands
through a decorator pattern. It aggregates functionality from different
modules to create a comprehensive CLI tool.

The module contains the following main components:

* :data:`cli` - Main CLI group that serves as the entry point for all commands

.. note::
   This module uses a decorator pattern to dynamically add subcommands to the
   main CLI group. New subcommands can be added by including their decorators
   in the _DECORATORS list.

Example::

    >>> # The CLI can be invoked from the command line:
    >>> # $ hbllmutils code pydoc --help
    >>> # $ hbllmutils code todo --help
    >>> 
    >>> # Or programmatically:
    >>> from hbllmutils.entry.cli import cli
    >>> cli()

"""

from .code.dispatch import _add_code_subcommand
from .dispatch import hbllmutils

_DECORATORS = [
    _add_code_subcommand,
]

cli = hbllmutils
for deco in _DECORATORS:
    cli = deco(cli)
