import click

from .pydoc import _add_pydoc_subcommand
from .todo import _add_todo_subcommand
from .unittest import _add_unittest_subcommand
from ..base import CONTEXT_SETTINGS


def _add_code_subcommand(cli: click.Group) -> click.Group:
    @cli.group('code', help='Python code completion and generation utilities',
               context_settings=CONTEXT_SETTINGS)
    def code():
        pass

    _DECORATORS = [
        _add_todo_subcommand,
        _add_pydoc_subcommand,
        _add_unittest_subcommand,
    ]

    cli_ = code
    for deco in _DECORATORS:
        cli_ = deco(cli_)

    return cli
