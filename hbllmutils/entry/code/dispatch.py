import click

from ..base import CONTEXT_SETTINGS


def _add_code_subcommand(cli: click.Group) -> click.Group:
    @cli.group('code', help='Python code completion and generation utilities',
               context_settings=CONTEXT_SETTINGS)
    def code():
        pass

    return cli
