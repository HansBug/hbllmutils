from .code.dispatch import _add_code_subcommand
from .dispatch import hbllmutils

_DECORATORS = [
    _add_code_subcommand,
]

cli = hbllmutils
for deco in _DECORATORS:
    cli = deco(cli)
