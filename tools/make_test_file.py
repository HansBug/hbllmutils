from pathlib import Path

import click


@click.command()
@click.option('-i', '--input', 'source_file', required=True, type=click.Path(exists=True), help='Source Python file')
@click.option('-s', '--src-dir', 'src_dir', type=click.Path(exists=True, file_okay=False),
              default='hbllmutils', help='Source directory path')
@click.option('-t', '--test-dir', 'test_dir', type=click.Path(exists=False, file_okay=False),
              default='test', help='Test directory path')
def make_test_file(source_file: str, src_dir: str, test_dir: str):
    """Create test file path from source file path."""
    source_path = Path(source_file).resolve()
    src_path = Path(src_dir).resolve()
    test_path = Path(test_dir).resolve()

    try:
        relative_path = source_path.relative_to(src_path)
    except ValueError:
        click.echo(f"Error: Source file {source_file} is not under directory {src_dir}", err=True)
        raise click.Abort()

    if source_path.name == '__init__.py':
        click.echo(f"Error: Source file cannot be __init__.py", err=True)
        raise click.Abort()

    parts = list(relative_path.parts)
    filename = parts[-1]

    if not filename.endswith('.py'):
        click.echo(f"Error: Source file must be a Python file (.py)", err=True)
        raise click.Abort()

    base_name = filename[:-3]
    test_filename = f'test_{base_name}.py'
    parts[-1] = test_filename
    test_relative_path = test_path / Path(*parts)

    click.echo(str(test_relative_path))


if __name__ == '__main__':
    make_test_file()
