"""
Resource file collection and management utilities.

This module provides utilities for collecting and managing resource files from
Python packages and the current project. It supports scanning package directories,
identifying non-Python resource files, and generating PyInstaller data mappings
for bundling resources with executables.

The module contains the following main components:

* :func:`get_resources_from_package` - Extract resource files from a Python package
* :func:`list_installed_packages` - List all installed Python packages
* :func:`list_resources` - List all resource files in the current project
* :func:`get_resources_from_mine` - Get resource files from the current project with relative paths
* :func:`get_resource_files` - Aggregate resource files from multiple sources
* :func:`print_resource_mappings` - Print PyInstaller-compatible resource mappings

.. note::
   This module is primarily designed for use with PyInstaller to bundle
   non-Python resource files with compiled executables.

.. warning::
   The module walks entire package directories, which may be slow for
   packages with many files.

Example::

    >>> from tools.resources import get_resources_from_package, list_resources
    >>> 
    >>> # Get resources from a specific package
    >>> for src_file, dst_path in get_resources_from_package('click'):
    ...     print(f"Resource: {src_file} -> {dst_path}")
    >>> 
    >>> # List all resources in current project
    >>> for resource_file in list_resources():
    ...     print(f"Found resource: {resource_file}")
    >>> 
    >>> # Generate PyInstaller mappings
    >>> print_resource_mappings()

"""

import logging
import os.path

import importlib_metadata
from hbutils.reflection import quick_import_object


def get_resources_from_package(package):
    """
    Extract non-Python resource files from a specified Python package.

    This function scans a package directory and yields all non-Python files
    along with their relative destination paths. It handles both regular packages
    and single-file modules.

    :param package: Name of the package to scan for resources
    :type package: str
    :yield: Tuple of (absolute_source_path, relative_destination_path)
    :rtype: Generator[tuple[str, str], None, None]

    .. note::
       Single-file packages (modules without __init__.py) are skipped as they
       typically don't contain separate resource files.

    .. warning::
       If the package cannot be imported, a warning is logged and the function
       returns without yielding any results.

    Example::

        >>> for src, dst in get_resources_from_package('click'):
        ...     print(f"Resource: {src} -> {dst}")
        Resource: /path/to/click/data.txt -> click

    """
    try:
        path, _, _ = quick_import_object(f'{package}.__file__')
    except ImportError:
        logging.warning(f'Cannot check {package!r} directory, skipped.')
        return

    if os.path.splitext(os.path.basename(path))[0] != '__init__':  # single file package
        return
    root_dir = os.path.dirname(path)

    for root, _, files in os.walk(root_dir):
        for file in files:
            src_file = os.path.abspath(os.path.join(root, file))
            _, ext = os.path.splitext(os.path.basename(src_file))
            if not ext.startswith('.py'):
                yield src_file, os.path.relpath(os.path.dirname(src_file), os.path.dirname(os.path.abspath(root_dir)))


def list_installed_packages():
    """
    List all installed Python packages in the current environment.

    This function queries the package metadata to retrieve names of all
    installed distributions in the current Python environment.

    :yield: Name of each installed package
    :rtype: Generator[str, None, None]

    Example::

        >>> packages = list(list_installed_packages())
        >>> print(f"Found {len(packages)} installed packages")
        >>> print(packages[:5])  # First 5 packages
        ['numpy', 'pandas', 'requests', 'click', 'setuptools']

    """
    installed_packages = importlib_metadata.distributions()
    for dist in installed_packages:
        yield dist.metadata['Name']


def list_resources():
    """
    List all non-Python resource files in the current project.

    This function walks through the project directory (determined by the location
    of hbllmutils package) and yields absolute paths to all non-Python files,
    excluding __pycache__ directories.

    :yield: Absolute path to each resource file
    :rtype: Generator[str, None, None]

    .. note::
       Python cache directories (__pycache__) are automatically excluded
       from the scan.

    Example::

        >>> for resource in list_resources():
        ...     print(f"Found: {resource}")
        Found: /project/data/config.json
        Found: /project/templates/index.html

    """
    from hbllmutils import __file__ as _mine_file

    proj_dir = os.path.abspath(os.path.normpath(os.path.join(_mine_file, '..')))
    for root, _, files in os.walk(proj_dir):
        if '__pycache__' in root:
            continue

        for file in files:
            _, ext = os.path.splitext(file)
            if ext != '.py':
                rfile = os.path.abspath(os.path.join(root, file))
                yield rfile


def get_resources_from_mine():
    """
    Get resource files from the current project with relative destination paths.

    This function retrieves all resource files from the current project and
    calculates their relative destination paths based on the current working
    directory.

    :yield: Tuple of (absolute_source_path, relative_destination_directory)
    :rtype: Generator[tuple[str, str], None, None]

    Example::

        >>> for src, dst in get_resources_from_mine():
        ...     print(f"Copy {src} to {dst}")
        Copy /project/data/file.txt to data

    """
    workdir = os.path.abspath('.')
    for rfile in list_resources():
        dst_file = os.path.dirname(os.path.relpath(rfile, workdir))
        yield rfile, dst_file


def get_resource_files():
    """
    Aggregate resource files from multiple package sources and the current project.

    This function collects resource files from predefined packages (click, jieba,
    markdown_it) and the current project, yielding all discovered resources.

    :yield: Tuple of (absolute_source_path, relative_destination_path)
    :rtype: Generator[tuple[str, str], None, None]

    .. note::
       Currently scans resources from: click, jieba, markdown_it packages
       and the current project. Additional packages can be added by
       uncommenting the loop at the end of the function.

    Example::

        >>> for src, dst in get_resource_files():
        ...     print(f"Resource mapping: {src} -> {dst}")
        Resource mapping: /path/to/click/data.txt -> click
        Resource mapping: /path/to/jieba/dict.txt -> jieba

    """
    yield from get_resources_from_package('click')
    yield from get_resources_from_package('jieba')
    yield from get_resources_from_package('markdown_it')
    yield from get_resources_from_mine()
    # for pack_name in list_installed_packages():
    #     yield from get_resource_files_from_package(pack_name)


def print_resource_mappings():
    """
    Print PyInstaller-compatible resource file mappings.

    This function generates and prints --add-data arguments for PyInstaller,
    mapping source resource files to their destination paths in the bundled
    executable.

    .. note::
       The output format uses the platform-specific path separator (os.pathsep)
       to separate source and destination paths, as required by PyInstaller.

    Example::

        >>> print_resource_mappings()
        --add-data '/path/to/click/data.txt:click'
        --add-data '/path/to/jieba/dict.txt:jieba'
        --add-data '/project/config.json:.'

    """
    for rfile, dst_file in get_resource_files():
        t = f'{rfile}{os.pathsep}{dst_file}'
        print(f'--add-data {t!r}')


if __name__ == '__main__':
    # print(list_installed_packages())
    print_resource_mappings()
