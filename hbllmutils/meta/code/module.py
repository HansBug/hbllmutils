"""
This module provides utilities for extracting Python module paths from source files.

It analyzes the file system structure to determine the appropriate PYTHONPATH and
module import path for a given Python source file by traversing up the directory
tree to find the package root (identified by the absence of __init__.py).

The module contains functions to:
- Determine the package root directory (PYTHONPATH) for a given source file
- Convert file paths to Python module import paths
- Resolve relative and absolute imports to their full module names
"""

import os.path
import re
from typing import Tuple, Optional


def _get_raw_pythonpath(source_file: str) -> Tuple[str, str]:
    """
    Get the normalized absolute path and its package root directory.

    This internal function normalizes the source file path and traverses up the
    directory tree to find the package root by checking for __init__.py files.
    The package root is identified as the first parent directory that does not
    contain an __init__.py file.

    :param source_file: The path to the Python source file.
    :type source_file: str

    :return: A tuple containing the normalized absolute source file path and the package root directory.
    :rtype: Tuple[str, str]

    Example::
        >>> _get_raw_pythonpath('/path/to/project/package/module.py')
        ('/path/to/project/package/module.py', '/path/to/project')
    """
    source_file = os.path.normpath(os.path.abspath(source_file))
    module_dir = os.path.dirname(source_file)
    while os.path.exists(os.path.join(module_dir, '__init__.py')):
        module_dir = os.path.dirname(module_dir)
    return source_file, module_dir


def get_package_name(source_file: str, pythonpath_dir: Optional[str] = None) -> str:
    """
    Convert a source file path to its corresponding Python module name.

    This function calculates the relative path from the PYTHONPATH directory to the
    source file, removes the file extension, and converts the path separators to dots
    to form a valid Python module import path. If the file is named __init__.py, it
    represents the package itself, so the last segment is removed.

    :param source_file: The absolute or relative path to the Python source file.
    :type source_file: str
    :param pythonpath_dir: The PYTHONPATH directory (package root) to calculate relative path from.
                          If not provided, it will be automatically determined.
    :type pythonpath_dir: Optional[str]

    :return: The Python module import path (e.g., 'package.subpackage.module').
    :rtype: str

    Example::
        >>> get_package_name('/path/to/project/package/module.py', '/path/to/project')
        'package.module'
        
        >>> get_package_name('/path/to/project/package/__init__.py', '/path/to/project')
        'package'
        
        >>> get_package_name('C:\\project\\pkg\\subpkg\\file.py', 'C:\\project')
        'pkg.subpkg.file'
    """
    pythonpath_dir = pythonpath_dir or _get_raw_pythonpath(source_file)[-1]
    rel_file = os.path.relpath(source_file, pythonpath_dir)
    segments_text, _ = os.path.splitext(rel_file)
    segments = [t for t in re.split(r'[\\/]+', segments_text) if t]
    if segments[-1] == '__init__':
        segments = segments[:-1]
    return '.'.join(segments)


def get_pythonpath_of_source_file(source_file: str) -> Tuple[str, str]:
    """
    Get the PYTHONPATH directory and module import path for a given Python source file.

    This function traverses up the directory tree from the source file location until
    it finds a directory without an __init__.py file, which is considered the package
    root. It then calculates the relative module path that can be used for imports.

    :param source_file: The path to the Python source file.
    :type source_file: str

    :return: A tuple containing the module directory (PYTHONPATH) and the module import path.
    :rtype: Tuple[str, str]

    Example::
        >>> get_pythonpath_of_source_file('/path/to/project/package/subpackage/module.py')
        ('/path/to/project', 'package.subpackage.module')
        
        >>> get_pythonpath_of_source_file('/path/to/standalone_script.py')
        ('/path/to', 'standalone_script')
    """
    source_file, module_dir = _get_raw_pythonpath(source_file)
    return module_dir, get_package_name(source_file, module_dir)


def get_package_from_import(source_file: str, import_: str) -> str:
    """
    Resolve an import statement to its full module name, handling both absolute and relative imports.

    This function takes a source file path and an import string, and resolves it to the
    full absolute module path. For absolute imports (not starting with a dot), it returns
    the import string as-is. For relative imports (starting with one or more dots), it
    resolves the import relative to the source file's package location.

    :param source_file: The path to the Python source file where the import occurs.
    :type source_file: str
    :param import_: The import string to resolve (e.g., '.module', '..package.module', 'absolute.module').
    :type import_: str

    :return: The fully resolved absolute module import path.
    :rtype: str

    Example::
        >>> get_package_from_import('/path/to/project/pkg/subpkg/file.py', 'external.module')
        'external.module'
        
        >>> get_package_from_import('/path/to/project/pkg/subpkg/file.py', '.sibling')
        'pkg.subpkg.sibling'

        >>> get_package_from_import('/path/to/project/pkg/subpkg/__init__.py', '.sibling')
        'pkg.subpkg.sibling'
        
        >>> get_package_from_import('/path/to/project/pkg/subpkg/file.py', '..parent_module')
        'pkg.parent_module'
    """
    imports = import_.split('.')
    if imports[0] != '':
        # is absolute import
        return import_
    else:
        # is relative import
        if imports[-1] == '':
            imports[-1] = '__init__'
        _, package_name = get_pythonpath_of_source_file(
            source_file=os.path.join(source_file, '/'.join([('..' if x == '' else x) for x in imports]))
        )
        return package_name
