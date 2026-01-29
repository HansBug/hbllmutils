"""
This module provides utilities for extracting Python module paths from source files.

It analyzes the file system structure to determine the appropriate PYTHONPATH and
module import path for a given Python source file by traversing up the directory
tree to find the package root (identified by the absence of __init__.py).
"""

import os.path
import re
from typing import Tuple, Optional


def _get_raw_pythonpath(source_file: str):
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
    :type pythonpath_dir: str

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
