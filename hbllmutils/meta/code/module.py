"""
This module provides utilities for extracting Python module paths from source files.

It analyzes the file system structure to determine the appropriate PYTHONPATH and
module import path for a given Python source file by traversing up the directory
tree to find the package root (identified by the absence of __init__.py).
"""

import os.path
import re
from typing import Tuple


def get_pythonpath_of_source_file(source_file: str) -> Tuple[str, str]:
    """
    Get the PYTHONPATH directory and module import path for a given Python source file.

    This function traverses up the directory tree from the source file location until
    it finds a directory without an __init__.py file, which is considered the package
    root. It then calculates the relative module path that can be used for imports.

    :param source_file: The path to the Python source file.
    :type source_file: str

    :return: A tuple containing the module directory (PYTHONPATH) and the module import path.
    :rtype: tuple[str, str]

    Example::
        >>> get_pythonpath_of_source_file('/path/to/project/package/subpackage/module.py')
        ('/path/to/project', 'package.subpackage.module')
        
        >>> get_pythonpath_of_source_file('/path/to/standalone_script.py')
        ('/path/to', 'standalone_script')
    """
    module_dir = os.path.normpath(os.path.abspath(os.path.dirname(source_file)))
    while os.path.exists(os.path.join(module_dir, '__init__.py')):
        module_dir = os.path.dirname(module_dir)

    rel_file = os.path.relpath(source_file, module_dir)
    segments_text, _ = os.path.splitext(rel_file)
    module_text = re.sub(r'[\\/]+', '.', segments_text)
    return module_dir, module_text
