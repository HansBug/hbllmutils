"""
This module provides utilities for inspecting Python objects and retrieving their source code information.

It includes functionality to extract source file paths, line numbers, and source code for various Python objects
such as functions, classes, and methods. The module is particularly useful for code analysis, documentation
generation, and debugging purposes.
"""

import inspect
import os
import pathlib
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ObjectInspect:
    """
    A dataclass that encapsulates inspection information about a Python object.

    This class stores metadata about a Python object including its source file location,
    line numbers, and source code. It provides convenient properties to access and
    manipulate this information.

    :param object: The Python object being inspected.
    :type object: Any
    :param source_file: The absolute path to the source file containing the object, or None if unavailable.
    :type source_file: Optional[str]
    :param start_line: The starting line number of the object in the source file, or None if unavailable.
    :type start_line: Optional[int]
    :param end_line: The ending line number of the object in the source file, or None if unavailable.
    :type end_line: Optional[int]
    :param source_lines: A list of source code lines for the object, or None if unavailable.
    :type source_lines: Optional[List[str]]
    """
    object: Any
    source_file: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]
    source_lines: Optional[List[str]]

    def __post_init__(self):
        """
        Normalize and convert the source file path to an absolute path after initialization.

        This method is automatically called after the dataclass is initialized. It ensures
        that the source_file path is normalized, case-normalized, and converted to an
        absolute path for consistency across different platforms.
        """
        if self.source_file is not None:
            self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the inspected object.

        :return: The name of the object if it has a '__name__' attribute, otherwise None.
        :rtype: Optional[str]

        Example::
            >>> def example_func():
            ...     pass
            >>> info = get_object_info(example_func)
            >>> info.name
            'example_func'
        """
        return getattr(self.object, '__name__', None)

    @property
    def source_code(self) -> Optional[str]:
        """
        Get the source code of the inspected object.

        :return: The complete source code as a string if available, otherwise None.
        :rtype: Optional[str]

        Example::
            >>> def example_func():
            ...     return 42
            >>> info = get_object_info(example_func)
            >>> print(info.source_code)
            def example_func():
                return 42
        """
        if self.has_source:
            return ''.join(self.source_lines)
        else:
            return None

    @property
    def source_file_code(self) -> Optional[str]:
        """
        Get the complete source code of the file containing the inspected object.

        :return: The entire file content as a string if the source file is available, otherwise None.
        :rtype: Optional[str]

        Example::
            >>> info = get_object_info(some_function)
            >>> file_content = info.source_file_code  # Gets entire file content
        """
        if self.source_file is not None:
            return pathlib.Path(self.source_file).read_text()
        else:
            return None

    @property
    def has_source(self) -> bool:
        """
        Check whether source code lines are available for the inspected object.

        :return: True if source lines are available, False otherwise.
        :rtype: bool

        Example::
            >>> info = get_object_info(print)  # Built-in function
            >>> info.has_source
            False
            >>> def custom_func():
            ...     pass
            >>> info = get_object_info(custom_func)
            >>> info.has_source
            True
        """
        return self.source_lines is not None

    @property
    def package_name(self) -> Optional[str]:
        from .module import get_package_name
        if self.source_file is not None:
            return get_package_name(self.source_file)
        else:
            return None


def get_object_info(obj: Any) -> ObjectInspect:
    """
    Retrieve comprehensive inspection information about a Python object.

    This function attempts to extract source file location, line numbers, and source code
    for the given object. If any information is unavailable (e.g., for built-in objects),
    the corresponding fields will be set to None.

    :param obj: The Python object to inspect (function, class, method, etc.).
    :type obj: Any

    :return: An ObjectInspect instance containing all available inspection information.
    :rtype: ObjectInspect

    Example::
        >>> def example_function():
        ...     '''A simple example function.'''
        ...     return "Hello, World!"
        >>> info = get_object_info(example_function)
        >>> info.name
        'example_function'
        >>> info.has_source
        True
        >>> print(info.source_code)  # doctest: +SKIP
        def example_function():
            '''A simple example function.'''
            return "Hello, World!"

        >>> # Built-in objects have limited information
        >>> info = get_object_info(print)
        >>> info.has_source
        False
        >>> info.source_file is None
        True
    """
    try:
        source_file = inspect.getfile(obj)
    except TypeError:
        source_file = None
    try:
        source_lines, start_line = inspect.getsourcelines(obj)
        end_line = start_line + len(source_lines) - 1
    except TypeError:
        source_lines, start_line, end_line = None, None, None
    return ObjectInspect(
        object=obj,
        source_file=source_file,
        start_line=start_line,
        end_line=end_line,
        source_lines=source_lines,
    )
