"""
Module for inspecting Python data model classes and generating meta prompts.

This module provides utilities for extracting source code information from Python classes,
particularly data model classes, and generating prompts based on their structure. It includes
functionality to:

- Inspect class source code, file locations, and line numbers
- Group related classes by their source files
- Generate meta prompts for data models including their related classes

The main components are:

- DataModelInspect: A dataclass storing inspection metadata for a single class
- RelatedReferencedFile: A dataclass grouping classes from the same source file
- Helper functions for class inspection and prompt generation
"""

import inspect
import os.path
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from typing import Optional

from natsort import natsorted

from ...template import PromptTemplate


@dataclass
class DataModelInspect:
    """
    Data class for storing inspection information about a Python class.

    This class holds metadata about a class's source code, including the file location,
    line numbers, and the actual source code lines.

    :ivar class_obj: The class object being inspected.
    :type class_obj: type
    :ivar source_file: The absolute path to the source file containing the class.
    :type source_file: str
    :ivar start_line: The line number where the class definition starts.
    :type start_line: int
    :ivar end_line: The line number where the class definition ends.
    :type end_line: int
    :ivar source_lines: List of source code lines for the class definition.
    :type source_lines: List[str]
    """
    class_obj: type
    source_file: str
    start_line: int
    end_line: int
    source_lines: List[str]

    def __post_init__(self):
        """
        Post-initialization processing to normalize the source file path.

        Converts the source file path to an absolute, normalized, and case-normalized
        path for consistent comparison across different platforms.
        """
        self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def class_name(self) -> str:
        """
        Get the name of the inspected class.

        :return: The class name.
        :rtype: str

        Example::
            >>> class MyClass:
            ...     pass
            >>> info = get_class_info(MyClass)
            >>> info.class_name
            'MyClass'
        """
        return self.class_obj.__name__

    @property
    def source_code(self) -> str:
        """
        Get the complete source code of the inspected class.

        :return: The concatenated source code lines of the class.
        :rtype: str

        Example::
            >>> inspect_info = get_class_info(MyClass)
            >>> print(inspect_info.source_code)
            class MyClass:
                def __init__(self):
                    pass
        """
        return ''.join(self.source_lines)

    @property
    def source_file_code(self) -> str:
        """
        Get the complete source code of the file containing the inspected class.

        :return: The entire content of the source file.
        :rtype: str

        Example::
            >>> inspect_info = get_class_info(MyClass)
            >>> file_content = inspect_info.source_file_code
            >>> print(len(file_content))
            1234
        """
        return pathlib.Path(self.source_file).read_text()


def get_class_info(cls: type) -> DataModelInspect:
    """
    Get inspection information for a given class.

    This function retrieves metadata about a class including its source file location,
    line numbers, and source code. It uses Python's inspect module to extract this
    information.

    :param cls: The class to inspect.
    :type cls: type

    :return: An object containing the class's source file, line numbers, and source code.
    :rtype: DataModelInspect

    :raises OSError: If the source file cannot be found or read.
    :raises TypeError: If the provided object is not a class or doesn't have source code.

    Example::
        >>> class MyClass:
        ...     def method(self):
        ...         pass
        >>> info = get_class_info(MyClass)
        >>> print(info.start_line)
        1
        >>> print(info.source_file)
        /path/to/file.py
        >>> print(info.source_code)
        class MyClass:
            def method(self):
                pass
    """
    source_file = inspect.getfile(cls)
    source_lines, start_line = inspect.getsourcelines(cls)
    return DataModelInspect(
        class_obj=cls,
        source_file=source_file,
        start_line=start_line,
        end_line=start_line + len(source_lines) - 1,
        source_lines=source_lines,
    )


@dataclass
class RelatedReferencedFile:
    """
    Data class for grouping related class inspections from the same source file.

    This class organizes multiple DataModelInspect objects that belong to the same
    source file, providing convenient access to their names and the file's content.

    :ivar source_file: The absolute path to the source file.
    :type source_file: str
    :ivar inspects: List of DataModelInspect objects for classes in this file.
    :type inspects: List[DataModelInspect]
    """
    source_file: str
    inspects: List[DataModelInspect]

    @property
    def class_names(self) -> List[str]:
        """
        Get the names of all classes in this file.

        :return: List of class names from all inspected classes in this file.
        :rtype: List[str]

        Example::
            >>> ref_file = RelatedReferencedFile(
            ...     source_file='/path/to/file.py',
            ...     inspects=[info1, info2]
            ... )
            >>> ref_file.class_names
            ['ClassA', 'ClassB']
        """
        return [insp.class_name for insp in self.inspects]

    @property
    def source_file_code(self) -> str:
        """
        Get the complete source code of the referenced file.

        :return: The entire content of the source file.
        :rtype: str

        Example::
            >>> ref_file = RelatedReferencedFile(
            ...     source_file='/path/to/file.py',
            ...     inspects=[info1]
            ... )
            >>> code = ref_file.source_file_code
            >>> print(len(code))
            5678
        """
        return pathlib.Path(self.source_file).read_text()


def create_meta_prompt_for_datamodel(
        datamodel_class: type,
        related_datamodel_classes: Optional[List[type]] = None,
) -> str:
    """
    Create a meta prompt for a data model class and its related classes.

    This function generates a prompt by inspecting a primary data model class and
    optionally related classes. It groups classes by their source files and renders
    them using a Jinja2 template.

    :param datamodel_class: The primary data model class to generate the prompt for.
    :type datamodel_class: type
    :param related_datamodel_classes: Optional list of related data model classes to include
                             in the prompt. Defaults to None.
    :type related_datamodel_classes: Optional[List[type]]

    :return: The rendered prompt string containing information about the data model
             and its related classes.
    :rtype: str

    :raises FileNotFoundError: If the prompt template file cannot be found.
    :raises OSError: If source files cannot be read.

    Example::
        >>> class UserModel:
        ...     name: str
        ...     age: int
        >>> class AddressModel:
        ...     street: str
        ...     city: str
        >>> prompt = create_meta_prompt_for_datamodel(
        ...     UserModel,
        ...     related_datamodel_classes=[AddressModel]
        ... )
        >>> print(prompt)
        # Generated prompt with class information
    """
    prompt_template_file = os.path.join(os.path.dirname(__file__), 'prompt.j2')
    t = PromptTemplate.from_file(prompt_template_file)
    related_datamodel_classes = list(related_datamodel_classes or [])

    # Sort all classes by name for consistent ordering
    classes = sorted({datamodel_class, *related_datamodel_classes}, key=lambda x: x.__name__)
    d_source_classes = defaultdict(list)

    # Group classes by their source file
    for cls in classes:
        cls_info = get_class_info(cls)
        d_source_classes[cls_info.source_file].append(cls_info)

    # Create RelatedReferencedFile objects for each source file
    relate_infos = []
    for source_file, inspects in natsorted(d_source_classes.items()):
        relate_infos.append(RelatedReferencedFile(
            source_file=source_file,
            inspects=inspects,
        ))

    # Render the template with the collected information
    return t.render(
        dm_info=get_class_info(datamodel_class),
        relate_infos=relate_infos,
    )
