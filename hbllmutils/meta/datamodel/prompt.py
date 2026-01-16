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
        self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def class_name(self) -> str:
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


def get_class_info(cls) -> DataModelInspect:
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
    source_file: str
    inspects: List[DataModelInspect]

    @property
    def class_names(self) -> List[str]:
        return [insp.class_name for insp in self.inspects]

    @property
    def source_file_code(self) -> str:
        return pathlib.Path(self.source_file).read_text()


def create_meta_prompt_for_datamodel(dm_cls: type, related_dm_clses: Optional[List[type]] = None):
    prompt_template_file = os.path.join(os.path.dirname(__file__), 'prompt.j2')
    t = PromptTemplate.from_file(prompt_template_file)
    related_dm_clses = list(related_dm_clses or [])

    classes = sorted({dm_cls, *related_dm_clses}, key=lambda x: x.__name__)
    d_source_classes = defaultdict(list)
    for cls in classes:
        cls_info = get_class_info(cls)
        d_source_classes[cls_info.source_file].append(cls_info)

    relate_infos = []
    for source_file, inspects in natsorted(d_source_classes.items()):
        relate_infos.append(RelatedReferencedFile(
            source_file=source_file,
            inspects=inspects,
        ))

    return t.render(
        dm_info=get_class_info(dm_cls),
        relate_infos=relate_infos,
    )
