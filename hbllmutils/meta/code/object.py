import inspect
import os
import pathlib
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ObjectInspect:
    object: Any
    source_file: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]
    source_lines: Optional[List[str]]

    def __post_init__(self):
        if self.source_file is not None:
            self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def name(self) -> Optional[str]:
        return getattr(self.object, '__name__', None)

    @property
    def source_code(self) -> Optional[str]:
        if self.has_source:
            return ''.join(self.source_lines)
        else:
            return None

    @property
    def source_file_code(self) -> Optional[str]:
        if self.source_file is not None:
            return pathlib.Path(self.source_file).read_text()
        else:
            return None

    @property
    def has_source(self) -> bool:
        return self.source_lines is not None


def get_object_info(obj: Any) -> ObjectInspect:
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
