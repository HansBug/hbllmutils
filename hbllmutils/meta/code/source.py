import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import List, Union

from hbutils.reflection import mount_pythonpath, quick_import_object

from .imp import analyze_imports, FromImportStatement, ImportStatement
from .module import get_pythonpath_of_source_file, get_package_name
from .object import get_object_info, ObjectInspect


@dataclass
class ImportSource:
    statement: Union[FromImportStatement, ImportStatement]
    inspect: ObjectInspect


@dataclass
class SourceInfo:
    source_file: str
    source_lines: List[str]
    imports: List[ImportSource]

    def __post_init__(self):
        self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def source_code(self) -> str:
        return ''.join(self.source_lines)

    @property
    def package_name(self) -> str:
        return get_package_name(self.source_file)


def get_source_info(source_file: str, skip_when_error: bool = False):
    source_code = pathlib.Path(source_file).read_text()
    source_lines = [line.rstrip() for line in source_code.splitlines(keepends=True)]
    import_statements = analyze_imports(source_code)

    from_imports: List[FromImportStatement] = []
    for import_item in import_statements:
        if isinstance(import_item, FromImportStatement):
            from_imports.append(import_item)

    pythonpath, pkg_name = get_pythonpath_of_source_file(source_file)

    with mount_pythonpath(pythonpath):
        import_inspects = []
        for import_item in from_imports:
            actual_name = import_item.alias or import_item.name
            try:
                obj, _, _ = quick_import_object(f'{pkg_name}.{actual_name}')
                inspect_obj = get_object_info(obj)
            except Exception as err:
                if not skip_when_error:
                    raise

                warnings.warn(
                    f"Failed to import object {actual_name!r} from module {pkg_name!r} "
                    f"in source file {source_file!r}: {type(err).__name__}: {err}",
                    ImportWarning,
                    stacklevel=2
                )
            else:
                import_inspects.append(ImportSource(
                    statement=import_item,
                    inspect=inspect_obj,
                ))

        return SourceInfo(
            source_file=source_file,
            source_lines=source_lines,
            imports=import_inspects,
        )
