"""
This module provides functionality for analyzing Python source files and their import statements.

It extracts information about imports, source code structure, and object inspection details
from Python source files. The module can parse import statements, resolve imported objects,
and provide comprehensive metadata about the source file and its dependencies.

Main components:

- ImportSource: Dataclass containing import statement and object inspection information
- SourceInfo: Dataclass containing comprehensive source file information
- get_source_info: Function to analyze and extract information from a Python source file
"""

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
    """
    Represents an import statement along with its corresponding object inspection information.
    
    This dataclass pairs an import statement (either a from-import or regular import)
    with the inspection details of the imported object, providing a complete view
    of both the import syntax and the actual imported object's metadata.
    
    :param statement: The import statement (either FromImportStatement or ImportStatement).
    :type statement: Union[FromImportStatement, ImportStatement]
    :param inspect: The inspection information of the imported object.
    :type inspect: ObjectInspect
    
    Example::
        >>> stmt = FromImportStatement(module='os', name='path', level=0)
        >>> obj_info = get_object_info(os.path)
        >>> import_src = ImportSource(statement=stmt, inspect=obj_info)
        >>> print(import_src.statement)
        from os import path
        >>> print(import_src.inspect.name)
        'path'
    """
    statement: Union[FromImportStatement, ImportStatement]
    inspect: ObjectInspect


@dataclass
class SourceInfo:
    """
    Contains comprehensive information about a Python source file.
    
    This dataclass stores the source file path, its content as lines, and information
    about all imports found in the file. It provides a complete snapshot of a Python
    source file's structure and dependencies.
    
    :param source_file: The path to the source file.
    :type source_file: str
    :param source_lines: List of source code lines from the file.
    :type source_lines: List[str]
    :param imports: List of import sources found in the file.
    :type imports: List[ImportSource]
    
    Example::
        >>> info = SourceInfo(
        ...     source_file='/path/to/module.py',
        ...     source_lines=['import os\\n', 'from typing import List\\n'],
        ...     imports=[]
        ... )
        >>> print(info.source_file)
        '/path/to/module.py'
        >>> print(info.source_code)
        import os
        from typing import List
    """
    source_file: str
    source_lines: List[str]
    imports: List[ImportSource]

    def __post_init__(self):
        """
        Post-initialization processing to normalize the source file path.
        
        Converts the source file path to an absolute, normalized, and case-normalized path
        to ensure consistency across different platforms and path representations.
        This is automatically called after the dataclass is initialized.
        
        Example::
            >>> info = SourceInfo(source_file='./relative/path.py', source_lines=[], imports=[])
            >>> # source_file is now normalized to absolute path
            >>> os.path.isabs(info.source_file)
            True
        """
        self.source_file = os.path.normpath(os.path.normcase(os.path.abspath(self.source_file)))

    @property
    def source_code(self) -> str:
        """
        Get the complete source code as a single string.
        
        This property concatenates all source lines into a single string,
        preserving the original line endings and formatting.
        
        :return: The concatenated source code from all lines.
        :rtype: str
        
        Example::
            >>> info = SourceInfo(
            ...     source_file='test.py',
            ...     source_lines=['import os\\n', 'print("hello")\\n'],
            ...     imports=[]
            ... )
            >>> print(info.source_code)
            import os
            print("hello")
        """
        return ''.join(self.source_lines)

    @property
    def package_name(self) -> str:
        """
        Get the package name of the source file.
        
        This property derives the Python package name from the source file's path
        by analyzing its location relative to the Python path root.
        
        :return: The package name derived from the source file path.
        :rtype: str
        
        Example::
            >>> # For a file at /project/mypackage/submodule/file.py
            >>> info = SourceInfo(source_file='/project/mypackage/submodule/file.py', 
            ...                   source_lines=[], imports=[])
            >>> info.package_name
            'mypackage.submodule.file'
        """
        return get_package_name(self.source_file)


def get_source_info(source_file: str, skip_when_error: bool = False) -> SourceInfo:
    """
    Analyze a Python source file and extract comprehensive information about it.
    
    This function reads the source file, parses its import statements, and attempts to
    inspect the imported objects. It returns a SourceInfo object containing all the
    extracted information including source code, imports, and object metadata.
    
    The function works by:
    
    1. Reading the source file content
    2. Parsing import statements using AST analysis
    3. Determining the Python path and package name
    4. Dynamically importing and inspecting each imported object
    5. Collecting all information into a SourceInfo object
    
    :param source_file: The path to the Python source file to analyze.
    :type source_file: str
    :param skip_when_error: If True, skip imports that fail to load and issue warnings
                           instead of raising exceptions. Defaults to False.
    :type skip_when_error: bool
    
    :return: A SourceInfo object containing the source file information and imports.
    :rtype: SourceInfo
    
    :raises Exception: If an import fails to load and skip_when_error is False.
    
    :warns ImportWarning: If an import fails to load and skip_when_error is True.
    
    Example::
        >>> # Analyze a simple module
        >>> info = get_source_info('mymodule.py')
        >>> print(info.package_name)
        'mypackage.mymodule'
        >>> print(len(info.imports))
        5
        >>> print(info.source_code[:50])
        'import os\\nimport sys\\nfrom typing import List\\n...'
        
        >>> # Skip errors when analyzing problematic imports
        >>> info = get_source_info('module_with_issues.py', skip_when_error=True)
        >>> # Warnings will be issued for failed imports, but processing continues
    """
    source_code = pathlib.Path(source_file).read_text()
    source_lines = [line for line in source_code.splitlines(keepends=True)]
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
