"""
Module for analyzing and extracting import statements from Python source code.

This module provides functionality to parse Python code and extract all import statements,
including both regular imports and from-imports. It uses the Abstract Syntax Tree (AST)
to analyze the code structure and collect import information.

The module includes:
- Data classes for representing import statements
- A custom AST visitor for collecting imports
- A main function for analyzing imports in code text

Example::
    >>> code = '''
    ... import os
    ... from typing import List
    ... '''
    >>> imports = analyze_imports(code)
    >>> print(imports)
    [ImportStatement(module='os', alias=None, line=2, col_offset=0),
     FromImportStatement(module='typing', name='List', alias=None, level=0, line=3, col_offset=0)]
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Optional, List, Union

from hbutils.reflection import quick_import_object


@dataclass
class ImportStatement:
    """
    Data class representing a regular import statement.

    This class stores information about an import statement of the form
    ``import module`` or ``import module as alias``.

    :param module: The name of the module being imported.
    :type module: str
    :param alias: The alias name for the imported module, if any.
    :type alias: Optional[str]
    :param line: The line number where the import statement appears.
    :type line: int
    :param col_offset: The column offset where the import statement starts.
    :type col_offset: int

    Example::
        >>> stmt = ImportStatement(module='os', alias='operating_system', line=1, col_offset=0)
        >>> print(stmt)
        import os as operating_system
    """
    module: str
    alias: Optional[str] = None
    line: int = 0
    col_offset: int = 0

    def __repr__(self) -> str:
        """
        Return a Python-readable representation of the import statement.

        :return: A string representation of the import statement.
        :rtype: str

        Example::
            >>> stmt = ImportStatement(module='os', alias='operating_system')
            >>> repr(stmt)
            'import os as operating_system'
        """
        if self.alias:
            return f"import {self.module} as {self.alias}"
        else:
            return f"import {self.module}"

    @property
    def root_module(self) -> str:
        """
        Get the root module name from a potentially nested module path.

        For example, for 'import os.path', this returns 'os' instead of 'os.path'.

        :return: The root module name.
        :rtype: str

        Example::
            >>> stmt = ImportStatement(module='os.path')
            >>> stmt.root_module
            'os'
        """
        return self.module.split('.')[0]

    @property
    def module_file(self) -> str:
        """
        Get the source code file path of the imported module.

        This property attempts to locate and return the file path where the
        module's source code is defined.

        :return: The file path of the module's source code.
        :rtype: str

        :raises TypeError: If the module is a built-in module without a source file.

        Example::
            >>> stmt = ImportStatement(module='os')
            >>> stmt.module_file  # doctest: +SKIP
            '/usr/lib/python3.x/os.py'
        """
        obj, _, _ = quick_import_object(self.module)
        return inspect.getsourcefile(obj)


@dataclass
class FromImportStatement:
    """
    Data class representing a from-import statement.

    This class stores information about an import statement of the form
    ``from module import name`` or ``from module import name as alias``.
    It also supports relative imports with the level parameter.

    :param module: The name of the module to import from.
    :type module: str
    :param name: The name of the object being imported.
    :type name: str
    :param alias: The alias name for the imported object, if any.
    :type alias: Optional[str]
    :param level: The level of relative import (0 for absolute, 1+ for relative).
    :type level: int
    :param line: The line number where the import statement appears.
    :type line: int
    :param col_offset: The column offset where the import statement starts.
    :type col_offset: int

    Example::
        >>> stmt = FromImportStatement(module='typing', name='List', alias=None, level=0, line=1, col_offset=0)
        >>> print(stmt)
        from typing import List
    """
    module: str
    name: str
    alias: Optional[str] = None
    level: int = 0
    line: int = 0
    col_offset: int = 0

    def __repr__(self) -> str:
        """
        Return a Python-readable representation of the from-import statement.

        This method constructs a string representation that includes relative import
        dots, module path, imported name, and optional alias.

        :return: A string representation of the from-import statement.
        :rtype: str

        Example::
            >>> stmt = FromImportStatement(module='typing', name='List', level=0)
            >>> repr(stmt)
            'from typing import List'
            >>> stmt = FromImportStatement(module='module', name='func', level=2)
            >>> repr(stmt)
            'from ..module import func'
        """
        # Build the relative import dot prefix
        level_str = "." * self.level

        # Build the module path
        if self.module:
            module_str = f"{level_str}{self.module}"
        else:
            module_str = level_str if level_str else ""

        # Build the alias part
        alias_str = f" as {self.alias}" if self.alias else ""

        # Build the complete from-import statement
        if module_str:
            return f"from {module_str} import {self.name}{alias_str}"
        else:
            return f"from . import {self.name}{alias_str}"


ImportStatementTyping = Union[ImportStatement, FromImportStatement]
"""Type alias for either ImportStatement or FromImportStatement."""


class ImportVisitor(ast.NodeVisitor):
    """
    Custom AST visitor class for collecting import information.

    This class extends ast.NodeVisitor to traverse the Abstract Syntax Tree
    and collect all import statements (both regular imports and from-imports)
    found in the code.

    :ivar imports: List of collected import statements.
    :vartype imports: List[ImportStatementTyping]

    Example::
        >>> tree = ast.parse("import os\\nfrom typing import List")
        >>> visitor = ImportVisitor()
        >>> visitor.visit(tree)
        >>> len(visitor.imports)
        2
    """

    def __init__(self):
        """
        Initialize the ImportVisitor.

        Creates an empty list to store collected import statements.
        """
        self.imports: List[ImportStatementTyping] = []

    def visit_Import(self, node: ast.Import) -> None:
        """
        Visit an Import node in the AST.

        This method is called when an import statement is encountered.
        It extracts information about each imported module and creates
        ImportStatement objects.

        :param node: The Import AST node to visit.
        :type node: ast.Import

        Example::
            >>> code = "import os, sys as system"
            >>> tree = ast.parse(code)
            >>> visitor = ImportVisitor()
            >>> visitor.visit(tree)
            >>> len(visitor.imports)
            2
        """
        for alias in node.names:
            import_stmt = ImportStatement(
                module=alias.name,
                alias=alias.asname,
                line=node.lineno,
                col_offset=node.col_offset
            )
            self.imports.append(import_stmt)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Visit an ImportFrom node in the AST.

        This method is called when a from-import statement is encountered.
        It extracts information about each imported name and creates
        FromImportStatement objects.

        :param node: The ImportFrom AST node to visit.
        :type node: ast.ImportFrom

        Example::
            >>> code = "from typing import List, Dict as D"
            >>> tree = ast.parse(code)
            >>> visitor = ImportVisitor()
            >>> visitor.visit(tree)
            >>> len(visitor.imports)
            2
        """
        module = node.module or ''
        level = node.level

        for alias in node.names:
            from_import_stmt = FromImportStatement(
                module=module,
                name=alias.name,
                alias=alias.asname,
                level=level,
                line=node.lineno,
                col_offset=node.col_offset
            )
            self.imports.append(from_import_stmt)
        self.generic_visit(node)


def analyze_imports(code_text: str) -> List[ImportStatementTyping]:
    """
    Analyze Python code text and extract all import statements.

    This function parses the provided Python code text using the AST module
    and collects all import statements (both regular imports and from-imports).

    :param code_text: The Python source code to analyze.
    :type code_text: str

    :return: A list of all import statements found in the code.
    :rtype: List[ImportStatementTyping]

    :raises SyntaxError: If the code_text contains invalid Python syntax.

    Example::
        >>> code = '''
        ... import os
        ... import sys as system
        ... from typing import List, Dict
        ... from ..module import func
        ... '''
        >>> imports = analyze_imports(code)
        >>> len(imports)
        4
        >>> print(imports[0])
        import os
        >>> print(imports[2])
        from typing import List
    """
    tree = ast.parse(code_text)
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports
