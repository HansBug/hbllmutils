"""
Module for analyzing and extracting import statements from Python source code.

This module provides comprehensive functionality to parse Python code and extract
all import statements, including both regular imports and from-imports. It uses
the Abstract Syntax Tree (AST) to analyze the code structure and collect detailed
import information with position tracking.

The module contains the following main components:

* :class:`ImportStatement` - Represents regular import statements (e.g., ``import os``)
* :class:`FromImportStatement` - Represents from-import statements (e.g., ``from typing import List``)
* :class:`ImportVisitor` - AST visitor for collecting import statements
* :func:`analyze_imports` - Main function for extracting imports from code text

Key Features:

* Support for both absolute and relative imports
* Wildcard import detection (``from module import *``)
* Import alias tracking
* Source location information (line number and column offset)
* Module classification (builtin, standard library, third-party)
* PyPI package popularity checking for filtering

.. note::
   This module requires the ``hbutils`` package for reflection utilities
   and depends on sibling modules for PyPI information retrieval.

.. warning::
   The module uses dynamic import mechanisms which may have security
   implications when analyzing untrusted code.

Example::

    >>> code = '''
    ... import os
    ... import sys as system
    ... from typing import List, Dict
    ... from collections import *
    ... '''
    >>> imports = analyze_imports(code)
    >>> len(imports)
    4
    >>> print(imports[0])
    import os
    >>> print(imports[2])
    from typing import List
    >>> imports[3].is_wildcard
    True
    >>> 
    >>> # Check if imports should be ignored based on popularity
    >>> stmt = ImportStatement(module='requests', line=1, col_offset=0)
    >>> stmt.check_ignore_or_not(min_last_month_downloads=1000000)
    True

"""

import ast
import inspect
from dataclasses import dataclass
from typing import Optional, List, Union, Iterable

from hbutils.reflection import quick_import_object

from .pypi import get_module_info
from .pypi_downloads import is_hot_pypi_project


@dataclass
class ImportStatement:
    """
    Data class representing a regular import statement.

    This class stores information about an import statement of the form
    ``import module`` or ``import module as alias``. It provides utilities
    for analyzing the import and determining whether it should be included
    in documentation or analysis based on module popularity and type.

    :param module: The name of the module being imported
    :type module: str
    :param alias: The alias name for the imported module, if any
    :type alias: Optional[str]
    :param line: The line number where the import statement appears
    :type line: int
    :param col_offset: The column offset where the import statement starts
    :type col_offset: int

    :ivar module: The full module path being imported
    :vartype module: str
    :ivar alias: Optional alias for the imported module
    :vartype alias: Optional[str]
    :ivar line: Line number in source code
    :vartype line: int
    :ivar col_offset: Column offset in source code
    :vartype col_offset: int

    Example::

        >>> stmt = ImportStatement(module='os', alias='operating_system', line=1, col_offset=0)
        >>> print(stmt)
        import os as operating_system
        >>> stmt.root_module
        'os'
        >>> 
        >>> # Nested module import
        >>> stmt = ImportStatement(module='os.path', line=2, col_offset=0)
        >>> stmt.root_module
        'os'
        >>> print(stmt)
        import os.path

    """
    module: str
    alias: Optional[str] = None
    line: int = 0
    col_offset: int = 0

    def __str__(self) -> str:
        """
        Return a Python-readable representation of the import statement.

        Constructs a valid Python import statement string that includes
        the module name and optional alias.

        :return: A string representation of the import statement
        :rtype: str

        Example::

            >>> stmt = ImportStatement(module='os', alias='operating_system')
            >>> str(stmt)
            'import os as operating_system'
            >>> stmt = ImportStatement(module='sys')
            >>> str(stmt)
            'import sys'

        """
        if self.alias:
            return f"import {self.module} as {self.alias}"
        else:
            return f"import {self.module}"

    @property
    def root_module(self) -> str:
        """
        Get the root module name from a potentially nested module path.

        Extracts the top-level module name from a dotted module path.
        For example, for ``import os.path``, this returns ``'os'`` instead
        of ``'os.path'``.

        :return: The root module name
        :rtype: str

        Example::

            >>> stmt = ImportStatement(module='os.path.join')
            >>> stmt.root_module
            'os'
            >>> stmt = ImportStatement(module='numpy')
            >>> stmt.root_module
            'numpy'

        """
        return self.module.split('.')[0]

    @property
    def module_file(self) -> str:
        """
        Get the source code file path of the imported module.

        This property attempts to locate and return the file path where the
        module's source code is defined. It uses dynamic import to resolve
        the module location.

        :return: The file path of the module's source code
        :rtype: str
        :raises TypeError: If the module is a built-in module without a source file
        :raises ImportError: If the module cannot be imported

        .. warning::
           This property performs dynamic imports which may have side effects
           if the module executes code at import time.

        Example::

            >>> stmt = ImportStatement(module='os')
            >>> stmt.module_file  # doctest: +SKIP
            '/usr/lib/python3.x/os.py'

        """
        obj, _, _ = quick_import_object(self.module)
        return inspect.getsourcefile(obj)

    def check_ignore_or_not(self, min_last_month_downloads: int = 1000000,
                            ignore_modules: Optional[Iterable[str]] = None,
                            no_ignore_modules: Optional[Iterable[str]] = None) -> bool:
        """
        Determine whether this import should be ignored in analysis or documentation.

        This method checks various criteria to decide if an import statement should
        be excluded from analysis, such as:
        
        * Module popularity (based on PyPI download statistics)
        * Module type (standard library vs third-party)
        * Explicit inclusion in no-ignore list

        :param min_last_month_downloads: Minimum monthly download threshold for
                                         considering a package as "hot" and ignorable,
                                         defaults to 1000000
        :type min_last_month_downloads: int, optional
        :param ignore_modules: Iterable of module names that should always be ignored
        :type ignore_modules: Optional[Iterable[str]], optional
        :param no_ignore_modules: Iterable of module names that should never be ignored
                                  regardless of other criteria
        :type no_ignore_modules: Optional[Iterable[str]], optional
        :return: True if the import should be ignored, False if it should be included
        :rtype: bool

        .. note::
           The logic for ignoring imports:
           
           * Modules in ``ignore_modules`` are always ignored
           * Modules in ``no_ignore_modules`` are never ignored
           * Unknown/unimportable modules are ignored
           * Standard library modules are ignored
           * Popular third-party packages (hot projects) are ignored
           * Less popular third-party packages are not ignored

        Example::

            >>> stmt = ImportStatement(module='requests', line=1, col_offset=0)
            >>> # Popular package, should be ignored
            >>> stmt.check_ignore_or_not(min_last_month_downloads=1000000)
            True
            >>> 
            >>> # Force inclusion with no_ignore_modules
            >>> stmt.check_ignore_or_not(no_ignore_modules={'requests'})
            False
            >>> 
            >>> # Standard library module
            >>> stmt = ImportStatement(module='os', line=1, col_offset=0)
            >>> stmt.check_ignore_or_not()
            True

        """
        if not isinstance(ignore_modules, set):
            ignore_modules = set(ignore_modules or [])
        if not isinstance(no_ignore_modules, set):
            no_ignore_modules = set(no_ignore_modules or [])
        root_module = self.root_module
        if root_module in ignore_modules:
            return True
        if root_module in no_ignore_modules:
            return False

        module_info = get_module_info(self.root_module)
        if not module_info:
            # unknown module, cannot import, so ignore
            return True
        if not module_info.is_third_party:
            # not third part module, so ignore
            return True
        if module_info.pypi_name and is_hot_pypi_project(module_info.pypi_name, min_last_month_downloads):
            # is a hot project, LLM must have know that, so ignore
            return True

        return False


@dataclass
class FromImportStatement:
    """
    Data class representing a from-import statement.

    This class stores information about an import statement of the form
    ``from module import name`` or ``from module import name as alias``.
    It supports both absolute and relative imports, as well as wildcard imports.

    :param module: The name of the module to import from
    :type module: str
    :param name: The name of the object being imported (can be '*' for wildcard imports)
    :type name: str
    :param alias: The alias name for the imported object, if any
    :type alias: Optional[str]
    :param level: The level of relative import (0 for absolute, 1+ for relative)
    :type level: int
    :param line: The line number where the import statement appears
    :type line: int
    :param col_offset: The column offset where the import statement starts
    :type col_offset: int

    :ivar module: Module path to import from
    :vartype module: str
    :ivar name: Name of the imported object or '*'
    :vartype name: str
    :ivar alias: Optional alias for the imported name
    :vartype alias: Optional[str]
    :ivar level: Relative import level (0=absolute, 1+=relative)
    :vartype level: int
    :ivar line: Line number in source code
    :vartype line: int
    :ivar col_offset: Column offset in source code
    :vartype col_offset: int

    Example::

        >>> stmt = FromImportStatement(module='typing', name='List', alias=None, level=0, line=1, col_offset=0)
        >>> print(stmt)
        from typing import List
        >>> 
        >>> # Wildcard import
        >>> stmt = FromImportStatement(module='collections', name='*', level=0)
        >>> print(stmt)
        from collections import *
        >>> stmt.is_wildcard
        True
        >>> 
        >>> # Relative import
        >>> stmt = FromImportStatement(module='module', name='func', level=2)
        >>> print(stmt)
        from ..module import func
        >>> stmt.is_relative
        True

    """
    module: str
    name: str
    alias: Optional[str] = None
    level: int = 0
    line: int = 0
    col_offset: int = 0

    def __str__(self) -> str:
        """
        Return a Python-readable representation of the from-import statement.

        This method constructs a string representation that includes relative import
        dots, module path, imported name, and optional alias. It correctly handles
        all import variations including relative imports and wildcards.

        :return: A string representation of the from-import statement
        :rtype: str

        Example::

            >>> stmt = FromImportStatement(module='typing', name='List', level=0)
            >>> str(stmt)
            'from typing import List'
            >>> 
            >>> # With alias
            >>> stmt = FromImportStatement(module='typing', name='Dict', alias='D', level=0)
            >>> str(stmt)
            'from typing import Dict as D'
            >>> 
            >>> # Relative import
            >>> stmt = FromImportStatement(module='module', name='func', level=2)
            >>> str(stmt)
            'from ..module import func'
            >>> 
            >>> # Wildcard import (no alias allowed)
            >>> stmt = FromImportStatement(module='collections', name='*', level=0)
            >>> str(stmt)
            'from collections import *'

        """
        # Build the relative import dot prefix
        level_str = "." * self.level

        # Build the module path
        if self.module:
            module_str = f"{level_str}{self.module}"
        else:
            module_str = level_str if level_str else ""

        # Build the alias part (wildcards cannot have aliases)
        alias_str = f" as {self.alias}" if self.alias and not self.is_wildcard else ""

        # Build the complete from-import statement
        if module_str:
            return f"from {module_str} import {self.name}{alias_str}"
        else:
            return f"from . import {self.name}{alias_str}"

    @property
    def is_relative(self) -> bool:
        """
        Check if this is a relative import statement.

        A relative import is one that uses dots to indicate the current and parent
        packages (e.g., ``from . import module`` or ``from ..package import func``),
        or has no module specified which implies the current package.

        :return: True if this is a relative import, False otherwise
        :rtype: bool

        Example::

            >>> # Absolute import
            >>> stmt = FromImportStatement(module='typing', name='List', level=0)
            >>> stmt.is_relative
            False
            >>> 
            >>> # Relative import with level
            >>> stmt = FromImportStatement(module='module', name='func', level=1)
            >>> stmt.is_relative
            True
            >>> 
            >>> # Current package import
            >>> stmt = FromImportStatement(module='', name='func', level=1)
            >>> stmt.is_relative
            True

        """
        return self.level > 0 or not self.module

    @property
    def is_wildcard(self) -> bool:
        """
        Check if this is a wildcard import statement.

        A wildcard import is one that uses '*' to import all public names from a module,
        such as ``from module import *``. This is generally discouraged in Python
        but is sometimes used for convenience.

        :return: True if this is a wildcard import, False otherwise
        :rtype: bool

        .. warning::
           Wildcard imports can pollute the namespace and make code harder to
           understand and maintain. They should be used sparingly.

        Example::

            >>> # Wildcard import
            >>> stmt = FromImportStatement(module='collections', name='*', level=0)
            >>> stmt.is_wildcard
            True
            >>> 
            >>> # Regular import
            >>> stmt = FromImportStatement(module='typing', name='List', level=0)
            >>> stmt.is_wildcard
            False

        """
        return self.name == '*'

    def check_ignore_or_not(self, min_last_month_downloads: int = 1000000,
                            ignore_modules: Optional[Iterable[str]] = None,
                            no_ignore_modules: Optional[Iterable[str]] = None) -> bool:
        """
        Determine whether this from-import should be ignored in analysis or documentation.

        This method checks various criteria to decide if a from-import statement should
        be excluded from analysis. The logic differs from regular imports in that
        relative imports are never ignored since they refer to project-internal modules.

        :param min_last_month_downloads: Minimum monthly download threshold for
                                         considering a package as "hot" and ignorable,
                                         defaults to 1000000
        :type min_last_month_downloads: int, optional
        :param ignore_modules: Iterable of module names that should always be ignored
        :type ignore_modules: Optional[Iterable[str]], optional
        :param no_ignore_modules: Iterable of module names that should never be ignored
                                  regardless of other criteria
        :type no_ignore_modules: Optional[Iterable[str]], optional
        :return: True if the import should be ignored, False if it should be included
        :rtype: bool

        .. note::
           The logic for ignoring from-imports:
           
           * Relative imports are never ignored (they're project-internal)
           * Modules in ``ignore_modules`` are always ignored
           * Modules in ``no_ignore_modules`` are never ignored
           * Unknown/unimportable modules are ignored
           * Standard library modules are ignored
           * Popular third-party packages (hot projects) are ignored
           * Less popular third-party packages are not ignored

        Example::

            >>> # Relative import - never ignored
            >>> stmt = FromImportStatement(module='module', name='func', level=1)
            >>> stmt.check_ignore_or_not()
            False
            >>> 
            >>> # Popular package - should be ignored
            >>> stmt = FromImportStatement(module='requests', name='get', level=0)
            >>> stmt.check_ignore_or_not(min_last_month_downloads=1000000)
            True
            >>> 
            >>> # Force inclusion with no_ignore_modules
            >>> stmt.check_ignore_or_not(no_ignore_modules={'requests'})
            False
            >>> 
            >>> # Standard library module
            >>> stmt = FromImportStatement(module='os', name='path', level=0)
            >>> stmt.check_ignore_or_not()
            True

        """
        if self.is_relative:
            # for relative modules, must not ignore
            return False

        if not isinstance(ignore_modules, set):
            ignore_modules = set(ignore_modules or [])
        if not isinstance(no_ignore_modules, set):
            no_ignore_modules = set(no_ignore_modules or [])
        root_module = self.module.split('.')[0]
        if root_module in ignore_modules:
            return True
        if root_module in no_ignore_modules:
            return False

        module_info = get_module_info(root_module)
        if not module_info:
            # unknown module, cannot import, so ignore
            return True
        if not module_info.is_third_party:
            # not third part module, so ignore
            return True
        if module_info.pypi_name and is_hot_pypi_project(module_info.pypi_name, min_last_month_downloads):
            # is a hot project, LLM must have know that, so ignore
            return True

        return False


ImportStatementTyping = Union[ImportStatement, FromImportStatement]
"""
Type alias for either ImportStatement or FromImportStatement.

This type alias is used throughout the module to represent any kind of
import statement, providing flexibility in function signatures and return types.

Example::

    >>> def process_import(stmt: ImportStatementTyping) -> str:
    ...     return str(stmt)
    >>> 
    >>> stmt1 = ImportStatement(module='os')
    >>> stmt2 = FromImportStatement(module='typing', name='List', level=0)
    >>> process_import(stmt1)
    'import os'
    >>> process_import(stmt2)
    'from typing import List'

"""


class ImportVisitor(ast.NodeVisitor):
    """
    Custom AST visitor class for collecting import information from Python code.

    This class extends :class:`ast.NodeVisitor` to traverse the Abstract Syntax Tree
    and collect all import statements (both regular imports and from-imports)
    found in the code. It handles all import variations including aliases,
    relative imports, and wildcard imports.

    :ivar imports: List of collected import statements
    :vartype imports: List[ImportStatementTyping]

    .. note::
       The visitor collects imports in the order they appear in the source code,
       preserving line numbers and column offsets for each import.

    Example::

        >>> code = '''
        ... import os
        ... import sys as system
        ... from typing import List, Dict
        ... from collections import *
        ... '''
        >>> tree = ast.parse(code)
        >>> visitor = ImportVisitor()
        >>> visitor.visit(tree)
        >>> len(visitor.imports)
        5
        >>> print(visitor.imports[0])
        import os
        >>> print(visitor.imports[2])
        from typing import List
        >>> visitor.imports[4].is_wildcard
        True

    """

    def __init__(self):
        """
        Initialize the ImportVisitor.

        Creates an empty list to store collected import statements during
        AST traversal.

        Example::

            >>> visitor = ImportVisitor()
            >>> len(visitor.imports)
            0

        """
        self.imports: List[ImportStatementTyping] = []

    def visit_Import(self, node: ast.Import) -> None:
        """
        Visit an Import node in the AST.

        This method is called automatically when an import statement is encountered
        during AST traversal. It extracts information about each imported module
        and creates :class:`ImportStatement` objects with position information.

        :param node: The Import AST node to visit
        :type node: ast.Import

        .. note::
           A single import statement can import multiple modules (e.g.,
           ``import os, sys``), so this method may create multiple ImportStatement
           objects from a single node.

        Example::

            >>> code = "import os, sys as system"
            >>> tree = ast.parse(code)
            >>> visitor = ImportVisitor()
            >>> visitor.visit(tree)
            >>> len(visitor.imports)
            2
            >>> print(visitor.imports[0])
            import os
            >>> print(visitor.imports[1])
            import sys as system

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

        This method is called automatically when a from-import statement is encountered
        during AST traversal. It extracts information about each imported name and
        creates :class:`FromImportStatement` objects. This includes support for
        wildcard imports (``from module import *``) and relative imports.

        :param node: The ImportFrom AST node to visit
        :type node: ast.ImportFrom

        .. note::
           A single from-import statement can import multiple names (e.g.,
           ``from typing import List, Dict``), so this method may create multiple
           FromImportStatement objects from a single node.

        Example::

            >>> code = '''
            ... from typing import List, Dict as D
            ... from collections import *
            ... from ..module import func
            ... '''
            >>> tree = ast.parse(code)
            >>> visitor = ImportVisitor()
            >>> visitor.visit(tree)
            >>> len(visitor.imports)
            4
            >>> print(visitor.imports[0])
            from typing import List
            >>> print(visitor.imports[1])
            from typing import Dict as D
            >>> visitor.imports[2].is_wildcard
            True
            >>> visitor.imports[3].is_relative
            True

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

    This function parses the provided Python source code using the AST module
    and collects all import statements (both regular imports and from-imports),
    including wildcard imports and relative imports. It returns a comprehensive
    list of all imports with their position information.

    :param code_text: The Python source code to analyze
    :type code_text: str
    :return: A list of all import statements found in the code, in order of appearance
    :rtype: List[ImportStatementTyping]
    :raises SyntaxError: If the code_text contains invalid Python syntax

    .. note::
       The function preserves the order of imports as they appear in the source code
       and includes detailed position information (line number and column offset)
       for each import statement.

    .. warning::
       The code_text must be syntactically valid Python code. If it contains
       syntax errors, a SyntaxError will be raised during parsing.

    Example::

        >>> code = '''
        ... import os
        ... import sys as system
        ... from typing import List, Dict
        ... from ..module import func
        ... from collections import *
        ... '''
        >>> imports = analyze_imports(code)
        >>> len(imports)
        6
        >>> 
        >>> # Check first import
        >>> print(imports[0])
        import os
        >>> imports[0].line
        2
        >>> 
        >>> # Check from-import
        >>> print(imports[2])
        from typing import List
        >>> 
        >>> # Check wildcard import
        >>> print(imports[5])
        from collections import *
        >>> imports[5].is_wildcard
        True
        >>> 
        >>> # Check relative import
        >>> imports[4].is_relative
        True
        >>> 
        >>> # Handle syntax error
        >>> try:
        ...     analyze_imports("import os as")
        ... except SyntaxError as e:
        ...     print("Syntax error detected")
        Syntax error detected

    """
    tree = ast.parse(code_text)
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports
