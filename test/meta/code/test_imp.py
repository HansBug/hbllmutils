import ast
import tempfile
import os
from unittest.mock import patch, MagicMock

import pytest

from hbllmutils.meta.code.imp import (
    ImportStatement,
    FromImportStatement,
    ImportVisitor,
    analyze_imports,
    ImportStatementTyping
)


@pytest.fixture
def simple_import_code():
    """Fixture providing simple import code."""
    return "import os"


@pytest.fixture
def import_with_alias_code():
    """Fixture providing import with alias code."""
    return "import os as operating_system"


@pytest.fixture
def multiple_imports_code():
    """Fixture providing multiple imports code."""
    return "import os, sys as system"


@pytest.fixture
def from_import_code():
    """Fixture providing from-import code."""
    return "from typing import List"


@pytest.fixture
def from_import_with_alias_code():
    """Fixture providing from-import with alias code."""
    return "from typing import List as ListType"


@pytest.fixture
def relative_import_code():
    """Fixture providing relative import code."""
    return "from ..module import func"


@pytest.fixture
def wildcard_import_code():
    """Fixture providing wildcard import code."""
    return "from collections import *"


@pytest.fixture
def wildcard_import_relative_code():
    """Fixture providing relative wildcard import code."""
    return "from .module import *"


@pytest.fixture
def wildcard_import_nested_relative_code():
    """Fixture providing nested relative wildcard import code."""
    return "from ..subpackage.module import *"


@pytest.fixture
def multiple_wildcard_imports_code():
    """Fixture providing multiple wildcard imports code."""
    return """
from collections import *
from itertools import *
from os import *
"""


@pytest.fixture
def mixed_imports_with_wildcard_code():
    """Fixture providing mixed imports with wildcard code."""
    return """
import os
from typing import List
from collections import *
from .local import func
"""


@pytest.fixture
def complex_code():
    """Fixture providing complex import code."""
    return """
import os
import sys as system
from typing import List, Dict
from ..module import func
from . import local_module
"""


@pytest.fixture
def invalid_syntax_code():
    """Fixture providing invalid syntax code."""
    return "import os import sys"


@pytest.fixture
def nested_module_code():
    """Fixture providing nested module import code."""
    return "import os.path.join"


@pytest.fixture
def empty_module_relative_import():
    """Fixture providing empty module relative import code."""
    return "from . import something"


@pytest.fixture
def mock_inspect_getsourcefile():
    """Mock inspect.getsourcefile to return a test path."""
    with patch('inspect.getsourcefile') as mock:
        mock.return_value = '/path/to/module.py'
        yield mock


@pytest.fixture
def mock_quick_import_object():
    """Mock quick_import_object to return a mock object."""
    with patch('hbllmutils.meta.code.imp.quick_import_object') as mock:
        mock_obj = MagicMock()
        mock.return_value = (mock_obj, None, None)
        yield mock


@pytest.fixture
def mock_get_module_info():
    """Mock get_module_info for testing check_ignore_or_not."""
    with patch('hbllmutils.meta.code.imp.get_module_info') as mock:
        yield mock


@pytest.fixture
def mock_is_hot_pypi_project():
    """Mock is_hot_pypi_project for testing check_ignore_or_not."""
    with patch('hbllmutils.meta.code.imp.is_hot_pypi_project') as mock:
        yield mock


@pytest.mark.unittest
class TestImportStatement:
    """Tests for the ImportStatement class."""

    def test_init_basic(self):
        """Test basic initialization of ImportStatement."""
        stmt = ImportStatement(module='os', line=1, col_offset=0)
        assert stmt.module == 'os'
        assert stmt.alias is None
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_init_with_alias(self):
        """Test initialization with alias."""
        stmt = ImportStatement(module='os', alias='operating_system', line=1, col_offset=0)
        assert stmt.module == 'os'
        assert stmt.alias == 'operating_system'
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_str_without_alias(self):
        """Test string representation without alias."""
        stmt = ImportStatement(module='os')
        assert str(stmt) == "import os"

    def test_str_with_alias(self):
        """Test string representation with alias."""
        stmt = ImportStatement(module='os', alias='operating_system')
        assert str(stmt) == "import os as operating_system"

    def test_root_module_simple(self):
        """Test root_module property with simple module."""
        stmt = ImportStatement(module='os')
        assert stmt.root_module == 'os'

    def test_root_module_nested(self):
        """Test root_module property with nested module."""
        stmt = ImportStatement(module='os.path.join')
        assert stmt.root_module == 'os'

    def test_module_file(self, mock_quick_import_object, mock_inspect_getsourcefile):
        """Test module_file property."""
        stmt = ImportStatement(module='os')
        result = stmt.module_file
        assert result == '/path/to/module.py'
        mock_quick_import_object.assert_called_once_with('os')
        mock_inspect_getsourcefile.assert_called_once()

    def test_check_ignore_or_not_unknown_module(self, mock_get_module_info):
        """Test check_ignore_or_not with unknown module."""
        mock_get_module_info.return_value = None
        stmt = ImportStatement(module='unknown_module')
        assert stmt.check_ignore_or_not() is True

    def test_check_ignore_or_not_standard_library(self, mock_get_module_info):
        """Test check_ignore_or_not with standard library module."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = False
        mock_get_module_info.return_value = mock_module_info
        
        stmt = ImportStatement(module='os')
        assert stmt.check_ignore_or_not() is True

    def test_check_ignore_or_not_hot_project(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with hot PyPI project."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'requests'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = True
        
        stmt = ImportStatement(module='requests')
        assert stmt.check_ignore_or_not(min_last_month_downloads=1000000) is True

    def test_check_ignore_or_not_not_hot_project(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with non-hot PyPI project."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'obscure_package'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = False
        
        stmt = ImportStatement(module='obscure_package')
        assert stmt.check_ignore_or_not() is False

    def test_check_ignore_or_not_with_ignore_modules(self, mock_get_module_info):
        """Test check_ignore_or_not with ignore_modules parameter."""
        stmt = ImportStatement(module='mymodule')
        assert stmt.check_ignore_or_not(ignore_modules={'mymodule'}) is True

    def test_check_ignore_or_not_with_no_ignore_modules(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with no_ignore_modules parameter."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'requests'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = True
        
        stmt = ImportStatement(module='requests')
        assert stmt.check_ignore_or_not(no_ignore_modules={'requests'}) is False

    def test_check_ignore_or_not_nested_module_prefix_match(self, mock_get_module_info):
        """Test check_ignore_or_not with nested module prefix matching."""
        stmt = ImportStatement(module='mypackage.submodule.module')
        assert stmt.check_ignore_or_not(ignore_modules={'mypackage.submodule'}) is True


@pytest.mark.unittest
class TestFromImportStatement:
    """Tests for the FromImportStatement class."""

    def test_init_basic(self):
        """Test basic initialization of FromImportStatement."""
        stmt = FromImportStatement(module='typing', name='List', line=1, col_offset=0)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias is None
        assert stmt.level == 0
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_init_with_alias(self):
        """Test initialization with alias."""
        stmt = FromImportStatement(module='typing', name='List', alias='ListType', level=0, line=1, col_offset=0)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias == 'ListType'
        assert stmt.level == 0

    def test_init_relative(self):
        """Test initialization with relative import."""
        stmt = FromImportStatement(module='module', name='func', level=2, line=1, col_offset=0)
        assert stmt.module == 'module'
        assert stmt.name == 'func'
        assert stmt.level == 2

    def test_init_wildcard(self):
        """Test initialization with wildcard import."""
        stmt = FromImportStatement(module='collections', name='*', level=0, line=1, col_offset=0)
        assert stmt.module == 'collections'
        assert stmt.name == '*'
        assert stmt.alias is None
        assert stmt.level == 0

    def test_init_wildcard_relative(self):
        """Test initialization with relative wildcard import."""
        stmt = FromImportStatement(module='module', name='*', level=1, line=1, col_offset=0)
        assert stmt.module == 'module'
        assert stmt.name == '*'
        assert stmt.level == 1

    def test_str_absolute_import(self):
        """Test string representation of absolute import."""
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert str(stmt) == "from typing import List"

    def test_str_with_alias(self):
        """Test string representation with alias."""
        stmt = FromImportStatement(module='typing', name='List', alias='ListType', level=0)
        assert str(stmt) == "from typing import List as ListType"

    def test_str_relative_import(self):
        """Test string representation of relative import."""
        stmt = FromImportStatement(module='module', name='func', level=2)
        assert str(stmt) == "from ..module import func"

    def test_str_relative_import_single_dot(self):
        """Test string representation with single dot relative import."""
        stmt = FromImportStatement(module='module', name='func', level=1)
        assert str(stmt) == "from .module import func"

    def test_str_empty_module_with_dots(self):
        """Test string representation with empty module and dots."""
        stmt = FromImportStatement(module='', name='something', level=1)
        assert str(stmt) == "from . import something"

    def test_str_no_module_no_level(self):
        """Test string representation with no module and no level."""
        stmt = FromImportStatement(module='', name='something', level=0)
        assert str(stmt) == "from . import something"

    def test_str_wildcard_absolute(self):
        """Test string representation of absolute wildcard import."""
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert str(stmt) == "from collections import *"

    def test_str_wildcard_relative(self):
        """Test string representation of relative wildcard import."""
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert str(stmt) == "from .module import *"

    def test_str_wildcard_nested_relative(self):
        """Test string representation of nested relative wildcard import."""
        stmt = FromImportStatement(module='subpackage.module', name='*', level=2)
        assert str(stmt) == "from ..subpackage.module import *"

    def test_str_wildcard_with_alias_ignored(self):
        """Test that wildcard imports ignore alias in string representation."""
        stmt = FromImportStatement(module='collections', name='*', alias='ignored', level=0)
        assert str(stmt) == "from collections import *"

    def test_str_wildcard_empty_module(self):
        """Test string representation of wildcard with empty module."""
        stmt = FromImportStatement(module='', name='*', level=1)
        assert str(stmt) == "from . import *"

    def test_is_relative_false(self):
        """Test is_relative property returns False for absolute import."""
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert stmt.is_relative is False

    def test_is_relative_true_with_level(self):
        """Test is_relative property returns True with level."""
        stmt = FromImportStatement(module='module', name='func', level=1)
        assert stmt.is_relative is True

    def test_is_relative_true_no_module(self):
        """Test is_relative property returns True with no module."""
        stmt = FromImportStatement(module='', name='func', level=0)
        assert stmt.is_relative is True

    def test_is_relative_wildcard_false(self):
        """Test is_relative property with absolute wildcard import."""
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert stmt.is_relative is False

    def test_is_relative_wildcard_true(self):
        """Test is_relative property with relative wildcard import."""
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert stmt.is_relative is True

    def test_is_wildcard_true(self):
        """Test is_wildcard property returns True."""
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert stmt.is_wildcard is True

    def test_is_wildcard_false(self):
        """Test is_wildcard property returns False."""
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert stmt.is_wildcard is False

    def test_is_wildcard_relative(self):
        """Test is_wildcard property with relative import."""
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert stmt.is_wildcard is True

    def test_is_wildcard_empty_module(self):
        """Test is_wildcard property with empty module."""
        stmt = FromImportStatement(module='', name='*', level=1)
        assert stmt.is_wildcard is True

    def test_check_ignore_or_not_relative_import(self):
        """Test check_ignore_or_not with relative import."""
        stmt = FromImportStatement(module='module', name='func', level=1)
        assert stmt.check_ignore_or_not() is False

    def test_check_ignore_or_not_unknown_module(self, mock_get_module_info):
        """Test check_ignore_or_not with unknown module."""
        mock_get_module_info.return_value = None
        stmt = FromImportStatement(module='unknown_module', name='func', level=0)
        assert stmt.check_ignore_or_not() is True

    def test_check_ignore_or_not_standard_library(self, mock_get_module_info):
        """Test check_ignore_or_not with standard library module."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = False
        mock_get_module_info.return_value = mock_module_info
        
        stmt = FromImportStatement(module='os', name='path', level=0)
        assert stmt.check_ignore_or_not() is True

    def test_check_ignore_or_not_hot_project(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with hot PyPI project."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'requests'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = True
        
        stmt = FromImportStatement(module='requests', name='get', level=0)
        assert stmt.check_ignore_or_not(min_last_month_downloads=1000000) is True

    def test_check_ignore_or_not_not_hot_project(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with non-hot PyPI project."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'obscure_package'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = False
        
        stmt = FromImportStatement(module='obscure_package', name='func', level=0)
        assert stmt.check_ignore_or_not() is False

    def test_check_ignore_or_not_with_ignore_modules(self, mock_get_module_info):
        """Test check_ignore_or_not with ignore_modules parameter."""
        stmt = FromImportStatement(module='mymodule', name='func', level=0)
        assert stmt.check_ignore_or_not(ignore_modules={'mymodule'}) is True

    def test_check_ignore_or_not_with_no_ignore_modules(self, mock_get_module_info, mock_is_hot_pypi_project):
        """Test check_ignore_or_not with no_ignore_modules parameter."""
        mock_module_info = MagicMock()
        mock_module_info.is_third_party = True
        mock_module_info.pypi_name = 'requests'
        mock_get_module_info.return_value = mock_module_info
        mock_is_hot_pypi_project.return_value = True
        
        stmt = FromImportStatement(module='requests', name='get', level=0)
        assert stmt.check_ignore_or_not(no_ignore_modules={'requests'}) is False


@pytest.mark.unittest
class TestImportVisitor:
    """Tests for the ImportVisitor class."""

    def test_init(self):
        """Test initialization of ImportVisitor."""
        visitor = ImportVisitor()
        assert visitor.imports == []

    def test_visit_import_single(self, simple_import_code):
        """Test visiting single import statement."""
        tree = ast.parse(simple_import_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os'
        assert stmt.alias is None
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_visit_import_with_alias(self, import_with_alias_code):
        """Test visiting import with alias."""
        tree = ast.parse(import_with_alias_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os'
        assert stmt.alias == 'operating_system'

    def test_visit_import_multiple(self, multiple_imports_code):
        """Test visiting multiple imports."""
        tree = ast.parse(multiple_imports_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 2

        stmt1 = visitor.imports[0]
        assert isinstance(stmt1, ImportStatement)
        assert stmt1.module == 'os'
        assert stmt1.alias is None

        stmt2 = visitor.imports[1]
        assert isinstance(stmt2, ImportStatement)
        assert stmt2.module == 'sys'
        assert stmt2.alias == 'system'

    def test_visit_import_from_single(self, from_import_code):
        """Test visiting single from-import statement."""
        tree = ast.parse(from_import_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias is None
        assert stmt.level == 0

    def test_visit_import_from_with_alias(self, from_import_with_alias_code):
        """Test visiting from-import with alias."""
        tree = ast.parse(from_import_with_alias_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias == 'ListType'

    def test_visit_import_from_relative(self, relative_import_code):
        """Test visiting relative from-import."""
        tree = ast.parse(relative_import_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'module'
        assert stmt.name == 'func'
        assert stmt.level == 2

    def test_visit_import_from_empty_module(self, empty_module_relative_import):
        """Test visiting from-import with empty module."""
        tree = ast.parse(empty_module_relative_import)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == ''
        assert stmt.name == 'something'
        assert stmt.level == 1

    def test_visit_wildcard_import(self, wildcard_import_code):
        """Test visiting wildcard import."""
        tree = ast.parse(wildcard_import_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'collections'
        assert stmt.name == '*'
        assert stmt.alias is None
        assert stmt.level == 0
        assert stmt.is_wildcard is True

    def test_visit_wildcard_import_relative(self, wildcard_import_relative_code):
        """Test visiting relative wildcard import."""
        tree = ast.parse(wildcard_import_relative_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'module'
        assert stmt.name == '*'
        assert stmt.level == 1
        assert stmt.is_wildcard is True
        assert stmt.is_relative is True

    def test_visit_wildcard_import_nested_relative(self, wildcard_import_nested_relative_code):
        """Test visiting nested relative wildcard import."""
        tree = ast.parse(wildcard_import_nested_relative_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'subpackage.module'
        assert stmt.name == '*'
        assert stmt.level == 2
        assert stmt.is_wildcard is True
        assert stmt.is_relative is True

    def test_visit_multiple_wildcard_imports(self, multiple_wildcard_imports_code):
        """Test visiting multiple wildcard imports."""
        tree = ast.parse(multiple_wildcard_imports_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 3

        for stmt in visitor.imports:
            assert isinstance(stmt, FromImportStatement)
            assert stmt.name == '*'
            assert stmt.is_wildcard is True
            assert stmt.level == 0

        assert visitor.imports[0].module == 'collections'
        assert visitor.imports[1].module == 'itertools'
        assert visitor.imports[2].module == 'os'

    def test_visit_mixed_imports_with_wildcard(self, mixed_imports_with_wildcard_code):
        """Test visiting mixed imports including wildcard."""
        tree = ast.parse(mixed_imports_with_wildcard_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 4

        # Regular import
        stmt1 = visitor.imports[0]
        assert isinstance(stmt1, ImportStatement)
        assert stmt1.module == 'os'

        # From import
        stmt2 = visitor.imports[1]
        assert isinstance(stmt2, FromImportStatement)
        assert stmt2.module == 'typing'
        assert stmt2.name == 'List'
        assert stmt2.is_wildcard is False

        # Wildcard import
        stmt3 = visitor.imports[2]
        assert isinstance(stmt3, FromImportStatement)
        assert stmt3.module == 'collections'
        assert stmt3.name == '*'
        assert stmt3.is_wildcard is True

        # Relative import
        stmt4 = visitor.imports[3]
        assert isinstance(stmt4, FromImportStatement)
        assert stmt4.module == 'local'
        assert stmt4.name == 'func'
        assert stmt4.level == 1
        assert stmt4.is_relative is True

    def test_visit_complex_code(self, complex_code):
        """Test visiting complex code with various import types."""
        tree = ast.parse(complex_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 6

        # Check first import
        stmt1 = visitor.imports[0]
        assert isinstance(stmt1, ImportStatement)
        assert stmt1.module == 'os'

        # Check second import
        stmt2 = visitor.imports[1]
        assert isinstance(stmt2, ImportStatement)
        assert stmt2.module == 'sys'
        assert stmt2.alias == 'system'

        # Check from imports
        stmt3 = visitor.imports[2]
        assert isinstance(stmt3, FromImportStatement)
        assert stmt3.module == 'typing'
        assert stmt3.name == 'List'

        stmt4 = visitor.imports[3]
        assert isinstance(stmt4, FromImportStatement)
        assert stmt4.module == 'typing'
        assert stmt4.name == 'Dict'

        stmt5 = visitor.imports[4]
        assert isinstance(stmt5, FromImportStatement)
        assert stmt5.module == 'module'
        assert stmt5.name == 'func'
        assert stmt5.level == 2


@pytest.mark.unittest
class TestAnalyzeImports:
    """Tests for the analyze_imports function."""

    def test_analyze_imports_simple(self, simple_import_code):
        """Test analyzing simple import code."""
        imports = analyze_imports(simple_import_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os'

    def test_analyze_imports_complex(self, complex_code):
        """Test analyzing complex import code."""
        imports = analyze_imports(complex_code)

        assert len(imports) == 6

        # Verify types and basic properties
        assert isinstance(imports[0], ImportStatement)
        assert isinstance(imports[1], ImportStatement)
        assert isinstance(imports[2], FromImportStatement)
        assert isinstance(imports[3], FromImportStatement)
        assert isinstance(imports[4], FromImportStatement)
        assert isinstance(imports[5], FromImportStatement)

    def test_analyze_imports_wildcard(self, wildcard_import_code):
        """Test analyzing wildcard import."""
        imports = analyze_imports(wildcard_import_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'collections'
        assert stmt.name == '*'
        assert stmt.is_wildcard is True

    def test_analyze_imports_multiple_wildcard(self, multiple_wildcard_imports_code):
        """Test analyzing multiple wildcard imports."""
        imports = analyze_imports(multiple_wildcard_imports_code)

        assert len(imports) == 3
        for stmt in imports:
            assert isinstance(stmt, FromImportStatement)
            assert stmt.is_wildcard is True

    def test_analyze_imports_mixed_with_wildcard(self, mixed_imports_with_wildcard_code):
        """Test analyzing mixed imports with wildcard."""
        imports = analyze_imports(mixed_imports_with_wildcard_code)

        assert len(imports) == 4
        wildcard_count = sum(1 for stmt in imports if isinstance(stmt, FromImportStatement) and stmt.is_wildcard)
        assert wildcard_count == 1

    def test_analyze_imports_wildcard_relative(self, wildcard_import_relative_code):
        """Test analyzing relative wildcard import."""
        imports = analyze_imports(wildcard_import_relative_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.is_wildcard is True
        assert stmt.is_relative is True
        assert stmt.level == 1

    def test_analyze_imports_empty_code(self):
        """Test analyzing empty code."""
        imports = analyze_imports("")
        assert len(imports) == 0

    def test_analyze_imports_no_imports(self):
        """Test analyzing code with no imports."""
        code = "x = 1\ny = 2"
        imports = analyze_imports(code)
        assert len(imports) == 0

    def test_analyze_imports_syntax_error(self, invalid_syntax_code):
        """Test analyzing code with syntax error."""
        with pytest.raises(SyntaxError):
            analyze_imports(invalid_syntax_code)

    def test_analyze_imports_nested_module(self, nested_module_code):
        """Test analyzing nested module import."""
        imports = analyze_imports(nested_module_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os.path.join'
        assert stmt.root_module == 'os'
