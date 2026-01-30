import ast
from unittest.mock import patch, MagicMock

import pytest

from hbllmutils.meta.code.imp import ImportStatement, FromImportStatement, ImportVisitor, analyze_imports


@pytest.fixture
def simple_import_code():
    return "import os"


@pytest.fixture
def import_with_alias_code():
    return "import os as operating_system"


@pytest.fixture
def multiple_imports_code():
    return "import os, sys as system"


@pytest.fixture
def from_import_code():
    return "from typing import List"


@pytest.fixture
def from_import_with_alias_code():
    return "from typing import List as ListType"


@pytest.fixture
def relative_import_code():
    return "from ..module import func"


@pytest.fixture
def wildcard_import_code():
    return "from collections import *"


@pytest.fixture
def wildcard_import_relative_code():
    return "from .module import *"


@pytest.fixture
def wildcard_import_nested_relative_code():
    return "from ..subpackage.module import *"


@pytest.fixture
def multiple_wildcard_imports_code():
    return """
from collections import *
from itertools import *
from os import *
"""


@pytest.fixture
def mixed_imports_with_wildcard_code():
    return """
import os
from typing import List
from collections import *
from .local import func
"""


@pytest.fixture
def complex_code():
    return """
import os
import sys as system
from typing import List, Dict
from ..module import func
from . import local_module
"""


@pytest.fixture
def invalid_syntax_code():
    return "import os import sys"


@pytest.fixture
def nested_module_code():
    return "import os.path.join"


@pytest.fixture
def empty_module_relative_import():
    return "from . import something"


@pytest.fixture
def mock_inspect_getsourcefile():
    with patch('inspect.getsourcefile') as mock:
        mock.return_value = '/path/to/module.py'
        yield mock


@pytest.fixture
def mock_quick_import_object():
    with patch('hbllmutils.meta.code.imp.quick_import_object') as mock:
        mock_obj = MagicMock()
        mock.return_value = (mock_obj, None, None)
        yield mock


@pytest.mark.unittest
class TestImportStatement:
    def test_init_basic(self):
        stmt = ImportStatement(module='os', line=1, col_offset=0)
        assert stmt.module == 'os'
        assert stmt.alias is None
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_init_with_alias(self):
        stmt = ImportStatement(module='os', alias='operating_system', line=1, col_offset=0)
        assert stmt.module == 'os'
        assert stmt.alias == 'operating_system'
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_str_without_alias(self):
        stmt = ImportStatement(module='os')
        assert str(stmt) == "import os"

    def test_str_with_alias(self):
        stmt = ImportStatement(module='os', alias='operating_system')
        assert str(stmt) == "import os as operating_system"

    def test_root_module_simple(self):
        stmt = ImportStatement(module='os')
        assert stmt.root_module == 'os'

    def test_root_module_nested(self):
        stmt = ImportStatement(module='os.path.join')
        assert stmt.root_module == 'os'

    def test_module_file(self, mock_quick_import_object, mock_inspect_getsourcefile):
        stmt = ImportStatement(module='os')
        result = stmt.module_file
        assert result == '/path/to/module.py'
        mock_quick_import_object.assert_called_once_with('os')
        mock_inspect_getsourcefile.assert_called_once()


@pytest.mark.unittest
class TestFromImportStatement:
    def test_init_basic(self):
        stmt = FromImportStatement(module='typing', name='List', line=1, col_offset=0)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias is None
        assert stmt.level == 0
        assert stmt.line == 1
        assert stmt.col_offset == 0

    def test_init_with_alias(self):
        stmt = FromImportStatement(module='typing', name='List', alias='ListType', level=0, line=1, col_offset=0)
        assert stmt.module == 'typing'
        assert stmt.name == 'List'
        assert stmt.alias == 'ListType'
        assert stmt.level == 0

    def test_init_relative(self):
        stmt = FromImportStatement(module='module', name='func', level=2, line=1, col_offset=0)
        assert stmt.module == 'module'
        assert stmt.name == 'func'
        assert stmt.level == 2

    def test_init_wildcard(self):
        stmt = FromImportStatement(module='collections', name='*', level=0, line=1, col_offset=0)
        assert stmt.module == 'collections'
        assert stmt.name == '*'
        assert stmt.alias is None
        assert stmt.level == 0

    def test_init_wildcard_relative(self):
        stmt = FromImportStatement(module='module', name='*', level=1, line=1, col_offset=0)
        assert stmt.module == 'module'
        assert stmt.name == '*'
        assert stmt.level == 1

    def test_str_absolute_import(self):
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert str(stmt) == "from typing import List"

    def test_str_with_alias(self):
        stmt = FromImportStatement(module='typing', name='List', alias='ListType', level=0)
        assert str(stmt) == "from typing import List as ListType"

    def test_str_relative_import(self):
        stmt = FromImportStatement(module='module', name='func', level=2)
        assert str(stmt) == "from ..module import func"

    def test_str_relative_import_single_dot(self):
        stmt = FromImportStatement(module='module', name='func', level=1)
        assert str(stmt) == "from .module import func"

    def test_str_empty_module_with_dots(self):
        stmt = FromImportStatement(module='', name='something', level=1)
        assert str(stmt) == "from . import something"

    def test_str_no_module_no_level(self):
        stmt = FromImportStatement(module='', name='something', level=0)
        assert str(stmt) == "from . import something"

    def test_str_wildcard_absolute(self):
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert str(stmt) == "from collections import *"

    def test_str_wildcard_relative(self):
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert str(stmt) == "from .module import *"

    def test_str_wildcard_nested_relative(self):
        stmt = FromImportStatement(module='subpackage.module', name='*', level=2)
        assert str(stmt) == "from ..subpackage.module import *"

    def test_str_wildcard_with_alias_ignored(self):
        # Wildcard imports cannot have aliases, so alias should be ignored
        stmt = FromImportStatement(module='collections', name='*', alias='ignored', level=0)
        assert str(stmt) == "from collections import *"

    def test_str_wildcard_empty_module(self):
        stmt = FromImportStatement(module='', name='*', level=1)
        assert str(stmt) == "from . import *"

    def test_is_relative_false(self):
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert stmt.is_relative is False

    def test_is_relative_true_with_level(self):
        stmt = FromImportStatement(module='module', name='func', level=1)
        assert stmt.is_relative is True

    def test_is_relative_true_no_module(self):
        stmt = FromImportStatement(module='', name='func', level=0)
        assert stmt.is_relative is True

    def test_is_relative_wildcard_false(self):
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert stmt.is_relative is False

    def test_is_relative_wildcard_true(self):
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert stmt.is_relative is True

    def test_is_wildcard_true(self):
        stmt = FromImportStatement(module='collections', name='*', level=0)
        assert stmt.is_wildcard is True

    def test_is_wildcard_false(self):
        stmt = FromImportStatement(module='typing', name='List', level=0)
        assert stmt.is_wildcard is False

    def test_is_wildcard_relative(self):
        stmt = FromImportStatement(module='module', name='*', level=1)
        assert stmt.is_wildcard is True

    def test_is_wildcard_empty_module(self):
        stmt = FromImportStatement(module='', name='*', level=1)
        assert stmt.is_wildcard is True


@pytest.mark.unittest
class TestImportVisitor:
    def test_init(self):
        visitor = ImportVisitor()
        assert visitor.imports == []

    def test_visit_import_single(self, simple_import_code):
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
        tree = ast.parse(import_with_alias_code)
        visitor = ImportVisitor()
        visitor.visit(tree)

        assert len(visitor.imports) == 1
        stmt = visitor.imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os'
        assert stmt.alias == 'operating_system'

    def test_visit_import_multiple(self, multiple_imports_code):
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
    def test_analyze_imports_simple(self, simple_import_code):
        imports = analyze_imports(simple_import_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os'

    def test_analyze_imports_complex(self, complex_code):
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
        imports = analyze_imports(wildcard_import_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.module == 'collections'
        assert stmt.name == '*'
        assert stmt.is_wildcard is True

    def test_analyze_imports_multiple_wildcard(self, multiple_wildcard_imports_code):
        imports = analyze_imports(multiple_wildcard_imports_code)

        assert len(imports) == 3
        for stmt in imports:
            assert isinstance(stmt, FromImportStatement)
            assert stmt.is_wildcard is True

    def test_analyze_imports_mixed_with_wildcard(self, mixed_imports_with_wildcard_code):
        imports = analyze_imports(mixed_imports_with_wildcard_code)

        assert len(imports) == 4
        wildcard_count = sum(1 for stmt in imports if isinstance(stmt, FromImportStatement) and stmt.is_wildcard)
        assert wildcard_count == 1

    def test_analyze_imports_wildcard_relative(self, wildcard_import_relative_code):
        imports = analyze_imports(wildcard_import_relative_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, FromImportStatement)
        assert stmt.is_wildcard is True
        assert stmt.is_relative is True
        assert stmt.level == 1

    def test_analyze_imports_empty_code(self):
        imports = analyze_imports("")
        assert len(imports) == 0

    def test_analyze_imports_no_imports(self):
        code = "x = 1\ny = 2"
        imports = analyze_imports(code)
        assert len(imports) == 0

    def test_analyze_imports_syntax_error(self, invalid_syntax_code):
        with pytest.raises(SyntaxError):
            analyze_imports(invalid_syntax_code)

    def test_analyze_imports_nested_module(self, nested_module_code):
        imports = analyze_imports(nested_module_code)

        assert len(imports) == 1
        stmt = imports[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == 'os.path.join'
        assert stmt.root_module == 'os'
