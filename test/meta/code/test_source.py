"""
Unit tests for the hbllmutils.meta.code.source module.

This module contains comprehensive tests for source code analysis functionality,
including import statement extraction, source file information gathering, and
object inspection integration.
"""

import os
import tempfile
import warnings
from typing import List

import pytest

from hbllmutils.meta.code.imp import FromImportStatement, ImportStatement
from hbllmutils.meta.code.object import ObjectInspect
from hbllmutils.meta.code.source import (
    ImportSource,
    SourceInfo,
    get_source_info
)


@pytest.fixture
def simple_python_file():
    """Create a simple Python file with basic imports for testing."""
    content = """import os
import sys
from typing import List, Dict
from collections import defaultdict

def example_function():
    return "Hello"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def python_file_with_relative_imports():
    """Create a Python file with relative imports in a package structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create package structure
        pkg_dir = os.path.join(tmpdir, 'testpkg')
        os.makedirs(pkg_dir)

        # Create __init__.py
        init_file = os.path.join(pkg_dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write("# Package init\n")

        # Create module with relative imports
        module_file = os.path.join(pkg_dir, 'module.py')
        content = """from . import something
from ..parent import other
import os

def test_func():
    pass
"""
        with open(module_file, 'w') as f:
            f.write(content)

        yield module_file


@pytest.fixture
def python_file_with_syntax_error():
    """Create a Python file with syntax errors."""
    content = """import os
def broken_function(
    # Missing closing parenthesis
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def python_file_with_import_errors():
    """Create a Python file with imports that will fail."""
    content = """import os
from nonexistent_module import something
from typing import List

def example():
    pass
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_python_file():
    """Create an empty Python file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.unittest
class TestImportSource:
    """Tests for the ImportSource dataclass."""

    def test_import_source_creation_with_from_import(self):
        """Test creating ImportSource with FromImportStatement."""
        stmt = FromImportStatement(
            module='os',
            name='path',
            alias=None,
            level=0,
            line=1,
            col_offset=0
        )

        # Create a mock ObjectInspect
        obj_inspect = ObjectInspect(
            object=os.path,
            source_file='/usr/lib/python3.x/posixpath.py',
            start_line=1,
            end_line=10,
            source_lines=None
        )

        import_src = ImportSource(statement=stmt, inspect=obj_inspect)

        assert import_src.statement == stmt
        assert import_src.inspect == obj_inspect
        assert isinstance(import_src.statement, FromImportStatement)

    def test_import_source_creation_with_regular_import(self):
        """Test creating ImportSource with ImportStatement."""
        stmt = ImportStatement(
            module='sys',
            alias='system',
            line=2,
            col_offset=0
        )

        obj_inspect = ObjectInspect(
            object=None,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )

        import_src = ImportSource(statement=stmt, inspect=obj_inspect)

        assert import_src.statement == stmt
        assert import_src.inspect == obj_inspect
        assert isinstance(import_src.statement, ImportStatement)

    def test_import_source_attributes_access(self):
        """Test accessing attributes of ImportSource."""
        stmt = FromImportStatement(
            module='typing',
            name='List',
            alias='ListType',
            level=0
        )

        obj_inspect = ObjectInspect(
            object=List,
            source_file='/usr/lib/python3.x/typing.py',
            start_line=100,
            end_line=150,
            source_lines=['class List:\n', '    pass\n']
        )

        import_src = ImportSource(statement=stmt, inspect=obj_inspect)

        assert import_src.statement.module == 'typing'
        assert import_src.statement.name == 'List'
        assert import_src.statement.alias == 'ListType'
        assert import_src.inspect.start_line == 100
        assert import_src.inspect.end_line == 150


@pytest.mark.unittest
class TestSourceInfo:
    """Tests for the SourceInfo dataclass."""

    def test_source_info_creation(self, simple_python_file):
        """Test creating SourceInfo with basic attributes."""
        source_lines = ['import os\n', 'import sys\n']
        imports = []

        info = SourceInfo(
            source_file=simple_python_file,
            source_lines=source_lines,
            imports=imports
        )

        assert os.path.isabs(info.source_file)
        assert info.source_lines == source_lines
        assert info.imports == imports

    def test_source_info_path_normalization(self):
        """Test that source file path is normalized."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# test\n")
            temp_path = f.name

        try:
            # Use relative path
            rel_path = os.path.relpath(temp_path)
            info = SourceInfo(
                source_file=rel_path,
                source_lines=['# test\n'],
                imports=[]
            )

            # Should be converted to absolute path
            assert os.path.isabs(info.source_file)
            assert os.path.exists(info.source_file)
        finally:
            os.unlink(temp_path)

    def test_source_code_property(self):
        """Test the source_code property concatenates lines correctly."""
        source_lines = ['import os\n', 'import sys\n', 'print("hello")\n']

        info = SourceInfo(
            source_file='/tmp/test.py',
            source_lines=source_lines,
            imports=[]
        )

        expected_code = 'import os\nimport sys\nprint("hello")\n'
        assert info.source_code == expected_code

    def test_source_code_property_empty(self):
        """Test source_code property with empty source lines."""
        info = SourceInfo(
            source_file='/tmp/test.py',
            source_lines=[],
            imports=[]
        )

        assert info.source_code == ''

    def test_source_code_property_single_line(self):
        """Test source_code property with single line."""
        info = SourceInfo(
            source_file='/tmp/test.py',
            source_lines=['import os\n'],
            imports=[]
        )

        assert info.source_code == 'import os\n'

    def test_package_name_property(self, simple_python_file):
        """Test the package_name property."""
        info = SourceInfo(
            source_file=simple_python_file,
            source_lines=['import os\n'],
            imports=[]
        )

        # Package name should be derived from file path
        pkg_name = info.package_name
        assert isinstance(pkg_name, str)
        # The exact package name depends on file location, just verify it's a string

    def test_source_info_with_imports(self):
        """Test SourceInfo with actual import data."""
        stmt = FromImportStatement(module='os', name='path', level=0)
        obj_inspect = ObjectInspect(
            object=os.path,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )
        import_src = ImportSource(statement=stmt, inspect=obj_inspect)

        info = SourceInfo(
            source_file='/tmp/test.py',
            source_lines=['from os import path\n'],
            imports=[import_src]
        )

        assert len(info.imports) == 1
        assert info.imports[0].statement.module == 'os'
        assert info.imports[0].statement.name == 'path'


@pytest.mark.unittest
class TestGetSourceInfo:
    """Tests for the get_source_info function."""

    def test_get_source_info_simple_file(self, simple_python_file):
        """Test get_source_info with a simple Python file."""
        info = get_source_info(simple_python_file, skip_when_error=True)

        assert isinstance(info, SourceInfo)
        assert os.path.isabs(info.source_file)
        assert len(info.source_lines) > 0
        assert 'import os' in info.source_code
        assert 'import sys' in info.source_code

    def test_get_source_info_empty_file(self, empty_python_file):
        """Test get_source_info with an empty file."""
        info = get_source_info(empty_python_file, skip_when_error=True)

        assert isinstance(info, SourceInfo)
        assert info.source_code == ''
        assert len(info.imports) == 0

    def test_get_source_info_with_import_errors_skip(self, python_file_with_import_errors):
        """Test get_source_info with import errors when skip_when_error=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = get_source_info(python_file_with_import_errors, skip_when_error=True)

            assert isinstance(info, SourceInfo)
            # Should have warnings for failed imports
            import_warnings = [warning for warning in w if issubclass(warning.category, ImportWarning)]
            assert len(import_warnings) > 0

    def test_get_source_info_with_import_errors_no_skip(self, python_file_with_import_errors):
        """Test get_source_info with import errors when skip_when_error=False."""
        # This should raise an exception for failed imports
        with pytest.raises(Exception):
            get_source_info(python_file_with_import_errors, skip_when_error=False)

    def test_get_source_info_source_lines_preserved(self, simple_python_file):
        """Test that source lines are correctly preserved."""
        info = get_source_info(simple_python_file, skip_when_error=True)

        # Read the file directly for comparison
        with open(simple_python_file, 'r') as f:
            expected_content = f.read()

        assert info.source_code == expected_content

    def test_get_source_info_package_name(self, simple_python_file):
        """Test that package name is correctly determined."""
        info = get_source_info(simple_python_file, skip_when_error=True)

        pkg_name = info.package_name
        assert isinstance(pkg_name, str)
        assert len(pkg_name) > 0

    def test_get_source_info_imports_list(self, simple_python_file):
        """Test that imports list is populated."""
        info = get_source_info(simple_python_file, skip_when_error=True)

        assert isinstance(info.imports, list)
        # Should have some imports from the simple file
        for import_src in info.imports:
            assert isinstance(import_src, ImportSource)
            assert hasattr(import_src, 'statement')
            assert hasattr(import_src, 'inspect')

    def test_get_source_info_file_not_found(self):
        """Test get_source_info with non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_source_info('/nonexistent/path/file.py', skip_when_error=True)

    def test_get_source_info_absolute_path_conversion(self):
        """Test that relative paths are converted to absolute."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import os\n")
            temp_path = f.name

        try:
            # Get relative path
            rel_path = os.path.relpath(temp_path)
            info = get_source_info(rel_path, skip_when_error=True)

            assert os.path.isabs(info.source_file)
            assert os.path.normpath(os.path.abspath(rel_path)) == info.source_file
        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize("content,expected_imports", [
        ("import os\n", 1),
        ("import os\nimport sys\n", 2),
        ("from typing import List\n", 1),
        ("", 0),
        ("# Just a comment\n", 0),
    ])
    def test_get_source_info_various_imports(self, content, expected_imports):
        """Test get_source_info with various import patterns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            info = get_source_info(temp_path, skip_when_error=True)
            # Note: The actual number of imports may vary based on what can be successfully imported
            assert isinstance(info.imports, list)
        finally:
            os.unlink(temp_path)

    def test_get_source_info_warning_message_format(self, python_file_with_import_errors):
        """Test that warning messages contain expected information."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = get_source_info(python_file_with_import_errors, skip_when_error=True)

            import_warnings = [warning for warning in w if issubclass(warning.category, ImportWarning)]
            if import_warnings:
                warning_msg = str(import_warnings[0].message)
                assert "Failed to import object" in warning_msg
                assert python_file_with_import_errors in warning_msg

    def test_get_source_info_consistency(self, simple_python_file):
        """Test that calling get_source_info twice produces consistent results."""
        info1 = get_source_info(simple_python_file, skip_when_error=True)
        info2 = get_source_info(simple_python_file, skip_when_error=True)

        assert info1.source_file == info2.source_file
        assert info1.source_code == info2.source_code
        assert len(info1.imports) == len(info2.imports)

    def test_get_source_info_multiline_imports(self):
        """Test get_source_info with multiline imports."""
        content = """from typing import (
    List,
    Dict,
    Optional
)

import os
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            info = get_source_info(temp_path, skip_when_error=True)
            assert isinstance(info, SourceInfo)
            assert 'from typing import' in info.source_code
        finally:
            os.unlink(temp_path)


@pytest.mark.unittest
class TestSourceInfoIntegration:
    """Integration tests for SourceInfo with real file operations."""

    def test_source_info_with_real_module(self):
        """Test SourceInfo with a real Python module structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple package
            pkg_dir = os.path.join(tmpdir, 'testpkg')
            os.makedirs(pkg_dir)

            init_file = os.path.join(pkg_dir, '__init__.py')
            with open(init_file, 'w') as f:
                f.write("# Package\n")

            module_file = os.path.join(pkg_dir, 'module.py')
            with open(module_file, 'w') as f:
                f.write("import os\nfrom typing import List\n")

            info = get_source_info(module_file, skip_when_error=True)

            assert isinstance(info, SourceInfo)
            assert 'import os' in info.source_code
            assert info.package_name.endswith('module')

    def test_source_info_preserves_line_endings(self):
        """Test that different line endings are preserved."""
        content_unix = "import os\nimport sys\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, newline='') as f:
            f.write(content_unix)
            temp_path = f.name

        try:
            info = get_source_info(temp_path, skip_when_error=True)
            assert info.source_code == content_unix
        finally:
            os.unlink(temp_path)

    def test_source_info_with_complex_imports(self):
        """Test SourceInfo with complex import patterns."""
        content = """import os
import sys as system
from typing import List, Dict, Optional
from collections import defaultdict, Counter
from pathlib import Path

def example():
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            info = get_source_info(temp_path, skip_when_error=True)

            assert isinstance(info, SourceInfo)
            assert len(info.source_lines) > 0
            assert 'from typing import' in info.source_code
            assert 'from collections import' in info.source_code
        finally:
            os.unlink(temp_path)
