"""
Unit tests for the hbllmutils.meta.code.prompt module.

This module contains comprehensive tests for code prompt generation functionality,
including validation of Python code, file checking, and prompt generation for LLM analysis.
"""

import ast
import io
import os
import pathlib
import tempfile
import warnings
from typing import List, Tuple
from unittest.mock import patch, MagicMock

import pytest

from hbllmutils.meta.code.prompt import (
    is_python_code,
    is_python_file,
    get_prompt_for_source_file,
)


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file with valid code."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("# Test module\n")
        f.write("import os\n")
        f.write("def hello():\n")
        f.write("    return 'world'\n")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_invalid_python_file():
    """Create a temporary file with invalid Python code."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("invalid python code {{{ }}}\n")
        f.write("def broken(\n")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_text_file():
    """Create a temporary non-Python text file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is plain text\n")
        f.write("Not Python code\n")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_binary_file():
    """Create a temporary binary file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        f.write(b'\x00\x01\x02\x03\x04\x05')
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_empty_file():
    """Create a temporary empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        pass
    yield f.name
    os.unlink(f.name)


@pytest.mark.unittest
class TestIsPythonCode:
    """Tests for the is_python_code function."""

    def test_valid_simple_statement(self):
        """Test that simple valid Python statements are recognized."""
        assert is_python_code("print('hello')")
        assert is_python_code("x = 1 + 2")
        assert is_python_code("import os")

    def test_valid_function_definition(self):
        """Test that function definitions are recognized as valid Python."""
        assert is_python_code("def foo(): return 42")
        assert is_python_code("def bar(x, y):\n    return x + y")

    def test_valid_class_definition(self):
        """Test that class definitions are recognized as valid Python."""
        assert is_python_code("class MyClass: pass")
        assert is_python_code("class Foo:\n    def __init__(self):\n        pass")

    def test_empty_string(self):
        """Test that empty string is considered valid Python."""
        assert is_python_code("")

    def test_multiline_code(self):
        """Test that multiline Python code is recognized."""
        code = """
def calculate(x, y):
    result = x + y
    return result

print(calculate(1, 2))
"""
        assert is_python_code(code)

    @pytest.mark.parametrize("invalid_code", [
        "invalid python code {{{",
        "def broken(",
        "if True",
        "class Foo",
        "import",
        # "1 + + 2",  # this is a valid python code
        "def foo() return 42",
    ])
    def test_invalid_syntax(self, invalid_code):
        """Test that invalid Python syntax is detected."""
        assert not is_python_code(invalid_code)

    def test_comments_only(self):
        """Test that comments-only code is valid Python."""
        assert is_python_code("# This is a comment")
        assert is_python_code("# Comment 1\n# Comment 2")

    def test_docstring_only(self):
        """Test that docstring-only code is valid Python."""
        assert is_python_code('"""This is a docstring"""')

    def test_whitespace_only(self):
        """Test that whitespace-only code is valid Python."""
        assert is_python_code("   \n   \n   ")


@pytest.mark.unittest
class TestIsPythonFile:
    """Tests for the is_python_file function."""

    def test_valid_python_file(self, temp_python_file):
        """Test that a file with valid Python code is recognized."""
        assert is_python_file(temp_python_file)

    def test_invalid_python_file(self, temp_invalid_python_file):
        """Test that a file with invalid Python code is detected."""
        assert not is_python_file(temp_invalid_python_file)

    def test_text_file(self, temp_text_file):
        """Test that a non-Python text file is not recognized as Python."""
        assert not is_python_file(temp_text_file)

    def test_binary_file(self, temp_binary_file):
        """Test that binary files are not recognized as Python."""
        assert not is_python_file(temp_binary_file)

    def test_empty_file(self, temp_empty_file):
        """Test that empty files are considered valid Python."""
        assert is_python_file(temp_empty_file)

    def test_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            is_python_file('/nonexistent/path/file.py')


@pytest.mark.unittest
class TestGetPromptForSourceFile:
    """Tests for the get_prompt_for_source_file function."""

    def test_basic_python_file_prompt(self, temp_python_file):
        """Test basic prompt generation for a Python file."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(temp_python_file, show_module_directory_tree=False)

                assert '## Primary Source Code Analysis' in prompt
                assert f'**Source File Location:** `{temp_python_file}`' in prompt
                assert '**Package Namespace:** `test_module`' in prompt
                assert '**Complete Source Code:**' in prompt
                assert 'def hello():' in prompt

    def test_custom_level(self, temp_python_file):
        """Test prompt generation with custom heading level."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(temp_python_file, level=3, show_module_directory_tree=False)

                assert '### Primary Source Code Analysis' in prompt

    def test_custom_code_name(self, temp_python_file):
        """Test prompt generation with custom code name."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    code_name='custom',
                    show_module_directory_tree=False
                )

                assert '## Custom Source Code Analysis' in prompt

    def test_no_code_name(self, temp_python_file):
        """Test prompt generation without code name prefix."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    code_name=None,
                    show_module_directory_tree=False
                )

                assert '## Source Code Analysis' in prompt
                assert 'Primary' not in prompt.split('\n')[0]

    def test_description_text(self, temp_python_file):
        """Test prompt generation with custom description text."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                description = "This is a test module for demonstration."
                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    description_text=description,
                    show_module_directory_tree=False
                )

                assert description in prompt

    def test_non_python_file(self, temp_text_file):
        """Test prompt generation for non-Python files."""
        prompt = get_prompt_for_source_file(temp_text_file)

        assert '## Primary Source Code Analysis' in prompt
        assert f'**Source File Location:** `{temp_text_file}`' in prompt
        assert '**Complete Source Code:**' in prompt
        assert 'This is plain text' in prompt
        assert 'Package Namespace' not in prompt

    def test_non_python_file_with_python_params_warning(self, temp_text_file):
        """Test that warnings are issued for Python-specific params on non-Python files."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            prompt = get_prompt_for_source_file(
                temp_text_file,
                show_module_directory_tree=True,
                no_imports=True,
                warning_when_not_python=True
            )

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert 'not a Python file' in str(w[0].message)
            assert 'show_module_directory_tree=True' in str(w[0].message)

    def test_non_python_file_no_warning(self, temp_text_file):
        """Test that no warnings are issued when warning_when_not_python is False."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            prompt = get_prompt_for_source_file(
                temp_text_file,
                show_module_directory_tree=True,
                warning_when_not_python=False
            )

            assert len(w) == 0

    def test_no_imports_flag(self, temp_python_file):
        """Test that no_imports flag excludes dependency analysis."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = [MagicMock()]  # Has imports but should be ignored
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    no_imports=True,
                    show_module_directory_tree=False
                )

                assert 'Dependency Analysis' not in prompt

    def test_return_imported_items(self, temp_python_file):
        """Test that return_imported_items returns tuple with imported items list."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                result = get_prompt_for_source_file(
                    temp_python_file,
                    return_imported_items=True,
                    show_module_directory_tree=False
                )

                assert isinstance(result, tuple)
                assert len(result) == 2
                prompt, imported_items = result
                assert isinstance(prompt, str)
                assert isinstance(imported_items, list)
                assert 'test_module' in imported_items

    def test_with_imports(self, temp_python_file):
        """Test prompt generation with imports."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_import = MagicMock()
            mock_import.statement = MagicMock()
            mock_import.statement.__str__ = lambda self: 'from os import path'
            mock_import.statement.name = 'path'
            mock_import.statement.check_ignore_or_not = MagicMock(return_value=False)
            mock_import.inspect = MagicMock()
            mock_import.inspect.source_file = '/usr/lib/python3.10/os.py'
            mock_import.inspect.has_source = True
            mock_import.inspect.source_code = 'def join(*args): pass'

            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = [mock_import]
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                with patch('hbllmutils.meta.code.prompt.get_package_name') as mock_pkg_name:
                    mock_pkg_name.return_value = 'os'

                    prompt = get_prompt_for_source_file(
                        temp_python_file,
                        show_module_directory_tree=False
                    )

                    assert 'Dependency Analysis' in prompt
                    assert 'from os import path' in prompt

    def test_ignore_modules(self, temp_python_file):
        """Test that ignore_modules filters out specified modules."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_import = MagicMock()
            mock_import.statement = MagicMock()
            mock_import.statement.check_ignore_or_not = MagicMock(return_value=True)

            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = [mock_import]
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    ignore_modules=['os'],
                    show_module_directory_tree=False
                )

                # Should not have dependency analysis section since import is ignored
                assert 'Dependency Analysis' not in prompt

    def test_skip_when_error(self, temp_python_file):
        """Test that skip_when_error is passed to get_source_info."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                get_prompt_for_source_file(
                    temp_python_file,
                    skip_when_error=True,
                    show_module_directory_tree=False
                )

                mock_source_info.assert_called_once_with(temp_python_file, skip_when_error=True)

    @pytest.mark.parametrize("level,expected_header", [
        (1, '# Primary Source Code Analysis'),
        (2, '## Primary Source Code Analysis'),
        (3, '### Primary Source Code Analysis'),
        (4, '#### Primary Source Code Analysis'),
    ])
    def test_various_levels(self, temp_python_file, level, expected_header):
        """Test prompt generation with various heading levels."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    level=level,
                    show_module_directory_tree=False
                )

                assert expected_header in prompt

    def test_import_without_source(self, temp_python_file):
        """Test handling of imports without available source code."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_import = MagicMock()
            mock_import.statement = MagicMock()
            mock_import.statement.__str__ = lambda self: 'from builtins import int'
            mock_import.statement.name = 'int'
            mock_import.statement.check_ignore_or_not = MagicMock(return_value=False)
            mock_import.inspect = MagicMock()
            mock_import.inspect.source_file = None
            mock_import.inspect.has_source = False
            mock_import.inspect.object = '<class int>'

            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = [mock_import]
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                prompt = get_prompt_for_source_file(
                    temp_python_file,
                    show_module_directory_tree=False
                )

                assert 'Source code is not available' in prompt
                assert '<class int>' in prompt

    def test_module_directory_tree(self, temp_python_file):
        """Test that module directory tree is included when requested."""
        with patch('hbllmutils.meta.code.prompt.get_source_info') as mock_source_info:
            mock_info = MagicMock()
            mock_info.source_file = temp_python_file
            mock_info.package_name = 'test_module'
            mock_info.source_code = pathlib.Path(temp_python_file).read_text()
            mock_info.imports = []
            mock_source_info.return_value = mock_info

            with patch('hbllmutils.meta.code.prompt.get_pythonpath_of_source_file') as mock_pythonpath:
                mock_pythonpath.return_value = (os.path.dirname(temp_python_file), 'test_module')

                with patch('hbllmutils.meta.code.prompt.get_python_project_tree_text') as mock_tree:
                    mock_tree.return_value = 'test_module/\n  test.py'

                    prompt = get_prompt_for_source_file(
                        temp_python_file,
                        show_module_directory_tree=True
                    )

                    assert 'Module directory tree:' in prompt
                    mock_tree.assert_called_once()
