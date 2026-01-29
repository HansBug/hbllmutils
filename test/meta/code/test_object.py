import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from hbllmutils.meta.code.object import ObjectInspect, get_object_info


@pytest.fixture
def temp_source_file():
    """Create a temporary source file for testing."""
    content = '''def test_function():
    """A test function."""
    return 42

class TestClass:
    def method(self):
        pass
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def sample_function():
    """Create a sample function for testing."""

    def example_func():
        """Example function docstring."""
        return "test"

    return example_func


@pytest.fixture
def sample_class():
    """Create a sample class for testing."""

    class ExampleClass:
        def method(self):
            return "method"

    return ExampleClass


@pytest.fixture
def object_with_name():
    """Create an object with __name__ attribute."""
    obj = MagicMock()
    obj.__name__ = "test_object"
    return obj


@pytest.fixture
def object_without_name():
    """Create an object without __name__ attribute."""
    obj = MagicMock()
    del obj.__name__
    return obj


@pytest.fixture
def mock_object_inspect():
    """Create a mock ObjectInspect instance."""

    return ObjectInspect(
        object=lambda: None,
        source_file="/path/to/file.py",
        start_line=1,
        end_line=3,
        source_lines=["def test():\n", "    pass\n"]
    )


@pytest.mark.unittest
class TestObjectInspect:

    def test_init_with_all_parameters(self, sample_function):
        """Test ObjectInspect initialization with all parameters."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file="/path/to/file.py",
            start_line=1,
            end_line=5,
            source_lines=["line1\n", "line2\n"]
        )

        assert obj_inspect.object == sample_function
        assert obj_inspect.source_file == os.path.normpath(os.path.normcase(os.path.abspath("/path/to/file.py")))
        assert obj_inspect.start_line == 1
        assert obj_inspect.end_line == 5
        assert obj_inspect.source_lines == ["line1\n", "line2\n"]

    def test_init_with_none_values(self, sample_function):
        """Test ObjectInspect initialization with None values."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )

        assert obj_inspect.object == sample_function
        assert obj_inspect.source_file is None
        assert obj_inspect.start_line is None
        assert obj_inspect.end_line is None
        assert obj_inspect.source_lines is None

    def test_post_init_normalizes_path(self, sample_function):
        """Test that __post_init__ normalizes the source file path."""

        relative_path = "relative/path/file.py"
        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=relative_path,
            start_line=1,
            end_line=5,
            source_lines=["line1\n"]
        )

        expected_path = os.path.normpath(os.path.normcase(os.path.abspath(relative_path)))
        assert obj_inspect.source_file == expected_path

    def test_post_init_with_none_source_file(self, sample_function):
        """Test that __post_init__ handles None source file."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=None,
            start_line=1,
            end_line=5,
            source_lines=["line1\n"]
        )

        assert obj_inspect.source_file is None

    def test_name_property_with_name_attribute(self, object_with_name):
        """Test name property when object has __name__ attribute."""

        obj_inspect = ObjectInspect(
            object=object_with_name,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )

        assert obj_inspect.name == "test_object"

    def test_name_property_without_name_attribute(self, object_without_name):
        """Test name property when object doesn't have __name__ attribute."""

        obj_inspect = ObjectInspect(
            object=object_without_name,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )

        assert obj_inspect.name is None

    def test_source_code_property_with_source_lines(self, sample_function):
        """Test source_code property when source_lines is available."""

        source_lines = ["def test():\n", "    return 42\n"]
        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file="/path/to/file.py",
            start_line=1,
            end_line=2,
            source_lines=source_lines
        )

        expected_source = "def test():\n    return 42\n"
        assert obj_inspect.source_code == expected_source

    def test_source_code_property_without_source_lines(self, sample_function):
        """Test source_code property when source_lines is None."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=None,
            start_line=None,
            end_line=None,
            source_lines=None
        )

        assert obj_inspect.source_code is None

    def test_source_file_code_property_with_file(self, temp_source_file, sample_function):
        """Test source_file_code property when source_file is available."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=temp_source_file,
            start_line=1,
            end_line=2,
            source_lines=["def test():\n"]
        )

        file_content = obj_inspect.source_file_code
        assert "def test_function():" in file_content
        assert "class TestClass:" in file_content

    def test_source_file_code_property_without_file(self, sample_function):
        """Test source_file_code property when source_file is None."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file=None,
            start_line=1,
            end_line=2,
            source_lines=["def test():\n"]
        )

        assert obj_inspect.source_file_code is None

    def test_has_source_property_with_source_lines(self, sample_function):
        """Test has_source property when source_lines is available."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file="/path/to/file.py",
            start_line=1,
            end_line=2,
            source_lines=["def test():\n"]
        )

        assert obj_inspect.has_source is True

    def test_has_source_property_without_source_lines(self, sample_function):
        """Test has_source property when source_lines is None."""

        obj_inspect = ObjectInspect(
            object=sample_function,
            source_file="/path/to/file.py",
            start_line=1,
            end_line=2,
            source_lines=None
        )

        assert obj_inspect.has_source is False

    def test_get_object_info_with_function(self, sample_function):
        """Test get_object_info with a regular function."""

        info = get_object_info(sample_function)

        assert info.object == sample_function
        assert info.source_file is not None
        assert info.start_line is not None
        assert info.end_line is not None
        assert info.source_lines is not None
        assert info.has_source is True

    def test_get_object_info_with_class(self, sample_class):
        """Test get_object_info with a class."""

        info = get_object_info(sample_class)

        assert info.object == sample_class
        assert info.source_file is not None
        assert info.start_line is not None
        assert info.end_line is not None
        assert info.source_lines is not None
        assert info.has_source is True

    def test_get_object_info_with_builtin_function(self):
        """Test get_object_info with a built-in function."""

        info = get_object_info(print)

        assert info.object == print
        assert info.source_file is None
        assert info.start_line is None
        assert info.end_line is None
        assert info.source_lines is None
        assert info.has_source is False

    @patch('inspect.getfile')
    def test_get_object_info_getfile_raises_typeerror(self, mock_getfile, sample_function):
        """Test get_object_info when inspect.getfile raises TypeError."""

        mock_getfile.side_effect = TypeError("No source file")

        info = get_object_info(sample_function)

        assert info.object == sample_function
        assert info.source_file is None

    @patch('inspect.getsourcelines')
    def test_get_object_info_getsourcelines_raises_typeerror(self, mock_getsourcelines, sample_function):
        """Test get_object_info when inspect.getsourcelines raises TypeError."""

        mock_getsourcelines.side_effect = TypeError("No source lines")

        info = get_object_info(sample_function)

        assert info.object == sample_function
        assert info.start_line is None
        assert info.end_line is None
        assert info.source_lines is None

    @patch('inspect.getfile')
    @patch('inspect.getsourcelines')
    def test_get_object_info_both_raise_typeerror(self, mock_getsourcelines, mock_getfile, sample_function):
        """Test get_object_info when both inspect functions raise TypeError."""

        mock_getfile.side_effect = TypeError("No source file")
        mock_getsourcelines.side_effect = TypeError("No source lines")

        info = get_object_info(sample_function)

        assert info.object == sample_function
        assert info.source_file is None
        assert info.start_line is None
        assert info.end_line is None
        assert info.source_lines is None
        assert info.has_source is False

    @patch('inspect.getsourcelines')
    def test_get_object_info_calculates_end_line(self, mock_getsourcelines, sample_function):
        """Test that get_object_info correctly calculates end_line."""

        source_lines = ["def test():\n", "    pass\n", "    return 42\n"]
        start_line = 10
        mock_getsourcelines.return_value = (source_lines, start_line)

        info = get_object_info(sample_function)

        expected_end_line = start_line + len(source_lines) - 1
        assert info.end_line == expected_end_line
        assert info.source_lines == source_lines
        assert info.start_line == start_line
