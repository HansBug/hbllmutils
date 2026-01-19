import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hbllmutils.template import BaseMatcher


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = [
        "image_001_test.png",
        "image_002_sample.png",
        "image_003_demo.png",
        "log_2023-01-01_error.txt",
        "log_2023-01-02_info.txt",
        "data_123_45.6_result.csv",
        "invalid_file.txt"
    ]

    for filename in files:
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write("test content")

    # Create subdirectory with files for recursive testing
    subdir = os.path.join(temp_dir, "subdir")
    os.makedirs(subdir)
    subfiles = ["image_004_sub.png", "log_2023-01-03_debug.txt"]
    for filename in subfiles:
        filepath = os.path.join(subdir, filename)
        with open(filepath, 'w') as f:
            f.write("test content")

    return temp_dir


@pytest.fixture
def image_matcher_class():
    """Create ImageMatcher class for testing."""

    class ImageMatcher(BaseMatcher):
        __pattern__ = "image_<id>_<name>.png"
        id: int
        name: str

    return ImageMatcher


@pytest.fixture
def log_matcher_class():
    """Create LogMatcher class for testing."""

    class LogMatcher(BaseMatcher):
        __pattern__ = "log_<date>_<level>.txt"
        date: str
        level: str

    return LogMatcher


@pytest.fixture
def data_matcher_class():
    """Create DataMatcher class for testing."""

    class DataMatcher(BaseMatcher):
        __pattern__ = "data_<id>_<value>_result.csv"
        id: int
        value: float

    return DataMatcher


@pytest.fixture
def recursive_matcher_class():
    """Create RecursiveMatcher class for testing."""

    class RecursiveMatcher(BaseMatcher):
        __pattern__ = "image_<id>_<name>.png"
        __recursively__ = True
        id: int
        name: str

    return RecursiveMatcher


@pytest.fixture
def bool_matcher_class():
    """Create BoolMatcher class for testing."""

    class BoolMatcher(BaseMatcher):
        __pattern__ = "config_<enabled>.cfg"
        enabled: bool

    return BoolMatcher


@pytest.fixture
def invalid_type_matcher_class():
    """Create InvalidTypeMatcher class for testing."""

    class InvalidTypeMatcher(BaseMatcher):
        __pattern__ = "test_<value>.txt"
        value: dict  # Unsupported type

    return InvalidTypeMatcher


@pytest.mark.unittest
class TestBaseMatcher:

    def test_metaclass_pattern_parsing(self, image_matcher_class):
        """Test metaclass pattern parsing and regex generation."""
        assert hasattr(image_matcher_class, '__regexp_pattern__')
        assert hasattr(image_matcher_class, '__fields__')
        assert hasattr(image_matcher_class, '__field_names__')
        assert hasattr(image_matcher_class, '__field_names_set__')

        assert image_matcher_class.__fields__ == {'id': int, 'name': str}
        assert image_matcher_class.__field_names__ == ['id', 'name']
        assert image_matcher_class.__field_names_set__ == {'id', 'name'}

    def test_init_success(self, image_matcher_class):
        """Test successful initialization."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")

        assert matcher.full_path == "/path/image_001_test.png"
        assert matcher.file_name == "image_001_test.png"
        assert matcher.dir_path == "/path"
        assert matcher.id == 1
        assert matcher.name == "test"

    def test_init_unknown_fields(self, image_matcher_class):
        """Test initialization with unknown fields."""
        with pytest.raises(ValueError, match="Unknown fields"):
            image_matcher_class("/path/file.png", id=1, name="test", unknown=123)

    def test_init_missing_fields(self, image_matcher_class):
        """Test initialization with missing fields."""
        with pytest.raises(ValueError, match="Non-included fields"):
            image_matcher_class("/path/file.png", id=1)

    def test_convert_value_int(self):
        """Test _convert_value with int type."""
        result = BaseMatcher._convert_value("123", int)
        assert result == 123
        assert isinstance(result, int)

    def test_convert_value_float(self):
        """Test _convert_value with float type."""
        result = BaseMatcher._convert_value("3.14", float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_value_str(self):
        """Test _convert_value with str type."""
        result = BaseMatcher._convert_value("hello", str)
        assert result == "hello"
        assert isinstance(result, str)

    def test_convert_value_bool_true_cases(self):
        """Test _convert_value with bool type - true cases."""
        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'YES', 'on', 'ON']
        for value in true_values:
            result = BaseMatcher._convert_value(value, bool)
            assert result is True

    def test_convert_value_bool_false_cases(self):
        """Test _convert_value with bool type - false cases."""
        false_values = ['false', 'False', 'FALSE', '0', 'no', 'NO', 'off', 'OFF', 'other']
        for value in false_values:
            result = BaseMatcher._convert_value(value, bool)
            assert result is False

    def test_convert_value_unsupported_type(self):
        """Test _convert_value with unsupported type."""
        with pytest.raises(TypeError, match="Unsupported target type"):
            BaseMatcher._convert_value("test", dict)

    def test_yield_match_nonexistent_directory(self, image_matcher_class):
        """Test _yield_match with non-existent directory."""
        results = list(image_matcher_class._yield_match("/nonexistent/path"))
        assert results == []

    def test_yield_match_success(self, image_matcher_class, sample_files):
        """Test _yield_match with matching files."""
        results = list(image_matcher_class._yield_match(sample_files))

        assert len(results) == 3
        assert all(isinstance(r, image_matcher_class) for r in results)
        assert results[0].id == 1
        assert results[0].name == "test"
        assert results[1].id == 2
        assert results[1].name == "sample"

    def test_yield_match_recursive(self, recursive_matcher_class, sample_files):
        """Test _yield_match with recursive search."""
        results = list(recursive_matcher_class._yield_match(sample_files))

        assert len(results) == 4  # 3 in root + 1 in subdir
        assert any("subdir" in r.full_path for r in results)

    def test_yield_match_type_conversion_failure(self, sample_files):
        """Test _yield_match with type conversion failure."""

        class FailMatcher(BaseMatcher):
            __pattern__ = "invalid_<id>.txt"
            id: int  # Will fail to convert "file" to int

        # Create a file that matches pattern but fails conversion
        filepath = os.path.join(sample_files, "invalid_file.txt")

        results = list(FailMatcher._yield_match(sample_files))
        assert len(results) == 0  # Should skip files with conversion failures

    def test_yield_match_no_matches(self, image_matcher_class, temp_dir):
        """Test _yield_match with no matching files."""
        # Create non-matching file
        filepath = os.path.join(temp_dir, "nomatch.txt")
        with open(filepath, 'w') as f:
            f.write("test")

        results = list(image_matcher_class._yield_match(temp_dir))
        assert len(results) == 0

    @patch('pathlib.Path.glob')
    def test_yield_match_with_directories(self, mock_glob, image_matcher_class, temp_dir):
        """Test _yield_match ignores directories."""
        # Mock glob to return both files and directories
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.name = "image_001_test.png"
        mock_file.__str__ = lambda x: os.path.join(temp_dir, "image_001_test.png")

        mock_dir = MagicMock()
        mock_dir.is_file.return_value = False

        mock_glob.return_value = [mock_file, mock_dir]

        results = list(image_matcher_class._yield_match(temp_dir))
        assert len(results) == 1  # Only file, not directory

    def test_match_success(self, image_matcher_class, sample_files):
        """Test match method success."""
        result = image_matcher_class.match(sample_files)

        assert result is not None
        assert isinstance(result, image_matcher_class)
        assert result.id == 1
        assert result.name == "test"

    def test_match_no_results(self, image_matcher_class, temp_dir):
        """Test match method with no results."""
        result = image_matcher_class.match(temp_dir)
        assert result is None

    def test_match_all_success(self, image_matcher_class, sample_files):
        """Test match_all method success."""
        results = image_matcher_class.match_all(sample_files)

        assert len(results) == 3
        assert all(isinstance(r, image_matcher_class) for r in results)

    def test_match_all_no_results(self, image_matcher_class, temp_dir):
        """Test match_all method with no results."""
        results = image_matcher_class.match_all(temp_dir)
        assert results == []

    def test_exists_true(self, image_matcher_class, sample_files):
        """Test exists method returns True."""
        assert image_matcher_class.exists(sample_files) is True

    def test_exists_false(self, image_matcher_class, temp_dir):
        """Test exists method returns False."""
        assert image_matcher_class.exists(temp_dir) is False

    def test_str_representation(self, image_matcher_class):
        """Test __str__ method."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        str_repr = str(matcher)

        assert "ImageMatcher" in str_repr
        assert "id=1" in str_repr
        assert "name='test'" in str_repr
        assert "full_path='/path/image_001_test.png'" in str_repr

    def test_repr_representation(self, image_matcher_class):
        """Test __repr__ method."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        assert repr(matcher) == str(matcher)

    def test_tuple_method(self, image_matcher_class):
        """Test tuple method."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        result = matcher.tuple()

        assert result == (1, "test")
        assert isinstance(result, tuple)

    def test_dict_method(self, image_matcher_class):
        """Test dict method."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        result = matcher.dict()

        assert result == {"id": 1, "name": "test"}
        assert isinstance(result, dict)

    def test_hash_method(self, image_matcher_class):
        """Test __hash__ method."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/other/image_001_test.png", id=1, name="test")
        matcher3 = image_matcher_class("/path/image_002_test.png", id=2, name="test")

        assert hash(matcher1) == hash(matcher2)  # Same field values
        assert hash(matcher1) != hash(matcher3)  # Different field values

    def test_cmpkey_method(self, image_matcher_class):
        """Test _cmpkey method."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        cmpkey = matcher._cmpkey()

        assert cmpkey == (1, "test")

    def test_comparison_equal_same_instance(self, image_matcher_class):
        """Test equality comparison with same instance."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        assert matcher == matcher
        assert not (matcher != matcher)

    def test_comparison_equal_different_instances(self, image_matcher_class):
        """Test equality comparison with different instances."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/other/image_001_test.png", id=1, name="test")

        assert matcher1 == matcher2
        assert not (matcher1 != matcher2)

    def test_comparison_not_equal(self, image_matcher_class):
        """Test inequality comparison."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/path/image_002_test.png", id=2, name="test")

        assert matcher1 != matcher2
        assert not (matcher1 == matcher2)

    def test_comparison_less_than(self, image_matcher_class):
        """Test less than comparison."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/path/image_002_test.png", id=2, name="test")

        assert matcher1 < matcher2
        assert not (matcher2 < matcher1)

    def test_comparison_less_equal(self, image_matcher_class):
        """Test less than or equal comparison."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/path/image_002_test.png", id=2, name="test")
        matcher3 = image_matcher_class("/other/image_001_test.png", id=1, name="test")

        assert matcher1 <= matcher2
        assert matcher1 <= matcher3
        assert not (matcher2 <= matcher1)

    def test_comparison_greater_than(self, image_matcher_class):
        """Test greater than comparison."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/path/image_002_test.png", id=2, name="test")

        assert matcher2 > matcher1
        assert not (matcher1 > matcher2)

    def test_comparison_greater_equal(self, image_matcher_class):
        """Test greater than or equal comparison."""
        matcher1 = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        matcher2 = image_matcher_class("/path/image_002_test.png", id=2, name="test")
        matcher3 = image_matcher_class("/other/image_002_test.png", id=2, name="test")

        assert matcher2 >= matcher1
        assert matcher2 >= matcher3
        assert not (matcher1 >= matcher2)

    def test_comparison_different_types(self, image_matcher_class, log_matcher_class):
        """Test comparison with different types."""
        image_matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        log_matcher = log_matcher_class("/path/log_2023-01-01_error.txt", date="2023-01-01", level="error")

        assert not (image_matcher == log_matcher)
        assert image_matcher != log_matcher
        assert not (image_matcher < log_matcher)
        assert not (image_matcher <= log_matcher)
        assert not (image_matcher > log_matcher)
        assert not (image_matcher >= log_matcher)

    def test_comparison_with_non_comparable(self, image_matcher_class):
        """Test comparison with non-comparable object."""
        matcher = image_matcher_class("/path/image_001_test.png", id=1, name="test")
        other = "not a matcher"

        assert not (matcher == other)
        assert matcher != other
        assert not (matcher < other)
        assert not (matcher <= other)
        assert not (matcher > other)
        assert not (matcher >= other)

    def test_float_type_matching(self, data_matcher_class, sample_files):
        """Test matching with float type."""
        results = data_matcher_class.match_all(sample_files)

        assert len(results) == 1
        assert results[0].id == 123
        assert results[0].value == 45.6
        assert isinstance(results[0].value, float)

    def test_string_type_matching(self, log_matcher_class, sample_files):
        """Test matching with string type."""
        results = log_matcher_class.match_all(sample_files)

        assert len(results) == 2
        assert results[0].date == "2023-01-01"
        assert results[0].level == "error"
        assert isinstance(results[0].date, str)
        assert isinstance(results[0].level, str)

    def test_bool_type_conversion_in_matching(self, bool_matcher_class, temp_dir):
        """Test bool type conversion during matching."""
        # Create test files with bool values
        bool_files = ["config_true.cfg", "config_false.cfg", "config_1.cfg", "config_0.cfg"]
        for filename in bool_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test")

        results = bool_matcher_class.match_all(temp_dir)

        assert len(results) == 4
        bool_values = [r.enabled for r in results]
        assert True in bool_values
        assert False in bool_values

    def test_path_object_input(self, image_matcher_class, sample_files):
        """Test methods accept Path objects as input."""
        path_obj = Path(sample_files)

        # Test all methods with Path object
        match_result = image_matcher_class.match(path_obj)
        assert match_result is not None

        match_all_results = image_matcher_class.match_all(path_obj)
        assert len(match_all_results) > 0

        exists_result = image_matcher_class.exists(path_obj)
        assert exists_result is True

    def test_file_path_properties(self, image_matcher_class):
        """Test file path related properties."""
        full_path = "/some/directory/image_001_test.png"
        matcher = image_matcher_class(full_path, id=1, name="test")

        assert matcher.full_path == full_path
        assert matcher.file_name == "image_001_test.png"
        assert matcher.dir_path == "/some/directory"

    def test_natural_sorting(self, image_matcher_class, temp_dir):
        """Test natural sorting of results."""
        # Create files with numbers that should be naturally sorted
        files = ["image_1_a.png", "image_2_b.png", "image_10_c.png", "image_20_d.png"]
        for filename in files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test")

        results = image_matcher_class.match_all(temp_dir)

        # Should be sorted naturally: 1, 2, 10, 20 (not 1, 10, 2, 20)
        ids = [r.id for r in results]
        assert ids == [1, 2, 10, 20]

    def test_empty_directory(self, image_matcher_class, temp_dir):
        """Test behavior with empty directory."""
        # temp_dir is empty by default in this test
        results = image_matcher_class.match_all(temp_dir)
        assert results == []

        result = image_matcher_class.match(temp_dir)
        assert result is None

        exists = image_matcher_class.exists(temp_dir)
        assert exists is False

    def test_class_attributes_inheritance(self):
        """Test class attributes are properly set by metaclass."""

        class TestMatcher(BaseMatcher):
            __pattern__ = "test_<id>.txt"
            __recursively__ = True
            id: int

        assert hasattr(TestMatcher, '__regexp_pattern__')
        assert hasattr(TestMatcher, '__fields__')
        assert hasattr(TestMatcher, '__field_names__')
        assert hasattr(TestMatcher, '__field_names_set__')
        assert TestMatcher.__recursively__ is True
        assert TestMatcher.__fields__ == {'id': int}
        assert TestMatcher.__field_names__ == ['id']
        assert TestMatcher.__field_names_set__ == {'id'}
