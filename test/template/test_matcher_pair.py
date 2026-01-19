import os
import tempfile
from pathlib import Path

import pytest

from hbllmutils.template import BaseMatcher, BaseMatcherPair


# Test matcher classes
class TestMatcher1(BaseMatcher):
    __pattern__ = "test_<id>_<name>.txt"
    id: int
    name: str


class TestMatcher2(BaseMatcher):
    __pattern__ = "data_<id>_<name>.dat"
    id: int
    name: str


class TestMatcher3(BaseMatcher):
    __pattern__ = "other_<value>.txt"
    value: str


class TestMatcherDifferentFields(BaseMatcher):
    __pattern__ = "diff_<id>_<category>.txt"
    id: int
    category: str


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = [
            "test_001_alpha.txt",
            "test_002_beta.txt",
            "data_001_alpha.dat",
            "data_002_beta.dat",
            "other_gamma.txt",
            "diff_001_cat1.txt"
        ]

        for filename in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")

        yield temp_dir


@pytest.fixture
def test_matcher_instances():
    """Create test matcher instances."""
    matcher1 = TestMatcher1("/path/test_001_alpha.txt", id=1, name="alpha")
    matcher2 = TestMatcher2("/path/data_001_alpha.dat", id=1, name="alpha")
    return {"matcher1": matcher1, "matcher2": matcher2}


@pytest.fixture
def test_values():
    """Create test values dictionary."""
    return {"id": 1, "name": "alpha"}


@pytest.mark.unittest
class TestBaseMatcherPair:

    def test_metaclass_field_initialization(self):
        """Test that metaclass properly initializes fields."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        assert hasattr(TestPair, '__fields__')
        assert hasattr(TestPair, '__field_names__')
        assert hasattr(TestPair, '__value_fields__')
        assert hasattr(TestPair, '__value_field_names__')
        assert hasattr(TestPair, '__field_names_set__')
        assert hasattr(TestPair, '__value_field_names_set__')

        assert TestPair.__fields__ == {'matcher1': TestMatcher1, 'matcher2': TestMatcher2}
        assert TestPair.__field_names__ == ['matcher1', 'matcher2']
        assert TestPair.__value_fields__ == {'id': int, 'name': str}
        assert TestPair.__value_field_names__ == ['id', 'name']
        assert TestPair.__field_names_set__ == {'matcher1', 'matcher2'}
        assert TestPair.__value_field_names_set__ == {'id', 'name'}

    def test_metaclass_non_matcher_field_error(self):
        """Test that metaclass raises error for non-matcher fields."""
        with pytest.raises(NameError, match="Field 'invalid' is not a matcher"):
            class InvalidPair(BaseMatcherPair):
                invalid: str

    def test_metaclass_inconsistent_value_fields_error(self):
        """Test that metaclass raises error for inconsistent value fields."""
        with pytest.raises(TypeError, match="Field not match"):
            class InconsistentPair(BaseMatcherPair):
                matcher1: TestMatcher1
                matcher3: TestMatcher3

    def test_init_success(self, test_values, test_matcher_instances):
        """Test successful initialization of matcher pair."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)

        assert pair.id == 1
        assert pair.name == "alpha"
        assert pair.matcher1 == test_matcher_instances["matcher1"]
        assert pair.matcher2 == test_matcher_instances["matcher2"]

    def test_init_unknown_instance_fields(self, test_values, test_matcher_instances):
        """Test initialization with unknown instance fields."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        invalid_instances = {**test_matcher_instances, "unknown": "value"}

        with pytest.raises(ValueError, match="Unknown fields for class TestPair"):
            TestPair(values=test_values, instances=invalid_instances)

    def test_init_missing_instance_fields(self, test_values):
        """Test initialization with missing instance fields."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        incomplete_instances = {"matcher1": TestMatcher1("/path/test.txt", id=1, name="test")}

        with pytest.raises(ValueError, match="Non-included fields of class TestPair"):
            TestPair(values=test_values, instances=incomplete_instances)

    def test_init_unknown_value_fields(self, test_matcher_instances):
        """Test initialization with unknown value fields."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        invalid_values = {"id": 1, "name": "test", "unknown": "value"}

        with pytest.raises(ValueError, match="Unknown value fields for class TestPair"):
            TestPair(values=invalid_values, instances=test_matcher_instances)

    def test_init_missing_value_fields(self, test_matcher_instances):
        """Test initialization with missing value fields."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        incomplete_values = {"id": 1}

        with pytest.raises(ValueError, match="Non-included value fields of class TestPair"):
            TestPair(values=incomplete_values, instances=test_matcher_instances)

    def test_match_all_success(self, temp_directory):
        """Test successful matching of all files in directory."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pairs = TestPair.match_all(temp_directory)

        assert len(pairs) == 2
        assert pairs[0].id == 1
        assert pairs[0].name == "alpha"
        assert pairs[1].id == 2
        assert pairs[1].name == "beta"

        for pair in pairs:
            assert isinstance(pair.matcher1, TestMatcher1)
            assert isinstance(pair.matcher2, TestMatcher2)

    def test_match_all_no_common_matches(self, temp_directory):
        """Test match_all when there are no common matches between matchers."""
        with pytest.raises(TypeError):
            class TestPair(BaseMatcherPair):
                matcher1: TestMatcher1
                matcher3: TestMatcher3

    def test_match_all_empty_directory(self):
        """Test match_all with empty directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            class TestPair(BaseMatcherPair):
                matcher1: TestMatcher1
                matcher2: TestMatcher2

            pairs = TestPair.match_all(empty_dir)
            assert len(pairs) == 0

    def test_match_all_nonexistent_directory(self):
        """Test match_all with nonexistent directory."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pairs = TestPair.match_all("/nonexistent/directory")
        assert len(pairs) == 0

    def test_str_representation(self, test_values, test_matcher_instances):
        """Test string representation of matcher pair."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        str_repr = str(pair)

        assert "TestPair(" in str_repr
        assert "id=1" in str_repr
        assert "name='alpha'" in str_repr
        assert "matcher1=" in str_repr
        assert "matcher2=" in str_repr

    def test_repr_representation(self, test_values, test_matcher_instances):
        """Test repr representation of matcher pair."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        repr_str = repr(pair)

        assert repr_str == str(pair)

    def test_values_tuple(self, test_values, test_matcher_instances):
        """Test values_tuple method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        values_tuple = pair.values_tuple()

        assert values_tuple == (1, "alpha")
        assert isinstance(values_tuple, tuple)

    def test_values_dict(self, test_values, test_matcher_instances):
        """Test values_dict method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        values_dict = pair.values_dict()

        assert values_dict == {"id": 1, "name": "alpha"}
        assert isinstance(values_dict, dict)

    def test_tuple(self, test_values, test_matcher_instances):
        """Test tuple method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        full_tuple = pair.tuple()

        expected = (1, "alpha", test_matcher_instances["matcher1"], test_matcher_instances["matcher2"])
        assert full_tuple == expected
        assert isinstance(full_tuple, tuple)

    def test_dict(self, test_values, test_matcher_instances):
        """Test dict method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        full_dict = pair.dict()

        expected = {
            "id": 1,
            "name": "alpha",
            "matcher1": test_matcher_instances["matcher1"],
            "matcher2": test_matcher_instances["matcher2"]
        }
        assert full_dict == expected
        assert isinstance(full_dict, dict)

    def test_hash(self, test_values, test_matcher_instances):
        """Test hash method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair1 = TestPair(values=test_values, instances=test_matcher_instances)
        pair2 = TestPair(values=test_values, instances=test_matcher_instances)

        assert hash(pair1) == hash(pair2)
        assert isinstance(hash(pair1), int)

    def test_cmpkey(self, test_values, test_matcher_instances):
        """Test _cmpkey method."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair = TestPair(values=test_values, instances=test_matcher_instances)
        cmpkey = pair._cmpkey()

        expected = (1, "alpha", test_matcher_instances["matcher1"], test_matcher_instances["matcher2"])
        assert cmpkey == expected

    def test_comparison_operations(self, test_values, test_matcher_instances):
        """Test comparison operations inherited from IComparable."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        pair1 = TestPair(values=test_values, instances=test_matcher_instances)
        pair2 = TestPair(values=test_values, instances=test_matcher_instances)

        # Test equality
        assert pair1 == pair2
        assert not (pair1 != pair2)

        # Test identity
        assert pair1 == pair1
        assert not (pair1 != pair1)

    def test_comparison_with_different_values(self, test_matcher_instances):
        """Test comparison with different values."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        values1 = {"id": 1, "name": "alpha"}
        values2 = {"id": 2, "name": "beta"}

        matcher_instances2 = {
            "matcher1": TestMatcher1("/path/test_002_beta.txt", id=2, name="beta"),
            "matcher2": TestMatcher2("/path/data_002_beta.dat", id=2, name="beta")
        }

        pair1 = TestPair(values=values1, instances=test_matcher_instances)
        pair2 = TestPair(values=values2, instances=matcher_instances2)

        assert pair1 != pair2
        assert pair1 < pair2
        assert pair1 <= pair2
        assert not (pair1 > pair2)
        assert not (pair1 >= pair2)

    def test_empty_class_annotations(self):
        """Test with empty class (no matchers defined)."""

        class EmptyPair(BaseMatcherPair):
            pass

        assert EmptyPair.__fields__ == {}
        assert EmptyPair.__field_names__ == []
        assert EmptyPair.__value_fields__ == {}
        assert EmptyPair.__value_field_names__ == []

    def test_single_matcher_class(self, temp_directory):
        """Test with single matcher class."""

        class SinglePair(BaseMatcherPair):
            matcher1: TestMatcher1

        pairs = SinglePair.match_all(temp_directory)
        assert len(pairs) == 2

        for pair in pairs:
            assert hasattr(pair, 'matcher1')
            assert hasattr(pair, 'id')
            assert hasattr(pair, 'name')

    def test_match_all_with_path_object(self, temp_directory):
        """Test match_all with Path object instead of string."""

        class TestPair(BaseMatcherPair):
            matcher1: TestMatcher1
            matcher2: TestMatcher2

        path_obj = Path(temp_directory)
        pairs = TestPair.match_all(path_obj)

        assert len(pairs) == 2
