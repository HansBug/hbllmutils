"""
Unit tests for the hashable module.

This module contains comprehensive tests for the obj_hashable function,
which converts Python objects into hashable representations.
"""

import pytest

from hbllmutils.utils.hashable import obj_hashable


@pytest.fixture
def simple_primitives():
    """Provide a list of simple primitive values for testing."""
    return [None, 42, 3.14, "hello", True]


@pytest.fixture
def simple_list():
    """Provide a simple list for testing."""
    return [1, 2, 3]


@pytest.fixture
def nested_list():
    """Provide a nested list structure for testing."""
    return [[1, 2], [3, 4], [5, [6, 7]]]


@pytest.fixture
def simple_tuple():
    """Provide a simple tuple for testing."""
    return (1, 2, 3)


@pytest.fixture
def nested_tuple():
    """Provide a nested tuple structure for testing."""
    return (1, [2, 3], 4)


@pytest.fixture
def simple_dict():
    """Provide a simple dictionary for testing."""
    return {'b': 2, 'a': 1}


@pytest.fixture
def nested_dict():
    """Provide a nested dictionary structure for testing."""
    return {'list': [1, 2], 'dict': {'x': 10}}


@pytest.fixture
def mixed_keys_dict():
    """Provide a dictionary with mixed numeric/string keys for testing."""
    return {'item10': 1, 'item2': 2, 'item1': 3}


@pytest.fixture
def complex_nested_structure():
    """Provide a complex nested structure for testing."""
    return {
        'level1': {
            'level2': [1, 2, {'level3': [3, 4]}]
        },
        'simple': 'value'
    }


@pytest.fixture
def hashable_custom_object():
    """Provide a custom hashable object for testing."""
    class HashableClass:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(self.value)

        def __eq__(self, other):
            return isinstance(other, HashableClass) and self.value == other.value

    return HashableClass(42)


@pytest.fixture
def non_hashable_custom_object():
    """Provide a custom non-hashable object for testing."""
    class NonHashableClass:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            raise TypeError("unhashable type")

    return NonHashableClass(42)


@pytest.mark.unittest
class TestObjHashable:
    """Tests for the obj_hashable function."""

    def test_none_input(self):
        """Test that None input returns None."""
        result = obj_hashable(None)
        assert result is None

    def test_primitive_types(self, simple_primitives):
        """Test that primitive types are returned unchanged."""
        for primitive in simple_primitives:
            result = obj_hashable(primitive)
            assert result == primitive
            assert type(result) == type(primitive)

    def test_int_type(self):
        """Test that integer values are returned unchanged."""
        result = obj_hashable(42)
        assert result == 42
        assert isinstance(result, int)

    def test_float_type(self):
        """Test that float values are returned unchanged."""
        result = obj_hashable(3.14)
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_type(self):
        """Test that string values are returned unchanged."""
        result = obj_hashable("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_simple_list_conversion(self, simple_list):
        """Test that simple lists are converted to tuples."""
        result = obj_hashable(simple_list)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_list_conversion(self, nested_list):
        """Test that nested lists are recursively converted to tuples."""
        result = obj_hashable(nested_list)
        expected = ((1, 2), (3, 4), (5, (6, 7)))
        assert result == expected
        assert isinstance(result, tuple)

    def test_simple_tuple_conversion(self, simple_tuple):
        """Test that simple tuples are processed correctly."""
        result = obj_hashable(simple_tuple)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_tuple_conversion(self, nested_tuple):
        """Test that nested tuples with lists are converted correctly."""
        result = obj_hashable(nested_tuple)
        expected = (1, (2, 3), 4)
        assert result == expected
        assert isinstance(result, tuple)

    def test_simple_dict_conversion(self, simple_dict):
        """Test that dictionaries are converted to sorted tuples of key-value pairs."""
        result = obj_hashable(simple_dict)
        expected = (('a', 1), ('b', 2))
        assert result == expected
        assert isinstance(result, tuple)

    def test_nested_dict_conversion(self, nested_dict):
        """Test that nested dictionaries are recursively converted."""
        result = obj_hashable(nested_dict)
        expected = (('dict', (('x', 10),)), ('list', (1, 2)))
        assert result == expected

    def test_dict_with_mixed_keys(self, mixed_keys_dict):
        """Test that dictionary keys are naturally sorted."""
        result = obj_hashable(mixed_keys_dict)
        expected = (('item1', 3), ('item2', 2), ('item10', 1))
        assert result == expected

    def test_complex_nested_structure(self, complex_nested_structure):
        """Test that complex nested structures are fully converted."""
        result = obj_hashable(complex_nested_structure)
        expected = (
            ('level1', (('level2', (1, 2, (('level3', (3, 4)),))),)),
            ('simple', 'value')
        )
        assert result == expected

    def test_hashable_custom_object(self, hashable_custom_object):
        """Test that hashable custom objects are returned unchanged."""
        result = obj_hashable(hashable_custom_object)
        assert result is hashable_custom_object

    def test_non_hashable_custom_object_raises_error(self, non_hashable_custom_object):
        """Test that non-hashable custom objects raise TypeError."""
        with pytest.raises(TypeError, match="Object of type 'NonHashableClass' is not hashable"):
            obj_hashable(non_hashable_custom_object)

    def test_empty_list(self):
        """Test that empty lists are converted to empty tuples."""
        result = obj_hashable([])
        assert result == ()
        assert isinstance(result, tuple)

    def test_empty_tuple(self):
        """Test that empty tuples remain empty tuples."""
        result = obj_hashable(())
        assert result == ()
        assert isinstance(result, tuple)

    def test_empty_dict(self):
        """Test that empty dictionaries are converted to empty tuples."""
        result = obj_hashable({})
        assert result == ()
        assert isinstance(result, tuple)

    def test_dict_with_none_values(self):
        """Test that dictionaries with None values are handled correctly."""
        data = {'a': None, 'b': 1}
        result = obj_hashable(data)
        expected = (('a', None), ('b', 1))
        assert result == expected

    def test_list_with_none_values(self):
        """Test that lists with None values are handled correctly."""
        data = [1, None, 3]
        result = obj_hashable(data)
        expected = (1, None, 3)
        assert result == expected

    def test_result_is_hashable(self):
        """Test that the result can be hashed and used as a dictionary key or set member."""
        data = {'list': [1, 2], 'dict': {'nested': 'value'}}
        result = obj_hashable(data)

        # Should be able to hash the result
        hash_value = hash(result)
        assert isinstance(hash_value, int)

        # Should be able to use as dict key
        test_dict = {result: 'test_value'}
        assert test_dict[result] == 'test_value'

        # Should be able to add to set
        test_set = {result}
        assert result in test_set
