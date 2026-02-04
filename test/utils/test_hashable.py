import pytest

from hbllmutils.utils.hashable import obj_hashable


@pytest.fixture
def simple_primitives():
    return [None, 42, 3.14, "hello", True]


@pytest.fixture
def simple_list():
    return [1, 2, 3]


@pytest.fixture
def nested_list():
    return [[1, 2], [3, 4], [5, [6, 7]]]


@pytest.fixture
def simple_tuple():
    return (1, 2, 3)


@pytest.fixture
def nested_tuple():
    return (1, [2, 3], 4)


@pytest.fixture
def simple_dict():
    return {'b': 2, 'a': 1}


@pytest.fixture
def nested_dict():
    return {'list': [1, 2], 'dict': {'x': 10}}


@pytest.fixture
def mixed_keys_dict():
    return {'item10': 1, 'item2': 2, 'item1': 3}


@pytest.fixture
def complex_nested_structure():
    return {
        'level1': {
            'level2': [1, 2, {'level3': [3, 4]}]
        },
        'simple': 'value'
    }


@pytest.fixture
def hashable_custom_object():
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
    class NonHashableClass:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            raise TypeError("unhashable type")

    return NonHashableClass(42)


@pytest.mark.unittest
class TestObjHashable:

    def test_none_input(self):
        result = obj_hashable(None)
        assert result is None

    def test_primitive_types(self, simple_primitives):
        for primitive in simple_primitives:
            result = obj_hashable(primitive)
            assert result == primitive
            assert type(result) == type(primitive)

    def test_int_type(self):
        result = obj_hashable(42)
        assert result == 42
        assert isinstance(result, int)

    def test_float_type(self):
        result = obj_hashable(3.14)
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_type(self):
        result = obj_hashable("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_simple_list_conversion(self, simple_list):
        result = obj_hashable(simple_list)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_list_conversion(self, nested_list):
        result = obj_hashable(nested_list)
        expected = ((1, 2), (3, 4), (5, (6, 7)))
        assert result == expected
        assert isinstance(result, tuple)

    def test_simple_tuple_conversion(self, simple_tuple):
        result = obj_hashable(simple_tuple)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_tuple_conversion(self, nested_tuple):
        result = obj_hashable(nested_tuple)
        expected = (1, (2, 3), 4)
        assert result == expected
        assert isinstance(result, tuple)

    def test_simple_dict_conversion(self, simple_dict):
        result = obj_hashable(simple_dict)
        expected = (('a', 1), ('b', 2))
        assert result == expected
        assert isinstance(result, tuple)

    def test_nested_dict_conversion(self, nested_dict):
        result = obj_hashable(nested_dict)
        expected = (('dict', (('x', 10),)), ('list', (1, 2)))
        assert result == expected

    def test_dict_with_mixed_keys(self, mixed_keys_dict):
        result = obj_hashable(mixed_keys_dict)
        expected = (('item1', 3), ('item2', 2), ('item10', 1))
        assert result == expected

    def test_complex_nested_structure(self, complex_nested_structure):
        result = obj_hashable(complex_nested_structure)
        expected = (
            ('level1', (('level2', (1, 2, (('level3', (3, 4)),))),)),
            ('simple', 'value')
        )
        assert result == expected

    def test_hashable_custom_object(self, hashable_custom_object):
        result = obj_hashable(hashable_custom_object)
        assert result is hashable_custom_object

    def test_non_hashable_custom_object_raises_error(self, non_hashable_custom_object):
        with pytest.raises(TypeError, match="Object of type 'NonHashableClass' is not hashable"):
            obj_hashable(non_hashable_custom_object)

    def test_empty_list(self):
        result = obj_hashable([])
        assert result == ()
        assert isinstance(result, tuple)

    def test_empty_tuple(self):
        result = obj_hashable(())
        assert result == ()
        assert isinstance(result, tuple)

    def test_empty_dict(self):
        result = obj_hashable({})
        assert result == ()
        assert isinstance(result, tuple)

    def test_dict_with_none_values(self):
        data = {'a': None, 'b': 1}
        result = obj_hashable(data)
        expected = (('a', None), ('b', 1))
        assert result == expected

    def test_list_with_none_values(self):
        data = [1, None, 3]
        result = obj_hashable(data)
        expected = (1, None, 3)
        assert result == expected

    def test_result_is_hashable(self):
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
