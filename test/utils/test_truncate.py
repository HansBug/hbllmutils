from unittest.mock import patch

import pytest

from hbllmutils.utils import truncate_dict, log_pformat


@pytest.fixture
def sample_string():
    return "a" * 300


@pytest.fixture
def sample_list():
    return [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def sample_dict():
    return {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
        "key4": "value4",
        "key5": "value5",
        "key6": "value6"
    }


@pytest.fixture
def nested_structure():
    return {
        "users": [
            {"name": "John" * 100, "age": 30},
            {"name": "Jane" * 100, "age": 25}
        ],
        "data": "x" * 500,
        "config": {
            "setting1": "value1",
            "setting2": "value2",
            "setting3": "value3",
            "setting4": "value4",
            "setting5": "value5",
            "setting6": "value6"
        }
    }


@pytest.fixture
def llm_history():
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello" * 1000},
        {"role": "assistant", "content": "Hi there!"}
    ]


@pytest.mark.unittest
class TestTruncateDict:

    def test_truncate_string_within_limit(self):
        result = truncate_dict("hello", max_string_len=10)
        assert result == "hello"

    def test_truncate_string_exceeds_limit(self, sample_string):
        result = truncate_dict(sample_string, max_string_len=10)
        assert result == "aaaaaaaaaa...<truncated, total 300 chars>"

    def test_truncate_empty_string(self):
        result = truncate_dict("", max_string_len=10)
        assert result == ""

    def test_truncate_list_within_limit(self):
        result = truncate_dict([1, 2, 3], max_list_items=5)
        assert result == [1, 2, 3]

    def test_truncate_list_exceeds_limit(self, sample_list):
        result = truncate_dict(sample_list, max_list_items=3)
        assert result == [1, 2, 3, "...<5 more items>"]

    def test_truncate_empty_list(self):
        result = truncate_dict([], max_list_items=3)
        assert result == []

    def test_truncate_tuple_within_limit(self):
        result = truncate_dict((1, 2, 3), max_list_items=5)
        assert result == [1, 2, 3]

    def test_truncate_tuple_exceeds_limit(self):
        result = truncate_dict((1, 2, 3, 4, 5), max_list_items=3)
        assert result == [1, 2, 3, "...<2 more items>"]

    def test_truncate_dict_within_limit(self):
        small_dict = {"a": 1, "b": 2}
        result = truncate_dict(small_dict, max_dict_keys=5)
        assert result == {"a": 1, "b": 2}

    def test_truncate_dict_exceeds_limit(self, sample_dict):
        result = truncate_dict(sample_dict, max_dict_keys=3)
        expected_keys = ["key1", "key2", "key3"]
        for key in expected_keys:
            assert key in result
        assert result["<truncated>"] == "3 more keys"
        assert len([k for k in result.keys() if not k.startswith("<")]) == 3

    def test_truncate_empty_dict(self):
        result = truncate_dict({}, max_dict_keys=3)
        assert result == {}

    def test_truncate_nested_structure(self, nested_structure):
        result = truncate_dict(nested_structure, max_string_len=50, max_list_items=1, max_dict_keys=2)

        # Check that data string is truncated
        assert "...<truncated, total 500 chars>" in result["data"]

        # Check that users list is truncated
        assert len([item for item in result["users"] if not isinstance(item, str)]) == 1
        assert "...<1 more items>" in result["users"]

        # Check that config dict is truncated
        assert result["<truncated>"] == "1 more keys"

    def test_truncate_other_types(self):
        # Test with int
        assert truncate_dict(42) == 42

        # Test with float
        assert truncate_dict(3.14) == 3.14

        # Test with None
        assert truncate_dict(None) is None

        # Test with boolean
        assert truncate_dict(True) is True
        assert truncate_dict(False) is False

    def test_current_depth_parameter(self):
        # Test that current_depth parameter is accepted and passed through
        result = truncate_dict({"a": "test"}, current_depth=5)
        assert result == {"a": "test"}

    def test_recursive_truncation_with_nested_lists(self):
        nested_list = [["a" * 300, "b" * 300], ["c" * 300, "d" * 300]]
        result = truncate_dict(nested_list, max_string_len=10, max_list_items=2)

        # Check that inner strings are truncated
        assert "...<truncated, total 300 chars>" in result[0][0]
        assert "...<truncated, total 300 chars>" in result[0][1]


@pytest.mark.unittest
class TestLogPformat:
    def test_log_pformat_basic(self, llm_history):
        result = log_pformat(llm_history, max_string_len=50)
        assert isinstance(result, str)
        assert "truncated" in result

    def test_log_pformat_with_custom_width(self, sample_dict):
        result = log_pformat(sample_dict, width=80)
        assert isinstance(result, str)

    @patch('shutil.get_terminal_size')
    def test_log_pformat_default_width(self, mock_terminal_size, sample_dict):
        mock_terminal_size.return_value.width = 120
        mock_terminal_size.return_value.__getitem__ = lambda self, index: 120

        result = log_pformat(sample_dict, width=None)
        assert isinstance(result, str)
        mock_terminal_size.assert_called_once()

    def test_log_pformat_with_kwargs(self, sample_list):
        result = log_pformat(sample_list, width=80, indent=4, depth=2)
        assert isinstance(result, str)

    def test_log_pformat_all_parameters(self):
        data = {
            "long_string": "x" * 500,
            "long_list": list(range(20)),
            "many_keys": {f"key{i}": f"value{i}" for i in range(10)}
        }

        result = log_pformat(
            data,
            max_string_len=100,
            max_list_items=3,
            max_dict_keys=2,
            width=120
        )

        assert isinstance(result, str)
        assert "truncated" in result

    def test_log_pformat_empty_object(self):
        result = log_pformat({})
        assert result == "{}"

        result = log_pformat([])
        assert result == "[]"

        result = log_pformat("")
        assert result == "''"

    def test_log_pformat_simple_types(self):
        assert log_pformat(42) == "42"
        assert log_pformat(3.14) == "3.14"
        assert log_pformat(True) == "True"
        assert log_pformat(None) == "None"
