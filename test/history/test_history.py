from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from hbllmutils.history import create_llm_message, LLMHistory


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def mock_to_blob_url():
    with patch('hbllmutils.history.history.to_blob_url') as mock:
        mock.return_value = "blob:mock_url"
        yield mock


@pytest.fixture
def sample_history():
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]


@pytest.mark.unittest
class TestHistoryHistoryCreateMessage:
    def test_create_llm_message_with_string(self):
        result = create_llm_message("Hello world")
        expected = {
            "role": "user",
            "content": "Hello world"
        }
        assert result == expected

    def test_create_llm_message_with_string_custom_role(self):
        result = create_llm_message("Hello world", role="assistant")
        expected = {
            "role": "assistant",
            "content": "Hello world"
        }
        assert result == expected

    def test_create_llm_message_with_image(self, mock_image, mock_to_blob_url):
        result = create_llm_message(mock_image)
        expected = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": "blob:mock_url"}]
        }
        assert result == expected
        mock_to_blob_url.assert_called_once_with(mock_image)

    def test_create_llm_message_with_list_strings_and_images(self, mock_image, mock_to_blob_url):
        message = ["Hello", mock_image, "World"]
        result = create_llm_message(message)
        expected = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": "blob:mock_url"},
                {"type": "text", "text": "World"}
            ]
        }
        assert result == expected

    def test_create_llm_message_with_tuple(self, mock_image, mock_to_blob_url):
        message = ("Hello", mock_image)
        result = create_llm_message(message)
        expected = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": "blob:mock_url"}
            ]
        }
        assert result == expected

    def test_create_llm_message_with_invalid_list_item(self):
        message = ["Hello", 123]
        with pytest.raises(TypeError, match="Unsupported type for message content item at #1 - 123"):
            create_llm_message(message)

    def test_create_llm_message_with_invalid_content_type(self):
        with pytest.raises(TypeError, match="Unsupported content type - 123"):
            create_llm_message(123)


@pytest.mark.unittest
class TestHistoryHistoryModule:
    def test_init_with_no_history(self):
        history = LLMHistory()
        assert len(history) == 0
        assert history._history == []

    def test_init_with_history(self, sample_history):
        history = LLMHistory(sample_history)
        assert len(history) == 3
        assert history._history == sample_history

    def test_init_with_none_history(self):
        history = LLMHistory(None)
        assert len(history) == 0
        assert history._history == []

    def test_getitem_single_index(self, sample_history):
        history = LLMHistory(sample_history)
        assert history[0] == sample_history[0]
        assert history[1] == sample_history[1]
        assert history[-1] == sample_history[-1]

    def test_getitem_slice_returns_llmhistory(self, sample_history):
        history = LLMHistory(sample_history)
        sliced = history[0:2]
        assert isinstance(sliced, LLMHistory)
        assert len(sliced) == 2
        assert sliced._history == sample_history[0:2]

    def test_len(self, sample_history):
        history = LLMHistory(sample_history)
        assert len(history) == 3

    def test_is_sequence(self):
        history = LLMHistory()
        assert isinstance(history, Sequence)

    def test_append_with_string(self, mock_to_blob_url):
        history = LLMHistory()
        new_history = history.with_message("user", "Hello")
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == "Hello"
        # Verify original is unchanged
        assert len(history) == 0

    def test_append_with_image(self, mock_image, mock_to_blob_url):
        history = LLMHistory()
        new_history = history.with_message("user", mock_image)
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == [{"type": "image_url", "image_url": "blob:mock_url"}]
        # Verify original is unchanged
        assert len(history) == 0

    def test_append_user(self, mock_to_blob_url):
        history = LLMHistory()
        new_history = history.with_user_message("Hello user")
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == "Hello user"
        # Verify original is unchanged
        assert len(history) == 0

    def test_append_assistant(self, mock_to_blob_url):
        history = LLMHistory()
        new_history = history.with_assistant_message("Hello assistant")
        assert len(new_history) == 1
        assert new_history[0]["role"] == "assistant"
        assert new_history[0]["content"] == "Hello assistant"
        # Verify original is unchanged
        assert len(history) == 0

    def test_set_system_prompt_empty_history(self, mock_to_blob_url):
        history = LLMHistory()
        new_history = history.with_system_prompt("System message")
        assert len(new_history) == 1
        assert new_history[0]["role"] == "system"
        assert new_history[0]["content"] == "System message"
        # Verify original is unchanged
        assert len(history) == 0

    def test_set_system_prompt_replace_existing(self, mock_to_blob_url):
        history = LLMHistory([{"role": "system", "content": "Old system"}])
        new_history = history.with_system_prompt("New system")
        assert len(new_history) == 1
        assert new_history[0]["role"] == "system"
        assert new_history[0]["content"] == "New system"
        # Verify original is unchanged
        assert history[0]["content"] == "Old system"

    def test_set_system_prompt_insert_at_beginning(self, mock_to_blob_url):
        history = LLMHistory([{"role": "user", "content": "Hello"}])
        new_history = history.with_system_prompt("System message")
        assert len(new_history) == 2
        assert new_history[0]["role"] == "system"
        assert new_history[0]["content"] == "System message"
        assert new_history[1]["role"] == "user"
        # Verify original is unchanged
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_to_json(self, sample_history):
        history = LLMHistory(sample_history)
        result = history.to_json()
        assert result == sample_history
        assert result is not history._history

    def test_clone_empty_history(self):
        history = LLMHistory()
        cloned = history.clone()
        assert isinstance(cloned, LLMHistory)
        assert len(cloned) == 0
        assert cloned._history == []
        assert cloned._history is not history._history

    def test_clone_with_history(self, sample_history):
        history = LLMHistory(sample_history)
        cloned = history.clone()
        assert isinstance(cloned, LLMHistory)
        assert len(cloned) == 3
        assert cloned._history == sample_history
        assert cloned._history is not history._history

    def test_clone_independence(self, mock_to_blob_url):
        history = LLMHistory()
        history_with_msg = history.with_user_message("Original message")
        cloned = history_with_msg.clone()

        # Modify original
        new_history = history_with_msg.with_user_message("New message in original")

        # Modify clone
        new_cloned = cloned.with_assistant_message("New message in clone")

        # Verify independence
        assert len(new_history) == 2
        assert len(new_cloned) == 2
        assert new_history[1]["content"] == "New message in original"
        assert new_cloned[1]["content"] == "New message in clone"
        # Verify original instances are unchanged
        assert len(history_with_msg) == 1
        assert len(cloned) == 1

    def test_method_chaining(self, mock_to_blob_url):
        history = LLMHistory()
        result = (history
                  .with_system_prompt("You are helpful")
                  .with_user_message("Hello")
                  .with_assistant_message("Hi there"))

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        # Verify original is unchanged
        assert len(history) == 0

    def test_immutability_verification(self, mock_to_blob_url):
        history = LLMHistory()

        # Test that all operations return new instances
        h1 = history.with_user_message("msg1")
        h2 = history.with_assistant_message("msg2")
        h3 = history.with_system_prompt("system")

        # All should be different instances
        assert h1 is not history
        assert h2 is not history
        assert h3 is not history
        assert h1 is not h2
        assert h1 is not h3
        assert h2 is not h3

        # Original should remain empty
        assert len(history) == 0

        # Each should have expected content
        assert len(h1) == 1
        assert len(h2) == 1
        assert len(h3) == 1

    def test_eq_same_empty_histories(self):
        history1 = LLMHistory()
        history2 = LLMHistory()
        assert history1 == history2

    def test_eq_same_histories_with_content(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hello")
        assert history1 == history2

    def test_eq_different_histories(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hi")
        assert history1 != history2

    def test_eq_different_lengths(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hello").with_assistant_message("Hi")
        assert history1 != history2

    def test_eq_with_non_llmhistory(self, mock_to_blob_url):
        history = LLMHistory().with_user_message("Hello")
        assert history != "not a history"
        assert history != []
        assert history != {"role": "user", "content": "Hello"}

    def test_eq_complex_histories(self, sample_history):
        history1 = LLMHistory(sample_history)
        history2 = LLMHistory(sample_history.copy())
        assert history1 == history2

    def test_hash_empty_histories(self):
        history1 = LLMHistory()
        history2 = LLMHistory()
        assert hash(history1) == hash(history2)

    def test_hash_same_content(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hello")
        assert hash(history1) == hash(history2)

    def test_hash_different_content(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hi")
        assert hash(history1) != hash(history2)

    def test_hash_with_nested_structures(self, mock_image, mock_to_blob_url):
        history1 = LLMHistory().with_message("user", ["Hello", mock_image])
        history2 = LLMHistory().with_message("user", ["Hello", mock_image])
        assert hash(history1) == hash(history2)

    def test_hash_with_different_roles(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_assistant_message("Hello")
        assert hash(history1) != hash(history2)

    def test_hash_consistency_after_operations(self, mock_to_blob_url):
        history = LLMHistory().with_user_message("Hello")
        hash1 = hash(history)
        hash2 = hash(history)
        assert hash1 == hash2

    def test_hash_with_complex_nested_data(self):
        complex_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": "blob:test"}
                ]
            },
            {
                "role": "assistant",
                "content": "Hi there!"
            }
        ]
        history1 = LLMHistory(complex_history)
        history2 = LLMHistory(complex_history.copy())
        assert hash(history1) == hash(history2)

    def test_hash_with_none_values(self):
        history_with_none = [{"role": "user", "content": None}]
        history1 = LLMHistory(history_with_none)
        history2 = LLMHistory(history_with_none.copy())
        assert hash(history1) == hash(history2)

    def test_hash_with_boolean_values(self):
        history_with_bool = [{"role": "user", "content": "Hello", "enabled": True}]
        history1 = LLMHistory(history_with_bool)
        history2 = LLMHistory(history_with_bool.copy())
        assert hash(history1) == hash(history2)

    def test_hash_with_numeric_values(self):
        history_with_numbers = [
            {"role": "user", "content": "Hello", "id": 123, "score": 45.6}
        ]
        history1 = LLMHistory(history_with_numbers)
        history2 = LLMHistory(history_with_numbers.copy())
        assert hash(history1) == hash(history2)

    def test_hash_usable_in_set(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hello")
        history3 = LLMHistory().with_user_message("Hi")

        history_set = {history1, history2, history3}
        assert len(history_set) == 2

    def test_hash_usable_as_dict_key(self, mock_to_blob_url):
        history1 = LLMHistory().with_user_message("Hello")
        history2 = LLMHistory().with_user_message("Hello")

        history_dict = {history1: "value1"}
        history_dict[history2] = "value2"

        assert len(history_dict) == 1
        assert history_dict[history1] == "value2"

    def test_hash_make_hashable_with_custom_object(self):
        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomObject({self.value})"

        custom_obj = CustomObject("test")
        history_with_custom = [{"role": "user", "content": custom_obj}]
        history = LLMHistory(history_with_custom)

        # Should not raise an exception
        hash_value = hash(history)
        assert isinstance(hash_value, int)

    def test_hash_deep_nested_structures(self):
        deeply_nested = [
            {
                "role": "user",
                "content": {
                    "messages": [
                        {"text": "Hello", "metadata": {"id": 1, "tags": ["greeting"]}},
                        {"text": "World", "metadata": {"id": 2, "tags": ["noun"]}}
                    ]
                }
            }
        ]
        history1 = LLMHistory(deeply_nested)
        history2 = LLMHistory(deeply_nested.copy())
        assert hash(history1) == hash(history2)
