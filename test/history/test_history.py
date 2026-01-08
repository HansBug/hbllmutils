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
        history.append("user", "Hello")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_append_with_image(self, mock_image, mock_to_blob_url):
        history = LLMHistory()
        history.append("user", mock_image)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == [{"type": "image_url", "image_url": "blob:mock_url"}]

    def test_append_user(self, mock_to_blob_url):
        history = LLMHistory()
        history.append_user("Hello user")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello user"

    def test_append_assistant(self, mock_to_blob_url):
        history = LLMHistory()
        history.append_assistant("Hello assistant")
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["content"] == "Hello assistant"

    def test_set_system_prompt_empty_history(self, mock_to_blob_url):
        history = LLMHistory()
        history.set_system_prompt("System message")
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert history[0]["content"] == "System message"

    def test_set_system_prompt_replace_existing(self, mock_to_blob_url):
        history = LLMHistory([{"role": "system", "content": "Old system"}])
        history.set_system_prompt("New system")
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert history[0]["content"] == "New system"

    def test_set_system_prompt_insert_at_beginning(self, mock_to_blob_url):
        history = LLMHistory([{"role": "user", "content": "Hello"}])
        history.set_system_prompt("System message")
        assert len(history) == 2
        assert history[0]["role"] == "system"
        assert history[0]["content"] == "System message"
        assert history[1]["role"] == "user"

    def test_to_json(self, sample_history):
        history = LLMHistory(sample_history)
        result = history.to_json()
        assert result == sample_history
        assert result is history._history
