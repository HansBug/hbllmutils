import json
import tempfile
from pathlib import Path
from typing import Sequence
from unittest.mock import Mock, patch

import pytest
import yaml
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


@pytest.fixture
def complex_history():
    return [
        {"role": "system", "content": "You are a helpful AI assistant specializing in data analysis."},
        {"role": "user", "content": "Can you help me analyze some data?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help you with data analysis."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's a chart showing sales data:"},
                {"type": "image_url", "image_url": "blob:chart_data_url"}
            ]
        },
        {
            "role": "assistant",
            "content": "I can see the chart. The sales show an upward trend with some seasonal variations."
        },
        {"role": "user", "content": "What about Q4 performance?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Q4 shows strong performance with 25% growth."},
                {"type": "image_url", "image_url": "blob:q4_analysis_url"}
            ]
        }
    ]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


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

    # JSON export/import tests
    def test_dump_json_simple(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        json_file = temp_dir / "test.json"

        history.dump_json(str(json_file))

        assert json_file.exists()
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data == sample_history

    def test_dump_json_complex(self, temp_dir, complex_history):
        history = LLMHistory(complex_history)
        json_file = temp_dir / "complex.json"

        history.dump_json(str(json_file))

        assert json_file.exists()
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data == complex_history

    def test_dump_json_with_custom_params(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        json_file = temp_dir / "custom.json"

        history.dump_json(str(json_file), indent=4, ensure_ascii=True, sort_keys=False)

        assert json_file.exists()
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check that custom formatting was applied
            assert "    " in content  # 4-space indentation
            data = json.loads(content)
        assert data == sample_history

    def test_dump_json_creates_directory(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        nested_dir = temp_dir / "nested" / "dir"
        json_file = nested_dir / "test.json"

        history.dump_json(str(json_file))

        assert json_file.exists()
        assert nested_dir.exists()

    def test_load_json_simple(self, temp_dir, sample_history):
        json_file = temp_dir / "test.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_history, f)

        loaded_history = LLMHistory.load_json(str(json_file))

        assert len(loaded_history) == len(sample_history)
        assert loaded_history._history == sample_history

    def test_load_json_complex(self, temp_dir, complex_history):
        json_file = temp_dir / "complex.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(complex_history, f)

        loaded_history = LLMHistory.load_json(str(json_file))

        assert len(loaded_history) == len(complex_history)
        assert loaded_history._history == complex_history

    def test_load_json_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="File not found"):
            LLMHistory.load_json(str(non_existent_file))

    def test_load_json_invalid_json(self, temp_dir):
        json_file = temp_dir / "invalid.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            LLMHistory.load_json(str(json_file))

    def test_load_json_invalid_structure_not_list(self, temp_dir):
        json_file = temp_dir / "invalid_structure.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="JSON file must contain a list of messages"):
            LLMHistory.load_json(str(json_file))

    def test_load_json_invalid_message_not_dict(self, temp_dir):
        json_file = temp_dir / "invalid_message.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(["not a dict"], f)

        with pytest.raises(ValueError, match="Message at index 0 must be a dictionary"):
            LLMHistory.load_json(str(json_file))

    def test_load_json_missing_required_fields(self, temp_dir):
        json_file = temp_dir / "missing_fields.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([{"role": "user"}], f)  # Missing 'content'

        with pytest.raises(ValueError, match="Message at index 0 must have 'role' and 'content' fields"):
            LLMHistory.load_json(str(json_file))

    def test_json_roundtrip_simple(self, temp_dir, sample_history):
        original_history = LLMHistory(sample_history)
        json_file = temp_dir / "roundtrip.json"

        # Export
        original_history.dump_json(str(json_file))

        # Import
        loaded_history = LLMHistory.load_json(str(json_file))

        # Verify equality
        assert original_history == loaded_history
        assert original_history._history == loaded_history._history

    def test_json_roundtrip_complex(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)
        json_file = temp_dir / "complex_roundtrip.json"

        # Export
        original_history.dump_json(str(json_file))

        # Import
        loaded_history = LLMHistory.load_json(str(json_file))

        # Verify equality
        assert original_history == loaded_history
        assert original_history._history == loaded_history._history

    def test_json_multiple_roundtrips(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)

        # Perform multiple export/import cycles
        current_history = original_history
        for i in range(5):
            json_file = temp_dir / f"roundtrip_{i}.json"

            # Export
            current_history.dump_json(str(json_file))

            # Import
            current_history = LLMHistory.load_json(str(json_file))

            # Verify data integrity after each cycle
            assert current_history == original_history
            assert current_history._history == original_history._history

    # YAML export/import tests
    def test_dump_yaml_simple(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        yaml_file = temp_dir / "test.yaml"

        history.dump_yaml(str(yaml_file))

        assert yaml_file.exists()
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert data == sample_history

    def test_dump_yaml_complex(self, temp_dir, complex_history):
        history = LLMHistory(complex_history)
        yaml_file = temp_dir / "complex.yaml"

        history.dump_yaml(str(yaml_file))

        assert yaml_file.exists()
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert data == complex_history

    def test_dump_yaml_with_custom_params(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        yaml_file = temp_dir / "custom.yaml"

        history.dump_yaml(str(yaml_file), default_flow_style=True, indent=4, sort_keys=False)

        assert yaml_file.exists()
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert data == sample_history

    def test_dump_yaml_creates_directory(self, temp_dir, sample_history):
        history = LLMHistory(sample_history)
        nested_dir = temp_dir / "nested" / "dir"
        yaml_file = nested_dir / "test.yaml"

        history.dump_yaml(str(yaml_file))

        assert yaml_file.exists()
        assert nested_dir.exists()

    def test_load_yaml_simple(self, temp_dir, sample_history):
        yaml_file = temp_dir / "test.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_history, f)

        loaded_history = LLMHistory.load_yaml(str(yaml_file))

        assert len(loaded_history) == len(sample_history)
        assert loaded_history._history == sample_history

    def test_load_yaml_complex(self, temp_dir, complex_history):
        yaml_file = temp_dir / "complex.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(complex_history, f)

        loaded_history = LLMHistory.load_yaml(str(yaml_file))

        assert len(loaded_history) == len(complex_history)
        assert loaded_history._history == complex_history

    def test_load_yaml_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="File not found"):
            LLMHistory.load_yaml(str(non_existent_file))

    def test_load_yaml_invalid_yaml(self, temp_dir):
        yaml_file = temp_dir / "invalid.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            LLMHistory.load_yaml(str(yaml_file))

    def test_load_yaml_invalid_structure_not_list(self, temp_dir):
        yaml_file = temp_dir / "invalid_structure.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="YAML file must contain a list of messages"):
            LLMHistory.load_yaml(str(yaml_file))

    def test_load_yaml_invalid_message_not_dict(self, temp_dir):
        yaml_file = temp_dir / "invalid_message.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(["not a dict"], f)

        with pytest.raises(ValueError, match="Message at index 0 must be a dictionary"):
            LLMHistory.load_yaml(str(yaml_file))

    def test_load_yaml_missing_required_fields(self, temp_dir):
        yaml_file = temp_dir / "missing_fields.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump([{"role": "user"}], f)  # Missing 'content'

        with pytest.raises(ValueError, match="Message at index 0 must have 'role' and 'content' fields"):
            LLMHistory.load_yaml(str(yaml_file))

    def test_yaml_roundtrip_simple(self, temp_dir, sample_history):
        original_history = LLMHistory(sample_history)
        yaml_file = temp_dir / "roundtrip.yaml"

        # Export
        original_history.dump_yaml(str(yaml_file))

        # Import
        loaded_history = LLMHistory.load_yaml(str(yaml_file))

        # Verify equality
        assert original_history == loaded_history
        assert original_history._history == loaded_history._history

    def test_yaml_roundtrip_complex(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)
        yaml_file = temp_dir / "complex_roundtrip.yaml"

        # Export
        original_history.dump_yaml(str(yaml_file))

        # Import
        loaded_history = LLMHistory.load_yaml(str(yaml_file))

        # Verify equality
        assert original_history == loaded_history
        assert original_history._history == loaded_history._history

    def test_yaml_multiple_roundtrips(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)

        # Perform multiple export/import cycles
        current_history = original_history
        for i in range(5):
            yaml_file = temp_dir / f"roundtrip_{i}.yaml"

            # Export
            current_history.dump_yaml(str(yaml_file))

            # Import
            current_history = LLMHistory.load_yaml(str(yaml_file))

            # Verify data integrity after each cycle
            assert current_history == original_history
            assert current_history._history == original_history._history

    def test_cross_format_compatibility_json_to_yaml(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)
        json_file = temp_dir / "test.json"
        yaml_file = temp_dir / "test.yaml"

        # Export as JSON
        original_history.dump_json(str(json_file))

        # Load from JSON and export as YAML
        json_loaded = LLMHistory.load_json(str(json_file))
        json_loaded.dump_yaml(str(yaml_file))

        # Load from YAML
        yaml_loaded = LLMHistory.load_yaml(str(yaml_file))

        # All should be equal
        assert original_history == json_loaded == yaml_loaded

    def test_cross_format_compatibility_yaml_to_json(self, temp_dir, complex_history):
        original_history = LLMHistory(complex_history)
        yaml_file = temp_dir / "test.yaml"
        json_file = temp_dir / "test.json"

        # Export as YAML
        original_history.dump_yaml(str(yaml_file))

        # Load from YAML and export as JSON
        yaml_loaded = LLMHistory.load_yaml(str(yaml_file))
        yaml_loaded.dump_json(str(json_file))

        # Load from JSON
        json_loaded = LLMHistory.load_json(str(json_file))

        # All should be equal
        assert original_history == yaml_loaded == json_loaded

    def test_empty_history_export_import(self, temp_dir):
        empty_history = LLMHistory()
        json_file = temp_dir / "empty.json"
        yaml_file = temp_dir / "empty.yaml"

        # Test JSON
        empty_history.dump_json(str(json_file))
        loaded_json = LLMHistory.load_json(str(json_file))
        assert empty_history == loaded_json
        assert len(loaded_json) == 0

        # Test YAML
        empty_history.dump_yaml(str(yaml_file))
        loaded_yaml = LLMHistory.load_yaml(str(yaml_file))
        assert empty_history == loaded_yaml
        assert len(loaded_yaml) == 0

    def test_unicode_content_preservation(self, temp_dir):
        unicode_history = [
            {"role": "user", "content": "Hello 世界! 🌍"},
            {"role": "assistant", "content": "Bonjour! Здравствуй! こんにちは! 🎉"},
            {"role": "user", "content": "Emoji test: 😀😃😄😁🥰🤔💭✨🚀"}
        ]
        original_history = LLMHistory(unicode_history)

        # Test JSON roundtrip
        json_file = temp_dir / "unicode.json"
        original_history.dump_json(str(json_file))
        json_loaded = LLMHistory.load_json(str(json_file))
        assert original_history == json_loaded

        # Test YAML roundtrip
        yaml_file = temp_dir / "unicode.yaml"
        original_history.dump_yaml(str(yaml_file))
        yaml_loaded = LLMHistory.load_yaml(str(yaml_file))
        assert original_history == yaml_loaded

        # Verify specific unicode content
        assert json_loaded[0]["content"] == "Hello 世界! 🌍"
        assert yaml_loaded[1]["content"] == "Bonjour! Здравствуй! こんにちは! 🎉"
