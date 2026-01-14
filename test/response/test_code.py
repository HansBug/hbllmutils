import json

import pytest

from hbllmutils.response import extract_code, parse_json


@pytest.mark.unittest
class TestExtractCodeFromMarkdown:
    def test_plain_code_without_fenced_blocks(self, text_aligner):
        """Test extracting plain code without fenced code block markers."""

        text = "print('hello world')"
        result = extract_code(text)
        text_aligner.assert_equal("print('hello world')", result)

    def test_plain_code_with_whitespace(self, text_aligner):
        """Test extracting plain code with leading/trailing whitespace."""

        text = "   print('hello world')   \n  "
        result = extract_code(text)
        text_aligner.assert_equal("print('hello world')", result)

    def test_fenced_code_block_without_language(self, text_aligner):
        """Test extracting code from fenced block without language specification."""

        text = "```\nprint('hello world')\nprint('goodbye')\n```"
        result = extract_code(text)
        text_aligner.assert_equal("print('hello world')\nprint('goodbye')\n", result)

    def test_fenced_code_block_with_language_match(self, text_aligner):
        """Test extracting code from fenced block with matching language."""

        text = "```python\nprint('hello world')\nprint('goodbye')\n```"
        result = extract_code(text, 'python')
        text_aligner.assert_equal("print('hello world')\nprint('goodbye')\n", result)

    def test_fenced_code_block_with_language_no_filter(self, text_aligner):
        """Test extracting code from fenced block with language but no filter."""

        text = "```python\nprint('hello world')\n```"
        result = extract_code(text)
        text_aligner.assert_equal("print('hello world')\n", result)

    def test_fenced_code_block_language_mismatch(self):
        """Test error when language doesn't match."""

        text = "```javascript\nconsole.log('hello');\n```"
        with pytest.raises(ValueError, match="No python code found in response."):
            extract_code(text, 'python')

    def test_no_code_blocks_found_with_language(self):
        """Test error when no code blocks found with language filter."""

        text = "```\nprint('hello')\n```"
        with pytest.raises(ValueError, match="No python code found in response."):
            extract_code(text, 'python')

    def test_multiple_code_blocks_with_language(self):
        """Test error when multiple code blocks found with language filter."""

        text = "```python\nprint('hello')\n```\n\n```python\nprint('world')\n```"
        with pytest.raises(ValueError, match="Non-unique python code blocks found in response."):
            extract_code(text, 'python')

    def test_multiple_code_blocks_without_language(self):
        """Test error when multiple code blocks found without language filter."""

        text = "```\nprint('hello')\n```\n\n```\nprint('world')\n```"
        with pytest.raises(ValueError, match="Non-unique code blocks found in response."):
            extract_code(text)

    def test_mixed_language_blocks_filter_specific(self, text_aligner):
        """Test extracting specific language from mixed language blocks."""

        text = "```javascript\nconsole.log('hello');\n```\n\n```python\nprint('world')\n```"
        result = extract_code(text, 'python')
        text_aligner.assert_equal("print('world')\n", result)

    def test_mixed_language_blocks_no_filter(self):
        """Test error with mixed language blocks and no filter."""

        text = "```javascript\nconsole.log('hello');\n```\n\n```python\nprint('world')\n```"
        with pytest.raises(ValueError, match="Non-unique code blocks found in response."):
            extract_code(text)

    def test_empty_plain_text(self, text_aligner):
        """Test empty plain text input."""

        text = ""
        result = extract_code(text)
        text_aligner.assert_equal("", result)

    def test_whitespace_only_plain_text(self, text_aligner):
        """Test whitespace-only plain text input."""

        text = "   \n\t  \n  "
        result = extract_code(text)
        text_aligner.assert_equal("", result)

    def test_fenced_block_with_whitespace_content(self, text_aligner):
        """Test fenced block with whitespace content."""

        text = "```python\n   \n\t\n```"
        result = extract_code(text, 'python')
        text_aligner.assert_equal("   \n\t\n", result)

    def test_language_case_sensitivity(self):
        """Test language parameter case sensitivity."""

        text = "```Python\nprint('hello')\n```"
        with pytest.raises(ValueError, match="No python code found in response."):
            extract_code(text, 'python')

    def test_language_exact_match(self, text_aligner):
        """Test language parameter exact match."""

        text = "```Python\nprint('hello')\n```"
        result = extract_code(text, 'Python')
        text_aligner.assert_equal("print('hello')\n", result)


@pytest.mark.unittest
class TestParseJson:
    def test_parse_valid_json_dict_with_repair(self):
        """Test parsing valid JSON dict with repair enabled."""
        text = '{"key": "value", "number": 42}'
        result = parse_json(text, with_repair=True)
        assert result == {"key": "value", "number": 42}

    def test_parse_valid_json_dict_without_repair(self):
        """Test parsing valid JSON dict without repair."""
        text = '{"key": "value", "number": 42}'
        result = parse_json(text, with_repair=False)
        assert result == {"key": "value", "number": 42}

    def test_parse_valid_json_list_with_repair(self):
        """Test parsing valid JSON list with repair enabled."""
        text = '[1, 2, 3, "test"]'
        result = parse_json(text, with_repair=True)
        assert result == [1, 2, 3, "test"]

    def test_parse_valid_json_list_without_repair(self):
        """Test parsing valid JSON list without repair."""
        text = '[1, 2, 3, "test"]'
        result = parse_json(text, with_repair=False)
        assert result == [1, 2, 3, "test"]

    def test_parse_valid_json_string_with_repair(self):
        """Test parsing valid JSON string with repair enabled."""
        text = '"hello world"'
        result = parse_json(text, with_repair=True)
        assert result == "hello world"

    def test_parse_valid_json_string_without_repair(self):
        """Test parsing valid JSON string without repair."""
        text = '"hello world"'
        result = parse_json(text, with_repair=False)
        assert result == "hello world"

    def test_parse_valid_json_number_with_repair(self):
        """Test parsing valid JSON number with repair enabled."""
        text = '42'
        result = parse_json(text, with_repair=True)
        assert result == 42

    def test_parse_valid_json_number_without_repair(self):
        """Test parsing valid JSON number without repair."""
        text = '42'
        result = parse_json(text, with_repair=False)
        assert result == 42

    def test_parse_valid_json_boolean_with_repair(self):
        """Test parsing valid JSON boolean with repair enabled."""
        text = 'true'
        result = parse_json(text, with_repair=True)
        assert result is True

    def test_parse_valid_json_boolean_without_repair(self):
        """Test parsing valid JSON boolean without repair."""
        text = 'false'
        result = parse_json(text, with_repair=False)
        assert result is False

    def test_parse_valid_json_null_with_repair(self):
        """Test parsing valid JSON null with repair enabled."""
        text = 'null'
        result = parse_json(text, with_repair=True)
        assert result is None

    def test_parse_valid_json_null_without_repair(self):
        """Test parsing valid JSON null without repair."""
        text = 'null'
        result = parse_json(text, with_repair=False)
        assert result is None

    def test_parse_malformed_json_missing_brace_with_repair(self):
        """Test parsing malformed JSON with missing closing brace using repair."""
        text = '{"key": "value"'
        result = parse_json(text, with_repair=True)
        assert result == {"key": "value"}

    def test_parse_malformed_json_missing_brace_without_repair(self):
        """Test parsing malformed JSON with missing closing brace without repair."""
        text = '{"key": "value"'
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_malformed_json_missing_bracket_with_repair(self):
        """Test parsing malformed JSON with missing closing bracket using repair."""
        text = '[1, 2, 3'
        result = parse_json(text, with_repair=True)
        assert result == [1, 2, 3]

    def test_parse_malformed_json_missing_bracket_without_repair(self):
        """Test parsing malformed JSON with missing closing bracket without repair."""
        text = '[1, 2, 3'
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_malformed_json_trailing_comma_with_repair(self):
        """Test parsing malformed JSON with trailing comma using repair."""
        text = '{"key": "value",}'
        result = parse_json(text, with_repair=True)
        assert result == {"key": "value"}

    def test_parse_malformed_json_trailing_comma_without_repair(self):
        """Test parsing malformed JSON with trailing comma without repair."""
        text = '{"key": "value",}'
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_malformed_json_unquoted_keys_with_repair(self):
        """Test parsing malformed JSON with unquoted keys using repair."""
        text = '{key: "value"}'
        result = parse_json(text, with_repair=True)
        assert result == {"key": "value"}

    def test_parse_malformed_json_unquoted_keys_without_repair(self):
        """Test parsing malformed JSON with unquoted keys without repair."""
        text = '{key: "value"}'
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_empty_string_with_repair(self):
        """Test parsing empty string with repair enabled."""
        assert parse_json('', with_repair=True) == ''

    def test_parse_empty_string_without_repair(self):
        """Test parsing empty string without repair."""
        text = ''
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_whitespace_only_with_repair(self):
        """Test parsing whitespace-only string with repair enabled."""
        assert parse_json('   \n\t  ') == ''

    def test_parse_whitespace_only_without_repair(self):
        """Test parsing whitespace-only string without repair."""
        text = '   \n\t  '
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_invalid_json_with_repair(self):
        """Test parsing completely invalid JSON with repair enabled."""
        assert parse_json('not json at all', with_repair=True) == ''

    def test_parse_invalid_json_without_repair(self):
        """Test parsing completely invalid JSON without repair."""
        text = 'not json at all'
        with pytest.raises(json.JSONDecodeError):
            parse_json(text, with_repair=False)

    def test_parse_json_default_with_repair_true(self):
        """Test parsing JSON with default with_repair=True parameter."""
        text = '{"key": "value"}'
        result = parse_json(text)
        assert result == {"key": "value"}

    def test_parse_complex_nested_json_with_repair(self):
        """Test parsing complex nested JSON with repair enabled."""
        text = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], "count": 2}'
        result = parse_json(text, with_repair=True)
        expected = {"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], "count": 2}
        assert result == expected

    def test_parse_complex_nested_json_without_repair(self):
        """Test parsing complex nested JSON without repair."""
        text = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], "count": 2}'
        result = parse_json(text, with_repair=False)
        expected = {"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], "count": 2}
        assert result == expected
