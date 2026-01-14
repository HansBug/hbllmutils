import pytest

from hbllmutils.response import extract_code


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
