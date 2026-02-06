"""
Unit tests for the hbllmutils.meta.code.task module.

This module contains comprehensive unit tests for Python code generation LLM tasks,
including both basic code generation with AST validation and detailed code generation
with source file analysis.
"""

import ast
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.meta.code.task import (
    PythonCodeGenerationLLMTask,
    PythonDetailedCodeGenerationLLMTask,
)
from hbllmutils.model import LLMModel
from hbllmutils.response import OutputParseFailed


@pytest.fixture
def mock_model():
    """Create a mock LLM model for testing."""
    model = Mock(spec=LLMModel)
    model._logger_name = "test_model"
    return model


@pytest.fixture
def sample_python_code():
    """Provide sample valid Python code for testing."""
    return """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""


@pytest.fixture
def sample_invalid_python_code():
    """Provide sample invalid Python code for testing."""
    return """def add(a, b)
    return a + b
"""


@pytest.fixture
def sample_code_with_fencing():
    """Provide sample Python code wrapped in markdown fencing."""
    return """```python
def multiply(x, y):
    return x * y
```"""


@pytest.fixture
def temporary_python_file(sample_python_code):
    """Create a temporary Python file with sample code."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write(sample_python_code)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.mark.unittest
class TestPythonCodeGenerationLLMTask:
    """Tests for the PythonCodeGenerationLLMTask class."""

    def test_initialization_default_params(self, mock_model):
        """Test initialization with default parameters."""
        task = PythonCodeGenerationLLMTask(mock_model)

        assert task.model == mock_model
        assert task.default_max_retries == 5
        assert task.force_ast_check is True
        assert isinstance(task.history, LLMHistory)

    def test_initialization_custom_params(self, mock_model):
        """Test initialization with custom parameters."""
        custom_history = LLMHistory().with_system_prompt("Test system prompt")
        task = PythonCodeGenerationLLMTask(
            mock_model,
            history=custom_history,
            default_max_retries=10,
            force_ast_check=False
        )

        assert task.model == mock_model
        assert task.default_max_retries == 10
        assert task.force_ast_check is False
        assert len(task.history) == 1

    def test_parse_and_validate_valid_code(self, mock_model, sample_python_code):
        """Test parsing and validating valid Python code."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=True)

        result = task._parse_and_validate(sample_python_code)

        assert result == sample_python_code.rstrip()
        # Verify it's valid Python by parsing with ast
        ast.parse(result)

    def test_parse_and_validate_invalid_code(self, mock_model, sample_invalid_python_code):
        """Test parsing invalid Python code raises SyntaxError."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=True)

        with pytest.raises(SyntaxError):
            task._parse_and_validate(sample_invalid_python_code)

    def test_parse_and_validate_with_fencing(self, mock_model, sample_code_with_fencing):
        """Test parsing code wrapped in markdown fencing."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=True)

        result = task._parse_and_validate(sample_code_with_fencing)

        assert "def multiply(x, y):" in result
        assert "return x * y" in result
        assert "```" not in result

    def test_parse_and_validate_without_ast_check(self, mock_model, sample_invalid_python_code):
        """Test parsing without AST check allows invalid code."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=False)

        # Should not raise exception even with invalid syntax
        result = task._parse_and_validate(sample_invalid_python_code)
        assert result == sample_invalid_python_code.rstrip()

    def test_parse_and_validate_strips_trailing_whitespace(self, mock_model):
        """Test that trailing whitespace is stripped from parsed code."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=True)
        code_with_whitespace = "x = 42\n\n\n   "

        result = task._parse_and_validate(code_with_whitespace)

        assert result == "x = 42"
        assert not result.endswith("\n")

    @pytest.mark.parametrize("code,expected", [
        ("x = 1", "x = 1"),
        ("def foo():\n    pass", "def foo():\n    pass"),
        ("# comment\nx = 1", "# comment\nx = 1"),
    ])
    def test_parse_and_validate_various_code(self, mock_model, code, expected):
        """Test parsing various valid Python code snippets."""
        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=True)

        result = task._parse_and_validate(code)

        assert result == expected

    def test_exceptions_attribute(self, mock_model):
        """Test that __exceptions__ attribute is properly set."""
        task = PythonCodeGenerationLLMTask(mock_model)

        assert hasattr(task, '__exceptions__')
        assert SyntaxError in task.__exceptions__
        assert ValueError in task.__exceptions__


@pytest.mark.unittest
class TestPythonDetailedCodeGenerationLLMTask:
    """Tests for the PythonDetailedCodeGenerationLLMTask class."""

    def test_initialization_default_params(self, mock_model):
        """Test initialization with default parameters."""
        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="test_code",
            description_text="Test description"
        )

        assert task.model == mock_model
        assert task.code_name == "test_code"
        assert task.description_text == "Test description"
        assert task.default_max_retries == 5
        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True
        assert task.ignore_modules is None
        assert task.no_ignore_modules is None

    def test_initialization_custom_params(self, mock_model):
        """Test initialization with custom parameters."""
        custom_history = LLMHistory().with_system_prompt("Custom prompt")
        ignore_mods = ["module1", "module2"]
        no_ignore_mods = ["important_module"]

        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="custom",
            description_text="Custom desc",
            history=custom_history,
            default_max_retries=10,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=False,
            ignore_modules=ignore_mods,
            no_ignore_modules=no_ignore_mods
        )

        assert task.code_name == "custom"
        assert task.description_text == "Custom desc"
        assert task.default_max_retries == 10
        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is False
        assert task.ignore_modules == ignore_mods
        assert task.no_ignore_modules == no_ignore_mods
        assert len(task.history) == 1

    @patch('hbllmutils.meta.code.task.get_prompt_for_source_file')
    def test_preprocess_input_content_valid_file(self, mock_get_prompt, mock_model, temporary_python_file):
        """Test preprocessing input content with valid file path."""
        mock_get_prompt.return_value = "Generated prompt"

        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="test",
            description_text="Test desc"
        )

        result = task._preprocess_input_content(temporary_python_file)

        assert result == "Generated prompt"
        mock_get_prompt.assert_called_once_with(
            source_file=temporary_python_file,
            level=1,
            code_name="test",
            description_text="Test desc",
            show_module_directory_tree=False,
            skip_when_error=True,
            ignore_modules=None,
            no_ignore_modules=None
        )

    @patch('hbllmutils.meta.code.task.get_prompt_for_source_file')
    def test_preprocess_input_content_with_all_options(self, mock_get_prompt, mock_model, temporary_python_file):
        """Test preprocessing with all options enabled."""
        mock_get_prompt.return_value = "Detailed prompt"
        ignore_mods = ["test_module"]
        no_ignore_mods = ["keep_module"]

        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="detailed",
            description_text="Detailed description",
            show_module_directory_tree=True,
            skip_when_error=False,
            ignore_modules=ignore_mods,
            no_ignore_modules=no_ignore_mods
        )

        result = task._preprocess_input_content(temporary_python_file)

        assert result == "Detailed prompt"
        mock_get_prompt.assert_called_once_with(
            source_file=temporary_python_file,
            level=1,
            code_name="detailed",
            description_text="Detailed description",
            show_module_directory_tree=True,
            skip_when_error=False,
            ignore_modules=ignore_mods,
            no_ignore_modules=no_ignore_mods
        )

    def test_preprocess_input_content_empty_raises_error(self, mock_model):
        """Test that empty input content raises ValueError."""
        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="test",
            description_text="Test"
        )

        with pytest.raises(ValueError, match="Empty content is not supported"):
            task._preprocess_input_content(None)

        with pytest.raises(ValueError, match="Empty content is not supported"):
            task._preprocess_input_content("")

    def test_inheritance_from_python_code_generation_task(self, mock_model):
        """Test that detailed task inherits from basic task."""
        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="test",
            description_text="Test"
        )

        assert isinstance(task, PythonCodeGenerationLLMTask)
        assert hasattr(task, '_parse_and_validate')
        assert hasattr(task, 'force_ast_check')

    def test_parse_and_validate_inherited(self, mock_model, sample_python_code):
        """Test that _parse_and_validate is inherited and works correctly."""
        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="test",
            description_text="Test",
            force_ast_check=True
        )

        result = task._parse_and_validate(sample_python_code)

        assert result == sample_python_code.rstrip()
        ast.parse(result)


@pytest.mark.unittest
class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_basic_code_generation_workflow(self, mock_model, sample_python_code):
        """Test complete workflow for basic code generation."""
        mock_model.ask.return_value = sample_python_code

        task = PythonCodeGenerationLLMTask(mock_model, default_max_retries=2)

        result = task.ask_then_parse(input_content="Generate a calculator")

        assert "def add" in result
        assert "def subtract" in result
        mock_model.ask.assert_called_once()

    def test_code_generation_with_retry(self, mock_model, sample_invalid_python_code, sample_python_code):
        """Test code generation with retry on invalid syntax."""
        # First call returns invalid code, second call returns valid code
        mock_model.ask.side_effect = [sample_invalid_python_code, sample_python_code]

        task = PythonCodeGenerationLLMTask(mock_model, default_max_retries=2)

        result = task.ask_then_parse(input_content="Generate code")

        assert "def add" in result
        assert mock_model.ask.call_count == 2

    def test_code_generation_max_retries_exceeded(self, mock_model, sample_invalid_python_code):
        """Test that max retries exceeded raises OutputParseFailed."""
        mock_model.ask.return_value = sample_invalid_python_code

        task = PythonCodeGenerationLLMTask(mock_model, default_max_retries=1)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="Generate invalid code")

        assert len(exc_info.value.tries) == 2  # initial + 1 retry
        assert mock_model.ask.call_count == 2

    @patch('hbllmutils.meta.code.task.get_prompt_for_source_file')
    def test_detailed_generation_workflow(self, mock_get_prompt, mock_model, temporary_python_file, sample_python_code):
        """Test complete workflow for detailed code generation."""
        mock_get_prompt.return_value = "# Detailed prompt\n\nGenerate tests"
        mock_model.ask.return_value = f"```python\n{sample_python_code}\n```"

        task = PythonDetailedCodeGenerationLLMTask(
            mock_model,
            code_name="calculator",
            description_text="Generate unit tests"
        )

        result = task.ask_then_parse(input_content=temporary_python_file)

        assert "def add" in result
        assert "def subtract" in result
        mock_get_prompt.assert_called_once()
        mock_model.ask.assert_called_once()

    def test_code_generation_without_ast_check(self, mock_model, sample_invalid_python_code):
        """Test code generation without AST validation."""
        mock_model.ask.return_value = sample_invalid_python_code

        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=False)

        # Should succeed even with invalid syntax
        result = task.ask_then_parse(input_content="Generate code")

        assert result == sample_invalid_python_code.rstrip()
        mock_model.ask.assert_called_once()

    def test_code_generation_with_markdown_fencing(self, mock_model):
        """Test code generation with markdown code fencing."""
        fenced_code = """```python
def greet(name):
    return f"Hello, {name}!"
```"""
        mock_model.ask.return_value = fenced_code

        task = PythonCodeGenerationLLMTask(mock_model)

        result = task.ask_then_parse(input_content="Generate greeting function")

        assert "def greet(name):" in result
        assert "```" not in result
        mock_model.ask.assert_called_once()


@pytest.mark.unittest
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_code_generation(self, mock_model):
        """Test handling of empty code response."""
        mock_model.ask.return_value = ""

        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=False)

        result = task.ask_then_parse(input_content="Generate nothing")

        assert result == ""

    def test_whitespace_only_code(self, mock_model):
        """Test handling of whitespace-only code response."""
        mock_model.ask.return_value = "   \n\n   "

        task = PythonCodeGenerationLLMTask(mock_model, force_ast_check=False)

        result = task.ask_then_parse(input_content="Generate whitespace")

        assert result == ""

    def test_code_with_unicode(self, mock_model):
        """Test handling of code with Unicode characters."""
        unicode_code = """def greet():
    return "Hello, 世界! 🌍"
"""
        mock_model.ask.return_value = unicode_code

        task = PythonCodeGenerationLLMTask(mock_model)

        result = task.ask_then_parse(input_content="Generate Unicode code")

        assert "世界" in result
        assert "🌍" in result

    def test_very_long_code(self, mock_model):
        """Test handling of very long code generation."""
        long_code = "\n".join([f"x{i} = {i}" for i in range(1000)])
        mock_model.ask.return_value = long_code

        task = PythonCodeGenerationLLMTask(mock_model)

        result = task.ask_then_parse(input_content="Generate long code")

        assert "x0 = 0" in result
        assert "x999 = 999" in result

    def test_code_with_special_characters(self, mock_model):
        """Test handling of code with special characters."""
        special_code = """def test():
    s = "Line with \\n newline and \\t tab"
    return s
"""
        mock_model.ask.return_value = special_code

        task = PythonCodeGenerationLLMTask(mock_model)

        result = task.ask_then_parse(input_content="Generate code with escapes")

        assert "\\n" in result
        assert "\\t" in result

    @pytest.mark.parametrize("max_retries", [0, 1, 5, 10, 100])
    def test_various_max_retries(self, mock_model, sample_python_code, max_retries):
        """Test with various max_retries values."""
        mock_model.ask.return_value = sample_python_code

        task = PythonCodeGenerationLLMTask(mock_model, default_max_retries=max_retries)

        result = task.ask_then_parse(input_content="Generate code")

        assert "def add" in result

    def test_none_history_initialization(self, mock_model):
        """Test initialization with None history creates new history."""
        task = PythonCodeGenerationLLMTask(mock_model, history=None)

        assert isinstance(task.history, LLMHistory)
        assert len(task.history) == 0
