"""
Unit tests for the hbllmutils.meta.code.task module.

This module provides comprehensive tests for Python code generation LLM tasks,
including basic code generation with AST validation and detailed code generation
with source file analysis.
"""

import ast
import os
import tempfile

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.meta.code.task import (
    PythonCodeGenerationLLMTask,
    PythonDetailedCodeGenerationLLMTask
)
from hbllmutils.model import FakeLLMModel
from hbllmutils.response import OutputParseFailed


@pytest.fixture
def valid_python_code():
    """Provide valid Python code for testing."""
    return """def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y"""


@pytest.fixture
def invalid_python_code():
    """Provide syntactically invalid Python code for testing."""
    return "def broken_function("


@pytest.fixture
def python_code_with_fencing():
    """Provide Python code wrapped in markdown fencing."""
    return """```python
def greet(name):
    return f"Hello, {name}!"
```"""


@pytest.fixture
def temporary_python_file(valid_python_code):
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write(valid_python_code)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def fake_model():
    """Create a basic fake LLM model for testing."""
    return FakeLLMModel(stream_wps=100)


@pytest.mark.unittest
class TestPythonCodeGenerationLLMTask:
    """Tests for the PythonCodeGenerationLLMTask class."""

    def test_initialization_defaults(self, fake_model):
        """Test that task initializes with default parameters."""
        task = PythonCodeGenerationLLMTask(fake_model)
        assert task.model == fake_model
        assert task.default_max_retries == 5
        assert task.force_ast_check is True
        assert len(task.history) == 0

    def test_initialization_custom_retries(self, fake_model):
        """Test initialization with custom max_retries."""
        task = PythonCodeGenerationLLMTask(fake_model, default_max_retries=10)
        assert task.default_max_retries == 10

    def test_initialization_with_history(self, fake_model):
        """Test initialization with existing history."""
        history = LLMHistory().with_system_prompt("You are a code generator.")
        task = PythonCodeGenerationLLMTask(fake_model, history=history)
        assert len(task.history) == 1
        assert task.history[0]['role'] == 'system'

    def test_initialization_without_ast_check(self, fake_model):
        """Test initialization with AST checking disabled."""
        task = PythonCodeGenerationLLMTask(fake_model, force_ast_check=False)
        assert task.force_ast_check is False

    def test_parse_and_validate_valid_code(self, fake_model, valid_python_code):
        """Test parsing and validation of valid Python code."""
        task = PythonCodeGenerationLLMTask(fake_model)
        result = task._parse_and_validate(valid_python_code)
        assert result == valid_python_code.rstrip()
        # Verify it's valid Python by parsing it
        ast.parse(result)

    def test_parse_and_validate_invalid_code(self, fake_model, invalid_python_code):
        """Test that invalid Python code raises SyntaxError."""
        task = PythonCodeGenerationLLMTask(fake_model, force_ast_check=True)
        with pytest.raises(SyntaxError):
            task._parse_and_validate(invalid_python_code)

    def test_parse_and_validate_code_with_fencing(self, fake_model, python_code_with_fencing):
        """Test parsing code from markdown fenced blocks."""
        task = PythonCodeGenerationLLMTask(fake_model)
        result = task._parse_and_validate(python_code_with_fencing)
        assert "def greet(name):" in result
        assert "```" not in result

    def test_parse_and_validate_without_ast_check(self, fake_model, invalid_python_code):
        """Test that invalid code passes when AST checking is disabled."""
        task = PythonCodeGenerationLLMTask(fake_model, force_ast_check=False)
        result = task._parse_and_validate(invalid_python_code)
        assert result == invalid_python_code.rstrip()

    def test_parse_and_validate_strips_trailing_whitespace(self, fake_model):
        """Test that trailing whitespace is removed from parsed code."""
        task = PythonCodeGenerationLLMTask(fake_model)
        code_with_whitespace = "x = 42\n\n\n"
        result = task._parse_and_validate(code_with_whitespace)
        assert result == "x = 42"

    def test_ask_then_parse_success(self, fake_model, valid_python_code):
        """Test successful code generation and parsing."""
        model = fake_model.response_always(valid_python_code)
        task = PythonCodeGenerationLLMTask(model)
        result = task.ask_then_parse(input_content="Write a calculator")
        assert "def add(a, b):" in result
        assert "class Calculator:" in result

    def test_ask_then_parse_with_fenced_response(self, fake_model, python_code_with_fencing):
        """Test parsing when model returns fenced code blocks."""
        model = fake_model.response_always(python_code_with_fencing)
        task = PythonCodeGenerationLLMTask(model)
        result = task.ask_then_parse(input_content="Write a greeting function")
        assert "def greet(name):" in result
        assert "```" not in result

    def test_ask_then_parse_retry_on_invalid_code(self, fake_model, invalid_python_code, valid_python_code):
        """Test that task retries when receiving invalid code."""
        model = fake_model.response_sequence([
            invalid_python_code,
            invalid_python_code,
            valid_python_code
        ])
        task = PythonCodeGenerationLLMTask(model, default_max_retries=5)
        result = task.ask_then_parse(input_content="Write code")
        assert "def add(a, b):" in result

    def test_ask_then_parse_max_retries_exceeded(self, fake_model, invalid_python_code):
        """Test that OutputParseFailed is raised when max retries exceeded."""
        model = fake_model.response_always(invalid_python_code)
        task = PythonCodeGenerationLLMTask(model, default_max_retries=2)
        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="Write code")
        assert len(exc_info.value.tries) == 3  # max_retries + 1

    def test_ask_then_parse_custom_max_retries(self, fake_model, invalid_python_code, valid_python_code):
        """Test using custom max_retries parameter."""
        model = fake_model.response_sequence([
            invalid_python_code,
            valid_python_code
        ])
        task = PythonCodeGenerationLLMTask(model, default_max_retries=5)
        result = task.ask_then_parse(input_content="Write code", max_retries=1)
        assert "def add(a, b):" in result

    def test_ask_then_parse_without_input_content(self, fake_model, valid_python_code):
        """Test asking without providing new input content."""
        model = fake_model.response_always(valid_python_code)
        history = LLMHistory().with_user_message("Generate a calculator")
        task = PythonCodeGenerationLLMTask(model, history=history)
        result = task.ask_then_parse()
        assert "def add(a, b):" in result

    @pytest.mark.parametrize("code,expected_valid", [
        ("x = 42", True),
        ("def foo(): pass", True),
        ("class Bar: pass", True),
        ("import os", True),
        ("def broken(", False),
        ("class Incomplete", False),
        ("if True", False),
    ])
    def test_parse_and_validate_various_code(self, fake_model, code, expected_valid):
        """Test parsing various code snippets."""
        task = PythonCodeGenerationLLMTask(fake_model, force_ast_check=True)
        if expected_valid:
            result = task._parse_and_validate(code)
            assert result == code.rstrip()
        else:
            with pytest.raises(SyntaxError):
                task._parse_and_validate(code)

    def test_exceptions_attribute(self):
        """Test that __exceptions__ is properly defined."""
        assert hasattr(PythonCodeGenerationLLMTask, '__exceptions__')
        assert PythonCodeGenerationLLMTask.__exceptions__ == (SyntaxError, ValueError)


@pytest.mark.unittest
class TestPythonDetailedCodeGenerationLLMTask:
    """Tests for the PythonDetailedCodeGenerationLLMTask class."""

    def test_initialization_required_parameters(self, fake_model):
        """Test initialization with required parameters."""
        task = PythonDetailedCodeGenerationLLMTask(
            model=fake_model,
            code_name="test_code",
            description_text="Test description"
        )
        assert task.code_name == "test_code"
        assert task.description_text == "Test description"
        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True

    def test_initialization_all_parameters(self, fake_model):
        """Test initialization with all parameters."""
        history = LLMHistory()
        task = PythonDetailedCodeGenerationLLMTask(
            model=fake_model,
            code_name="my_module",
            description_text="Generate tests",
            history=history,
            default_max_retries=10,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=False,
            ignore_modules=["numpy", "pandas"],
            no_ignore_modules=["mypackage"]
        )
        assert task.code_name == "my_module"
        assert task.description_text == "Generate tests"
        assert task.default_max_retries == 10
        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is False
        assert task.ignore_modules == ["numpy", "pandas"]
        assert task.no_ignore_modules == ["mypackage"]

    def test_preprocess_input_content_with_file(self, fake_model, temporary_python_file, valid_python_code):
        """Test preprocessing with a valid Python file."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="calculator",
            description_text="Generate unit tests"
        )
        result = task._preprocess_input_content(temporary_python_file)
        assert result is not None
        assert "Source File Location:" in result
        assert "Package Namespace:" in result
        assert "Complete Source Code:" in result
        assert temporary_python_file in result

    def test_preprocess_input_content_empty_raises_error(self, fake_model):
        """Test that empty content raises ValueError."""
        task = PythonDetailedCodeGenerationLLMTask(
            model=fake_model,
            code_name="test",
            description_text="Test"
        )
        with pytest.raises(ValueError, match="Empty content is not supported"):
            task._preprocess_input_content(None)
        with pytest.raises(ValueError, match="Empty content is not supported"):
            task._preprocess_input_content("")

    def test_preprocess_input_content_with_directory_tree(self, fake_model, temporary_python_file, valid_python_code):
        """Test preprocessing with directory tree enabled."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="module",
            description_text="Analyze code",
            show_module_directory_tree=True
        )
        result = task._preprocess_input_content(temporary_python_file)
        assert "Module directory tree:" in result

    def test_ask_then_parse_with_file(self, fake_model, temporary_python_file, valid_python_code):
        """Test complete workflow with file input."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="calculator",
            description_text="Generate comprehensive tests"
        )
        result = task.ask_then_parse(input_content=temporary_python_file)
        assert "def add(a, b):" in result
        assert "class Calculator:" in result

    def test_ask_then_parse_with_custom_retries(self, fake_model, temporary_python_file,
                                                invalid_python_code, valid_python_code):
        """Test retry mechanism with file input."""
        model = fake_model.response_sequence([
            invalid_python_code,
            valid_python_code
        ])
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="test",
            description_text="Test",
            default_max_retries=5
        )
        result = task.ask_then_parse(input_content=temporary_python_file, max_retries=2)
        assert "def add(a, b):" in result

    def test_code_name_none_uses_default(self, fake_model, temporary_python_file, valid_python_code):
        """Test that None code_name results in default title."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name=None,
            description_text="Test"
        )
        # This should work without error and use default title
        result = task._preprocess_input_content(temporary_python_file)
        assert "Source Code Analysis" in result

    def test_ignore_modules_parameter(self, fake_model, temporary_python_file, valid_python_code):
        """Test that ignore_modules parameter is stored correctly."""
        model = fake_model.response_always(valid_python_code)
        ignore_list = ["numpy", "pandas", "scipy"]
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="test",
            description_text="Test",
            ignore_modules=ignore_list
        )
        assert task.ignore_modules == ignore_list

    def test_no_ignore_modules_parameter(self, fake_model, temporary_python_file, valid_python_code):
        """Test that no_ignore_modules parameter is stored correctly."""
        model = fake_model.response_always(valid_python_code)
        no_ignore_list = ["mypackage", "mymodule"]
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="test",
            description_text="Test",
            no_ignore_modules=no_ignore_list
        )
        assert task.no_ignore_modules == no_ignore_list

    def test_inherits_from_python_code_generation_task(self, fake_model):
        """Test that class properly inherits from PythonCodeGenerationLLMTask."""
        task = PythonDetailedCodeGenerationLLMTask(
            model=fake_model,
            code_name="test",
            description_text="Test"
        )
        assert isinstance(task, PythonCodeGenerationLLMTask)

    def test_preprocess_maintains_immutability(self, fake_model, temporary_python_file, valid_python_code):
        """Test that preprocessing doesn't modify task state."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="test",
            description_text="Test"
        )
        original_code_name = task.code_name
        original_description = task.description_text

        task._preprocess_input_content(temporary_python_file)

        assert task.code_name == original_code_name
        assert task.description_text == original_description

    @pytest.mark.parametrize("show_tree,skip_error", [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ])
    def test_various_flag_combinations(self, fake_model, temporary_python_file,
                                       valid_python_code, show_tree, skip_error):
        """Test various combinations of boolean flags."""
        model = fake_model.response_always(valid_python_code)
        task = PythonDetailedCodeGenerationLLMTask(
            model=model,
            code_name="test",
            description_text="Test",
            show_module_directory_tree=show_tree,
            skip_when_error=skip_error
        )
        assert task.show_module_directory_tree == show_tree
        assert task.skip_when_error == skip_error

        # Should work without errors
        result = task._preprocess_input_content(temporary_python_file)
        assert result is not None
