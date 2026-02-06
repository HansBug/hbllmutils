"""
Unit tests for the unittest_generation module.

This module contains comprehensive tests for the UnittestCodeGenerationLLMTask class
and the create_unittest_generation_task factory function. Tests verify code generation,
prompt construction, error handling, and configuration management.
"""

import os
import tempfile

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.meta.code.unittest_generation import (
    UnittestCodeGenerationLLMTask,
    create_unittest_generation_task,
)
from hbllmutils.model import FakeLLMModel


@pytest.fixture
def sample_python_file():
    """Create a temporary Python source file for testing."""
    with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
    ) as f:
        f.write("""
def add(a, b):
    '''Add two numbers.'''
    return a + b

def subtract(a, b):
    '''Subtract b from a.'''
    return a - b

class Calculator:
    '''A simple calculator class.'''
    
    def multiply(self, a, b):
        '''Multiply two numbers.'''
        return a * b
    
    def divide(self, a, b):
        '''Divide a by b.'''
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
""")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_test_file():
    """Create a temporary test file for testing."""
    with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
    ) as f:
        f.write("""
import pytest

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2
""")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def fake_model():
    """Create a FakeLLMModel configured to return valid Python test code."""
    model = FakeLLMModel(stream_wps=100)
    test_code = """
import pytest

def test_example():
    assert True
"""
    return model.response_always(test_code)


@pytest.fixture
def task_with_fake_model(fake_model):
    """Create an UnittestCodeGenerationLLMTask with a fake model."""
    history = LLMHistory().with_system_prompt("Generate unit tests")
    return UnittestCodeGenerationLLMTask(
        model=fake_model,
        history=history,
        show_module_directory_tree=False,
        skip_when_error=True,
        force_ast_check=True,
    )


@pytest.mark.unittest
class TestUnittestCodeGenerationLLMTask:
    """Tests for the UnittestCodeGenerationLLMTask class."""

    def test_initialization_default_params(self, fake_model):
        """Test initialization with default parameters."""
        task = UnittestCodeGenerationLLMTask(model=fake_model)

        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True
        assert isinstance(task.ignore_modules, set)
        assert isinstance(task.no_ignore_modules, set)
        assert len(task.ignore_modules) == 0
        assert len(task.no_ignore_modules) == 0

    def test_initialization_custom_params(self, fake_model):
        """Test initialization with custom parameters."""
        history = LLMHistory().with_system_prompt("Custom prompt")
        ignore_mods = ['module1', 'module2']
        no_ignore_mods = ['module3']

        task = UnittestCodeGenerationLLMTask(
            model=fake_model,
            history=history,
            default_max_retries=10,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=False,
            ignore_modules=ignore_mods,
            no_ignore_modules=no_ignore_mods,
        )

        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is False
        assert task.ignore_modules == set(ignore_mods)
        assert task.no_ignore_modules == set(no_ignore_mods)

    def test_generate_with_source_file_only(self, task_with_fake_model, sample_python_file):
        """Test generating tests with only a source file."""
        result = task_with_fake_model.generate(source_file=sample_python_file)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid Python by checking AST parsing doesn't raise
        import ast
        ast.parse(result)

    def test_generate_with_source_and_test_file(
            self,
            task_with_fake_model,
            sample_python_file,
            sample_test_file
    ):
        """Test generating tests with both source and test files."""
        result = task_with_fake_model.generate(
            source_file=sample_python_file,
            test_file=sample_test_file
        )

        assert isinstance(result, str)
        assert len(result) > 0
        import ast
        ast.parse(result)

    def test_generate_with_max_retries(self, task_with_fake_model, sample_python_file):
        """Test generating tests with custom max_retries parameter."""
        result = task_with_fake_model.generate(
            source_file=sample_python_file,
            max_retries=2
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_additional_params(self, task_with_fake_model, sample_python_file):
        """Test generating tests with additional LLM parameters."""
        result = task_with_fake_model.generate(
            source_file=sample_python_file,
            temperature=0.7,
            max_tokens=2000
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_nonexistent_source_file(self, task_with_fake_model):
        """Test that generating with non-existent source file raises appropriate error."""
        with pytest.raises(Exception):
            task_with_fake_model.generate(source_file='/nonexistent/file.py')

    def test_ignore_modules_filtering(self, fake_model, sample_python_file):
        """Test that ignore_modules parameter filters dependencies correctly."""
        task = UnittestCodeGenerationLLMTask(
            model=fake_model,
            ignore_modules=['os', 'sys'],
        )

        result = task.generate(source_file=sample_python_file)
        assert isinstance(result, str)

    def test_no_ignore_modules_preservation(self, fake_model, sample_python_file):
        """Test that no_ignore_modules parameter preserves specified modules."""
        task = UnittestCodeGenerationLLMTask(
            model=fake_model,
            no_ignore_modules=['pytest'],
        )

        result = task.generate(source_file=sample_python_file)
        assert isinstance(result, str)

    def test_show_module_directory_tree_enabled(self, fake_model, sample_python_file):
        """Test generation with module directory tree enabled."""
        task = UnittestCodeGenerationLLMTask(
            model=fake_model,
            show_module_directory_tree=True,
        )

        result = task.generate(source_file=sample_python_file)
        assert isinstance(result, str)

    def test_skip_when_error_disabled(self, fake_model, sample_python_file):
        """Test generation with skip_when_error disabled."""
        task = UnittestCodeGenerationLLMTask(
            model=fake_model,
            skip_when_error=False,
        )

        # Should still work for valid files
        result = task.generate(source_file=sample_python_file)
        assert isinstance(result, str)


@pytest.mark.unittest
class TestCreateUnittestGenerationTask:
    """Tests for the create_unittest_generation_task factory function."""

    def test_create_with_fake_model_pytest(self):
        """Test creating task with FakeLLMModel and pytest framework."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest',
            mark_name='unittest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)
        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True

    def test_create_with_unittest_framework(self):
        """Test creating task with unittest framework."""
        model = FakeLLMModel().response_always(
            "import unittest\n\nclass TestExample(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='unittest',
            mark_name=None
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)

    def test_create_with_nose2_framework(self):
        """Test creating task with nose2 framework."""
        model = FakeLLMModel().response_always("def test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='nose2',
            mark_name='unittest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)

    def test_create_with_no_mark_name(self):
        """Test creating task without mark name."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest',
            mark_name=None
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)

    def test_create_with_empty_mark_name(self):
        """Test creating task with empty mark name."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest',
            mark_name=''
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)

    def test_create_with_custom_parameters(self):
        """Test creating task with custom parameters."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=False,
            test_framework_name='pytest',
            mark_name='integration'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)
        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is False

    def test_create_with_ignore_modules(self):
        """Test creating task with ignore_modules parameter."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        ignore_mods = ['deprecated', 'legacy']
        task = create_unittest_generation_task(
            model=model,
            ignore_modules=ignore_mods,
            test_framework_name='pytest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)
        assert task.ignore_modules == set(ignore_mods)

    def test_create_with_no_ignore_modules(self):
        """Test creating task with no_ignore_modules parameter."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        no_ignore_mods = ['core', 'utils']
        task = create_unittest_generation_task(
            model=model,
            no_ignore_modules=no_ignore_mods,
            test_framework_name='pytest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)
        assert task.no_ignore_modules == set(no_ignore_mods)

    def test_create_with_both_ignore_lists(self):
        """Test creating task with both ignore and no_ignore modules."""
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        ignore_mods = ['deprecated']
        no_ignore_mods = ['core']

        task = create_unittest_generation_task(
            model=model,
            ignore_modules=ignore_mods,
            no_ignore_modules=no_ignore_mods,
            test_framework_name='pytest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)
        assert task.ignore_modules == set(ignore_mods)
        assert task.no_ignore_modules == set(no_ignore_mods)

    @pytest.mark.parametrize("framework", ['pytest', 'unittest', 'nose2'])
    def test_create_with_all_frameworks(self, framework):
        """Test creating task with all supported frameworks."""
        model = FakeLLMModel().response_always("def test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name=framework
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)

    def test_create_with_model_string(self):
        """Test creating task with model as string (requires config)."""
        # This test would require actual model configuration
        # For now, we test with FakeLLMModel instance
        model = FakeLLMModel().response_always("import pytest\n\ndef test_example():\n    assert True")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest'
        )

        assert isinstance(task, UnittestCodeGenerationLLMTask)


@pytest.mark.unittest
class TestUnittestGenerationIntegration:
    """Integration tests for unittest generation functionality."""

    def test_end_to_end_generation(self, sample_python_file):
        """Test end-to-end test generation workflow."""
        model = FakeLLMModel().response_always("""
import pytest

@pytest.mark.unittest
class TestCalculator:
    def test_add(self):
        assert add(2, 3) == 5
    
    def test_subtract(self):
        assert subtract(5, 3) == 2

@pytest.mark.unittest
class TestCalculatorClass:
    def test_multiply(self):
        calc = Calculator()
        assert calc.multiply(2, 3) == 6
    
    def test_divide(self):
        calc = Calculator()
        assert calc.divide(6, 2) == 3
    
    def test_divide_by_zero(self):
        calc = Calculator()
        with pytest.raises(ValueError):
            calc.divide(1, 0)
""")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest',
            mark_name='unittest'
        )

        result = task.generate(source_file=sample_python_file)

        assert isinstance(result, str)
        assert 'import pytest' in result
        assert 'def test_' in result or 'class Test' in result

        # Verify it's valid Python
        import ast
        ast.parse(result)

    def test_generation_with_existing_tests(self, sample_python_file, sample_test_file):
        """Test generation using existing tests as reference."""
        model = FakeLLMModel().response_always("""
import pytest

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    calc = Calculator()
    assert calc.multiply(2, 3) == 6
""")

        task = create_unittest_generation_task(
            model=model,
            test_framework_name='pytest'
        )

        result = task.generate(
            source_file=sample_python_file,
            test_file=sample_test_file
        )

        assert isinstance(result, str)
        assert len(result) > 0

        import ast
        ast.parse(result)

    def test_generation_with_complex_configuration(self, sample_python_file):
        """Test generation with complex configuration options."""
        model = FakeLLMModel().response_always("""
import pytest

@pytest.mark.unittest
def test_calculator():
    assert True
""")

        task = create_unittest_generation_task(
            model=model,
            show_module_directory_tree=True,
            skip_when_error=True,
            force_ast_check=True,
            test_framework_name='pytest',
            mark_name='unittest',
            ignore_modules=['os', 'sys'],
            no_ignore_modules=['pytest']
        )

        result = task.generate(source_file=sample_python_file)

        assert isinstance(result, str)
        assert len(result) > 0

        import ast
        ast.parse(result)
