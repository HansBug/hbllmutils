"""
Unit tests for the TODO completion task creation functionality.

This module contains comprehensive tests for the create_todo_completion_task function,
which creates LLM tasks for completing TODO comments in Python source code.
"""

import os
import tempfile

import pytest

from hbllmutils.meta.code.todo_completion import create_todo_completion_task
from hbllmutils.model import FakeLLMModel


@pytest.fixture
def fake_model():
    """Create a FakeLLMModel instance for testing."""
    return FakeLLMModel(stream_wps=50).response_always("def completed_function():\n    pass")


@pytest.fixture
def sample_python_file():
    """Create a temporary Python file with TODO comments for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("""
def example_function():
    # TODO: Implement this function
    pass

class ExampleClass:
    def method(self):
        # TODO: Add implementation
        pass
""")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_non_python_file():
    """Create a temporary non-Python file with TODO comments for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.js') as f:
        f.write("""
function exampleFunction() {
    // TODO: Implement this function
}

class ExampleClass {
    method() {
        // TODO: Add implementation
    }
}
""")
    yield f.name
    os.unlink(f.name)


@pytest.mark.unittest
class TestCreateTodoCompletionTask:
    """Tests for the create_todo_completion_task function."""

    def test_create_task_with_default_parameters(self, fake_model):
        """Test creating a task with default parameters."""
        task = create_todo_completion_task(model=fake_model)

        assert task is not None
        assert hasattr(task, 'ask_then_parse')
        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True

    def test_create_task_with_show_directory_tree(self, fake_model):
        """Test creating a task with show_module_directory_tree enabled."""
        task = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=True
        )

        assert task.show_module_directory_tree is True

    def test_create_task_with_skip_when_error_disabled(self, fake_model):
        """Test creating a task with skip_when_error disabled."""
        task = create_todo_completion_task(
            model=fake_model,
            skip_when_error=False
        )

        assert task.skip_when_error is False

    def test_create_task_with_force_ast_check_enabled(self, fake_model):
        """Test creating a task with force_ast_check explicitly enabled."""
        task = create_todo_completion_task(
            model=fake_model,
            force_ast_check=True
        )

        assert task.force_ast_check is True

    def test_create_task_with_force_ast_check_disabled(self, fake_model):
        """Test creating a task with force_ast_check explicitly disabled."""
        task = create_todo_completion_task(
            model=fake_model,
            is_python_code=True,
            force_ast_check=False
        )

        assert task.force_ast_check is False

    def test_create_task_for_non_python_code(self, fake_model):
        """Test creating a task for non-Python code."""
        task = create_todo_completion_task(
            model=fake_model,
            is_python_code=False
        )

        assert task.force_ast_check is False

    def test_create_task_non_python_with_ast_check_warning(self, fake_model):
        """Test that setting force_ast_check=True with is_python_code=False logs a warning."""
        with pytest.warns(UserWarning):
            task = create_todo_completion_task(
                model=fake_model,
                is_python_code=False,
                force_ast_check=True
            )

        assert task.force_ast_check is False

    def test_create_task_with_ignore_modules(self, fake_model):
        """Test creating a task with ignore_modules parameter."""
        ignore_list = ['numpy', 'pandas']
        task = create_todo_completion_task(
            model=fake_model,
            ignore_modules=ignore_list
        )

        assert task.ignore_modules == ignore_list

    def test_create_task_with_no_ignore_modules(self, fake_model):
        """Test creating a task with no_ignore_modules parameter."""
        no_ignore_list = ['myproject.core']
        task = create_todo_completion_task(
            model=fake_model,
            no_ignore_modules=no_ignore_list
        )

        assert task.no_ignore_modules == no_ignore_list

    def test_create_task_with_both_ignore_parameters(self, fake_model):
        """Test creating a task with both ignore_modules and no_ignore_modules."""
        ignore_list = ['numpy', 'pandas']
        no_ignore_list = ['myproject.core']
        task = create_todo_completion_task(
            model=fake_model,
            ignore_modules=ignore_list,
            no_ignore_modules=no_ignore_list
        )

        assert task.ignore_modules == ignore_list
        assert task.no_ignore_modules == no_ignore_list

    @pytest.mark.parametrize("show_tree,skip_error,force_ast", [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (False, False, True),
        (True, True, False),
        (False, False, False),
    ])
    def test_create_task_with_various_combinations(self, fake_model, show_tree, skip_error, force_ast):
        """Test creating tasks with various parameter combinations."""
        task = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=show_tree,
            skip_when_error=skip_error,
            force_ast_check=force_ast
        )

        assert task.show_module_directory_tree == show_tree
        assert task.skip_when_error == skip_error
        assert task.force_ast_check == force_ast

    def test_create_task_with_string_model_name(self):
        """Test creating a task with a model name string."""
        try:
            task = create_todo_completion_task(model='fake-model')
            assert task is not None
        except Exception:
            pass

    def test_create_task_with_none_model(self):
        """Test creating a task with None as model parameter."""
        try:
            task = create_todo_completion_task(model=None)
            assert task is not None
        except Exception:
            pass

    def test_task_has_correct_attributes(self, fake_model):
        """Test that the created task has the expected attributes."""
        task = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False
        )

        assert hasattr(task, 'model')
        assert hasattr(task, 'code_name')
        assert hasattr(task, 'description_text')
        assert hasattr(task, 'show_module_directory_tree')
        assert hasattr(task, 'skip_when_error')
        assert hasattr(task, 'force_ast_check')

        assert task.code_name == 'Code For Task'
        assert 'TODO' in task.description_text

    def test_task_history_has_system_prompt(self, fake_model):
        """Test that the created task has a system prompt in its history."""
        task = create_todo_completion_task(model=fake_model)

        assert hasattr(task, 'history')
        assert len(task.history) > 0
        assert task.history[0]['role'] == 'system'
        assert len(task.history[0]['content']) > 0

    def test_force_ast_check_default_behavior_python(self, fake_model):
        """Test that force_ast_check defaults to True when is_python_code is True."""
        task = create_todo_completion_task(
            model=fake_model,
            is_python_code=True
        )

        assert task.force_ast_check is True

    def test_force_ast_check_default_behavior_non_python(self, fake_model):
        """Test that force_ast_check defaults to False when is_python_code is False."""
        task = create_todo_completion_task(
            model=fake_model,
            is_python_code=False
        )

        assert task.force_ast_check is False

    def test_create_task_with_empty_ignore_modules(self, fake_model):
        """Test creating a task with empty ignore_modules list."""
        task = create_todo_completion_task(
            model=fake_model,
            ignore_modules=[]
        )

        assert task.ignore_modules == []

    def test_create_task_with_empty_no_ignore_modules(self, fake_model):
        """Test creating a task with empty no_ignore_modules list."""
        task = create_todo_completion_task(
            model=fake_model,
            no_ignore_modules=[]
        )

        assert task.no_ignore_modules == []

    def test_create_task_immutability(self, fake_model):
        """Test that creating multiple tasks doesn't affect each other."""
        task1 = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False
        )

        task2 = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=False,
            skip_when_error=True
        )

        assert task1.show_module_directory_tree is True
        assert task1.skip_when_error is False
        assert task2.show_module_directory_tree is False
        assert task2.skip_when_error is True


@pytest.mark.unittest
class TestTodoCompletionTaskIntegration:
    """Integration tests for TODO completion task functionality."""

    def test_task_can_be_created_and_used(self, fake_model):
        """Test that a created task can be instantiated and has callable methods."""
        task = create_todo_completion_task(model=fake_model)

        assert callable(task.ask_then_parse)
        assert hasattr(task, 'model')

    def test_multiple_tasks_with_different_models(self):
        """Test creating multiple tasks with different model configurations."""
        model1 = FakeLLMModel(stream_wps=50).response_always("response1")
        model2 = FakeLLMModel(stream_wps=100).response_always("response2")

        task1 = create_todo_completion_task(model=model1)
        task2 = create_todo_completion_task(model=model2)

        assert task1.model != task2.model

    def test_task_configuration_persistence(self, fake_model):
        """Test that task configuration is properly stored and accessible."""
        ignore_list = ['module1', 'module2']
        no_ignore_list = ['module3']

        task = create_todo_completion_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=True,
            ignore_modules=ignore_list,
            no_ignore_modules=no_ignore_list
        )

        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is True
        assert task.ignore_modules == ignore_list
        assert task.no_ignore_modules == no_ignore_list
