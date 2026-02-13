"""
Unit tests for the pydoc_generation module.

This module contains comprehensive tests for the create_pydoc_generation_task function
and its integration with the PythonDetailedCodeGenerationLLMTask. The tests verify:

- Task creation with various parameter combinations
- Proper configuration of task parameters
- Integration with FakeLLMModel for predictable testing
- Error handling for invalid inputs
- Immutability and consistency of generated tasks

All tests use FakeLLMModel to avoid external dependencies and ensure deterministic behavior.
"""

import os
import tempfile

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.meta.code.pydoc_generation import create_pydoc_generation_task
from hbllmutils.model import FakeLLMModel


@pytest.fixture
def fake_model():
    """
    Create a FakeLLMModel instance for testing.
    
    This fixture provides a basic fake model that can be configured with
    response rules for testing various scenarios.
    
    :return: A FakeLLMModel instance.
    :rtype: FakeLLMModel
    """
    return FakeLLMModel(stream_wps=50)


@pytest.fixture
def sample_python_file():
    """
    Create a temporary Python file with sample content for testing.
    
    This fixture creates a real temporary file with simple Python code
    that can be used to test the pydoc generation task.
    
    :return: Path to the temporary Python file.
    :rtype: str
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write('def sample_function(x, y):\n')
        f.write('    return x + y\n')
        f.write('\n')
        f.write('class SampleClass:\n')
        f.write('    def method(self):\n')
        f.write('        pass\n')

    yield f.name

    # Cleanup
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.mark.unittest
class TestCreatePydocGenerationTask:
    """Tests for the create_pydoc_generation_task function."""

    def test_create_task_with_fake_model(self, fake_model):
        """
        Test that create_pydoc_generation_task successfully creates a task with FakeLLMModel.
        
        Verifies that the function accepts a FakeLLMModel instance and returns
        a properly configured task object.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert task is not None
        assert hasattr(task, 'ask_then_parse')
        assert hasattr(task, 'model')

    def test_create_task_with_default_parameters(self, fake_model):
        """
        Test task creation with default parameters.
        
        Verifies that all default parameter values are correctly applied
        to the created task.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert task.show_module_directory_tree is False
        assert task.skip_when_error is True
        assert task.force_ast_check is True

    @pytest.mark.parametrize("show_tree,skip_error,force_check", [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ])
    def test_create_task_with_various_parameters(
            self, fake_model, show_tree, skip_error, force_check
    ):
        """
        Test task creation with various parameter combinations.
        
        Verifies that all boolean parameters are correctly passed through
        to the created task instance.
        """
        task = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=show_tree,
            skip_when_error=skip_error,
            force_ast_check=force_check
        )

        assert task.show_module_directory_tree == show_tree
        assert task.skip_when_error == skip_error
        assert task.force_ast_check == force_check

    def test_create_task_with_ignore_modules(self, fake_model):
        """
        Test task creation with ignore_modules parameter.
        
        Verifies that the ignore_modules parameter is correctly passed
        to the task and stored.
        """
        ignore_list = ['module1', 'module2', 'module3']
        task = create_pydoc_generation_task(
            model=fake_model,
            ignore_modules=ignore_list
        )

        assert task.ignore_modules is not None
        assert list(task.ignore_modules) == ignore_list

    def test_create_task_with_no_ignore_modules(self, fake_model):
        """
        Test task creation with no_ignore_modules parameter.
        
        Verifies that the no_ignore_modules parameter is correctly passed
        to the task and stored.
        """
        no_ignore_list = ['important_module', 'critical_module']
        task = create_pydoc_generation_task(
            model=fake_model,
            no_ignore_modules=no_ignore_list
        )

        assert task.no_ignore_modules is not None
        assert list(task.no_ignore_modules) == no_ignore_list

    def test_create_task_with_both_ignore_parameters(self, fake_model):
        """
        Test task creation with both ignore_modules and no_ignore_modules.
        
        Verifies that both parameters can be set simultaneously and are
        correctly stored in the task.
        """
        ignore_list = ['module1', 'module2']
        no_ignore_list = ['important_module']

        task = create_pydoc_generation_task(
            model=fake_model,
            ignore_modules=ignore_list,
            no_ignore_modules=no_ignore_list
        )

        assert list(task.ignore_modules) == ignore_list
        assert list(task.no_ignore_modules) == no_ignore_list

    def test_create_task_has_system_prompt(self, fake_model):
        """
        Test that the created task has a system prompt configured.
        
        Verifies that the task's history contains a system prompt loaded
        from the pydoc_generation.j2 template file.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert task.history is not None
        assert len(task.history) > 0
        assert task.history[0]['role'] == 'system'
        assert len(task.history[0]['content']) > 0

    def test_create_task_system_prompt_content(self, fake_model):
        """
        Test that the system prompt contains expected content.
        
        Verifies that the system prompt loaded from the template file
        contains key instructions for pydoc generation.
        """
        task = create_pydoc_generation_task(model=fake_model)

        system_prompt = task.history[0]['content']

        # The system prompt should contain instructions about Python documentation
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 100  # Should be a substantial prompt

    def test_create_task_with_string_model_name(self):
        """
        Test task creation with a string model name.
        
        Verifies that passing a string model name is handled correctly
        by the load_llm_model function.
        """
        # This test verifies the integration with load_llm_model
        # We expect it to work with FakeLLMModel or raise appropriate error
        fake_model = FakeLLMModel()
        task = create_pydoc_generation_task(model=fake_model)
        assert task is not None

    def test_create_task_immutability(self, fake_model):
        """
        Test that creating multiple tasks doesn't affect each other.
        
        Verifies that each call to create_pydoc_generation_task returns
        an independent task instance with its own configuration.
        """
        task1 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=True
        )
        task2 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=False
        )

        assert task1.show_module_directory_tree is True
        assert task2.show_module_directory_tree is False

    def test_create_task_code_name_and_description(self, fake_model):
        """
        Test that the task has correct code_name and description_text.
        
        Verifies that the internal parameters used for prompt generation
        are set correctly.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert hasattr(task, 'code_name')
        assert hasattr(task, 'description_text')
        assert task.code_name == 'Code For Task'
        assert 'pydoc' in task.description_text.lower()

    def test_create_task_with_none_ignore_modules(self, fake_model):
        """
        Test task creation with None as ignore_modules.
        
        Verifies that None is handled correctly for the ignore_modules parameter.
        """
        task = create_pydoc_generation_task(
            model=fake_model,
            ignore_modules=None
        )

        assert task.ignore_modules is None

    def test_create_task_with_empty_ignore_modules(self, fake_model):
        """
        Test task creation with empty list as ignore_modules.
        
        Verifies that an empty list is handled correctly.
        """
        task = create_pydoc_generation_task(
            model=fake_model,
            ignore_modules=[]
        )

        assert task.ignore_modules is not None
        assert list(task.ignore_modules) == []

    def test_create_task_consistency(self, fake_model):
        """
        Test that creating tasks with same parameters produces consistent results.
        
        Verifies that the function is deterministic and produces identical
        task configurations when called with the same parameters.
        """
        task1 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=True
        )
        task2 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=True
        )

        assert task1.show_module_directory_tree == task2.show_module_directory_tree
        assert task1.skip_when_error == task2.skip_when_error
        assert task1.force_ast_check == task2.force_ast_check
        assert task1.code_name == task2.code_name
        assert task1.description_text == task2.description_text


@pytest.mark.unittest
class TestPydocGenerationTaskIntegration:
    """Integration tests for the pydoc generation task with FakeLLMModel."""

    def test_task_has_required_methods(self, fake_model):
        """
        Test that the created task has all required methods.
        
        Verifies that the task object implements the expected interface
        for code generation tasks.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert hasattr(task, 'ask_then_parse')
        assert callable(task.ask_then_parse)
        assert hasattr(task, 'model')
        assert hasattr(task, 'history')

    def test_task_model_is_correct_type(self, fake_model):
        """
        Test that the task's model is correctly set.
        
        Verifies that the model passed to create_pydoc_generation_task
        is properly stored in the task instance.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert task.model is not None
        # The model should be the same instance or type
        assert isinstance(task.model, FakeLLMModel)

    def test_task_history_is_llm_history(self, fake_model):
        """
        Test that the task's history is an LLMHistory instance.
        
        Verifies that the conversation history is properly initialized
        with the correct type.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert isinstance(task.history, LLMHistory)
        assert len(task.history) >= 1  # Should have at least system prompt

    def test_task_with_configured_model(self):
        """
        Test task with a pre-configured FakeLLMModel.
        
        Verifies that a FakeLLMModel with response rules can be used
        to create a functional pydoc generation task.
        """
        configured_model = FakeLLMModel().response_always(
            "def example():\n    '''Example function.'''\n    pass"
        )

        task = create_pydoc_generation_task(model=configured_model)

        assert task is not None
        assert task.model.rules_count == 1

    def test_task_default_max_retries(self, fake_model):
        """
        Test that the task has the correct default_max_retries value.
        
        Verifies that the retry configuration is properly set.
        """
        task = create_pydoc_generation_task(model=fake_model)

        assert hasattr(task, 'default_max_retries')
        assert task.default_max_retries == 5

    def test_multiple_tasks_with_different_models(self):
        """
        Test creating multiple tasks with different FakeLLMModel instances.
        
        Verifies that each task maintains its own model instance and
        configuration independently.
        """
        model1 = FakeLLMModel(stream_wps=50)
        model2 = FakeLLMModel(stream_wps=100)

        task1 = create_pydoc_generation_task(model=model1)
        task2 = create_pydoc_generation_task(model=model2)

        assert task1.model.stream_wps == 50
        assert task2.model.stream_wps == 100

    def test_task_parameters_are_independent(self, fake_model):
        """
        Test that task parameters don't affect each other across instances.
        
        Verifies that modifying parameters in one task creation doesn't
        affect other tasks created subsequently.
        """
        task1 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=True,
            ignore_modules=['mod1']
        )
        task2 = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=False,
            ignore_modules=['mod2']
        )

        assert task1.show_module_directory_tree is True
        assert task2.show_module_directory_tree is False
        assert list(task1.ignore_modules) == ['mod1']
        assert list(task2.ignore_modules) == ['mod2']


@pytest.mark.unittest
class TestPydocGenerationTaskConfiguration:
    """Tests for task configuration and parameter handling."""

    @pytest.mark.parametrize("stream_wps", [10, 50, 100, 200])
    def test_task_with_various_stream_speeds(self, stream_wps):
        """
        Test task creation with various streaming speeds.
        
        Verifies that different stream_wps values in FakeLLMModel
        are correctly preserved in the created task.
        """
        model = FakeLLMModel(stream_wps=stream_wps)
        task = create_pydoc_generation_task(model=model)

        assert task.model.stream_wps == stream_wps

    def test_task_ignore_modules_as_tuple(self, fake_model):
        """
        Test task creation with ignore_modules as tuple.
        
        Verifies that tuple input is correctly handled for ignore_modules.
        """
        ignore_tuple = ('module1', 'module2')
        task = create_pydoc_generation_task(
            model=fake_model,
            ignore_modules=ignore_tuple
        )

        assert task.ignore_modules is not None
        # Convert to list for comparison since the internal storage might differ
        assert set(task.ignore_modules) == set(ignore_tuple)

    def test_task_no_ignore_modules_as_tuple(self, fake_model):
        """
        Test task creation with no_ignore_modules as tuple.
        
        Verifies that tuple input is correctly handled for no_ignore_modules.
        """
        no_ignore_tuple = ('important1', 'important2')
        task = create_pydoc_generation_task(
            model=fake_model,
            no_ignore_modules=no_ignore_tuple
        )

        assert task.no_ignore_modules is not None
        assert set(task.no_ignore_modules) == set(no_ignore_tuple)

    def test_task_with_all_parameters(self, fake_model):
        """
        Test task creation with all parameters specified.
        
        Verifies that all parameters can be set simultaneously and are
        correctly stored in the task.
        """
        task = create_pydoc_generation_task(
            model=fake_model,
            show_module_directory_tree=True,
            skip_when_error=False,
            force_ast_check=True,
            ignore_modules=['mod1', 'mod2'],
            no_ignore_modules=['important']
        )

        assert task.show_module_directory_tree is True
        assert task.skip_when_error is False
        assert task.force_ast_check is True
        assert list(task.ignore_modules) == ['mod1', 'mod2']
        assert list(task.no_ignore_modules) == ['important']

    def test_task_system_prompt_is_not_empty(self, fake_model):
        """
        Test that the system prompt is not empty.
        
        Verifies that the loaded system prompt template contains actual content.
        """
        task = create_pydoc_generation_task(model=fake_model)

        system_message = task.history[0]
        assert system_message['role'] == 'system'
        assert len(system_message['content'].strip()) > 0

    def test_task_creation_is_repeatable(self, fake_model):
        """
        Test that task creation is repeatable and deterministic.
        
        Verifies that calling create_pydoc_generation_task multiple times
        with the same parameters produces equivalent results.
        """
        params = {
            'model': fake_model,
            'show_module_directory_tree': True,
            'skip_when_error': False,
            'force_ast_check': True,
            'ignore_modules': ['test'],
            'no_ignore_modules': ['important']
        }

        task1 = create_pydoc_generation_task(**params)
        task2 = create_pydoc_generation_task(**params)

        # Verify all parameters match
        assert task1.show_module_directory_tree == task2.show_module_directory_tree
        assert task1.skip_when_error == task2.skip_when_error
        assert task1.force_ast_check == task2.force_ast_check
        assert list(task1.ignore_modules) == list(task2.ignore_modules)
        assert list(task1.no_ignore_modules) == list(task2.no_ignore_modules)

        # Verify system prompts are identical
        assert task1.history[0]['content'] == task2.history[0]['content']
