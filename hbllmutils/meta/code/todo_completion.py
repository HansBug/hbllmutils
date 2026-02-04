"""
TODO completion task creation utilities for Python code generation.

This module provides functionality for creating LLM tasks that automatically
complete TODO comments in Python source code. It leverages detailed code analysis
and LLM capabilities to generate contextually appropriate code completions.

The module contains the following main components:

* :func:`create_todo_completion_task` - Factory function for creating TODO completion tasks

.. note::
   This module requires a valid LLM model configuration to function properly.
   The TODO completion prompt template is loaded from 'todo-completion-req.md'.

Example::

    >>> from hbllmutils.meta.code.todo_completion import create_todo_completion_task
    >>> from hbllmutils.model import load_llm_model
    >>> 
    >>> # Create a TODO completion task
    >>> model = load_llm_model('gpt-4')
    >>> task = create_todo_completion_task(
    ...     model=model,
    ...     show_module_directory_tree=True,
    ...     skip_when_error=False
    ... )
    >>> 
    >>> # Use the task to complete TODOs in a source file
    >>> completed_code = task.ask_then_parse(input_content='path/to/source.py')
    >>> print(completed_code)

"""

import os

from .task import PythonDetailedCodeGenerationLLMTask, PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model
from ...template import PromptTemplate


def create_todo_completion_task(model: LLMModelTyping, show_module_directory_tree: bool = False,
                                skip_when_error: bool = True,
                                force_ast_check: bool = True) -> PythonCodeGenerationLLMTask:
    """
    Create a configured LLM task for completing TODO comments in Python code.

    This factory function creates a specialized :class:`PythonDetailedCodeGenerationLLMTask`
    configured specifically for TODO completion. It loads the TODO completion prompt
    template from the module's resource files and initializes the task with appropriate
    settings for code generation.

    The created task will:
    
    - Analyze the source file and its dependencies
    - Identify TODO comments in the code
    - Generate appropriate code completions using the LLM
    - Validate the generated code for Python syntax correctness
    - Optionally display the module directory tree for context
    - Handle import errors gracefully based on skip_when_error setting

    The task uses a pre-defined system prompt from 'todo-completion-req.md' that
    instructs the LLM on how to properly complete TODO comments while maintaining
    code quality, consistency, and adherence to existing patterns in the codebase.

    :param model: The LLM model to use for code generation. Can be:
        - A string representing the model name (e.g., 'gpt-4', 'claude-3')
        - An LLMModel instance for direct use
        - None to use the default model from configuration
    :type model: LLMModelTyping
    :param show_module_directory_tree: If True, include a directory tree visualization
        of the module structure in the analysis prompt. This provides additional
        context about the project structure to help the LLM understand the codebase
        organization. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load during dependency
        analysis and issue warnings instead of raising exceptions. This allows the
        task to proceed even when some dependencies are unavailable. Defaults to True.
    :type skip_when_error: bool
    :param force_ast_check: If True, enforce Python AST validation on the generated
        code to ensure syntactic correctness. This helps catch syntax errors before
        the code is returned. Defaults to True.
    :type force_ast_check: bool

    :return: A configured task instance ready to process Python source files and
        complete TODO comments. The task can be used by calling its ask_then_parse()
        method with a source file path.
    :rtype: PythonCodeGenerationLLMTask

    :raises FileNotFoundError: If the TODO completion prompt template file
        'todo-completion-req.md' is not found in the module directory.
    :raises ValueError: If the model parameter is invalid or cannot be loaded.
    :raises TypeError: If model is not a string, LLMModel instance, or None.

    .. note::
       The TODO completion prompt template is loaded from 'todo-completion-req.md'
       located in the same directory as this module. Ensure this file exists and
       contains valid prompt instructions for the LLM.

    .. warning::
       When skip_when_error is False, the task will fail if any import dependencies
       cannot be resolved. Set to True for more robust operation with incomplete
       dependency information.

    .. warning::
       The quality of TODO completions depends heavily on the LLM model used.
       More capable models (e.g., GPT-4, Claude-3) typically produce better results
       than smaller models.

    Example::

        >>> from hbllmutils.meta.code.todo_completion import create_todo_completion_task
        >>> 
        >>> # Create task with default settings
        >>> task = create_todo_completion_task('gpt-4')
        >>> 
        >>> # Create task with full context and strict error handling
        >>> task = create_todo_completion_task(
        ...     model='gpt-4',
        ...     show_module_directory_tree=True,
        ...     skip_when_error=False,
        ...     force_ast_check=True
        ... )
        >>> 
        >>> # Process a source file with TODOs
        >>> completed_code = task.ask_then_parse(
        ...     input_content='myproject/module_with_todos.py'
        ... )
        >>> 
        >>> # Save the completed code
        >>> with open('myproject/module_completed.py', 'w') as f:
        ...     f.write(completed_code)
        >>> 
        >>> # Use with existing model instance
        >>> from hbllmutils.model import RemoteLLMModel
        >>> custom_model = RemoteLLMModel(
        ...     base_url='https://api.example.com',
        ...     api_token='your-token',
        ...     model_name='custom-model'
        ... )
        >>> task = create_todo_completion_task(
        ...     model=custom_model,
        ...     show_module_directory_tree=False
        ... )
        >>> 
        >>> # Process multiple files
        >>> source_files = [
        ...     'src/utils.py',
        ...     'src/models.py',
        ...     'src/views.py'
        ... ]
        >>> for source_file in source_files:
        ...     try:
        ...         completed = task.ask_then_parse(input_content=source_file)
        ...         with open(source_file, 'w') as f:
        ...             f.write(completed)
        ...         print(f"Completed TODOs in {source_file}")
        ...     except Exception as e:
        ...         print(f"Failed to process {source_file}: {e}")

    """
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'todo-completion-req.md')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render()

    return PythonDetailedCodeGenerationLLMTask(
        model=load_llm_model(model),
        code_name='Code For Task',
        description_text='This is the source code for you to complete the TODOs',
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
    )
