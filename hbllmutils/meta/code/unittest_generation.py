"""
Unit test generation utilities for Python code using LLM models.

This module provides comprehensive tools for automatically generating unit tests
from Python source code using Large Language Models. It leverages LLM capabilities
to analyze source code and generate appropriate test cases with configurable
test frameworks and marking strategies.

The module contains the following main components:

* :class:`UnittestCodeGenerationLLMTask` - Main task class for generating unit tests
* :func:`create_unittest_generation_task` - Factory function for creating configured test generation tasks

.. note::
   This module requires a configured LLM model and supports multiple test frameworks
   including pytest, unittest, and nose2.

.. warning::
   Generated tests should be reviewed and validated before use in production.
   The LLM may not cover all edge cases or generate semantically correct tests.

Example::

    >>> from hbllmutils.meta.code.unittest_generation import create_unittest_generation_task
    >>> 
    >>> # Create a task with pytest framework
    >>> task = create_unittest_generation_task(
    ...     model='gpt-4',
    ...     test_framework_name='pytest',
    ...     mark_name='unittest'
    ... )
    >>> 
    >>> # Generate tests for a source file
    >>> test_code = task.generate(
    ...     source_file='mypackage/calculator.py',
    ...     max_retries=3
    ... )
    >>> print(test_code)
    
    >>> # Generate tests with existing test file as reference
    >>> test_code = task.generate(
    ...     source_file='mypackage/calculator.py',
    ...     test_file='tests/test_calculator_old.py'
    ... )

"""

import io
import os
from typing import Optional

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

from .prompt import get_prompt_for_source_file
from .task import PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model, LLMModel
from ...template import PromptTemplate


class UnittestCodeGenerationLLMTask(PythonCodeGenerationLLMTask):
    """
    LLM task for generating unit test code from Python source files.

    This class extends :class:`PythonCodeGenerationLLMTask` to provide specialized
    functionality for generating unit tests. It analyzes source code and optionally
    existing test files to generate comprehensive test cases using an LLM model.

    The task supports:
    
    - Generating tests from source code with full dependency analysis
    - Using existing test files as reference for test style and patterns
    - Optional module directory tree visualization for context
    - Configurable error handling during import analysis
    - Automatic AST validation of generated test code

    :param model: The LLM model to use for test generation.
    :type model: LLMModel
    :param history: Optional conversation history with system prompt. If None, creates new history.
    :type history: Optional[LLMHistory]
    :param default_max_retries: Maximum number of retry attempts for generation and parsing.
                               Defaults to 5.
    :type default_max_retries: int
    :param show_module_directory_tree: If True, include module directory tree in the prompt
                                      to provide structural context. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load during analysis
                           instead of raising exceptions. Defaults to True.
    :type skip_when_error: bool
    :param force_ast_check: If True, validate generated code with AST parsing.
                           Defaults to True.
    :type force_ast_check: bool

    :ivar show_module_directory_tree: Whether to include directory tree in prompts.
    :vartype show_module_directory_tree: bool
    :ivar skip_when_error: Whether to skip failed imports during analysis.
    :vartype skip_when_error: bool

    .. note::
       The generated tests should be reviewed for correctness and completeness.
       The LLM may not generate tests for all edge cases or complex scenarios.

    .. warning::
       Large source files with many dependencies may generate very large prompts,
       potentially exceeding model context limits.

    Example::

        >>> from hbllmutils.model import LLMModel
        >>> from hbllmutils.history import LLMHistory
        >>> 
        >>> # Create task with custom configuration
        >>> model = LLMModel(...)
        >>> history = LLMHistory().with_system_prompt("Generate comprehensive pytest tests")
        >>> task = UnittestCodeGenerationLLMTask(
        ...     model=model,
        ...     history=history,
        ...     show_module_directory_tree=True,
        ...     skip_when_error=True
        ... )
        >>> 
        >>> # Generate tests for a module
        >>> test_code = task.generate('mypackage/calculator.py')
        >>> print(test_code)
        
        >>> # Generate with existing tests as reference
        >>> test_code = task.generate(
        ...     source_file='mypackage/calculator.py',
        ...     test_file='tests/test_calculator_old.py',
        ...     max_retries=3
        ... )

    """

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 show_module_directory_tree: bool = False, skip_when_error: bool = True,
                 force_ast_check: bool = True):
        """
        Initialize the UnittestCodeGenerationLLMTask.

        :param model: The LLM model to use for test generation.
        :type model: LLMModel
        :param history: Optional conversation history. If None, creates new history.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Maximum retry attempts for parsing. Defaults to 5.
        :type default_max_retries: int
        :param show_module_directory_tree: Whether to include directory tree. Defaults to False.
        :type show_module_directory_tree: bool
        :param skip_when_error: Whether to skip failed imports. Defaults to True.
        :type skip_when_error: bool
        :param force_ast_check: Whether to enforce AST validation. Defaults to True.
        :type force_ast_check: bool
        """
        super().__init__(model, history, default_max_retries, force_ast_check)
        self.show_module_directory_tree = show_module_directory_tree
        self.skip_when_error = skip_when_error

    def generate(self, source_file: str, test_file: Optional[str] = None, max_retries: Optional[int] = None, **params):
        """
        Generate unit test code for the specified source file.

        This method analyzes the source file and optionally an existing test file
        to generate comprehensive unit tests. It creates a detailed prompt containing:
        
        - Complete source code analysis with dependencies
        - Optional module directory tree for structural context
        - Optional existing test file for reference patterns
        - All imported dependencies and their implementations

        The generated prompt is then sent to the LLM model, which produces test code
        that is validated and returned.

        :param source_file: Path to the Python source file to generate tests for.
        :type source_file: str
        :param test_file: Optional path to existing test file to use as reference
                         for test style and patterns. If provided, the existing tests
                         will be included in the prompt to guide generation.
        :type test_file: Optional[str]
        :param max_retries: Maximum number of retry attempts if generation fails.
                           If None, uses the default_max_retries value.
        :type max_retries: Optional[int]
        :param params: Additional parameters to pass to the LLM model during generation.
                      These may include temperature, max_tokens, etc.
        :type params: dict

        :return: The generated unit test code as a string, validated with AST parsing.
        :rtype: str

        :raises OutputParseFailed: If test generation fails after all retry attempts.
        :raises FileNotFoundError: If source_file or test_file does not exist.
        :raises SyntaxError: If the generated code has syntax errors (after retries exhausted).

        .. note::
           The method uses :func:`get_prompt_for_source_file` to analyze both the
           source and test files. Import failures can be handled gracefully with
           the skip_when_error parameter.

        .. warning::
           Very large source files or complex dependency trees may generate prompts
           that exceed the model's context window, potentially causing failures.

        Example::

            >>> task = UnittestCodeGenerationLLMTask(model, history)
            >>> 
            >>> # Generate tests for a simple module
            >>> test_code = task.generate('mypackage/calculator.py')
            >>> print(test_code)
            import pytest
            from mypackage.calculator import Calculator
            
            @pytest.mark.unittest
            def test_calculator_add():
                calc = Calculator()
                assert calc.add(2, 3) == 5
            
            >>> # Generate with existing tests as reference
            >>> test_code = task.generate(
            ...     source_file='mypackage/calculator.py',
            ...     test_file='tests/test_calculator_old.py'
            ... )
            >>> # Generated tests will follow patterns from the existing test file
            
            >>> # Generate with custom retry limit
            >>> test_code = task.generate(
            ...     source_file='mypackage/complex_module.py',
            ...     max_retries=10
            ... )
            
            >>> # Generate with model parameters
            >>> test_code = task.generate(
            ...     source_file='mypackage/calculator.py',
            ...     temperature=0.7,
            ...     max_tokens=2000
            ... )

        """
        with io.StringIO() as sf:
            source_prompt, imported_items = get_prompt_for_source_file(
                source_file=source_file,
                level=1,
                code_name='Code For Unittest Generation',
                description_text='This is the source code for you to generate unittest code',
                show_module_directory_tree=self.show_module_directory_tree,
                skip_when_error=self.skip_when_error,
                return_imported_items=True,
            )
            print(source_prompt, file=sf)
            print(f'', file=sf)

            if test_file:
                test_prompt = get_prompt_for_source_file(
                    source_file=test_file,
                    level=1,
                    code_name='Code Of Existing Unittest',
                    description_text='This is the source code of existing unittest',
                    show_module_directory_tree=self.show_module_directory_tree,
                    skip_when_error=self.skip_when_error,
                    ignore_modules=imported_items,
                )
                print(test_prompt, file=sf)
                print(f'', file=sf)

            prompt = sf.getvalue().rstrip()

        return self.ask_then_parse(
            input_content=prompt,
            max_retries=max_retries,
            **params,
        )


def create_unittest_generation_task(
        model: LLMModelTyping,
        show_module_directory_tree: bool = False,
        skip_when_error: bool = True,
        force_ast_check: bool = True,
        test_framework_name: Literal['pytest', 'unittest', 'nose2'] = "pytest",
        mark_name: Optional[str] = 'unittest',
) -> UnittestCodeGenerationLLMTask:
    """
    Create a configured unit test generation task with appropriate system prompt.

    This factory function creates an :class:`UnittestCodeGenerationLLMTask` instance
    with a system prompt tailored for the specified test framework. The system prompt
    is loaded from a Jinja2 template and rendered with the provided configuration.

    The function handles:
    
    - Loading and initializing the specified LLM model
    - Creating a system prompt from template with framework-specific instructions
    - Configuring test marking strategies (e.g., @pytest.mark.unittest)
    - Setting up error handling and validation options

    :param model: The LLM model to use. Can be a model name string, an LLMModel instance,
                 or None to use the default model from configuration.
    :type model: LLMModelTyping
    :param show_module_directory_tree: If True, include module directory tree in prompts
                                      to provide structural context. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load during analysis
                           instead of raising exceptions. Defaults to True.
    :type skip_when_error: bool
    :param force_ast_check: If True, validate generated code with AST parsing.
                           Defaults to True.
    :type force_ast_check: bool
    :param test_framework_name: The test framework to generate tests for.
                               Must be one of 'pytest', 'unittest', or 'nose2'.
                               Defaults to 'pytest'.
    :type test_framework_name: Literal['pytest', 'unittest', 'nose2']
    :param mark_name: The pytest mark name to use for generated tests (e.g., 'unittest'
                     will generate @pytest.mark.unittest decorators). If None or empty,
                     no mark decorators will be added. Only applies to pytest framework.
                     Defaults to 'unittest'.
    :type mark_name: Optional[str]

    :return: A configured UnittestCodeGenerationLLMTask instance ready for test generation.
    :rtype: UnittestCodeGenerationLLMTask

    :raises ValueError: If test_framework_name is not one of the supported frameworks.
    :raises FileNotFoundError: If the system prompt template file cannot be found.
    :raises TypeError: If model parameter is of invalid type.

    .. note::
       The system prompt template is loaded from 'unittest_generation.j2' in the
       same directory as this module. The template is rendered with the specified
       test framework and mark name.

    .. warning::
       Different test frameworks have different capabilities and syntax. Ensure
       the LLM model is capable of generating tests for the specified framework.

    Example::

        >>> # Create task with pytest framework
        >>> task = create_unittest_generation_task(
        ...     model='gpt-4',
        ...     test_framework_name='pytest',
        ...     mark_name='unittest'
        ... )
        >>> test_code = task.generate('mypackage/calculator.py')
        
        >>> # Create task with unittest framework
        >>> task = create_unittest_generation_task(
        ...     model='gpt-4',
        ...     test_framework_name='unittest',
        ...     mark_name=None  # No marks for unittest framework
        ... )
        >>> test_code = task.generate('mypackage/calculator.py')
        
        >>> # Create task without pytest marks
        >>> task = create_unittest_generation_task(
        ...     model='gpt-4',
        ...     test_framework_name='pytest',
        ...     mark_name=None  # Will not add @pytest.mark decorators
        ... )
        
        >>> # Create task with directory tree visualization
        >>> task = create_unittest_generation_task(
        ...     model='gpt-4',
        ...     show_module_directory_tree=True,
        ...     test_framework_name='pytest'
        ... )
        
        >>> # Create task with custom error handling
        >>> task = create_unittest_generation_task(
        ...     model='gpt-4',
        ...     skip_when_error=False,  # Raise on import errors
        ...     force_ast_check=True
        ... )
        
        >>> # Use existing model instance
        >>> from hbllmutils.model import RemoteLLMModel
        >>> my_model = RemoteLLMModel(base_url='...', api_token='...', model_name='gpt-4')
        >>> task = create_unittest_generation_task(
        ...     model=my_model,
        ...     test_framework_name='pytest'
        ... )
        
        >>> # Use default model from configuration
        >>> task = create_unittest_generation_task(
        ...     model=None,  # Uses default from config
        ...     test_framework_name='pytest'
        ... )

    """
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'unittest_generation.j2')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render(
        test_framework_name=test_framework_name,
        mark_name=mark_name,
    )

    return UnittestCodeGenerationLLMTask(
        model=load_llm_model(model),
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
    )
