"""
Unit test generation command-line interface for Python source files.

This module provides a CLI tool for generating comprehensive unit tests for Python
source files using Large Language Models (LLMs). It supports multiple test frameworks
(pytest, unittest, nose2) and offers extensive configuration options for customizing
the test generation process.

The module contains the following main components:

* :func:`generate_unittest_for_file` - Core function for generating unit tests for a single file
* :func:`_get_llm_task` - Internal cached function for creating LLM task instances
* :func:`_add_unittest_subcommand` - Registers the unittest CLI subcommand

.. note::
   This module requires an LLM model configuration to function. Models can be
   specified via command-line arguments, environment variables, or configuration files.

.. warning::
   Test generation may consume significant API tokens/credits depending on the
   complexity of the source code and the chosen LLM model.

Example::

    >>> from hbllmutils.entry.code.unittest import generate_unittest_for_file
    >>> 
    >>> # Generate tests for a source file
    >>> test_code = generate_unittest_for_file(
    ...     source_file='mypackage/calculator.py',
    ...     test_file='tests/test_calculator.py',
    ...     model_name='gpt-4',
    ...     test_framework_name='pytest',
    ...     mark_name='unittest'
    ... )
    >>> 
    >>> # Save the generated test code
    >>> with open('tests/test_calculator.py', 'w') as f:
    ...     f.write(test_code)

"""

import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Union, Tuple

from ...utils import obj_hashable

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

import click
from hbutils.logging import ColoredFormatter

from ..base import CONTEXT_SETTINGS, parse_key_value_params
from ...meta.code.unittest_generation import create_unittest_generation_task
from ...utils.logging import get_global_logger
from ...model import load_llm_model_from_config


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None, timeout: int = 240,
                  extra_params: Tuple[Tuple[str, Union[str, int, float]], ...] = (),
                  show_module_directory_tree: bool = False,
                  skip_when_error: bool = True,
                  force_ast_check: bool = True,
                  test_framework_name: Literal['pytest', 'unittest', 'nose2'] = "pytest",
                  mark_name: Optional[str] = 'unittest'):
    """
    Create and cache an LLM task instance for Python unit test generation.

    This function creates a unittest generation task using the specified LLM model
    and parameters. Results are cached using LRU caching to avoid recreating
    identical task instances, improving performance when processing multiple files
    with the same configuration.

    The function loads the LLM model from configuration and creates a specialized
    task for generating Python unit tests for the specified test framework.

    :param model_name: Name of the LLM model to use (e.g., 'gpt-4', 'claude-2').
                      If None, uses the default model from configuration.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
    :type timeout: int
    :param extra_params: Additional parameters as tuple of (key, value) pairs to pass
                        to the LLM model. Must be hashable for caching purposes.
    :type extra_params: Tuple[Tuple[str, Union[str, int, float]], ...]
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

    :return: Configured LLM task ready to generate Python unit tests
    :rtype: UnittestCodeGenerationLLMTask

    :raises ValueError: If model configuration is invalid or test_framework_name is not supported
    :raises RuntimeError: If no model parameters are specified and no local configuration exists

    .. note::
       This function uses LRU caching to reuse task instances with identical parameters.
       The cache is based on all input parameters, so changing any parameter will create
       a new task instance.

    .. note::
       The extra_params parameter must be a tuple of tuples (not a dict) to maintain
       hashability for the LRU cache.

    Example::

        >>> from hbllmutils.entry.code.unittest import _get_llm_task
        >>> 
        >>> # Create a basic task
        >>> task = _get_llm_task(model_name='gpt-4', timeout=300)
        >>> 
        >>> # Create a task with extra parameters
        >>> extra = (('max_tokens', 128000), ('temperature', 0.7))
        >>> task = _get_llm_task(
        ...     model_name='gpt-4',
        ...     timeout=300,
        ...     extra_params=extra,
        ...     test_framework_name='pytest',
        ...     mark_name='unittest'
        ... )
        >>> 
        >>> # Subsequent calls with same parameters return cached instance
        >>> task2 = _get_llm_task(
        ...     model_name='gpt-4',
        ...     timeout=300,
        ...     extra_params=extra,
        ...     test_framework_name='pytest',
        ...     mark_name='unittest'
        ... )
        >>> assert task is task2  # Same object from cache

    """
    params = dict(extra_params)
    return create_unittest_generation_task(
        model=load_llm_model_from_config(
            model_name=model_name,
            timeout=timeout,
            **params
        ),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
        test_framework_name=test_framework_name,
        mark_name=mark_name,
    )


def generate_unittest_for_file(source_file: str, test_file: Optional[str] = None,
                               model_name: Optional[str] = None, timeout: int = 240,
                               extra_params: Optional[Dict[str, Union[str, int, float]]] = None,
                               show_module_directory_tree: bool = False,
                               skip_when_error: bool = True,
                               force_ast_check: bool = True,
                               test_framework_name: Literal['pytest', 'unittest', 'nose2'] = "pytest",
                               mark_name: Optional[str] = 'unittest') -> str:
    """
    Generate unit test code for a single Python file using LLM.

    This function reads a Python source file and optionally an existing test file,
    then generates comprehensive unit tests using an LLM model. The generated test
    code follows the specified test framework conventions and can use existing tests
    as reference for style and patterns.

    The function performs the following steps:
    
    1. Loads or creates an LLM task with the specified configuration
    2. Analyzes the source file to understand its structure and functionality
    3. Optionally uses an existing test file as a style reference
    4. Generates comprehensive unit tests covering the source code
    5. Returns the generated test code as a string

    :param source_file: Path to the Python source file to generate tests for.
                       Must be a valid Python file that can be parsed.
    :type source_file: str
    :param test_file: Optional path to existing test file to use as reference
                     for test style and patterns. If provided and exists, the LLM
                     will attempt to match the existing test style.
    :type test_file: Optional[str]
    :param model_name: Name of the LLM model to use (e.g., 'gpt-4', 'claude-2').
                      If None, uses default from configuration or environment variables.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
    :type timeout: int
    :param extra_params: Additional parameters to pass to the LLM model as a dictionary.
                        Common parameters include 'max_tokens', 'temperature', etc.
    :type extra_params: Optional[Dict[str, Union[str, int, float]]]
    :param show_module_directory_tree: If True, include module directory tree in prompts
                                      to provide structural context about the package layout.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load during analysis
                           instead of raising exceptions. Useful for handling optional dependencies.
    :type skip_when_error: bool
    :param force_ast_check: If True, validate generated code with AST parsing to ensure
                           syntactic correctness before returning.
    :type force_ast_check: bool
    :param test_framework_name: The test framework to generate tests for.
                               Must be one of 'pytest', 'unittest', or 'nose2'.
    :type test_framework_name: Literal['pytest', 'unittest', 'nose2']
    :param mark_name: The pytest mark name to use for generated tests (e.g., 'unittest'
                     will generate @pytest.mark.unittest decorators). If None or empty,
                     no mark decorators will be added. Only applies to pytest framework.
    :type mark_name: Optional[str]

    :return: The generated unit test code as a string, with trailing whitespace removed
    :rtype: str

    :raises FileNotFoundError: If the specified source file does not exist
    :raises RuntimeError: If test generation fails due to LLM errors or configuration issues
    :raises ValueError: If model configuration is invalid

    .. note::
       The function uses LRU caching internally via :func:`_get_llm_task` to reuse
       task instances with identical configurations, improving performance when
       generating tests for multiple files.

    .. warning::
       Test generation may take significant time depending on source file complexity
       and LLM model response time. Consider using appropriate timeout values.

    Example::

        >>> from hbllmutils.entry.code.unittest import generate_unittest_for_file
        >>> 
        >>> # Basic usage
        >>> test_code = generate_unittest_for_file(
        ...     source_file='mypackage/calculator.py',
        ...     model_name='gpt-4'
        ... )
        >>> print(test_code[:100])  # Preview generated code
        
        >>> # With existing test file as reference
        >>> test_code = generate_unittest_for_file(
        ...     source_file='mypackage/calculator.py',
        ...     test_file='tests/test_calculator.py',
        ...     model_name='gpt-4'
        ... )
        
        >>> # With custom parameters
        >>> test_code = generate_unittest_for_file(
        ...     source_file='mypackage/calculator.py',
        ...     model_name='gpt-4',
        ...     timeout=300,
        ...     extra_params={'max_tokens': 128000, 'temperature': 0.7},
        ...     test_framework_name='pytest',
        ...     mark_name='unittest'
        ... )
        
        >>> # For unittest framework
        >>> test_code = generate_unittest_for_file(
        ...     source_file='mypackage/calculator.py',
        ...     model_name='gpt-4',
        ...     test_framework_name='unittest',
        ...     mark_name=None  # No marks for unittest
        ... )
        
        >>> # With directory tree context
        >>> test_code = generate_unittest_for_file(
        ...     source_file='mypackage/calculator.py',
        ...     model_name='gpt-4',
        ...     show_module_directory_tree=True
        ... )

    """
    get_global_logger().info(f'Generate unittest for {source_file!r} ...')

    task = _get_llm_task(
        model_name=model_name,
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
        test_framework_name=test_framework_name,
        mark_name=mark_name,
        extra_params=obj_hashable(extra_params or {}),
        timeout=timeout,
    )

    test_code = task.generate(
        source_file=source_file,
        test_file=test_file,
        max_retries=0,
    )

    return test_code.rstrip()


def _add_unittest_subcommand(cli: click.Group) -> click.Group:
    """
    Register the unittest subcommand to a Click CLI group.

    This function adds a 'unittest' subcommand to the provided Click command group,
    enabling unit test generation functionality through the command line interface.
    The subcommand supports processing individual Python source files and generating
    corresponding test files with extensive configuration options.

    The registered subcommand provides the following features:
    
    * Generate unit tests for Python source files using LLM models
    * Support for multiple test frameworks (pytest, unittest, nose2)
    * Configurable test marking (e.g., @pytest.mark.unittest)
    * Optional use of existing test files as style references
    * Extensive model configuration through command-line parameters
    * Logging configuration for debugging and monitoring

    :param cli: Click command group to which the unittest subcommand will be added.
               This should be the main CLI group for the application.
    :type cli: click.Group

    :return: The modified Click command group with the unittest subcommand registered.
            The returned group is the same object as the input parameter.
    :rtype: click.Group

    .. note::
       This function modifies the provided CLI group in-place by adding a new
       subcommand. The modification is done through Click's decorator mechanism.

    .. note::
       The subcommand automatically configures logging with colored output when
       executed, using the ColoredFormatter from hbutils.

    Example::

        >>> import click
        >>> from hbllmutils.entry.code.unittest import _add_unittest_subcommand
        >>> 
        >>> # Create a CLI group
        >>> @click.group()
        >>> def cli():
        ...     '''Main CLI application'''
        ...     pass
        >>> 
        >>> # Add the unittest subcommand
        >>> cli = _add_unittest_subcommand(cli)
        >>> 
        >>> # Now the CLI has a unittest subcommand
        >>> if __name__ == '__main__':
        ...     cli()

    """

    @cli.command('unittest', help='Generate unit test code for Python files using LLM.',
                 context_settings=CONTEXT_SETTINGS)
    @click.option('-i', '--input', 'input_path', type=str, required=True,
                  help='Input Python source file to generate tests for.')
    @click.option('-o', '--output', 'output_path', type=str, required=True,
                  help='Output test file path. If file exists, it will be used as reference for test style.')
    @click.option('-m', '--model-name', 'model_name', type=str, required=False, default=None,
                  help='LLM model name to use for test generation.')
    @click.option('--timeout', 'timeout', type=int, required=False, default=240,
                  help='Timeout in seconds for LLM API requests.')
    @click.option('--show-tree', 'show_tree', is_flag=True, default=False,
                  help='Include module directory tree in the prompt for structural context.')
    @click.option('--no-skip-error', 'no_skip_error', is_flag=True, default=False,
                  help='Raise exceptions on import errors instead of skipping them.')
    @click.option('--no-ast-check', 'no_ast_check', is_flag=True, default=False,
                  help='Skip AST validation of generated code.')
    @click.option('--framework', 'test_framework', type=click.Choice(['pytest', 'unittest', 'nose2']),
                  default='pytest', help='Test framework to generate tests for.')
    @click.option('--mark', 'mark_name', type=str, required=False, default='unittest',
                  help='Pytest mark name to use for generated tests (e.g., "unittest" for @pytest.mark.unittest). '
                       'Use empty string for no marks.')
    @click.option('-p', '--param', 'params', type=str, multiple=True,
                  help='Additional parameters in key=value format (e.g., --param max_tokens=128000). '
                       'Can be used multiple times.',
                  callback=lambda ctx, param, value: dict(parse_key_value_params(v) for v in value) if value else {})
    def unittest(input_path, output_path, model_name, timeout, show_tree,
                 no_skip_error, no_ast_check, test_framework, mark_name, params):
        """
        Generate unit test code for a Python source file using LLM.

        This command processes a Python source file and generates comprehensive
        unit tests using an LLM model. It can optionally use an existing test file
        as reference for test style and patterns.

        The command performs the following operations:
        
        1. Validates the input source file exists and is a regular file
        2. Configures logging for the generation process
        3. Loads the LLM model from configuration or command-line parameters
        4. Generates unit tests using the specified test framework
        5. Saves the generated test code to the output file

        :raises FileNotFoundError: If the input source file does not exist
        :raises RuntimeError: If the input path is not a file or test generation fails
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        get_global_logger().debug(f'Starting unittest generation for input path: {input_path!r}')
        get_global_logger().debug(f'Model name: {model_name or "default"}')
        get_global_logger().debug(f'Timeout: {timeout}s')
        get_global_logger().debug(f'Test framework: {test_framework}')

        extra_params = params
        if extra_params:
            get_global_logger().info(f'Extra parameters: {extra_params}')

        llm_model = (model_name or os.environ.get('OPENAI_MODEL_NAME')
                     or os.environ.get('LLM_MODEL_NAME') or os.environ.get('MODEL_NAME'))
        get_global_logger().info(f'Using LLM model: {llm_model or "default"}')

        if not os.path.exists(input_path):
            get_global_logger().error(f'Source file not found - {input_path!r}.')
            raise FileNotFoundError(f'Source file not found - {input_path!r}.')

        if not os.path.isfile(input_path):
            get_global_logger().error(f'Input path must be a file - {input_path!r}.')
            raise RuntimeError(f'Input path must be a file - {input_path!r}.')

        test_file = None
        if os.path.exists(output_path):
            get_global_logger().info(f'Using existing test file as reference: {output_path!r}')
            test_file = output_path

        mark_name_value = mark_name if mark_name else None

        get_global_logger().info(f'Processing source file: {input_path!r}')

        try:
            test_code = generate_unittest_for_file(
                source_file=input_path,
                test_file=test_file,
                model_name=llm_model,
                timeout=timeout,
                extra_params=extra_params,
                show_module_directory_tree=show_tree,
                skip_when_error=not no_skip_error,
                force_ast_check=not no_ast_check,
                test_framework_name=test_framework,
                mark_name=mark_name_value,
            )

            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                print(test_code.rstrip(), file=f)
            get_global_logger().info(f'Successfully generated test code to {output_path!r}')

        except Exception as e:
            get_global_logger().exception(f'Failed to generate test code for {input_path!r}: {e}')
            raise

    return cli
