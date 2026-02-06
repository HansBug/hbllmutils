"""
Python Documentation Generation Module

This module provides command line interface functionality for generating Python documentation
using LLM models. It includes functionality to process Python files or directories and generate
comprehensive documentation with proper formatting and structure.

The module contains the following main components:

* :func:`generate_pydoc_for_file` - Generate documentation for a single Python file
* :func:`_add_pydoc_subcommand` - Register the pydoc CLI subcommand
* :func:`_get_llm_task` - Create and cache LLM task instances for documentation generation

.. note::
   This module requires an LLM model configuration to be available either through
   command-line parameters or environment variables (OPENAI_MODEL_NAME, LLM_MODEL_NAME,
   or MODEL_NAME).

.. warning::
   Documentation generation may consume significant API tokens for large files or
   directories. Monitor your API usage when processing multiple files.

Example::

    >>> # Command line usage
    >>> # Generate docs for a single file
    >>> # hbllmutils code pydoc -i mymodule.py -m gpt-4
    >>> 
    >>> # Generate docs for a directory
    >>> # hbllmutils code pydoc -i mypackage/ -m gpt-4 --timeout 300
    >>> 
    >>> # With additional parameters
    >>> # hbllmutils code pydoc -i myfile.py --param max_tokens=128000

"""

import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Union, Tuple

import click
from hbutils.logging import ColoredFormatter, tqdm

from ..base import CONTEXT_SETTINGS, parse_key_value_params
from ...meta.code import create_pydoc_generation_task
from ...model import load_llm_model_from_config
from ...utils import obj_hashable, get_global_logger


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None, timeout: int = 240,
                  extra_params: Tuple[Tuple[str, Union[str, int, float]], ...] = (),
                  ignore_modules: Tuple[str, ...] = (),
                  no_ignore_modules: Tuple[str, ...] = ()):
    """
    Create and cache an LLM task instance for Python documentation generation.

    This function creates a pydoc generation task using the specified LLM model
    and parameters. Results are cached using LRU caching to avoid recreating
    identical task instances, improving performance when processing multiple files
    with the same configuration.

    The function loads the LLM model from configuration and creates a specialized
    task for generating Python documentation in reStructuredText format.

    :param model_name: Name of the LLM model to use (e.g., 'gpt-4', 'claude-2').
                      If None, uses the default model from configuration.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
    :type timeout: int
    :param extra_params: Additional parameters as tuple of (key, value) pairs to pass
                        to the LLM model. Must be hashable for caching purposes.
    :type extra_params: Tuple[Tuple[str, Union[str, int, float]], ...]
    :param ignore_modules: Tuple of module names to explicitly ignore during dependency analysis.
    :type ignore_modules: Tuple[str, ...]
    :param no_ignore_modules: Tuple of module names to never ignore during dependency analysis.
    :type no_ignore_modules: Tuple[str, ...]

    :return: Configured LLM task ready to generate Python documentation
    :rtype: PythonCodeGenerationLLMTask

    :raises ValueError: If model configuration is invalid
    :raises RuntimeError: If no model parameters are specified and no local configuration exists

    .. note::
       This function uses LRU caching to reuse task instances with identical parameters.
       The cache is based on all input parameters, so changing any parameter will create
       a new task instance.

    .. note::
       The extra_params, ignore_modules, and no_ignore_modules parameters must be tuples
       (not lists or dicts) to maintain hashability for the LRU cache.

    Example::

        >>> from hbllmutils.entry.code.pydoc import _get_llm_task
        >>> 
        >>> # Create a basic task
        >>> task = _get_llm_task(model_name='gpt-4', timeout=300)
        >>> 
        >>> # Create a task with extra parameters
        >>> extra = (('max_tokens', 128000), ('temperature', 0.7))
        >>> task = _get_llm_task(model_name='gpt-4', timeout=300, extra_params=extra)
        >>> 
        >>> # Create a task with module filtering
        >>> task = _get_llm_task(
        ...     model_name='gpt-4',
        ...     ignore_modules=('numpy', 'pandas'),
        ...     no_ignore_modules=('mypackage',)
        ... )
        >>> 
        >>> # Subsequent calls with same parameters return cached instance
        >>> task2 = _get_llm_task(model_name='gpt-4', timeout=300, extra_params=extra)
        >>> assert task is task2  # Same object from cache

    """
    params = dict(extra_params)
    return create_pydoc_generation_task(
        model=load_llm_model_from_config(
            model_name=model_name,
            timeout=timeout,
            **params
        ),
        ignore_modules=ignore_modules or None,
        no_ignore_modules=no_ignore_modules or None
    )


def generate_pydoc_for_file(file: str, model_name: Optional[str] = None, timeout: int = 240,
                            extra_params: Optional[Dict[str, Union[str, int, float]]] = None,
                            ignore_modules: Optional[Tuple[str, ...]] = None,
                            no_ignore_modules: Optional[Tuple[str, ...]] = None) -> None:
    """
    Generate Python documentation for a single file using LLM.

    This function reads a Python source file, generates comprehensive documentation
    using an LLM model, and writes the documented code back to the same file,
    replacing the original content. The documentation includes module-level docstrings,
    class and function documentation in reStructuredText format.

    The function uses the cached LLM task from :func:`_get_llm_task` to perform the
    documentation generation. The generated documentation is automatically formatted
    and validated before being written back to the file.

    :param file: Path to the Python file to document
    :type file: str
    :param model_name: Name of the LLM model to use. If None, uses default from configuration.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
    :type timeout: int
    :param extra_params: Additional parameters to pass to the LLM model as a dictionary.
                        Common parameters include 'max_tokens', 'temperature', etc.
    :type extra_params: Optional[Dict[str, Union[str, int, float]]]
    :param ignore_modules: Tuple of module names to explicitly ignore during dependency analysis.
    :type ignore_modules: Optional[Tuple[str, ...]]
    :param no_ignore_modules: Tuple of module names to never ignore during dependency analysis.
    :type no_ignore_modules: Optional[Tuple[str, ...]]

    :raises FileNotFoundError: If the specified file does not exist
    :raises PermissionError: If the file cannot be read or written
    :raises ValueError: If the file is not a valid Python file
    :raises RuntimeError: If documentation generation fails

    .. warning::
       This function overwrites the original file with the documented version.
       Ensure you have backups or version control in place before running.

    .. note::
       The function uses max_retries=0 when calling the LLM task, meaning it will
       not retry on failure. Any errors during generation will be propagated.

    Example::

        >>> from hbllmutils.entry.code.pydoc import generate_pydoc_for_file
        >>> 
        >>> # Generate docs for a single file
        >>> generate_pydoc_for_file('mymodule.py', model_name='gpt-4')
        >>> 
        >>> # With custom timeout and parameters
        >>> params = {'max_tokens': 128000, 'temperature': 0.7}
        >>> generate_pydoc_for_file(
        ...     'mymodule.py',
        ...     model_name='gpt-4',
        ...     timeout=300,
        ...     extra_params=params
        ... )
        >>> 
        >>> # With module filtering
        >>> generate_pydoc_for_file(
        ...     'mymodule.py',
        ...     model_name='gpt-4',
        ...     ignore_modules=('numpy', 'pandas'),
        ...     no_ignore_modules=('mypackage',)
        ... )

    """
    get_global_logger().info(f'Make docs for {file!r} ...')
    extra_params_hashable = obj_hashable(extra_params or {})
    ignore_modules_hashable = tuple(ignore_modules) if ignore_modules else ()
    no_ignore_modules_hashable = tuple(no_ignore_modules) if no_ignore_modules else ()
    new_docs = _get_llm_task(
        model_name=model_name,
        timeout=timeout,
        extra_params=extra_params_hashable,
        ignore_modules=ignore_modules_hashable,
        no_ignore_modules=no_ignore_modules_hashable
    ).ask_then_parse(file, max_retries=0)
    new_docs = new_docs.rstrip()
    with open(file, 'w') as f:
        print(new_docs, file=f)


def _add_pydoc_subcommand(cli: click.Group) -> click.Group:
    """
    Register the pydoc subcommand to a Click CLI group.

    This function adds a 'pydoc' subcommand to the provided Click command group,
    enabling Python documentation generation functionality through the command line.
    The subcommand supports processing both individual files and entire directories
    of Python source files.

    The registered command provides the following features:

    * Single file or directory processing
    * Configurable LLM model selection
    * Adjustable API timeout settings
    * Additional model parameters via key=value pairs
    * Module filtering options for dependency analysis
    * Progress tracking for directory processing
    * Comprehensive logging with colored output
    * Error handling and recovery for batch operations

    :param cli: Click command group to which the pydoc subcommand will be added
    :type cli: click.Group

    :return: The modified Click command group with the pydoc subcommand registered
    :rtype: click.Group

    .. note::
       The function modifies the provided Click group in-place by adding the
       pydoc command, but also returns the group for method chaining.

    .. note::
       The pydoc command uses environment variables as fallbacks for model name:
       OPENAI_MODEL_NAME, LLM_MODEL_NAME, or MODEL_NAME (checked in that order).

    Example::

        >>> import click
        >>> from hbllmutils.entry.code.pydoc import _add_pydoc_subcommand
        >>> 
        >>> # Create a CLI group and add pydoc command
        >>> @click.group()
        >>> def cli():
        ...     '''Main CLI application'''
        ...     pass
        >>> 
        >>> cli = _add_pydoc_subcommand(cli)
        >>> 
        >>> # Now the CLI has a pydoc subcommand
        >>> # Usage: python script.py pydoc -i myfile.py -m gpt-4

    Command Line Usage::

        # Generate docs for a single file
        $ hbllmutils code pydoc -i mymodule.py -m gpt-4

        # Generate docs for a directory
        $ hbllmutils code pydoc -i mypackage/ -m gpt-4 --timeout 300

        # With additional parameters
        $ hbllmutils code pydoc -i myfile.py -m gpt-4 -p max_tokens=128000 -p temperature=0.7

        # With module filtering
        $ hbllmutils code pydoc -i myfile.py -m gpt-4 --ignore-module numpy --ignore-module pandas --no-ignore-module mypackage

        # Using environment variable for model
        $ export OPENAI_MODEL_NAME=gpt-4
        $ hbllmutils code pydoc -i myfile.py

    """

    @cli.command('pydoc', help='Generate Python documentation for code files using LLM.',
                 context_settings=CONTEXT_SETTINGS)
    @click.option('-i', '--input', 'input_path', type=str, required=True,
                  help='Input Python file or directory to process for documentation generation.')
    @click.option('-m', '--model-name', 'model_name', type=str, required=False, default=None,
                  help='LLM model name to use for documentation generation.')
    @click.option('--timeout', 'timeout', type=int, required=False, default=210,
                  help='Timeout in seconds for LLM API requests.')
    @click.option('-p', '--param', 'params', type=str, multiple=True,
                  help='Additional parameters in key=value format (e.g., --param max_tokens=128000). '
                       'Can be used multiple times.',
                  callback=lambda ctx, param, value: dict(parse_key_value_params(v) for v in value) if value else {})
    @click.option('--ignore-module', 'ignore_modules', type=str, multiple=True,
                  help='Module names to explicitly ignore during dependency analysis. Can be used multiple times.')
    @click.option('--no-ignore-module', 'no_ignore_modules', type=str, multiple=True,
                  help='Module names to never ignore during dependency analysis. Can be used multiple times.')
    def pydoc(input_path, model_name, timeout, params, ignore_modules, no_ignore_modules):
        """
        Generate Python documentation for files or directories using LLM.

        This command processes Python source files and generates comprehensive
        documentation in reStructuredText format. It can handle both individual
        files and entire directory trees, with progress tracking and error handling.

        The command performs the following steps:

        1. Sets up logging with colored output for better readability
        2. Validates input path and determines processing mode (file/directory)
        3. Determines LLM model from parameters or environment variables
        4. For directories: discovers all .py files recursively
        5. Generates documentation for each file using the LLM
        6. Overwrites original files with documented versions
        7. Reports progress and handles errors gracefully

        :param input_path: Path to Python file or directory to process
        :type input_path: str
        :param model_name: LLM model name (or None to use environment variable)
        :type model_name: Optional[str]
        :param timeout: Timeout in seconds for API requests
        :type timeout: int
        :param params: Dictionary of additional model parameters
        :type params: Dict[str, Union[str, int, float]]
        :param ignore_modules: Tuple of module names to explicitly ignore
        :type ignore_modules: Tuple[str, ...]
        :param no_ignore_modules: Tuple of module names to never ignore
        :type no_ignore_modules: Tuple[str, ...]

        :raises FileNotFoundError: If input_path does not exist
        :raises RuntimeError: If input_path is neither a file nor a directory
        :raises Exception: If documentation generation fails for any file

        .. note::
           When processing directories, the command will process all .py files
           found recursively, including files in subdirectories.

        .. warning::
           Original files are overwritten with documented versions. Ensure you
           have backups or use version control before running this command.

        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        get_global_logger().debug(f'Starting pydoc generation for input path: {input_path!r}')
        get_global_logger().debug(f'Model name: {model_name or "default"}')
        get_global_logger().debug(f'Timeout: {timeout}s')

        extra_params = params
        if extra_params:
            get_global_logger().info(f'Extra parameters: {extra_params}')

        if ignore_modules:
            get_global_logger().info(f'Ignoring modules: {list(ignore_modules)}')
        if no_ignore_modules:
            get_global_logger().info(f'Not ignoring modules: {list(no_ignore_modules)}')

        llm_model = (model_name or os.environ.get('OPENAI_MODEL_NAME')
                     or os.environ.get('LLM_MODEL_NAME') or os.environ.get('MODEL_NAME'))
        get_global_logger().info(f'Using LLM model: {llm_model or "default"}')

        if not os.path.exists(input_path):
            get_global_logger().error(f'File not found - {input_path!r}.')
            raise FileNotFoundError(f'File not found - {input_path!r}.')
        elif os.path.isfile(input_path):
            get_global_logger().info(f'Processing single file: {input_path!r}')
            generate_pydoc_for_file(
                input_path,
                model_name=llm_model,
                timeout=timeout,
                extra_params=extra_params,
                ignore_modules=tuple(ignore_modules) if ignore_modules else None,
                no_ignore_modules=tuple(no_ignore_modules) if no_ignore_modules else None
            )
            get_global_logger().info(f'Successfully generated documentation for {input_path!r}')
        elif os.path.isdir(input_path):
            get_global_logger().info(f'Processing directory: {input_path!r}')
            py_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    _, ext = os.path.splitext(os.path.normcase(file))
                    if ext == '.py':
                        file_path = os.path.join(root, file)
                        py_files.append(file_path)

            get_global_logger().info(f'Found {len(py_files)} Python files in {input_path!r}')
            for file_path in tqdm(py_files, desc=f'Generate Docs in {input_path!r}', total=len(py_files)):
                try:
                    generate_pydoc_for_file(
                        file_path,
                        model_name=llm_model,
                        timeout=timeout,
                        extra_params=extra_params,
                        ignore_modules=tuple(ignore_modules) if ignore_modules else None,
                        no_ignore_modules=tuple(no_ignore_modules) if no_ignore_modules else None
                    )
                except Exception as e:
                    get_global_logger().exception(f'Failed to generate documentation for {file_path!r}: {e}')
                    raise
            get_global_logger().info(f'Completed documentation generation for directory {input_path!r}')
        else:
            get_global_logger().error(f'Unknown input - {input_path!r}.')
            raise RuntimeError(f'Unknown input - {input_path!r}.')

    return cli
