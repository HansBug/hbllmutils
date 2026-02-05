"""
TODO Completion Command Line Interface Module.

This module provides command line interface functionality for completing TODO comments
in Python source code files using Large Language Models (LLMs). It processes Python files
or directories, identifies TODO comments, and generates appropriate code completions while
maintaining code quality and consistency.

The module contains the following main components:

* :func:`complete_todo_for_file` - Process a single Python file to complete its TODO comments
* :func:`_get_llm_task` - Create and cache LLM task instances for TODO completion
* :func:`_add_todo_subcommand` - Register the 'todo' subcommand with the CLI

The TODO completion process includes:

* Source code analysis and dependency resolution
* TODO comment identification and context extraction
* LLM-based code generation with validation
* AST-based syntax verification
* Automatic file updates with completed code

.. note::
   This module requires an LLM model to be configured either through command-line
   parameters or environment variables (OPENAI_MODEL_NAME, LLM_MODEL_NAME, or MODEL_NAME).

.. warning::
   The quality of TODO completions depends on the LLM model used. More capable
   models typically produce better results. Processing large directories may
   consume significant API quota and time.

Example::

    >>> # Process a single file
    >>> from hbllmutils.entry.code.todo import complete_todo_for_file
    >>> complete_todo_for_file(
    ...     'myproject/utils.py',
    ...     model_name='gpt-4',
    ...     timeout=240
    ... )
    
    >>> # Use via command line
    >>> # hbllmutils code todo -i myproject/utils.py -m gpt-4
    >>> # hbllmutils code todo -i myproject/ -m gpt-4 --timeout 300
    >>> # hbllmutils code todo -i src/ -p max_tokens=128000 -p temperature=0.7

"""

import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Union, Tuple

import click
from hbutils.logging import ColoredFormatter, tqdm

from ..base import CONTEXT_SETTINGS, parse_key_value_params
from ...meta.code import create_todo_completion_task, is_python_file
from ...model import load_llm_model_from_config
from ...utils import obj_hashable, get_global_logger


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None, timeout: int = 240,
                  is_python_code: bool = True,
                  extra_params: Tuple[Tuple[str, Union[str, int, float]], ...] = ()):
    """
    Create and cache an LLM task instance for TODO completion.

    This function creates a TODO completion task configured with the specified LLM model
    and parameters. Results are cached using lru_cache to avoid recreating identical
    task instances, improving performance when processing multiple files with the same
    configuration.

    The function converts extra_params from tuple format (required for hashability) back
    to a dictionary for model initialization. The caching mechanism uses the combination
    of model_name, timeout, and extra_params as the cache key.

    :param model_name: Name of the LLM model to use (e.g., 'gpt-4', 'claude-3').
                       If None, uses the default model from configuration.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
    :type timeout: int
    :param extra_params: Additional model parameters as a tuple of (key, value) pairs.
                        Must be a tuple for hashability and caching. Common parameters
                        include max_tokens, temperature, top_p, etc.
    :type extra_params: Tuple[Tuple[str, Union[str, int, float]], ...]

    :return: Configured TODO completion task instance ready for processing files
    :rtype: PythonCodeGenerationLLMTask

    :raises ValueError: If model_name is invalid or model cannot be loaded
    :raises TypeError: If extra_params contains non-hashable values
    :raises RuntimeError: If model configuration fails

    .. note::
       This function uses lru_cache for performance optimization. The cache is
       unbounded by default, so identical parameter combinations will reuse the
       same task instance across multiple calls.

    .. note::
       The extra_params parameter must be a tuple of tuples for hashability.
       Use :func:`obj_hashable` to convert dictionaries to this format when
       calling this function.

    .. warning::
       Cached task instances persist for the lifetime of the Python process.
       Changes to model configuration files will not affect already-cached tasks.

    Example::

        >>> from hbllmutils.entry.code.todo import _get_llm_task
        >>> 
        >>> # Create task with default settings
        >>> task = _get_llm_task('gpt-4')
        >>> 
        >>> # Create task with custom timeout
        >>> task = _get_llm_task('gpt-4', timeout=300)
        >>> 
        >>> # Create task with extra parameters
        >>> extra = (('max_tokens', 128000), ('temperature', 0.7))
        >>> task = _get_llm_task('gpt-4', timeout=240, extra_params=extra)
        >>> 
        >>> # Subsequent calls with same parameters return cached instance
        >>> task2 = _get_llm_task('gpt-4', timeout=240, extra_params=extra)
        >>> assert task is task2  # Same object instance

    """
    params = dict(extra_params)
    return create_todo_completion_task(
        model=load_llm_model_from_config(
            model_name=model_name,
            timeout=timeout,
            **params
        ),
        is_python_code=is_python_code,
    )


def complete_todo_for_file(file: str, model_name: Optional[str] = None, timeout: int = 240,
                           extra_params: Optional[Dict[str, Union[str, int, float]]] = None) -> None:
    """
    Complete TODO comments in a single Python source file using LLM.

    This function processes a Python source file to identify and complete TODO comments.
    It uses an LLM to generate appropriate code completions based on the context of each
    TODO, then validates and writes the completed code back to the original file.

    The completion process includes:
    
    * Loading and analyzing the source file
    * Extracting TODO comments and their context
    * Generating completions using the configured LLM
    * Validating generated code for syntax correctness
    * Updating the file with completed code

    :param file: Path to the Python source file to process. Must be a valid file path
                 pointing to a .py file with TODO comments to complete.
    :type file: str
    :param model_name: Name of the LLM model to use for completion (e.g., 'gpt-4').
                       If None, uses the default model from configuration.
    :type model_name: Optional[str]
    :param timeout: Timeout in seconds for LLM API requests. Defaults to 240 seconds.
                    Increase for complex files or slower models.
    :type timeout: int
    :param extra_params: Additional parameters to pass to the LLM model as a dictionary.
                        Common parameters include max_tokens, temperature, top_p, etc.
                        If None, uses default model parameters.
    :type extra_params: Optional[Dict[str, Union[str, int, float]]]

    :return: None. The function modifies the input file in place.
    :rtype: None

    :raises FileNotFoundError: If the specified file does not exist
    :raises ValueError: If the file is not a valid Python source file
    :raises RuntimeError: If TODO completion fails or generated code is invalid
    :raises PermissionError: If the file cannot be written due to permission issues

    .. note::
       This function overwrites the original file with the completed code.
       Consider backing up important files before processing.

    .. warning::
       The function uses max_retries=0, meaning it will not retry on failure.
       Ensure stable network connectivity and sufficient API quota before processing.

    .. warning::
       Generated code is validated for syntax but not for semantic correctness.
       Review completed TODOs to ensure they match intended functionality.

    Example::

        >>> from hbllmutils.entry.code.todo import complete_todo_for_file
        >>> 
        >>> # Complete TODOs in a single file
        >>> complete_todo_for_file('myproject/utils.py', model_name='gpt-4')
        
        >>> # With custom timeout and parameters
        >>> complete_todo_for_file(
        ...     'myproject/models.py',
        ...     model_name='gpt-4',
        ...     timeout=300,
        ...     extra_params={'max_tokens': 128000, 'temperature': 0.7}
        ... )
        
        >>> # Process with default model from environment
        >>> import os
        >>> os.environ['OPENAI_MODEL_NAME'] = 'gpt-4'
        >>> complete_todo_for_file('myproject/views.py')

    """
    get_global_logger().info(f'Complete TODOs for {file!r} ...')
    extra_params = obj_hashable(extra_params or {})

    print(file, is_python_file(file))

    new_docs = _get_llm_task(
        model_name=model_name,
        timeout=timeout,
        is_python_code=is_python_file(file),
        extra_params=extra_params,
    ).ask_then_parse(file, max_retries=0)
    new_docs = new_docs.rstrip()
    with open(file, 'w') as f:
        print(new_docs, file=f)


def _add_todo_subcommand(cli: click.Group) -> click.Group:
    """
    Register the 'todo' subcommand with the CLI application.

    This function adds a 'todo' subcommand to the provided Click command group,
    enabling TODO completion functionality through the command line interface.
    The subcommand supports processing both individual files and entire directories
    of Python source code.

    The registered command provides the following options:
    
    * Input path specification (file or directory)
    * LLM model selection
    * API timeout configuration
    * Additional model parameters

    :param cli: Click command group to which the 'todo' subcommand will be added.
                This is typically the main CLI application group.
    :type cli: click.Group

    :return: The modified Click command group with the 'todo' subcommand registered.
             Returns the same group object that was passed in.
    :rtype: click.Group

    .. note::
       This function is typically called during CLI initialization to register
       the subcommand. It should not be called directly by end users.

    .. note::
       The function sets up comprehensive logging with colored output for better
       visibility of processing status and any issues encountered.

    Example::

        >>> import click
        >>> from hbllmutils.entry.code.todo import _add_todo_subcommand
        >>> 
        >>> @click.group()
        >>> def cli():
        ...     pass
        >>> 
        >>> # Register the todo subcommand
        >>> cli = _add_todo_subcommand(cli)
        >>> 
        >>> # Now the CLI has the 'todo' command available
        >>> # Usage: cli todo -i myproject/ -m gpt-4

    """

    @cli.command('todo', help='Complete TODO items in Python code files using LLM.',
                 context_settings=CONTEXT_SETTINGS)
    @click.option('-i', '--input', 'input_path', type=str, required=True,
                  help='Input Python file or directory to process for TODO completion.')
    @click.option('-m', '--model-name', 'model_name', type=str, required=False, default=None,
                  help='LLM model name to use for TODO completion.')
    @click.option('--timeout', 'timeout', type=int, required=False, default=210,
                  help='Timeout in seconds for LLM API requests.')
    @click.option('-p', '--param', 'params', type=str, multiple=True,
                  help='Additional parameters in key=value format (e.g., --param max_tokens=128000). '
                       'Can be used multiple times.',
                  callback=lambda ctx, param, value: dict(parse_key_value_params(v) for v in value) if value else {})
    def todo(input_path, model_name, timeout, params):
        """
        Complete TODO comments in Python source files using LLM.

        This command processes Python source files to identify and complete TODO comments
        using a Large Language Model. It can handle both individual files and entire
        directories, processing all .py files found recursively.

        The command performs the following steps:
        
        1. Validates the input path and determines if it's a file or directory
        2. Configures the LLM model from command-line options or environment variables
        3. For directories, discovers all Python files recursively
        4. Processes each file to complete TODO comments
        5. Updates files in place with completed code
        6. Provides progress feedback and error reporting

        :param input_path: Path to the Python file or directory to process.
                          For directories, all .py files are processed recursively.
        :type input_path: str
        :param model_name: Name of the LLM model to use. If None, attempts to load
                          from environment variables in order: OPENAI_MODEL_NAME,
                          LLM_MODEL_NAME, MODEL_NAME.
        :type model_name: Optional[str]
        :param timeout: Timeout in seconds for each LLM API request.
        :type timeout: int
        :param params: Dictionary of additional parameters parsed from --param options.
                      Supports parameters like max_tokens, temperature, top_p, etc.
        :type params: dict

        :raises FileNotFoundError: If the input path does not exist
        :raises RuntimeError: If the input path is neither a file nor a directory
        :raises Exception: If TODO completion fails for any file (re-raised after logging)

        .. note::
           The command uses colored logging output for better visibility of different
           log levels and processing status.

        .. note::
           When processing directories, a progress bar shows completion status for
           all discovered Python files.

        .. warning::
           Files are modified in place. Ensure you have backups or version control
           before processing important code.

        .. warning::
           Processing large directories may consume significant API quota and time.
           Monitor progress and consider processing in batches if needed.

        Example::

            # Process a single file with default model
            $ hbllmutils code todo -i myproject/utils.py
            
            # Process with specific model and timeout
            $ hbllmutils code todo -i myproject/models.py -m gpt-4 --timeout 300
            
            # Process entire directory with extra parameters
            $ hbllmutils code todo -i src/ -m gpt-4 -p max_tokens=128000 -p temperature=0.7
            
            # Process with model from environment variable
            $ export OPENAI_MODEL_NAME=gpt-4
            $ hbllmutils code todo -i myproject/

        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        get_global_logger().debug(f'Starting TODO completion with input: {input_path!r}')
        get_global_logger().debug(f'Model name: {model_name!r}, timeout: {timeout}s')

        extra_params = params
        if extra_params:
            get_global_logger().info(f'Extra parameters: {extra_params}')

        llm_model = (model_name or os.environ.get('OPENAI_MODEL_NAME')
                     or os.environ.get('LLM_MODEL_NAME') or os.environ.get('MODEL_NAME'))
        get_global_logger().info(f'Using LLM model: {llm_model!r}')

        if not os.path.exists(input_path):
            get_global_logger().error(f'File not found - {input_path!r}')
            raise FileNotFoundError(f'File not found - {input_path!r}.')
        elif os.path.isfile(input_path):
            get_global_logger().info(f'Processing single file: {input_path!r}')
            complete_todo_for_file(input_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
            get_global_logger().info(f'Successfully completed TODOs in {input_path!r}')
        elif os.path.isdir(input_path):
            get_global_logger().info(f'Processing directory: {input_path!r}')
            py_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    _, ext = os.path.splitext(os.path.normcase(file))
                    if ext == '.py':
                        file_path = os.path.join(root, file)
                        py_files.append(file_path)

            get_global_logger().info(f'Found {len(py_files)} Python files to process')
            for file_path in tqdm(py_files, desc=f'Complete Codes in {input_path!r}', total=len(py_files)):
                try:
                    complete_todo_for_file(file_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
                except Exception as e:
                    get_global_logger().exception(f'Failed to complete TODOs in {file_path!r}: {e}')
                    raise
            get_global_logger().info(f'Successfully completed TODOs in all {len(py_files)} files')
        else:
            get_global_logger().error(f'Unknown input type - {input_path!r}')
            raise RuntimeError(f'Unknown input - {input_path!r}.')

    return cli
