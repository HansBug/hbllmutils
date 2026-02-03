"""
Module for automatically generating Python documentation using OpenAI's language models.

This module provides functionality to automatically generate reStructuredText format
documentation (pydoc) for Python code files using OpenAI's API. It processes Python
source files and adds comprehensive docstrings to functions, methods, and classes
while preserving existing comments and code structure.

The module supports:

- Single file documentation generation
- Batch processing of entire directories
- Automatic translation of non-English comments
- Conversion to reStructuredText format
- Type annotation suggestions

Main Features:

- Command-line interface for easy integration into workflows
- Configurable LLM model selection via arguments or environment variables
- Caching mechanism to optimize repeated calls with the same model
- Recursive directory processing for bulk documentation updates
"""
import argparse
import glob
import os
from functools import lru_cache
from typing import Optional

from hbllmutils.meta.code.pydoc_generation import create_pydoc_generation_task


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None):
    """
    Get or create a cached LLM task for documentation generation.

    This function uses LRU caching to avoid creating duplicate task instances
    for the same model. The cache ensures efficient reuse of task configurations
    when processing multiple files with the same model.

    :param model_name: Name of the LLM model to use for documentation generation.
                      If None, uses the default model configured in the system.
    :type model_name: Optional[str]

    :return: A configured pydoc generation task ready for use.
    :rtype: PythonCodeGenerationLLMTask

    Example::
        >>> task1 = _get_llm_task('gpt-4')
        >>> task2 = _get_llm_task('gpt-4')
        >>> task1 is task2  # Returns True due to caching
        True
    """
    return create_pydoc_generation_task(
        model=model_name
    )


def make_doc_for_file(file: str, model_name: Optional[str] = None) -> None:
    """
    Generate documentation for a single Python file.

    This function reads a Python source file, generates comprehensive documentation
    using an LLM, and overwrites the original file with the documented version.
    The generated documentation includes module-level descriptions, function/method
    docstrings in reStructuredText format, and type annotations.

    :param file: Path to the Python file to document.
    :type file: str
    :param model_name: Name of the LLM model to use for generation. If None,
                      uses the default model.
    :type model_name: Optional[str]

    :raises FileNotFoundError: If the specified file does not exist.
    :raises PermissionError: If the file cannot be read or written.
    :raises RuntimeError: If documentation generation fails.

    .. warning::
        This function overwrites the original file. Ensure you have backups
        or version control before running.

    Example::
        >>> make_doc_for_file('mypackage/utils.py')
        Make docs for 'mypackage/utils.py' ...
        >>> # File is now documented with generated docstrings
        >>> make_doc_for_file('mypackage/core.py', model_name='gpt-4')
        Make docs for 'mypackage/core.py' ...
        >>> # File documented using GPT-4 model
    """
    print(f'Make docs for {file!r} ...')
    new_docs = _get_llm_task(model_name).ask_then_parse(file)
    with open(file, 'w') as f:
        print(new_docs, file=f)


def make_doc_file_directory(directory: str, model_name: Optional[str] = None) -> None:
    """
    Generate documentation for all Python files in a directory recursively.

    This function walks through the specified directory and all its subdirectories,
    finding all Python (.py) files and generating documentation for each one.
    The process is performed recursively, documenting all Python files in the
    entire directory tree. Each file is processed independently, and the operation
    continues even if individual files fail (depending on the task configuration).

    :param directory: Path to the directory containing Python files. The function
                     will process all .py files in this directory and subdirectories.
    :type directory: str
    :param model_name: Name of the LLM model to use for generation. If None,
                      uses the default model.
    :type model_name: Optional[str]

    :raises FileNotFoundError: If the specified directory does not exist.
    :raises PermissionError: If files cannot be read or written.
    :raises NotADirectoryError: If the path exists but is not a directory.

    .. warning::
        This function modifies all Python files in the directory tree.
        Ensure you have backups or version control before running.

    Example::
        >>> make_doc_file_directory('./my_project')
        Make docs for './my_project/module.py' ...
        Make docs for './my_project/utils/helpers.py' ...
        Make docs for './my_project/tests/test_module.py' ...
        >>> # All .py files in my_project and subdirectories are now documented
        >>> make_doc_file_directory('./src', model_name='gpt-4')
        Make docs for './src/main.py' ...
        >>> # All files documented using GPT-4 model
    """
    for file in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        make_doc_for_file(file, model_name=model_name)


def main():
    """
    Main entry point for the command-line documentation generation interface.

    This function serves as the CLI entry point for the documentation generator.
    It parses command-line arguments to determine the input path (file or directory)
    and optional model name, then calls the appropriate documentation generation
    function. The model name can be specified via command-line argument or through
    environment variables (OPENAI_MODEL_NAME or LLM_MODEL_NAME).

    Command-line Arguments:
        -i, --input: Required. Path to Python file or directory to document.
        -m, --model-name: Optional. LLM model name to use for generation.

    Environment Variables:
        OPENAI_MODEL_NAME: Default model name if not specified via argument.
        LLM_MODEL_NAME: Fallback model name if OPENAI_MODEL_NAME not set.

    :raises FileNotFoundError: If the specified input path does not exist.
    :raises RuntimeError: If the input path is neither a file nor a directory.
    :raises SystemExit: If required arguments are missing (handled by argparse).

    Example::
        >>> # Command line usage for single file:
        >>> # python remake_docs_via_llm.py -i my_file.py
        >>> # Command line usage for directory:
        >>> # python remake_docs_via_llm.py -i ./my_project
        >>> # Command line usage with specific model:
        >>> # python remake_docs_via_llm.py -i ./src -m gpt-4
        >>> # Using environment variable:
        >>> # export OPENAI_MODEL_NAME=gpt-4
        >>> # python remake_docs_via_llm.py -i ./src
    """
    parser = argparse.ArgumentParser(description='Auto create/update docs for file or directory')
    parser.add_argument('-i', '--input', required=True, help='Input code file or directory')
    parser.add_argument('-m', '--model-name', required=False, help='Model name for LLM', default=None)
    args = parser.parse_args()

    llm_model = args.model_name or os.environ.get('OPENAI_MODEL_NAME') or os.environ.get('LLM_MODEL_NAME')
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'File not found - {args.input!r}.')
    elif os.path.isfile(args.input):
        make_doc_for_file(args.input, model_name=llm_model)
    elif os.path.isdir(args.input):
        make_doc_file_directory(args.input, model_name=llm_model)
    else:
        raise RuntimeError(f'Unknown input - {args.input!r}.')


if __name__ == "__main__":
    main()
