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
"""
import argparse
import glob
import os
from functools import lru_cache
from typing import Optional

from hbllmutils.meta.code.pydoc_generation import create_pydoc_generation_task


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None):
    return create_pydoc_generation_task(
        model=model_name
    )


def make_doc_for_file(file: str, model_name: Optional[str] = None) -> None:
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
    entire directory tree.

    :param directory: Path to the directory containing Python files.
    :type directory: str

    :raises FileNotFoundError: If the specified directory does not exist.
    :raises PermissionError: If files cannot be read or written.

    Example::
        >>> make_doc_file_directory('./my_project')
        Make docs for './my_project/module.py' ...
        Make docs for './my_project/subdir/another.py' ...
        >>> # All .py files in my_project and subdirectories are now documented
    """
    for file in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        make_doc_for_file(file, model_name=model_name)


def main():
    """
    Main function to parse command-line arguments and generate documentation.

    This function serves as the entry point for the command-line interface.
    It parses arguments to determine whether to process a single file or
    an entire directory, then calls the appropriate documentation generation
    function.

    :raises FileNotFoundError: If the specified input path does not exist.
    :raises RuntimeError: If the input path is neither a file nor a directory.

    Example::
        >>> # Command line usage:
        >>> # python script.py -i my_file.py
        >>> # python script.py -i ./my_project
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
