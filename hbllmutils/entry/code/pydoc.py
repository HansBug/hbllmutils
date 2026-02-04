"""
Python Documentation Generation Module

This module provides command line interface functionality for generating Python documentation
using LLM models. It includes functionality to process Python files or directories and generate
comprehensive documentation with proper formatting and structure.
"""

import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Union, Tuple

import click
from hbutils.logging import ColoredFormatter, tqdm

from ..base import CONTEXT_SETTINGS
from ...meta.code import create_pydoc_generation_task
from ...model import load_llm_model_from_config
from ...utils import obj_hashable, get_global_logger


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None, timeout: int = 240,
                  extra_params: Tuple[Tuple[str, Union[str, int, float]], ...] = ()):
    params = dict(extra_params)
    return create_pydoc_generation_task(
        model=load_llm_model_from_config(
            model_name=model_name,
            timeout=timeout,
            **params
        )
    )


def generate_pydoc_for_file(file: str, model_name: Optional[str] = None, timeout: int = 240,
                            extra_params: Optional[Dict[str, Union[str, int, float]]] = None) -> None:
    get_global_logger().info(f'Make docs for {file!r} ...')
    extra_params = obj_hashable(extra_params or {})
    new_docs = _get_llm_task(model_name, timeout, extra_params).ask_then_parse(file, max_retries=0)
    new_docs = new_docs.rstrip()
    with open(file, 'w') as f:
        print(new_docs, file=f)


def _add_pydoc_subcommand(cli: click.Group) -> click.Group:
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
                       'Can be used multiple times.')
    def pydoc(input_path, model_name, timeout, params):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        get_global_logger().debug(f'Starting pydoc generation for input path: {input_path!r}')
        get_global_logger().debug(f'Model name: {model_name or "default"}')
        get_global_logger().debug(f'Timeout: {timeout}s')

        extra_params = {}
        for param in params:
            if '=' not in param:
                get_global_logger().error(f'Invalid parameter format: {param!r}. Expected format: key=value')
                raise ValueError(f'Invalid parameter format: {param!r}. Expected format: key=value')
            key, value = param.split('=', 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            extra_params[key] = value

        if extra_params:
            get_global_logger().info(f'Extra parameters: {extra_params}')

        llm_model = (model_name or os.environ.get('OPENAI_MODEL_NAME')
                     or os.environ.get('LLM_MODEL_NAME') or os.environ.get('MODEL_NAME'))
        get_global_logger().info(f'Using LLM model: {llm_model or "default"}')

        if not os.path.exists(input_path):
            get_global_logger().error(f'File not found - {input_path!r}.')
            raise FileNotFoundError(f'File not found - {input_path!r}.')
        elif os.path.isfile(input_path):
            get_global_logger().info(f'Processing single file: {input_path!r}')
            generate_pydoc_for_file(input_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
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
                    generate_pydoc_for_file(file_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
                except Exception as e:
                    get_global_logger().exception(f'Failed to generate documentation for {file_path!r}: {e}')
                    raise
            get_global_logger().info(f'Completed documentation generation for directory {input_path!r}')
        else:
            get_global_logger().error(f'Unknown input - {input_path!r}.')
            raise RuntimeError(f'Unknown input - {input_path!r}.')

    return cli
