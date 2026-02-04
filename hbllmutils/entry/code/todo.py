"""
State Machine Code Generation Module

This module provides command line interface functionality for generating code from a state machine DSL.
It includes functionality to parse DSL code, convert it to a state machine model, and render the model
using templates to generate output code.
"""

import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Union, Tuple

import click
from hbutils.logging import ColoredFormatter, tqdm

from ..base import CONTEXT_SETTINGS
from ...meta.code import create_todo_completion_task
from ...model import load_llm_model_from_config
from ...utils import obj_hashable, get_global_logger


@lru_cache()
def _get_llm_task(model_name: Optional[str] = None, timeout: int = 240,
                  extra_params: Tuple[Tuple[str, Union[str, int, float]], ...] = ()):
    params = dict(extra_params)
    return create_todo_completion_task(
        model=load_llm_model_from_config(
            model_name=model_name,
            timeout=timeout,
            **params
        )
    )


def complete_todo_for_file(file: str, model_name: Optional[str] = None, timeout: int = 240,
                           extra_params: Optional[Dict[str, Union[str, int, float]]] = None) -> None:
    get_global_logger().info(f'Complete TODOs for {file!r} ...')
    extra_params = obj_hashable(extra_params or {})
    new_docs = _get_llm_task(model_name, timeout, extra_params).ask_then_parse(file, max_retries=0)
    new_docs = new_docs.rstrip()
    with open(file, 'w') as f:
        print(new_docs, file=f)


def _add_todo_subcommand(cli: click.Group) -> click.Group:
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
                       'Can be used multiple times.')
    def todo(input_path, model_name, timeout, params):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        get_global_logger().debug(f'Starting TODO completion with input: {input_path!r}')
        get_global_logger().debug(f'Model name: {model_name!r}, timeout: {timeout}s')

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
