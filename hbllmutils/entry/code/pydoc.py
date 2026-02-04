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
    new_docs = _get_llm_task(model_name, timeout, extra_params).ask_then_parse(file)
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

        extra_params = {}
        for param in params:
            if '=' not in param:
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

        llm_model = (model_name or os.environ.get('OPENAI_MODEL_NAME')
                     or os.environ.get('LLM_MODEL_NAME') or os.environ.get('MODEL_NAME'))
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'File not found - {input_path!r}.')
        elif os.path.isfile(input_path):
            generate_pydoc_for_file(input_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
        elif os.path.isdir(input_path):
            py_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    _, ext = os.path.splitext(os.path.normcase(file))
                    if ext == '.py':
                        file_path = os.path.join(root, file)
                        py_files.append(file_path)

            for file_path in tqdm(py_files, desc=f'Generate Docs in {input_path!r}', total=len(py_files)):
                generate_pydoc_for_file(file_path, model_name=llm_model, timeout=timeout, extra_params=extra_params)
        else:
            raise RuntimeError(f'Unknown input - {input_path!r}.')

    return cli
