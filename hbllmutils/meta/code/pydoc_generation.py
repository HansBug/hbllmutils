"""
Module for generating Python documentation (pydoc) using LLM-based code analysis.

This module provides functionality to create specialized LLM tasks for generating
comprehensive Python documentation in reStructuredText format. It leverages detailed
source code analysis to produce high-quality docstrings that include:

- Functional analysis for modules, classes, methods, and functions
- Parameter descriptions with type annotations
- Return value documentation
- Exception documentation
- Usage examples

The generated documentation follows reStructuredText conventions and can be used
directly in Python source files for tools like Sphinx.

Example::
    >>> from hbllmutils.model import load_llm_model
    >>> model = load_llm_model('gpt-4')
    >>> task = create_pydoc_generation_task(model, show_module_directory_tree=True)
    >>> documented_code = task.ask_then_parse(input_content='mymodule.py')
    >>> print(documented_code)
    # Output will contain original code with comprehensive pydoc
"""

import os

from .task import PythonDetailedCodeGenerationLLMTask, PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model
from ...template import PromptTemplate


def create_pydoc_generation_task(model: LLMModelTyping, show_module_directory_tree: bool = False,
                                 skip_when_error: bool = True) -> PythonCodeGenerationLLMTask:
    """
    Create an LLM task for generating Python documentation (pydoc) in reStructuredText format.

    This function creates a specialized code generation task that analyzes Python source files
    and generates comprehensive documentation including:

    - Module-level docstrings with functional descriptions
    - Class, method, and function docstrings in reStructuredText format
    - Parameter and return value documentation with type hints
    - Exception documentation
    - Usage examples where appropriate

    The task uses a predefined system prompt (rst-doc-req.md) that instructs the LLM on
    documentation requirements and formatting conventions.

    :param model: The LLM model to use for documentation generation. Can be a model instance,
                 model name string, or None to use the default configured model.
    :type model: LLMModelTyping
    :param show_module_directory_tree: If True, includes the module's directory tree structure
                                      in the analysis prompt to provide additional context.
                                      Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skips over imports that fail to load during dependency
                           analysis and issues warnings instead of raising exceptions.
                           Defaults to True.
    :type skip_when_error: bool

    :return: A configured LLM task ready to generate Python documentation for source files.
    :rtype: PythonCodeGenerationLLMTask

    :raises FileNotFoundError: If the system prompt file (rst-doc-req.md) is not found.
    :raises ValueError: If the model specification is invalid.

    Example::
        >>> from hbllmutils.model import load_llm_model
        >>> # Create a pydoc generation task with GPT-4
        >>> model = load_llm_model('gpt-4')
        >>> task = create_pydoc_generation_task(
        ...     model=model,
        ...     show_module_directory_tree=True,
        ...     skip_when_error=False
        ... )
        >>> # Generate documentation for a Python file
        >>> documented_code = task.ask_then_parse(input_content='mypackage/module.py')
        >>> # Save the documented code
        >>> with open('mypackage/module.py', 'w') as f:
        ...     f.write(documented_code)
    """
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'rst-doc-req.md')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render()

    return PythonDetailedCodeGenerationLLMTask(
        model=load_llm_model(model),
        code_name='Code For Task',
        description_text='This is the source code for you to generate new code with pydoc',
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
    )
