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

The module contains the following main components:

* :func:`create_pydoc_generation_task` - Factory function for creating pydoc generation tasks

.. note::
   This module requires access to the source file system to analyze code structure
   and dependencies. Ensure proper file permissions when processing source files.

.. warning::
   The generated documentation quality depends on the LLM model capabilities.
   More advanced models typically produce better structured and more accurate
   documentation.

Example::

    >>> from hbllmutils.model import load_llm_model
    >>> from hbllmutils.meta.code.pydoc_generation import create_pydoc_generation_task
    >>> 
    >>> # Create a pydoc generation task with GPT-4
    >>> model = load_llm_model('gpt-4')
    >>> task = create_pydoc_generation_task(model, show_module_directory_tree=True)
    >>> 
    >>> # Generate documentation for a Python file
    >>> documented_code = task.ask_then_parse(input_content='mymodule.py')
    >>> print(documented_code)
    # Output will contain original code with comprehensive pydoc
    >>> 
    >>> # Save the documented code back to file
    >>> with open('mymodule.py', 'w') as f:
    ...     f.write(documented_code)

"""

import os
from typing import Optional, Iterable

from .task import PythonDetailedCodeGenerationLLMTask, PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model
from ...template import PromptTemplate


def create_pydoc_generation_task(
        model: LLMModelTyping,
        show_module_directory_tree: bool = False,
        skip_when_error: bool = True,
        force_ast_check: bool = True,
        ignore_modules: Optional[Iterable[str]] = None,
        no_ignore_modules: Optional[Iterable[str]] = None
) -> PythonCodeGenerationLLMTask:
    """
    Create an LLM task for generating Python documentation (pydoc) in reStructuredText format.

    This function creates a specialized code generation task that analyzes Python source files
    and generates comprehensive documentation including:

    - Module-level docstrings with functional descriptions and component listings
    - Class docstrings with attribute and inheritance information
    - Method and function docstrings in reStructuredText format
    - Parameter and return value documentation with type hints
    - Exception documentation with descriptions
    - Usage examples demonstrating typical use cases

    The task uses a predefined system prompt template (pydoc_generation.md) that instructs the LLM
    on documentation requirements and formatting conventions. The prompt template provides
    detailed guidelines for:

    - reStructuredText syntax and formatting standards
    - Documentation structure for different code elements
    - Cross-referencing conventions using Sphinx directives
    - Type annotation standards
    - Example code formatting

    The generated task performs comprehensive source code analysis including:

    - Full source file content extraction
    - Package namespace and file location identification
    - Optional module directory tree visualization
    - Dependency analysis showing all imports and their implementations
    - AST-based syntax validation of generated code

    :param model: The LLM model to use for documentation generation. Can be:
                 - A string representing the model name (e.g., 'gpt-4', 'claude-2')
                 - An LLMModel instance for direct use
                 - None to use the default configured model
    :type model: LLMModelTyping
    :param show_module_directory_tree: If True, includes the module's directory tree structure
                                      in the analysis prompt to provide additional context about
                                      the file's location within the project hierarchy.
                                      This helps the LLM understand the module's organizational
                                      context. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skips over imports that fail to load during dependency
                           analysis and issues warnings instead of raising exceptions.
                           This allows documentation generation to proceed even when some
                           dependencies are unavailable. Defaults to True.
    :type skip_when_error: bool
    :param force_ast_check: If True, validates the generated code using Python's AST parser
                           to ensure syntactic correctness. The task will retry generation
                           if validation fails. Defaults to True.
    :type force_ast_check: bool
    :param ignore_modules: Optional iterable of module names that should be explicitly ignored
                          during dependency analysis regardless of download count or other criteria.
    :type ignore_modules: Optional[Iterable[str]]
    :param no_ignore_modules: Optional iterable of module names that should never be ignored
                             during dependency analysis regardless of download count or other
                             filtering criteria.
    :type no_ignore_modules: Optional[Iterable[str]]

    :return: A configured LLM task ready to generate Python documentation for source files.
            The task can be used with the ask_then_parse() method to process Python files
            and return documented code.
    :rtype: PythonCodeGenerationLLMTask

    :raises FileNotFoundError: If the system prompt template file (pydoc_generation.md) is not found
                              in the module directory.
    :raises ValueError: If the model specification is invalid or cannot be loaded.
    :raises TypeError: If model parameter is not of type LLMModelTyping.

    .. note::
       The generated documentation quality and style depend on the capabilities of the
       selected LLM model. More advanced models (e.g., GPT-4) typically produce more
       accurate and well-structured documentation.

    .. warning::
       For large modules with many dependencies, the analysis prompt can become very long
       and may exceed token limits for some LLM models. Consider the model's context window
       when using this function with complex codebases.

    Example::

        >>> from hbllmutils.model import load_llm_model
        >>> from hbllmutils.meta.code.pydoc_generation import create_pydoc_generation_task
        >>> 
        >>> # Create a basic pydoc generation task
        >>> model = load_llm_model('gpt-4')
        >>> task = create_pydoc_generation_task(model)
        >>> 
        >>> # Generate documentation for a Python file
        >>> documented_code = task.ask_then_parse(input_content='mypackage/module.py')
        >>> 
        >>> # Save the documented code
        >>> with open('mypackage/module.py', 'w') as f:
        ...     f.write(documented_code)
        >>> 
        >>> # Create a task with directory tree visualization
        >>> task = create_pydoc_generation_task(
        ...     model=model,
        ...     show_module_directory_tree=True,
        ...     skip_when_error=False
        ... )
        >>> 
        >>> # Process multiple files in a batch
        >>> files = ['module1.py', 'module2.py', 'module3.py']
        >>> for file in files:
        ...     try:
        ...         documented = task.ask_then_parse(input_content=file)
        ...         with open(file, 'w') as f:
        ...             f.write(documented)
        ...         print(f"Successfully documented {file}")
        ...     except Exception as e:
        ...         print(f"Failed to document {file}: {e}")
        >>> 
        >>> # Use with default model from configuration
        >>> task = create_pydoc_generation_task(
        ...     model=None,
        ...     show_module_directory_tree=True
        ... )
        >>> documented = task.ask_then_parse(input_content='utils.py')
    """
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'pydoc_generation.md')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render()

    return PythonDetailedCodeGenerationLLMTask(
        model=load_llm_model(model),
        code_name='Code For Task',
        description_text='This is the source code for you to generate new code with pydoc',
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
        ignore_modules=ignore_modules,
        no_ignore_modules=no_ignore_modules,
    )
