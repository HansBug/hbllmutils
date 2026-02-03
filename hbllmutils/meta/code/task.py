"""
This module provides LLM task implementations for Python code generation with validation and parsing capabilities.

It includes specialized tasks for generating and validating Python code using LLM models,
with features like:

- Automatic syntax validation of generated Python code using AST parsing
- Support for detailed code generation with customizable prompts
- Integration with source file analysis and dependency tracking
- Configurable retry mechanisms for code generation failures
- Module directory tree visualization support

The module extends the base ParsableLLMTask to provide Python-specific code generation
and validation functionality.
"""

import ast
from typing import Optional

from .prompt import get_prompt_for_source_file
from ...history import LLMHistory
from ...model import LLMModel
from ...response import ParsableLLMTask, extract_code


class PythonCodeGenerationLLMTask(ParsableLLMTask):
    """
    An LLM task for generating and validating Python code with automatic syntax checking.

    This task extends ParsableLLMTask to provide Python-specific code generation capabilities.
    It automatically extracts code from the model's response and validates it using Python's
    AST parser to ensure syntactic correctness. The task will retry on parsing failures up
    to the configured maximum number of retries.

    The validation process:
    1. Extracts code from the model's response (handles both plain code and fenced code blocks)
    2. Parses the code using ast.parse() to validate Python syntax
    3. Returns the validated code if successful
    4. Raises an exception and retries if parsing fails

    Example::
        >>> from hbllmutils.model import LLMModel
        >>> model = LLMModel(...)
        >>> task = PythonCodeGenerationLLMTask(model, default_max_retries=3)
        >>> code = task.ask_then_parse(input_content="Write a function to add two numbers")
        >>> print(code)
        def add(a, b):
            return a + b
    """

    def _parse_and_validate(self, content: str) -> str:
        """
        Parse and validate Python code from the model's response.

        This method extracts code from the response content and validates it by
        attempting to parse it with Python's AST parser. If the code is syntactically
        valid, it returns the extracted code. Otherwise, it raises a SyntaxError that
        will trigger a retry.

        :param content: The raw output string from the model, potentially containing
                       Python code in plain text or fenced code blocks.
        :type content: str

        :return: The extracted and validated Python code.
        :rtype: str

        :raises SyntaxError: If the extracted code is not valid Python syntax.
        :raises ValueError: If no code blocks are found in the response.

        Example::
            >>> task = PythonCodeGenerationLLMTask(model)
            >>> code = task._parse_and_validate("```python\\ndef foo():\\n    pass\\n```")
            >>> print(code)
            def foo():
                pass
        """
        code = extract_code(content)
        ast.parse(code)
        return code


class PythonDetailedCodeGenerationLLMTask(PythonCodeGenerationLLMTask):
    """
    An advanced LLM task for generating Python code with comprehensive source file analysis.

    This task extends PythonCodeGenerationLLMTask to provide detailed code generation
    capabilities that include:

    - Full source file analysis with package namespace information
    - Dependency analysis showing all imports and their implementations
    - Optional module directory tree visualization
    - Customizable code generation prompts and descriptions
    - Configurable error handling behavior

    The task preprocesses input by generating a comprehensive prompt that includes
    the source file content, its dependencies, and optional contextual information
    like the module directory structure. This enriched context helps the LLM generate
    more accurate and contextually appropriate code.

    :param model: The LLM model to use for code generation.
    :type model: LLMModel
    :param code_name: The name/label for the code section in the generated prompt.
                     Used as a prefix for the title (e.g., "primary" results in
                     "primary Source Code Analysis").
    :type code_name: str
    :param description_text: Optional descriptive text to include in the prompt,
                            providing additional context or instructions for code generation.
    :type description_text: str
    :param history: Optional conversation history. If None, a new history will be created.
    :type history: Optional[LLMHistory]
    :param default_max_retries: Maximum number of retry attempts for code generation and parsing.
                               Defaults to 5.
    :type default_max_retries: int
    :param show_module_directory_tree: If True, include a directory tree visualization of the
                                      module structure in the generated prompt. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load during analysis and issue
                           warnings instead of raising exceptions. Defaults to True.
    :type skip_when_error: bool

    Example::
        >>> from hbllmutils.model import LLMModel
        >>> model = LLMModel(...)
        >>> task = PythonDetailedCodeGenerationLLMTask(
        ...     model=model,
        ...     code_name="calculator",
        ...     description_text="Generate unit tests for this calculator module",
        ...     show_module_directory_tree=True,
        ...     default_max_retries=3
        ... )
        >>> code = task.ask_then_parse(input_content="path/to/calculator.py")
        >>> print(code)
        import unittest
        from calculator import add, subtract
        ...
    """

    def __init__(self, model: LLMModel, code_name: str, description_text: str,
                 history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 show_module_directory_tree: bool = False, skip_when_error: bool = True):
        """
        Initialize the PythonDetailedCodeGenerationLLMTask.

        :param model: The LLM model to use for code generation.
        :type model: LLMModel
        :param code_name: The name/label for the code section in the generated prompt.
        :type code_name: str
        :param description_text: Descriptive text providing context for code generation.
        :type description_text: str
        :param history: Optional conversation history. If None, creates a new history.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Maximum retry attempts for parsing. Defaults to 5.
        :type default_max_retries: int
        :param show_module_directory_tree: Whether to include directory tree in the prompt.
                                          Defaults to False.
        :type show_module_directory_tree: bool
        :param skip_when_error: Whether to skip failed imports during analysis. Defaults to True.
        :type skip_when_error: bool
        """
        super().__init__(model, history, default_max_retries)
        self.code_name = code_name
        self.description_text = description_text
        self.show_module_directory_tree = show_module_directory_tree
        self.skip_when_error = skip_when_error

    def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
        """
        Preprocess the input by generating a comprehensive code analysis prompt.

        This method transforms a simple file path into a rich, structured prompt containing:

        - Source file location and package namespace
        - Complete source code
        - Optional module directory tree visualization
        - Dependency analysis with all imports and their implementations
        - Custom description text for context

        The generated prompt provides extensive context to help the LLM understand the code
        structure and generate appropriate responses.

        :param input_content: The path to the Python source file to analyze.
                             Must not be None or empty.
        :type input_content: Optional[str]

        :return: A comprehensive Markdown-formatted prompt containing source code analysis
                and dependency information.
        :rtype: str

        :raises ValueError: If input_content is None or empty.

        Example::
            >>> task = PythonDetailedCodeGenerationLLMTask(
            ...     model=model,
            ...     code_name="example",
            ...     description_text="Analyze this code"
            ... )
            >>> prompt = task._preprocess_input_content("mypackage/module.py")
            >>> print(prompt[:50])
            '# Example Source Code Analysis\\n\\nAnalyze this code'
        """
        if input_content:
            return get_prompt_for_source_file(
                source_file=input_content,
                level=1,
                code_name=self.code_name,
                description_text=self.description_text,
                show_module_directory_tree=self.show_module_directory_tree,
                skip_when_error=self.skip_when_error,
            )
        else:
            raise ValueError('Empty content is not supported.')

