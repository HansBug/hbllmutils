"""
LLM task implementations for Python code generation with validation and parsing capabilities.

This module provides specialized LLM task classes for generating and validating Python code
using Large Language Models. It extends the base parsable task framework to provide
Python-specific functionality with automatic syntax validation and comprehensive source
file analysis.

The module contains the following main components:

* :class:`PythonCodeGenerationLLMTask` - Basic Python code generation with AST validation
* :class:`PythonDetailedCodeGenerationLLMTask` - Advanced code generation with source analysis

Key features include:

- Automatic Python syntax validation using AST parsing
- Configurable retry mechanisms for code generation failures
- Integration with source file analysis and dependency tracking
- Support for detailed code generation with customizable prompts
- Module directory tree visualization support
- Comprehensive error handling and logging

.. note::
   All code generation tasks validate Python syntax using the ast module before
   returning results. This ensures generated code is syntactically correct.

.. warning::
   Code generation may require multiple LLM API calls if parsing fails, which
   can increase costs and response times. Configure max_retries appropriately.

Example::

    >>> from hbllmutils.model import LLMModel
    >>> from hbllmutils.meta.code.task import PythonCodeGenerationLLMTask
    >>> 
    >>> # Basic code generation
    >>> model = LLMModel(...)
    >>> task = PythonCodeGenerationLLMTask(model, default_max_retries=3)
    >>> code = task.ask_then_parse(input_content="Write a function to add two numbers")
    >>> print(code)
    def add(a, b):
        return a + b
    >>> 
    >>> # Detailed code generation with source analysis
    >>> from hbllmutils.meta.code.task import PythonDetailedCodeGenerationLLMTask
    >>> task = PythonDetailedCodeGenerationLLMTask(
    ...     model=model,
    ...     code_name="calculator",
    ...     description_text="Generate comprehensive unit tests",
    ...     show_module_directory_tree=True
    ... )
    >>> code = task.ask_then_parse(input_content="path/to/calculator.py")
"""

import ast
from typing import Optional, Iterable

from .prompt import get_prompt_for_source_file
from ...history import LLMHistory
from ...model import LLMModel
from ...response import ParsableLLMTask, extract_code


class PythonCodeGenerationLLMTask(ParsableLLMTask):
    """
    An LLM task for generating and validating Python code with automatic syntax checking.

    This task extends :class:`ParsableLLMTask` to provide Python-specific code generation
    capabilities. It automatically extracts code from the model's response and validates
    it using Python's AST parser to ensure syntactic correctness. The task will retry on
    parsing failures up to the configured maximum number of retries.

    The validation process:
    
    1. Extracts code from the model's response (handles both plain code and fenced code blocks)
    2. Parses the code using ast.parse() to validate Python syntax
    3. Returns the validated code if successful
    4. Raises an exception and retries if parsing fails

    The task catches :exc:`SyntaxError` and :exc:`ValueError` exceptions during parsing,
    which trigger automatic retries. Other exceptions will propagate immediately.

    :param model: The LLM model to use for code generation.
    :type model: LLMModel
    :param history: Optional conversation history. If None, a new history will be created.
    :type history: Optional[LLMHistory]
    :param default_max_retries: Maximum number of retry attempts for code generation and parsing.
                               Defaults to 5.
    :type default_max_retries: int
    :param force_ast_check: If True, always validate code with AST parsing. If False, skip
                           AST validation (useful for code snippets that may not be complete
                           valid Python modules). Defaults to True.
    :type force_ast_check: bool

    :ivar force_ast_check: Whether to enforce AST validation on generated code.
    :vartype force_ast_check: bool

    .. note::
       The task preserves trailing whitespace stripping on extracted code to ensure
       clean output formatting.

    .. warning::
       AST validation only checks syntax, not semantic correctness. The generated code
       may still contain logical errors or runtime issues.

    Example::

        >>> from hbllmutils.model import LLMModel
        >>> from hbllmutils.meta.code.task import PythonCodeGenerationLLMTask
        >>> 
        >>> model = LLMModel(...)
        >>> task = PythonCodeGenerationLLMTask(model, default_max_retries=3)
        >>> 
        >>> # Generate a simple function
        >>> code = task.ask_then_parse(input_content="Write a function to add two numbers")
        >>> print(code)
        def add(a, b):
            return a + b
        >>> 
        >>> # Generate with forced AST checking
        >>> task = PythonCodeGenerationLLMTask(model, force_ast_check=True)
        >>> code = task.ask_then_parse(input_content="Write a class for a calculator")
        >>> print(code)
        class Calculator:
            def add(self, a, b):
                return a + b
            def subtract(self, a, b):
                return a - b
        >>> 
        >>> # Handle generation failures
        >>> try:
        ...     code = task.ask_then_parse(
        ...         input_content="Generate invalid code",
        ...         max_retries=2
        ...     )
        ... except OutputParseFailed as e:
        ...     print(f"Failed after {len(e.tries)} attempts")
    """
    __exceptions__ = (SyntaxError, ValueError)

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 force_ast_check: bool = True):
        """
        Initialize the PythonCodeGenerationLLMTask.

        :param model: The LLM model to use for code generation.
        :type model: LLMModel
        :param history: Optional conversation history. If None, creates a new history.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Maximum retry attempts for parsing. Defaults to 5.
        :type default_max_retries: int
        :param force_ast_check: Whether to enforce AST validation. Defaults to True.
        :type force_ast_check: bool
        """
        super().__init__(model, history, default_max_retries)
        self.force_ast_check = force_ast_check

    def _parse_and_validate(self, content: str) -> str:
        """
        Parse and validate Python code from the model's response.

        This method extracts code from the response content and validates it by
        attempting to parse it with Python's AST parser (if force_ast_check is True).
        If the code is syntactically valid, it returns the extracted code. Otherwise,
        it raises a SyntaxError that will trigger a retry.

        The method uses :func:`extract_code` to handle both plain code and fenced
        code blocks in Markdown format. The extracted code is stripped of trailing
        whitespace before being returned.

        :param content: The raw output string from the model, potentially containing
                       Python code in plain text or fenced code blocks.
        :type content: str

        :return: The extracted and validated Python code with trailing whitespace removed.
        :rtype: str

        :raises SyntaxError: If the extracted code is not valid Python syntax (when
                            force_ast_check is True).
        :raises ValueError: If no code blocks are found in the response or if multiple
                           ambiguous code blocks are present.

        .. note::
           The method strips trailing whitespace from the extracted code but preserves
           internal formatting and indentation.

        Example::

            >>> task = PythonCodeGenerationLLMTask(model, force_ast_check=True)
            >>> 
            >>> # Parse valid Python code
            >>> code = task._parse_and_validate("```python\\ndef foo():\\n    pass\\n```")
            >>> print(code)
            def foo():
                pass
            >>> 
            >>> # Parse plain code without fencing
            >>> code = task._parse_and_validate("x = 42")
            >>> print(code)
            x = 42
            >>> 
            >>> # Invalid syntax raises SyntaxError
            >>> try:
            ...     code = task._parse_and_validate("def foo(")
            ... except SyntaxError as e:
            ...     print(f"Syntax error: {e}")
        """
        code = extract_code(content)
        if self.force_ast_check:
            ast.parse(code)
        return code.rstrip()


class PythonDetailedCodeGenerationLLMTask(PythonCodeGenerationLLMTask):
    """
    An advanced LLM task for generating Python code with comprehensive source file analysis.

    This task extends :class:`PythonCodeGenerationLLMTask` to provide detailed code generation
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

    The generated prompt includes:
    
    - Source file location and package namespace
    - Complete source code of the target file
    - Optional directory tree visualization
    - Dependency analysis with import statements and implementations
    - Custom description text for additional context

    :param model: The LLM model to use for code generation.
    :type model: LLMModel
    :param code_name: The name/label for the code section in the generated prompt.
                     Used as a prefix for the title (e.g., "primary" results in
                     "Primary Source Code Analysis"). If None, uses "Source Code Analysis".
    :type code_name: str
    :param description_text: Descriptive text to include in the prompt, providing
                            additional context or instructions for code generation.
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
    :param force_ast_check: If True, always validate generated code with AST parsing.
                           Defaults to True.
    :type force_ast_check: bool
    :param ignore_modules: Optional iterable of module names that should be explicitly ignored
                          during dependency analysis regardless of download count or other criteria.
    :type ignore_modules: Optional[Iterable[str]]
    :param no_ignore_modules: Optional iterable of module names that should never be ignored
                             during dependency analysis regardless of download count or other
                             filtering criteria.
    :type no_ignore_modules: Optional[Iterable[str]]

    :ivar code_name: The name/label for the code section in prompts.
    :vartype code_name: str
    :ivar description_text: Description text for prompt context.
    :vartype description_text: str
    :ivar show_module_directory_tree: Whether to include directory tree in prompts.
    :vartype show_module_directory_tree: bool
    :ivar skip_when_error: Whether to skip failed imports during analysis.
    :vartype skip_when_error: bool
    :ivar ignore_modules: Module names to explicitly ignore during analysis.
    :vartype ignore_modules: Optional[Iterable[str]]
    :ivar no_ignore_modules: Module names to never ignore during analysis.
    :vartype no_ignore_modules: Optional[Iterable[str]]

    .. note::
       This task is particularly useful for generating documentation, unit tests,
       or refactored code that requires understanding of the full module context.

    .. warning::
       Analyzing large modules with many dependencies can generate very long prompts,
       which may exceed token limits for some LLM models. Consider the model's context
       window when using this task.

    Example::

        >>> from hbllmutils.model import LLMModel
        >>> from hbllmutils.meta.code.task import PythonDetailedCodeGenerationLLMTask
        >>> 
        >>> model = LLMModel(...)
        >>> 
        >>> # Generate unit tests with full context
        >>> task = PythonDetailedCodeGenerationLLMTask(
        ...     model=model,
        ...     code_name="calculator",
        ...     description_text="Generate comprehensive unit tests for this module",
        ...     show_module_directory_tree=True,
        ...     default_max_retries=3
        ... )
        >>> code = task.ask_then_parse(input_content="path/to/calculator.py")
        >>> print(code)
        import unittest
        from calculator import add, subtract, multiply, divide
        
        class TestCalculator(unittest.TestCase):
            def test_add(self):
                self.assertEqual(add(2, 3), 5)
            ...
        >>> 
        >>> # Generate documentation with context
        >>> task = PythonDetailedCodeGenerationLLMTask(
        ...     model=model,
        ...     code_name="api_handler",
        ...     description_text="Generate comprehensive API documentation",
        ...     skip_when_error=True
        ... )
        >>> docs = task.ask_then_parse(input_content="mypackage/api.py")
        >>> 
        >>> # Handle analysis errors gracefully
        >>> task = PythonDetailedCodeGenerationLLMTask(
        ...     model=model,
        ...     code_name="module",
        ...     description_text="Analyze this code",
        ...     skip_when_error=False
        ... )
        >>> try:
        ...     code = task.ask_then_parse(input_content="problematic_module.py")
        ... except Exception as e:
        ...     print(f"Analysis failed: {e}")
    """

    def __init__(self, model: LLMModel, code_name: str, description_text: str,
                 history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 show_module_directory_tree: bool = False, skip_when_error: bool = True,
                 force_ast_check: bool = True, ignore_modules: Optional[Iterable[str]] = None,
                 no_ignore_modules: Optional[Iterable[str]] = None):
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
        :param force_ast_check: Whether to enforce AST validation. Defaults to True.
        :type force_ast_check: bool
        :param ignore_modules: Optional iterable of module names to explicitly ignore during
                              dependency analysis.
        :type ignore_modules: Optional[Iterable[str]]
        :param no_ignore_modules: Optional iterable of module names to never ignore during
                                 dependency analysis.
        :type no_ignore_modules: Optional[Iterable[str]]
        """
        super().__init__(model, history, default_max_retries, force_ast_check)
        self.code_name = code_name
        self.description_text = description_text
        self.show_module_directory_tree = show_module_directory_tree
        self.skip_when_error = skip_when_error
        self.ignore_modules: Optional[Iterable[str]] = ignore_modules
        self.no_ignore_modules: Optional[Iterable[str]] = no_ignore_modules

    def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
        """
        Preprocess the input by generating a comprehensive code analysis prompt.

        This method transforms a simple file path into a rich, structured prompt containing:

        - Source file location and package namespace information
        - Complete source code of the target file
        - Optional module directory tree visualization showing file location
        - Comprehensive dependency analysis with all imports and their implementations
        - Custom description text providing context for the generation task

        The generated prompt is formatted in Markdown with proper code blocks and
        hierarchical headers, making it easy for LLMs to parse and understand the
        code structure and dependencies.

        The method uses :func:`get_prompt_for_source_file` to perform the analysis
        and generate the structured prompt.

        :param input_content: The path to the Python source file to analyze.
                             Must not be None or empty.
        :type input_content: Optional[str]

        :return: A comprehensive Markdown-formatted prompt containing source code analysis
                and dependency information.
        :rtype: str

        :raises ValueError: If input_content is None or empty string.

        .. note::
           The generated prompt can be quite large for modules with many dependencies.
           Ensure your LLM model has sufficient context window to handle the prompt.

        .. warning::
           If skip_when_error is False, the method will raise exceptions for any
           imports that fail to load during analysis.

        Example::

            >>> task = PythonDetailedCodeGenerationLLMTask(
            ...     model=model,
            ...     code_name="example",
            ...     description_text="Analyze this code for documentation"
            ... )
            >>> 
            >>> # Generate prompt for a module
            >>> prompt = task._preprocess_input_content("mypackage/module.py")
            >>> print(prompt[:100])
            '# Example Source Code Analysis
            
            Analyze this code for documentation
            
            **Source File Location:** `mypackage/module.py`'
            >>> 
            >>> # With directory tree visualization
            >>> task = PythonDetailedCodeGenerationLLMTask(
            ...     model=model,
            ...     code_name="api",
            ...     description_text="Generate API docs",
            ...     show_module_directory_tree=True
            ... )
            >>> prompt = task._preprocess_input_content("mypackage/api.py")
            >>> 
            >>> # Error handling for empty input
            >>> try:
            ...     prompt = task._preprocess_input_content(None)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Empty content is not supported.
        """
        if input_content:
            return get_prompt_for_source_file(
                source_file=input_content,
                level=1,
                code_name=self.code_name,
                description_text=self.description_text,
                show_module_directory_tree=self.show_module_directory_tree,
                skip_when_error=self.skip_when_error,
                ignore_modules=self.ignore_modules,
                no_ignore_modules=self.no_ignore_modules,
            )
        else:
            raise ValueError('Empty content is not supported.')
