"""
Response parsing and extraction utilities for LLM outputs.

This package provides comprehensive tools for parsing and extracting structured data
from Large Language Model (LLM) responses. It handles common challenges in working with
LLM-generated content, including malformed output, inconsistent formatting, and the need
for automatic retry mechanisms.

The package is organized into the following modules:

* :mod:`code` - Code block extraction and JSON parsing utilities
* :mod:`datamodel` - Data model-based task creation and validation
* :mod:`parsable` - Parsable LLM tasks with automatic retry mechanisms

Key Features:
    - **Code Extraction**: Extract code blocks from Markdown-formatted text with
      optional language filtering
    - **Robust JSON Parsing**: Parse JSON with automatic repair for malformed output
    - **Data Model Integration**: Create LLM tasks that automatically validate outputs
      against Pydantic models or dataclasses
    - **Automatic Retry**: Configurable retry mechanisms for handling parse failures
    - **Comprehensive Error Tracking**: Detailed exception information for debugging
      failed parsing attempts

Public API:
    The package exposes the following public interfaces through this module:

    **Functions:**
        - :func:`extract_code` - Extract code blocks from Markdown text
        - :func:`parse_json` - Parse JSON with optional automatic repair
        - :func:`create_datamodel_task` - Create data model-based LLM tasks

    **Classes:**
        - :class:`ParsableLLMTask` - LLM task with automatic retry on parse failure
        - :class:`OutputParseFailed` - Exception for parse failure tracking
        - :class:`OutputParseWithException` - Data class for failed parse attempts

Common Usage Patterns:
    The most common usage involves extracting structured data from LLM responses:

    .. code-block:: python

        from hbllmutils.response import extract_code, parse_json, ParsableLLMTask

        # Extract code from Markdown
        markdown_text = '''```python
        def hello():
            print("Hello, world!")
        ```'''
        code = extract_code(markdown_text, language='python')

        # Parse JSON with automatic repair
        malformed_json = '{"name": "Alice", "age": 30'  # Missing closing brace
        data = parse_json(malformed_json, with_repair=True)

        # Create custom parsable task
        class MyTask(ParsableLLMTask):
            def _parse_and_validate(self, content: str):
                return parse_json(extract_code(content))

Workflow Integration:
    This package is designed to integrate seamlessly with LLM workflows:

    1. **Send Request**: Use LLM model to generate response
    2. **Extract Content**: Use :func:`extract_code` to extract code blocks
    3. **Parse Data**: Use :func:`parse_json` to parse structured data
    4. **Validate**: Use :class:`ParsableLLMTask` for automatic validation and retry
    5. **Handle Errors**: Catch :exc:`OutputParseFailed` for comprehensive error info

.. note::
   This package is designed to work with the hbllmutils LLM framework. It requires
   the base LLM model and history classes from the parent package.

.. warning::
   Automatic retry mechanisms may incur additional API costs. Set appropriate retry
   limits based on your use case and budget constraints.

Example::

    >>> from hbllmutils.response import (
    ...     extract_code, parse_json, create_datamodel_task,
    ...     ParsableLLMTask, OutputParseFailed
    ... )
    >>> from pydantic import BaseModel
    >>> 
    >>> # Extract code from Markdown
    >>> markdown = '''```python
    ... def greet(name):
    ...     return f"Hello, {name}!"
    ... ```'''
    >>> code = extract_code(markdown, language='python')
    >>> print(code)
    def greet(name):
        return f"Hello, {name}!"
    
    >>> # Parse JSON with repair
    >>> malformed = '{"name": "Bob", "age": 25'
    >>> data = parse_json(malformed, with_repair=True)
    >>> print(data)
    {'name': 'Bob', 'age': 25}
    
    >>> # Create data model task
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    >>> 
    >>> task = create_datamodel_task(
    ...     model=my_llm_model,
    ...     datamodel_class=Person,
    ...     task_requirements="Extract person information from text"
    ... )
    >>> result = task.ask_then_parse("John Doe is 30 years old")
    >>> print(result.name, result.age)
    John Doe 30
    
    >>> # Custom parsable task with retry
    >>> class JSONTask(ParsableLLMTask):
    ...     __exceptions__ = (ValueError, KeyError)
    ...     
    ...     def _parse_and_validate(self, content: str):
    ...         data = parse_json(extract_code(content))
    ...         if 'result' not in data:
    ...             raise KeyError("Missing 'result' field")
    ...         return data['result']
    >>> 
    >>> task = JSONTask(my_llm_model, default_max_retries=3)
    >>> try:
    ...     result = task.ask_then_parse("Calculate 2+2")
    ... except OutputParseFailed as e:
    ...     print(f"Failed after {len(e.tries)} attempts")
    ...     for attempt in e.tries:
    ...         print(f"Error: {attempt.exception}")

"""

from .code import extract_code, parse_json
from .datamodel import create_datamodel_task
from .parsable import OutputParseWithException, OutputParseFailed, ParsableLLMTask
