"""
This module provides comprehensive utilities for analyzing Python source code and its dependencies.

The ``hbllmutils.meta.code`` package offers a collection of tools for extracting, analyzing, and
inspecting Python source code, import statements, module structures, and object metadata. It is
designed to facilitate code analysis, documentation generation, and understanding code dependencies.

Main components exported by this module:

- **Import Analysis:**
  
  - :class:`ImportStatement`: Represents a regular import statement
  - :func:`analyze_imports`: Extracts all import statements from Python code

- **Module Path Resolution:**
  
  - :func:`get_package_name`: Converts a source file path to its Python module name
  - :func:`get_pythonpath_of_source_file`: Determines the PYTHONPATH and module path for a file

- **Object Inspection:**
  
  - :class:`ObjectInspect`: Contains inspection information about a Python object
  - :func:`get_object_info`: Retrieves comprehensive metadata about any Python object

- **Prompt Generation:**
  
  - :func:`get_prompt_for_source_file`: Generates structured prompts for LLM analysis

- **Source Analysis:**
  
  - :class:`ImportSource`: Pairs import statements with object inspection data
  - :class:`SourceInfo`: Contains comprehensive information about a Python source file
  - :func:`get_source_info`: Analyzes a source file and extracts all relevant information

Example Usage::
    >>> from hbllmutils.meta.code import analyze_imports, get_source_info, get_prompt_for_source_file
    
    >>> # Analyze imports in code
    >>> code = '''
    ... import os
    ... from typing import List, Dict
    ... '''
    >>> imports = analyze_imports(code)
    >>> print(imports[0])
    import os
    
    >>> # Get comprehensive source file information
    >>> info = get_source_info('mymodule.py')
    >>> print(info.package_name)
    'mypackage.mymodule'
    >>> print(len(info.imports))
    5
    
    >>> # Generate LLM prompt for code analysis
    >>> prompt = get_prompt_for_source_file('calculator.py')
    >>> # Use this prompt for documentation generation, testing, etc.

This module is particularly useful for:

- Static code analysis and introspection
- Automated documentation generation
- Dependency tracking and visualization
- Code understanding and refactoring tools
- LLM-assisted code analysis and generation
"""

from .imp import ImportStatement, analyze_imports
from .module import get_package_name, get_pythonpath_of_source_file, get_package_from_import
from .object import get_object_info, ObjectInspect
from .prompt import get_prompt_for_source_file
from .pydoc_generation import create_pydoc_generation_task
from .source import ImportSource, SourceInfo, get_source_info
from .task import PythonCodeGenerationLLMTask, PythonDetailedCodeGenerationLLMTask
from .tree import is_file_should_ignore, build_python_project_tree, get_python_project_tree_text
