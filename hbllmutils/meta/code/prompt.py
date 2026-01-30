"""
This module provides functionality for generating comprehensive code prompts for LLM analysis.

It creates structured prompts containing source code and its dependencies, suitable for various
LLM tasks such as documentation generation, unit testing, code analysis, and code understanding.
The generated prompts include the primary source code, package information, and detailed
dependency analysis with import statements and their implementations.
"""

import io
from typing import Optional

from hbutils.string import titleize

from .module import get_package_name
from .source import get_source_info


def get_prompt_for_source_file(source_file: str, level: int = 2, code_name: Optional[str] = 'primary') -> str:
    """
    Generate a comprehensive code prompt for LLM analysis.

    This function creates a structured prompt containing source code and its dependencies,
    suitable for various LLM tasks like documentation generation, unit testing, or code analysis.
    The prompt includes:
    
    - Primary source code analysis section with file location, package namespace, and complete source
    - Dependency analysis section with all imported dependencies and their implementations
    - For each import, includes the import statement, source file location, full package path,
      and either the implementation source code or object representation
    
    The generated prompt is formatted in Markdown with code blocks and hierarchical headers,
    making it easy for LLMs to parse and understand the code structure and dependencies.

    :param source_file: The path to the Python source file to generate a prompt for.
    :type source_file: str
    :param level: The heading level for the main sections in the generated Markdown.
                  Defaults to 2 (##). Subsections will use level+1.
    :type level: int
    :param code_name: The name to use for the code section title. If None, uses 'Source Code Analysis'.
                      Defaults to 'primary'.
    :type code_name: Optional[str]

    :return: A formatted Markdown string containing the comprehensive code prompt.
    :rtype: str

    Example::
        >>> # Generate a prompt for a Python module
        >>> prompt = get_prompt_for_source_file('mypackage/mymodule.py')
        >>> print(prompt[:100])
        '## Primary Source Code Analysis\\n\\n**Source File Location:** `mypackage/mymodule.py`\\n\\n**Package...'
        
        >>> # Generate with custom heading level
        >>> prompt = get_prompt_for_source_file('mymodule.py', level=3)
        >>> # Will use ### for main sections and #### for subsections
        
        >>> # Use the prompt for LLM tasks
        >>> prompt = get_prompt_for_source_file('calculator.py')
        >>> # Feed this prompt to an LLM for documentation generation, testing, etc.
        
        >>> # Generate without code name prefix
        >>> prompt = get_prompt_for_source_file('mymodule.py', code_name=None)
        >>> # Title will be 'Source Code Analysis' instead of 'Primary Source Code Analysis'
    """
    source_info = get_source_info(source_file)

    with io.StringIO() as sf:
        # Main source code section
        if code_name:
            title = f'{code_name} Source Code Analysis'
        else:
            title = f'Source Code Analysis'
        print(f'{"#" * level} {titleize(title)}', file=sf)
        print(f'', file=sf)
        print(f'**Source File Location:** `{source_info.source_file}`', file=sf)
        print(f'', file=sf)
        print(f'**Package Namespace:** `{source_info.package_name}`', file=sf)
        print(f'', file=sf)
        print(f'**Complete Source Code:**', file=sf)
        print(f'', file=sf)
        print(f'```python', file=sf)
        print(source_info.source_code, file=sf)
        print(f'```', file=sf)
        print(f'', file=sf)

        # Import dependencies section
        if source_info.imports:
            print(f'{"#" * level} Dependency Analysis - Import Statements and Their Implementations', file=sf)
            print(f'', file=sf)
            print(f'The following section contains all imported dependencies for package `{source_info.package_name}` '
                  f'along with their source code implementations. This information can be used as reference context '
                  f'for understanding the main code\'s functionality and dependencies.', file=sf)
            print(f'', file=sf)

            for imp in source_info.imports:
                print(f'{"#" * (level + 1)} Import: `{imp.statement}`', file=sf)
                print(f'', file=sf)

                # Source file information
                if imp.inspect.source_file:
                    print(f'**Source File:** `{imp.inspect.source_file}`', file=sf)
                    print(f'', file=sf)
                    print(f'**Full Package Path:** `{get_package_name(imp.inspect.source_file)}.{imp.statement.name}`',
                          file=sf)
                    print(f'', file=sf)

                # Source code or object representation
                if imp.inspect.has_source:
                    print(f'**Implementation Source Code:**', file=sf)
                    print(f'', file=sf)
                    print(f'```python', file=sf)
                    print(imp.inspect.source_code, file=sf)
                    print(f'```', file=sf)
                    print(f'', file=sf)
                else:
                    print(
                        f'**Note:** Source code is not available through Python\'s inspection mechanism. Below is the object representation:',
                        file=sf)
                    print(f'', file=sf)
                    print(f'```', file=sf)
                    print(imp.inspect.object, file=sf)
                    print(f'```', file=sf)
                    print(f'', file=sf)

        return sf.getvalue()
