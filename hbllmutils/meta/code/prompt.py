"""
Code prompt generation for LLM analysis and documentation tasks.

This module provides functionality for generating comprehensive code prompts suitable
for Large Language Model (LLM) analysis. It creates structured prompts containing
source code and its dependencies, formatted in Markdown with clear sections for
primary source code and dependency analysis.

The module is designed to support various LLM tasks including:

* Automated documentation generation
* Unit test creation
* Code analysis and review
* Code understanding and explanation
* Refactoring suggestions

The generated prompts include:

* Primary source code with file location and package namespace information
* Optional module directory tree visualization
* Comprehensive dependency analysis with import statements
* Full implementation source code for each dependency
* Proper Markdown formatting with hierarchical headers and code blocks

.. note::
   This module requires the source file to be part of a valid Python package
   structure with proper __init__.py files for accurate package name resolution.

.. warning::
   Large projects with many dependencies may generate very large prompts that
   could exceed token limits of some LLM models. Consider filtering dependencies
   or processing files individually.

Example::

    >>> from hbllmutils.meta.code.prompt import get_prompt_for_source_file
    >>> 
    >>> # Generate a basic prompt for documentation
    >>> prompt = get_prompt_for_source_file('mymodule.py')
    >>> print(prompt[:200])
    '## Primary Source Code Analysis\\n\\n**Source File Location:** `mymodule.py`...'
    >>> 
    >>> # Generate with custom description and directory tree
    >>> prompt = get_prompt_for_source_file(
    ...     'mymodule.py',
    ...     description_text='Generate comprehensive pydoc for this module.',
    ...     show_module_directory_tree=True
    ... )

"""

import io
import os.path
import re
from typing import Optional, Iterable

from hbutils.string import titleize

from .module import get_package_name, get_pythonpath_of_source_file
from .source import get_source_info
from .tree import get_python_project_tree_text


def get_prompt_for_source_file(source_file: str, level: int = 2, code_name: Optional[str] = 'primary',
                               description_text: Optional[str] = None, show_module_directory_tree: bool = False,
                               skip_when_error: bool = True, min_last_month_downloads: int = 1000000,
                               no_ignore_modules: Optional[Iterable[str]] = None) -> str:
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
    :param description_text: Optional description text to include after the title and before
                            the source file information. Can be used to provide context or
                            instructions for the LLM.
    :type description_text: Optional[str]
    :param show_module_directory_tree: If True, include a directory tree visualization of the module
                                      structure with the current file highlighted. Defaults to False.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load and issue warnings
                           instead of raising exceptions. Defaults to True.
    :type skip_when_error: bool
    :param min_last_month_downloads: Minimum monthly downloads threshold for including a dependency
                                    in the prompt. Dependencies with fewer downloads may be ignored
                                    to reduce prompt size. Defaults to 1000000.
    :type min_last_month_downloads: int
    :param no_ignore_modules: Optional iterable of module names that should never be ignored
                             regardless of download count or other filtering criteria.
    :type no_ignore_modules: Optional[Iterable[str]]

    :return: A formatted Markdown string containing the comprehensive code prompt.
    :rtype: str

    .. note::
       The function uses :func:`get_source_info` to analyze the source file and extract
       import information. Import failures can be handled gracefully with skip_when_error.

    .. warning::
       Large dependency trees can generate very large prompts. Consider using
       min_last_month_downloads to filter out less common dependencies.

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
        
        >>> # Skip errors when analyzing problematic imports
        >>> prompt = get_prompt_for_source_file('module_with_issues.py', skip_when_error=True)
        >>> # Warnings will be issued for failed imports, but processing continues
        
        >>> # Add custom description text
        >>> prompt = get_prompt_for_source_file(
        ...     'mymodule.py',
        ...     description_text='This module implements core business logic for user authentication.'
        ... )
        >>> # The description will appear after the title
        
        >>> # Include module directory tree visualization
        >>> prompt = get_prompt_for_source_file('mymodule.py', show_module_directory_tree=True)
        >>> # The prompt will include a tree view showing the module's location in the project structure
        
        >>> # Filter dependencies and preserve specific modules
        >>> prompt = get_prompt_for_source_file(
        ...     'mymodule.py',
        ...     min_last_month_downloads=5000000,
        ...     no_ignore_modules=['mypackage.utils', 'mypackage.config']
        ... )
        >>> # Only includes popular dependencies (>5M downloads) plus the specified modules

    """
    no_ignore_modules = no_ignore_modules or set()
    if not isinstance(no_ignore_modules, set):
        no_ignore_modules = set(no_ignore_modules or [])
    source_info = get_source_info(source_file, skip_when_error=skip_when_error)

    with io.StringIO() as sf:
        # Main source code section
        if code_name:
            title = f'{code_name} Source Code Analysis'
        else:
            title = f'Source Code Analysis'
        print(f'{"#" * level} {titleize(title)}', file=sf)
        print(f'', file=sf)
        if description_text:
            print(description_text, file=sf)
            print(f'', file=sf)
        print(f'**Source File Location:** `{source_info.source_file}`', file=sf)
        print(f'', file=sf)
        print(f'**Package Namespace:** `{source_info.package_name}`', file=sf)
        print(f'', file=sf)

        pythonpath, _ = get_pythonpath_of_source_file(source_info.source_file)
        rel_source_file = os.path.relpath(source_info.source_file, pythonpath)
        print(f'**Relative Source File Location:** `{rel_source_file}`', file=sf)
        print(f'', file=sf)

        if show_module_directory_tree:
            root_path = os.path.join(pythonpath, re.split(r'[\\/]+', rel_source_file)[0])
            print('Module directory tree:', file=sf)
            print(f'```', file=sf)
            print(get_python_project_tree_text(
                root_path=root_path,
                focus_items={
                    'My Location': source_info.source_file
                }
            ), file=sf)
            print(f'```', file=sf)
            print(f'', file=sf)

        print(f'**Complete Source Code:**', file=sf)
        print(f'', file=sf)
        print(f'```python', file=sf)
        print(source_info.source_code, file=sf)
        print(f'```', file=sf)
        print(f'', file=sf)

        # Import dependencies section
        imports_to_show = [
            imp for imp in source_info.imports
            if not imp.statement.check_ignore_or_not(
                min_last_month_downloads=min_last_month_downloads,
                no_ignore_modules=no_ignore_modules,
            )
        ]

        if imports_to_show:
            print(f'{"#" * (level + 1)} Dependency Analysis - Import Statements and Their Implementations', file=sf)
            print(f'', file=sf)
            print(f'The following section contains all imported dependencies for package `{source_info.package_name}` '
                  f'along with their source code implementations. This information can be used as reference context '
                  f'for understanding the main code\'s functionality and dependencies.', file=sf)
            print(f'', file=sf)

            for imp in imports_to_show:
                print(f'{"#" * (level + 2)} Import: `{imp.statement}`', file=sf)
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
