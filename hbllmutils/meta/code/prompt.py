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

The module contains the following main components:

* :func:`is_python_code` - Validate if text is syntactically correct Python code
* :func:`is_python_file` - Check if a file contains valid Python code
* :func:`get_prompt_for_source_file` - Generate comprehensive code prompts for LLM analysis

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
import ast
import io
import os.path
import pathlib
import re
import warnings
from typing import Optional, Iterable, Union, Tuple, List

from hbutils.string import titleize
from hbutils.system import is_binary_file

from .module import get_package_name, get_pythonpath_of_source_file
from .source import get_source_info
from .tree import get_python_project_tree_text


def is_python_code(code_text: str) -> bool:
    """
    Check if the given text is valid Python code.

    This function attempts to parse the provided text using Python's AST parser
    to determine if it represents syntactically valid Python code. It does not
    execute the code or check for semantic correctness, only syntax validity.

    :param code_text: The text string to check for Python code validity.
    :type code_text: str

    :return: True if the text is valid Python code, False otherwise.
    :rtype: bool

    .. note::
       This function only validates syntax, not semantics. Code that parses
       successfully may still have runtime errors or logical issues.

    Example::

        >>> is_python_code("print('hello')")
        True
        >>> is_python_code("def foo(): return 42")
        True
        >>> is_python_code("invalid python code {{{")
        False
        >>> is_python_code("x = 1 + 2")
        True
        >>> is_python_code("")
        True

    """
    try:
        ast.parse(code_text)
        return True
    except:
        return False


def is_python_file(code_file: str) -> bool:
    """
    Check if a file contains valid Python code.

    This function first checks if the file is binary, then reads its content
    and validates whether it contains syntactically valid Python code using
    AST parsing. It combines binary file detection with Python syntax validation.

    :param code_file: The path to the file to check.
    :type code_file: str

    :return: True if the file is a text file containing valid Python code,
             False if it's binary or contains invalid Python syntax.
    :rtype: bool

    :raises FileNotFoundError: If the specified file does not exist.
    :raises PermissionError: If the file cannot be read due to permissions.

    .. note::
       This function reads the entire file content into memory, which may be
       inefficient for very large files.

    Example::

        >>> is_python_file('module.py')
        True
        >>> is_python_file('data.json')
        False
        >>> is_python_file('script.sh')
        False
        >>> is_python_file('image.png')
        False

    """
    if is_binary_file(code_file):
        return False

    return is_python_code(pathlib.Path(code_file).read_text())


def get_prompt_for_source_file(
        source_file: str,
        level: int = 2,
        code_name: Optional[str] = 'primary',
        description_text: Optional[str] = None,
        show_module_directory_tree: bool = True,
        skip_when_error: bool = True,
        min_last_month_downloads: int = 1000000,
        no_imports: bool = False,
        ignore_modules: Optional[Iterable[str]] = None,
        no_ignore_modules: Optional[Iterable[str]] = None,
        warning_when_not_python: bool = True,
        return_imported_items: bool = False,
) -> Union[str, Tuple[str, List[str]]]:
    """
    Generate a comprehensive code prompt for LLM analysis.

    This function creates a structured prompt containing source code and its dependencies,
    suitable for various LLM tasks like documentation generation, unit testing, or code analysis.
    The prompt includes:
    
    - Primary source code analysis section with file location, package namespace, and complete source
    - Optional module directory tree visualization showing the file's location in the project structure
    - Dependency analysis section with all imported dependencies and their implementations
    - For each import, includes the import statement, source file location, full package path,
      and either the implementation source code or object representation
    
    The generated prompt is formatted in Markdown with code blocks and hierarchical headers,
    making it easy for LLMs to parse and understand the code structure and dependencies.
    
    Dependencies can be filtered based on popularity (download count) and explicit inclusion/exclusion
    lists to control the size and relevance of the generated prompt.

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
                                      structure with the current file highlighted. Defaults to True.
                                      For non-Python files, this parameter is ignored.
    :type show_module_directory_tree: bool
    :param skip_when_error: If True, skip imports that fail to load and issue warnings
                           instead of raising exceptions. Defaults to True.
                           For non-Python files, this parameter is ignored.
    :type skip_when_error: bool
    :param min_last_month_downloads: Minimum monthly downloads threshold for including a dependency
                                    in the prompt. Dependencies with higher downloads may be ignored
                                    to reduce prompt size. Defaults to 1000000.
                                    For non-Python files, this parameter is ignored.
    :type min_last_month_downloads: int
    :param no_imports: If True, skip the dependency analysis section entirely and only include
                      the primary source code. Defaults to False.
                      For non-Python files, this parameter is ignored.
    :type no_imports: bool
    :param ignore_modules: Optional iterable of module names that should be explicitly ignored
                          regardless of download count or other criteria.
                          For non-Python files, this parameter is ignored.
    :type ignore_modules: Optional[Iterable[str]]
    :param no_ignore_modules: Optional iterable of module names that should never be ignored
                             regardless of download count or other filtering criteria.
                             For non-Python files, this parameter is ignored.
    :type no_ignore_modules: Optional[Iterable[str]]
    :param warning_when_not_python: If True, issue warnings when Python-specific parameters
                                   are set to non-default values for non-Python files.
                                   Defaults to True.
    :type warning_when_not_python: bool
    :param return_imported_items: If True, return a tuple of (prompt_text, imported_items_list)
                                 instead of just the prompt text. The imported_items_list contains
                                 the full package paths of all imported dependencies that were included
                                 in the prompt. Defaults to False.
    :type return_imported_items: bool

    :return: A formatted Markdown string containing the comprehensive code prompt.
             If return_imported_items is True, returns a tuple of (prompt_text, imported_items_list).
    :rtype: str or tuple[str, list[str]]

    :warns UserWarning: When Python-specific parameters are set for non-Python files and
                       warning_when_not_python is True.

    .. note::
       The function uses :func:`get_source_info` to analyze the source file and extract
       import information. Import failures can be handled gracefully with skip_when_error.

    .. warning::
       Large dependency trees can generate very large prompts. Consider using
       min_last_month_downloads to filter out common dependencies or set no_imports=True
       to exclude all dependencies.

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
        
        >>> # Explicitly ignore certain modules
        >>> prompt = get_prompt_for_source_file(
        ...     'mymodule.py',
        ...     ignore_modules=['deprecated_module', 'legacy_code']
        ... )
        >>> # The specified modules will be excluded from the dependency analysis
        
        >>> # Generate prompt without any dependencies
        >>> prompt = get_prompt_for_source_file('mymodule.py', no_imports=True)
        >>> # Only the primary source code will be included, no dependency analysis
        
        >>> # Generate prompt for a non-Python file
        >>> prompt = get_prompt_for_source_file('config.yaml')
        >>> # Will generate a simplified prompt without Python-specific analysis
        
        >>> # Get both prompt and list of imported items
        >>> prompt, imports = get_prompt_for_source_file(
        ...     'mymodule.py',
        ...     return_imported_items=True
        ... )
        >>> print(f"Generated prompt with {len(imports)} dependencies")
        >>> print(imports)
        ['mypackage.utils.helper', 'mypackage.config.settings']

    """
    if not isinstance(no_ignore_modules, set):
        no_ignore_modules = set(no_ignore_modules or [])
    if not isinstance(ignore_modules, set):
        ignore_modules = set(ignore_modules or [])

    is_python = is_python_file(source_file)

    if not is_python and warning_when_not_python:
        python_specific_params = []
        if show_module_directory_tree:
            python_specific_params.append('show_module_directory_tree=True')
        if not skip_when_error:
            python_specific_params.append(f'skip_when_error={skip_when_error}')
        if min_last_month_downloads != 1000000:
            python_specific_params.append(f'min_last_month_downloads={min_last_month_downloads}')
        if no_imports:
            python_specific_params.append(f'no_imports={no_imports}')
        if ignore_modules:
            python_specific_params.append(f'ignore_modules={ignore_modules}')
        if no_ignore_modules:
            python_specific_params.append(f'no_ignore_modules={no_ignore_modules}')

        if python_specific_params:
            warnings.warn(
                f"The file {source_file!r} is not a Python file, but Python-specific parameters "
                f"were set to non-default values: {', '.join(python_specific_params)}. "
                f"These parameters will be ignored for non-Python files.",
                UserWarning,
                stacklevel=2
            )

    imported_items = []
    with io.StringIO() as sf:
        if code_name:
            title = f'{code_name} Source Code Analysis'
        else:
            title = f'Source Code Analysis'
        print(f'{"#" * level} {titleize(title)}', file=sf)
        print(f'', file=sf)
        if description_text:
            print(description_text, file=sf)
            print(f'', file=sf)

        if is_python:
            source_info = get_source_info(source_file, skip_when_error=skip_when_error)

            print(f'**Source File Location:** `{source_info.source_file}`', file=sf)
            print(f'', file=sf)
            print(f'**Package Namespace:** `{source_info.package_name}`', file=sf)
            print(f'', file=sf)
            imported_items.append(source_info.package_name)

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

            if no_imports:
                imports_to_show = []
            else:
                imports_to_show = [
                    imp for imp in source_info.imports
                    if not imp.statement.check_ignore_or_not(
                        min_last_month_downloads=min_last_month_downloads,
                        ignore_modules=ignore_modules,
                        no_ignore_modules=no_ignore_modules,
                    )
                ]

            if imports_to_show:
                print(f'{"#" * (level + 1)} Dependency Analysis - Import Statements and Their Implementations', file=sf)
                print(f'', file=sf)
                print(
                    f'The following section contains all imported dependencies for package `{source_info.package_name}` '
                    f'along with their source code implementations. This information can be used as reference context '
                    f'for understanding the main code\'s functionality and dependencies.', file=sf)
                print(f'', file=sf)

                for imp in imports_to_show:
                    print(f'{"#" * (level + 2)} Import: `{imp.statement}`', file=sf)
                    print(f'', file=sf)

                    if imp.inspect.source_file:
                        print(f'**Source File:** `{imp.inspect.source_file}`', file=sf)
                        print(f'', file=sf)
                        imported_item_name = f'{get_package_name(imp.inspect.source_file)}.{imp.statement.name}'
                        imported_items.append(imported_item_name)
                        print(f'**Full Package Path:** `{imported_item_name}`', file=sf)
                        print(f'', file=sf)

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

        else:
            print(f'**Source File Location:** `{source_file}`', file=sf)
            print(f'', file=sf)
            source_code = pathlib.Path(source_file).read_text()
            print(f'**Complete Source Code:**', file=sf)
            print(f'', file=sf)
            print(f'```', file=sf)
            print(source_code, file=sf)
            print(f'```', file=sf)
            print(f'', file=sf)

        if return_imported_items:
            return sf.getvalue(), imported_items
        else:
            return sf.getvalue()
