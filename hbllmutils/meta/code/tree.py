"""
This module provides functionality for managing file path patterns and determining which files
should be ignored based on Python project conventions and custom patterns.

The module includes:
- A comprehensive list of Python gitignore patterns
- Functions to check if files should be ignored based on these patterns
- Support for custom additional ignore patterns
- Functions to build directory tree structures while respecting ignore patterns
"""
import os
import pathlib
from functools import lru_cache
from typing import Optional, List, Tuple, Union

from natsort import natsorted
from pathspec import patterns, PathSpec

_PYTHON_GITIGNORE_PATTERNS = [
    # Byte-compiled / optimized / DLL files
    "__pycache__/",
    "*.py[cod]",
    "*$py.class",

    # C extensions
    # "*.so",

    # Distribution / packaging
    ".Python",
    "build/",
    "develop-eggs/",
    "dist/",
    "downloads/",
    "eggs/",
    ".eggs/",
    # "lib/",
    # "lib64/",
    "parts/",
    "sdist/",
    "var/",
    "wheels/",
    "pip-wheel-metadata/",
    "share/python-wheels/",
    "*.egg-info/",
    ".installed.cfg",
    "*.egg",
    "MANIFEST",

    # PyInstaller
    "*.manifest",
    "*.spec",

    # Installer logs
    "pip-log.txt",
    "pip-delete-this-directory.txt",

    # Unit test / coverage reports
    "htmlcov/",
    ".tox/",
    ".nox/",
    ".coverage",
    ".coverage.*",
    ".cache",
    "nosetests.xml",
    "coverage.xml",
    "*.cover",
    "*.py,cover",
    ".hypothesis/",
    ".pytest_cache/",

    # Translations
    "*.mo",
    "*.pot",

    # Django stuff:
    "*.log",
    "local_settings.py",
    "db.sqlite3",
    "db.sqlite3-journal",
    "media/",
    "staticfiles/",

    # Flask stuff:
    "instance/",
    ".webassets-cache",

    # Scrapy stuff:
    ".scrapy",

    # Sphinx documentation
    "docs/_build/",

    # PyBuilder
    "target/",

    # Jupyter Notebook
    ".ipynb_checkpoints",

    # IPython
    "profile_default/",
    "ipython_config.py",

    # pyenv
    ".python-version",

    # pipenv
    "Pipfile.lock",

    # poetry
    "poetry.lock",
    ".pypoetry/",

    # pdm
    ".pdm.toml",

    # PEP 582; used by e.g. github.com/David-OConnor/pyflow
    "__pypackages__/",

    # Celery stuff
    "celerybeat-schedule",
    "celerybeat.pid",

    # SageMath parsed files
    "*.sage.py",

    # Environments
    ".env",
    ".venv",
    "env/",
    "venv/",
    "ENV/",
    "env.bak/",
    "venv.bak/",

    # Spyder project settings
    ".spyderproject",
    ".spyproject",

    # Rope project settings
    ".ropeproject",

    # mkdocs documentation
    "/site",

    # mypy
    ".mypy_cache/",
    ".dmypy.json",
    "dmypy.json",

    # Pyre type checker
    ".pyre/",

    # pytype static type analyzer
    ".pytype/",

    # Cython debug symbols
    "cython_debug/",

    # IDE specific files
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",

    # OS generated files
    ".DS_Store?",
    "._*",
    ".Spotlight-V100",
    ".Trashes",
    "ehthumbs.db",
    "[Tt]humbs.db",

    # Temporary files
    "*.tmp",
    "*.bak",

    # Logs
    "logs/",
]


@lru_cache
def _get_ignore_matcher(extra_patterns: Tuple[str, ...]) -> PathSpec:
    """
    Create and cache a PathSpec matcher for file ignore patterns.
    
    This function combines the default Python gitignore patterns with any additional
    custom patterns provided, and returns a PathSpec object that can be used to match
    file paths against these patterns. The result is cached using LRU cache for
    performance optimization.
    
    :param extra_patterns: Additional patterns to include beyond the default Python gitignore patterns.
    :type extra_patterns: Tuple[str, ...]
    
    :return: A PathSpec object configured with all ignore patterns.
    :rtype: PathSpec
    
    Example::
        >>> matcher = _get_ignore_matcher(('*.txt', 'temp/'))
        >>> matcher.match_file('test.txt')
        True
    """
    return PathSpec.from_lines(
        pattern_factory=patterns.GitWildMatchPattern,
        lines=[*_PYTHON_GITIGNORE_PATTERNS, *extra_patterns],
    )


def is_file_should_ignore(path: Union[str, pathlib.Path], extra_patterns: Optional[List[str]] = None) -> bool:
    """
    Determine whether a file should be ignored based on Python gitignore patterns.
    
    This function checks if the given file path matches any of the default Python
    gitignore patterns or any additional custom patterns provided. It uses a cached
    PathSpec matcher for efficient pattern matching.
    
    :param path: The file path to check against ignore patterns.
    :type path: Union[str, pathlib.Path]
    :param extra_patterns: Optional list of additional patterns to check beyond the default Python gitignore patterns.
    :type extra_patterns: Optional[List[str]]
    
    :return: True if the file should be ignored, False otherwise.
    :rtype: bool
    
    Example::
        >>> is_file_should_ignore('__pycache__/test.pyc')
        True
        >>> is_file_should_ignore('main.py')
        False
        >>> is_file_should_ignore('test.txt', extra_patterns=['*.txt'])
        True
    """
    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    extra_patterns = tuple(natsorted(extra_patterns or []))
    return bool(_get_ignore_matcher(extra_patterns).match_file(path))


def build_python_project_tree(root_path: str, extra_patterns: Optional[List[str]] = None) -> Tuple[str, List]:
    """
    Build a directory tree structure for a Python project while respecting ignore patterns.
    
    This function recursively traverses the directory structure starting from the root path,
    filtering out files and directories that match the Python gitignore patterns or any
    additional custom patterns provided. It returns a tree structure representation of the
    project.
    
    :param root_path: The root directory path to start building the tree from.
    :type root_path: str
    :param extra_patterns: Optional list of additional patterns to ignore beyond the default Python gitignore patterns.
    :type extra_patterns: Optional[List[str]]
    
    :return: A tuple containing the relative path of the root directory and a list of tree nodes.
             Each tree node is a tuple of (name, children) where children is a list of child nodes.
    :rtype: Tuple[str, List]
    
    Example::
        >>> root, tree = build_python_project_tree('/path/to/project')
        >>> print(root)
        'project'
        >>> print(tree)
        [('src', [('main.py', []), ('utils.py', [])]), ('tests', [('test_main.py', [])])]
    """
    root_path = pathlib.Path(root_path)

    def build_node(path: pathlib.Path):
        """
        Recursively build a tree node for the given path.
        
        :param path: The path to build a node for.
        :type path: pathlib.Path
        
        :return: A tuple containing the node name and its children list, or None if the path should be ignored.
        :rtype: Optional[Tuple[str, List]]
        """
        rel_path = path.relative_to(root_path)
        if path.is_file() and not is_file_should_ignore(rel_path, extra_patterns=extra_patterns):
            return path.name, []
        elif path.is_dir():
            children = []
            try:
                # Get all items in the directory
                for item in sorted(path.iterdir()):
                    if is_file_should_ignore(item.relative_to(root_path), extra_patterns=extra_patterns):
                        continue
                    if item.is_file():
                        children.append((item.name, []))
                    elif item.is_dir():
                        # Recursively check if subdirectory contains files
                        sub_node = build_node(item)
                        if sub_node[1]:  # If subdirectory has content
                            children.append(sub_node)
                return path.name, children
            except PermissionError:
                return f"{path.name} (Permission Denied)", []
        return None

    nx = build_node(root_path)
    return os.path.relpath(os.path.normpath(str(root_path)), os.path.abspath('.')), nx[1]
