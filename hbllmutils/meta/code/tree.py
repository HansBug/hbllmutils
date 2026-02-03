"""
This module provides functionality for managing file path patterns and determining which files
should be ignored based on Python project conventions and custom patterns.

The module includes:
- A comprehensive list of Python gitignore patterns
- Functions to check if files should be ignored based on these patterns
- Support for custom additional ignore patterns
"""

from functools import lru_cache
from typing import Optional, List, Tuple

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


def is_file_should_ignore(path: str, extra_patterns: Optional[List[str]] = None) -> bool:
    """
    Determine whether a file should be ignored based on Python gitignore patterns.
    
    This function checks if the given file path matches any of the default Python
    gitignore patterns or any additional custom patterns provided. It uses a cached
    PathSpec matcher for efficient pattern matching.
    
    :param path: The file path to check against ignore patterns.
    :type path: str
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
    extra_patterns = tuple(natsorted(extra_patterns or []))
    return bool(_get_ignore_matcher(extra_patterns).match_file(path))
