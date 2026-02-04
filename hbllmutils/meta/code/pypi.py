"""
Python module metadata and PyPI package information utilities.

This module provides comprehensive functionality for analyzing Python modules
and determining their origin, type, and PyPI package information. It can
distinguish between built-in modules, standard library modules, and third-party
packages, and retrieve associated metadata such as package names and versions.

The module contains the following main components:

* :class:`PyPIModuleInfo` - Data class containing module metadata and type information
* :func:`get_module_info` - Main function to retrieve comprehensive module information
* :func:`is_standard_library` - Utility to determine if a module is from the standard library
* :func:`get_pypi_info` - Utility to extract PyPI package name and version information

.. note::
   This module uses multiple fallback mechanisms to ensure robust package
   detection across different Python versions and installation methods.

.. warning::
   Module analysis may fail for dynamically loaded modules or modules with
   non-standard installation paths.

Example::

    >>> from hbllmutils.meta.code.pypi import get_module_info
    >>> 
    >>> # Analyze a built-in module
    >>> info = get_module_info('sys')
    >>> print(f"Type: {info.type}, Module: {info.module_name}")
    Type: builtin, Module: sys
    >>> 
    >>> # Analyze a third-party package
    >>> info = get_module_info('requests')
    >>> if info and info.is_third_party:
    ...     print(f"PyPI: {info.pypi_name}, Version: {info.version}")
    PyPI: requests, Version: 2.28.0

"""

import importlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pkg_resources

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal


@dataclass
class PyPIModuleInfo:
    """
    Data class containing comprehensive information about a Python module.

    This class encapsulates all relevant metadata about a Python module including
    its type (builtin, standard library, or third-party), location, and PyPI
    package information if applicable.

    :param type: Classification of the module type
    :type type: Literal['builtin', 'standard', 'third_party']
    :param module_name: The import name of the module
    :type module_name: str
    :param pypi_name: The PyPI package name if it's a third-party module, None otherwise
    :type pypi_name: Optional[str]
    :param location: File system path to the module, None for built-in modules
    :type location: Optional[str]
    :param version: Version string of the package, None if not available
    :type version: Optional[str]

    :ivar type: Module classification type
    :vartype type: Literal['builtin', 'standard', 'third_party']
    :ivar module_name: Module import name
    :vartype module_name: str
    :ivar pypi_name: PyPI package name
    :vartype pypi_name: Optional[str]
    :ivar location: Module file path
    :vartype location: Optional[str]
    :ivar version: Package version
    :vartype version: Optional[str]

    Example::

        >>> info = PyPIModuleInfo(
        ...     type='third_party',
        ...     module_name='requests',
        ...     pypi_name='requests',
        ...     location='/usr/lib/python3.10/site-packages/requests/__init__.py',
        ...     version='2.28.0'
        ... )
        >>> print(info.is_third_party)
        True

    """

    type: Literal['builtin', 'standard', 'third_party']
    module_name: str
    pypi_name: Optional[str]
    location: Optional[str]
    version: Optional[str]

    @property
    def is_third_party(self) -> bool:
        """
        Check if the module is a third-party package.

        :return: True if the module is a third-party package, False otherwise
        :rtype: bool

        Example::

            >>> info = PyPIModuleInfo(type='third_party', module_name='numpy',
            ...                       pypi_name='numpy', location=None, version='1.21.0')
            >>> info.is_third_party
            True
            >>> 
            >>> builtin_info = PyPIModuleInfo(type='builtin', module_name='sys',
            ...                               pypi_name=None, location=None, version=None)
            >>> builtin_info.is_third_party
            False

        """
        return self.type == 'third_party'


def _is_relative_to(path: Path, base: Path) -> bool:
    """
    Check if path is relative to base (compatibility function for Python < 3.9).

    :param path: The path to check
    :type path: Path
    :param base: The base path to check against
    :type base: Path
    :return: True if path is relative to base, False otherwise
    :rtype: bool
    """
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def get_module_info(module_name: str) -> Optional[PyPIModuleInfo]:
    """
    Get detailed information about a module including its type and PyPI package name.

    This function analyzes a Python module to determine its classification (builtin,
    standard library, or third-party), location, and associated PyPI package information.
    It handles various edge cases and uses multiple fallback mechanisms for robust detection.

    :param module_name: The name of the module to analyze
    :type module_name: str
    :return: Module information object containing all metadata, or None if analysis fails
    :rtype: Optional[PyPIModuleInfo]

    .. note::
       The function first checks if the module is built-in, then attempts to import it
       to determine its location and type. For third-party packages, it retrieves
       PyPI metadata using multiple detection methods.

    .. warning::
       If the module cannot be imported or analyzed, a warning is issued and None is returned.

    Example::

        >>> # Analyze a standard library module
        >>> info = get_module_info('json')
        >>> print(f"Type: {info.type}")
        Type: standard
        >>> 
        >>> # Analyze a third-party module
        >>> info = get_module_info('numpy')
        >>> if info:
        ...     print(f"PyPI: {info.pypi_name}, Version: {info.version}")
        PyPI: numpy, Version: 1.21.0
        >>> 
        >>> # Handle non-existent module
        >>> info = get_module_info('nonexistent_module')
        >>> print(info)
        None

    """
    try:
        if module_name in sys.builtin_module_names:
            return PyPIModuleInfo(
                type='builtin',
                module_name=module_name,
                pypi_name=None,
                location=None,
                version=None
            )

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None

        module_file = getattr(module, '__file__', None)
        if not module_file:
            return PyPIModuleInfo(
                type='builtin',
                module_name=module_name,
                pypi_name=None,
                location=None,
                version=None
            )

        module_path = Path(module_file).resolve()

        if is_standard_library(module_path):
            return PyPIModuleInfo(
                type='standard',
                module_name=module_name,
                pypi_name=None,
                location=str(module_path),
                version=None
            )

        pypi_name, version = get_pypi_info(module_name)

        return PyPIModuleInfo(
            type='third_party',
            module_name=module_name,
            pypi_name=pypi_name,
            location=str(module_path),
            version=version
        )

    except Exception as e:
        warnings.warn(f"Error analyzing module {module_name}: {e}", stacklevel=2)
        return None


def is_standard_library(module_path: Union[str, Path]) -> bool:
    """
    Check if a module is part of the Python standard library.

    This function determines whether a given module path belongs to the Python
    standard library by comparing it against known standard library locations.
    It handles both Unix-like and Windows systems, and excludes site-packages
    directories to avoid false positives.

    :param module_path: Resolved file system path to the module
    :type module_path: Path
    :return: True if the module is part of the standard library, False otherwise
    :rtype: bool

    .. note::
       The function checks multiple potential standard library locations including
       both sys.prefix and sys.base_prefix to handle virtual environments correctly.

    .. warning::
       Modules in site-packages directories are explicitly excluded even if they
       reside within standard library paths.

    Example::

        >>> from pathlib import Path
        >>> import json
        >>> 
        >>> # Check a standard library module
        >>> json_path = Path(json.__file__).resolve()
        >>> is_standard_library(json_path)
        True
        >>> 
        >>> # Check a third-party module
        >>> import requests
        >>> requests_path = Path(requests.__file__).resolve()
        >>> is_standard_library(requests_path)
        False

    """
    if not isinstance(module_path, Path):
        module_path = Path(module_path)

    stdlib_paths = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
    ]

    if sys.platform == "win32":
        stdlib_paths.extend([
            Path(sys.prefix) / "Lib",
            Path(sys.base_prefix) / "Lib",
        ])

    for stdlib_path in stdlib_paths:
        try:
            if stdlib_path.exists() and _is_relative_to(module_path, stdlib_path):
                if "site-packages" not in str(module_path):
                    return True
        except (ValueError, OSError):
            continue

    return False


def get_pypi_info(module_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get PyPI package name and version for a third-party module.

    This function attempts to retrieve the PyPI package name and version for a
    given module using multiple detection strategies. It tries pkg_resources first,
    then falls back to importlib.metadata (Python 3.8+), and finally attempts to
    read the __version__ attribute from the module itself.

    :param module_name: The name of the module to look up
    :type module_name: str
    :return: Tuple containing (pypi_name, version), where either or both may be None
    :rtype: tuple[Optional[str], Optional[str]]

    .. note::
       The function uses multiple fallback mechanisms to maximize compatibility
       across different Python versions and package installation methods.

    .. warning::
       Some packages may not be detected if they don't follow standard naming
       conventions or lack proper metadata.

    Example::

        >>> # Get info for a well-known package
        >>> pypi_name, version = get_pypi_info('requests')
        >>> print(f"Package: {pypi_name}, Version: {version}")
        Package: requests, Version: 2.28.0
        >>> 
        >>> # Handle package with different module name
        >>> pypi_name, version = get_pypi_info('PIL')
        >>> print(f"Package: {pypi_name}")
        Package: Pillow
        >>> 
        >>> # Handle unknown package
        >>> pypi_name, version = get_pypi_info('unknown_module')
        >>> print(f"Package: {pypi_name}, Version: {version}")
        Package: None, Version: None

    """
    pypi_name = None
    version = None

    try:
        dist = pkg_resources.get_distribution(module_name)
        pypi_name = dist.project_name
        version = dist.version
        return pypi_name, version
    except pkg_resources.DistributionNotFound:
        pass

    try:
        for dist in pkg_resources.working_set:
            try:
                top_level = dist._get_metadata('top_level.txt')
                if module_name in [mod.replace('-', '_') for mod in top_level]:
                    pypi_name = dist.project_name
                    version = dist.version
                    return pypi_name, version
            except Exception:
                continue
    except Exception:
        pass

    if sys.version_info >= (3, 8):
        try:
            import importlib.metadata as metadata

            try:
                dist = metadata.distribution(module_name)
                pypi_name = dist.metadata['Name']
                version = dist.version
                return pypi_name, version
            except metadata.PackageNotFoundError:
                pass

            try:
                for dist in metadata.distributions():
                    try:
                        if dist.files:
                            top_level = set()
                            for file in dist.files:
                                if file.suffix == '.py' or not file.suffix:
                                    parts = file.parts
                                    if parts:
                                        top_level.add(parts[0])

                            if module_name in top_level:
                                pypi_name = dist.metadata['Name']
                                version = dist.version
                                return pypi_name, version
                    except Exception:
                        continue
            except Exception:
                pass
        except ImportError:
            pass

    if not version:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', None)
        except Exception:
            pass

    return pypi_name, version
