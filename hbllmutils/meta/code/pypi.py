import importlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pkg_resources

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal


@dataclass
class PyPIModuleInfo:
    """Module information data class"""
    type: Literal['builtin', 'standard', 'third_party']
    module_name: str
    pypi_name: Optional[str]
    location: Optional[str]
    version: Optional[str]

    @property
    def is_third_party(self) -> bool:
        return self.type == 'third_party'


def get_module_info(module_name):
    """
    Get detailed information about a module, including its type and PyPI package name

    Returns:
        PyPIModuleInfo: Module information object
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


def is_standard_library(module_path):
    """Check if a module is part of the Python standard library"""
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
            if stdlib_path.exists() and module_path.is_relative_to(stdlib_path):
                if "site-packages" not in str(module_path):
                    return True
        except (ValueError, OSError):
            continue

    return False


def get_pypi_info(module_name):
    """Get PyPI package name and version for a third-party module"""
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
