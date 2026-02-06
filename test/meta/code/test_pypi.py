"""
Comprehensive unit tests for the hbllmutils.meta.code.pypi module.

This test module provides complete coverage for Python module metadata and PyPI
package information utilities, testing all public interfaces including module
type detection, standard library identification, and PyPI package information
retrieval.

The tests are organized into the following classes:

* TestPyPIModuleInfo - Tests for the PyPIModuleInfo data class
* TestGetModuleInfo - Tests for the get_module_info function
* TestIsStandardLibrary - Tests for the is_standard_library function
* TestGetPypiInfo - Tests for the get_pypi_info function

All tests focus on public API behavior and use real objects where practical,
with mocking only when necessary for testing edge cases and error conditions.
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pkg_resources
import pytest

from hbllmutils.meta.code.pypi import PyPIModuleInfo, get_module_info, is_standard_library, get_pypi_info


@pytest.fixture
def builtin_module():
    """Fixture providing a built-in module name for testing."""
    return 'sys'


@pytest.fixture
def standard_module():
    """Fixture providing a standard library module name for testing."""
    return 'json'


@pytest.fixture
def third_party_module():
    """Fixture providing a third-party module name for testing."""
    return 'openai'


@pytest.fixture
def nonexistent_module():
    """Fixture providing a non-existent module name for testing."""
    return 'nonexistent_module_xyz'


@pytest.fixture
def mock_module_without_file():
    """Fixture providing a mock module without a __file__ attribute."""
    module = MagicMock()
    module.__file__ = None
    return module


@pytest.fixture
def mock_stdlib_path():
    """Fixture providing a mock standard library path."""
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"


@pytest.fixture
def mock_third_party_path():
    """Fixture providing a mock third-party package path."""
    return Path("/some/path/site-packages/test_module")


@pytest.mark.unittest
class TestPyPIModuleInfo:
    """Tests for the PyPIModuleInfo data class public interface."""

    def test_init(self):
        """Test PyPIModuleInfo initialization with all fields."""
        info = PyPIModuleInfo(
            type='third_party',
            module_name='test',
            pypi_name='test-pkg',
            location='/path/to/test',
            version='1.0.0'
        )
        assert info.type == 'third_party'
        assert info.module_name == 'test'
        assert info.pypi_name == 'test-pkg'
        assert info.location == '/path/to/test'
        assert info.version == '1.0.0'

    def test_is_third_party_true(self):
        """Test is_third_party property returns True for third-party modules."""
        info = PyPIModuleInfo(
            type='third_party',
            module_name='test',
            pypi_name='test-pkg',
            location='/path/to/test',
            version='1.0.0'
        )
        assert info.is_third_party is True

    def test_is_third_party_false_builtin(self):
        """Test is_third_party property returns False for built-in modules."""
        info = PyPIModuleInfo(
            type='builtin',
            module_name='sys',
            pypi_name=None,
            location=None,
            version=None
        )
        assert info.is_third_party is False

    def test_is_third_party_false_standard(self):
        """Test is_third_party property returns False for standard library modules."""
        info = PyPIModuleInfo(
            type='standard',
            module_name='json',
            pypi_name=None,
            location='/path/to/json',
            version=None
        )
        assert info.is_third_party is False


@pytest.mark.unittest
class TestGetModuleInfo:
    """Tests for the get_module_info function public interface."""

    def test_builtin_module(self, builtin_module):
        """Test get_module_info correctly identifies built-in modules."""
        info = get_module_info(builtin_module)
        assert info is not None
        assert info.type == 'builtin'
        assert info.module_name == builtin_module
        assert info.pypi_name is None
        assert info.location is None
        assert info.version is None

    def test_nonexistent_module(self, nonexistent_module):
        """Test get_module_info returns None for non-existent modules."""
        info = get_module_info(nonexistent_module)
        assert info is None

    def test_module_without_file(self):
        """Test get_module_info handles modules without __file__ attribute."""
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__file__ = None
            mock_import.return_value = mock_module

            info = get_module_info('test_module')
            assert info is not None
            assert info.type == 'builtin'
            assert info.module_name == 'test_module'
            assert info.pypi_name is None
            assert info.location is None
            assert info.version is None

    def test_standard_library_module(self, standard_module):
        """Test get_module_info correctly identifies standard library modules."""
        info = get_module_info(standard_module)
        assert info is not None
        assert info.type == 'standard'
        assert info.module_name == standard_module
        assert info.pypi_name is None
        assert info.location is not None
        assert info.version is None

    def test_third_party_module(self, third_party_module):
        """Test get_module_info correctly identifies third-party modules."""
        info = get_module_info(third_party_module)
        assert info is not None
        assert info.type == 'third_party'
        assert info.module_name == third_party_module
        assert info.location is not None

    def test_exception_handling(self):
        """Test get_module_info handles exceptions gracefully and issues warnings."""
        with patch('importlib.import_module', side_effect=Exception("Test error")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                info = get_module_info('test_module')
                assert info is None
                assert len(w) == 1
                assert "Error analyzing module test_module" in str(w[0].message)


@pytest.mark.unittest
class TestIsStandardLibrary:
    """Tests for the is_standard_library function public interface."""

    def test_standard_library_path(self, mock_stdlib_path):
        """Test is_standard_library returns True for standard library paths."""
        test_path = mock_stdlib_path / "json" / "__init__.py"
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', return_value=True):
                result = is_standard_library(test_path)
                assert result is True

    def test_site_packages_path(self, mock_stdlib_path):
        """Test is_standard_library returns False for site-packages paths."""
        test_path = mock_stdlib_path / "site-packages" / "test_module"
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', return_value=True):
                result = is_standard_library(test_path)
                assert result is False

    def test_non_stdlib_path(self, mock_third_party_path):
        """Test is_standard_library returns False for non-standard library paths."""
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', return_value=False):
                result = is_standard_library(mock_third_party_path)
                assert result is False

    def test_path_not_exists(self, mock_stdlib_path):
        """Test is_standard_library returns False for non-existent paths."""
        test_path = mock_stdlib_path / "nonexistent"
        with patch.object(Path, 'exists', return_value=False):
            result = is_standard_library(test_path)
            assert result is False

    def test_windows_stdlib_path(self, mock_stdlib_path):
        """Test is_standard_library handles Windows-specific paths correctly."""
        test_path = Path(sys.prefix) / "Lib" / "json"
        with patch('sys.platform', 'win32'):
            with patch.object(Path, 'exists', return_value=True):
                with patch('hbllmutils.meta.code.pypi._is_relative_to', return_value=True):
                    result = is_standard_library(test_path)
                    assert result is True

    def test_value_error_exception(self, mock_stdlib_path):
        """Test is_standard_library handles ValueError exceptions gracefully."""
        test_path = mock_stdlib_path / "test"
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', side_effect=ValueError("Test error")):
                result = is_standard_library(test_path)
                assert result is False

    def test_os_error_exception(self, mock_stdlib_path):
        """Test is_standard_library handles OSError exceptions gracefully."""
        test_path = mock_stdlib_path / "test"
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', side_effect=OSError("Test error")):
                result = is_standard_library(test_path)
                assert result is False

    def test_string_path_input(self, mock_stdlib_path):
        """Test is_standard_library accepts string paths and converts them to Path objects."""
        test_path = str(mock_stdlib_path / "json" / "__init__.py")
        with patch.object(Path, 'exists', return_value=True):
            with patch('hbllmutils.meta.code.pypi._is_relative_to', return_value=True):
                result = is_standard_library(test_path)
                assert result is True


@pytest.mark.unittest
class TestGetPypiInfo:
    """Tests for the get_pypi_info function public interface."""

    def test_direct_distribution_found(self):
        """Test get_pypi_info retrieves package info using pkg_resources directly."""
        pypi_name, version = get_pypi_info('openai')
        assert pypi_name is not None
        assert version is not None

    def test_distribution_not_found_fallback_working_set(self):
        """Test get_pypi_info falls back to working_set when direct lookup fails."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            mock_dist = MagicMock()
            mock_dist.project_name = 'test-package'
            mock_dist.version = '1.0.0'
            mock_dist._get_metadata.return_value = ['test_module']

            with patch('pkg_resources.working_set', [mock_dist]):
                pypi_name, version = get_pypi_info('test_module')
                assert pypi_name == 'test-package'
                assert version == '1.0.0'

    def test_working_set_with_dash_replacement(self):
        """Test get_pypi_info handles module name dash-to-underscore conversion."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            mock_dist = MagicMock()
            mock_dist.project_name = 'test-package'
            mock_dist.version = '1.0.0'
            mock_dist._get_metadata.return_value = ['test-module']

            with patch('pkg_resources.working_set', [mock_dist]):
                pypi_name, version = get_pypi_info('test_module')
                assert pypi_name == 'test-package'
                assert version == '1.0.0'

    def test_working_set_exception_handling(self):
        """Test get_pypi_info handles exceptions in working_set iteration gracefully."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            mock_dist = MagicMock()
            mock_dist._get_metadata.side_effect = Exception("Test error")

            with patch('pkg_resources.working_set', [mock_dist]):
                pypi_name, version = get_pypi_info('test_module')
                assert pypi_name is None

    def test_working_set_general_exception(self):
        """Test get_pypi_info handles general exceptions from working_set."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                pypi_name, version = get_pypi_info('test_module')
                assert pypi_name is None

    def test_importlib_metadata_direct_distribution(self):
        """Test get_pypi_info uses importlib.metadata when pkg_resources fails."""
        if sys.version_info >= (3, 8):
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    pypi_name, version = get_pypi_info('openai')
                    assert pypi_name is not None
                    assert version is not None

    def test_importlib_metadata_package_not_found(self):
        """Test get_pypi_info falls back to distributions() when package not found."""
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    with patch('importlib.metadata.distribution', side_effect=metadata.PackageNotFoundError()):
                        mock_dist = MagicMock()
                        mock_dist.metadata = {'Name': 'test-package'}
                        mock_dist.version = '1.0.0'
                        mock_file = MagicMock()
                        mock_file.suffix = '.py'
                        mock_file.parts = ['test_module', '__init__.py']
                        mock_dist.files = [mock_file]

                        with patch('importlib.metadata.distributions', return_value=[mock_dist]):
                            pypi_name, version = get_pypi_info('test_module')
                            assert pypi_name == 'test-package'
                            assert version == '1.0.0'

    def test_importlib_metadata_files_no_suffix(self):
        """Test get_pypi_info handles files without suffix in importlib.metadata."""
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    with patch('importlib.metadata.distribution', side_effect=metadata.PackageNotFoundError()):
                        mock_dist = MagicMock()
                        mock_dist.metadata = {'Name': 'test-package'}
                        mock_dist.version = '1.0.0'
                        mock_file = MagicMock()
                        mock_file.suffix = ''
                        mock_file.parts = ['test_module']
                        mock_dist.files = [mock_file]

                        with patch('importlib.metadata.distributions', return_value=[mock_dist]):
                            pypi_name, version = get_pypi_info('test_module')
                            assert pypi_name == 'test-package'
                            assert version == '1.0.0'

    def test_importlib_metadata_no_files(self):
        """Test get_pypi_info handles distributions without files attribute."""
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    with patch('importlib.metadata.distribution', side_effect=metadata.PackageNotFoundError()):
                        mock_dist = MagicMock()
                        mock_dist.files = None

                        with patch('importlib.metadata.distributions', return_value=[mock_dist]):
                            pypi_name, version = get_pypi_info('test_module')
                            assert pypi_name is None

    def test_importlib_metadata_distributions_exception(self):
        """Test get_pypi_info handles exceptions during distributions() iteration."""
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    with patch('importlib.metadata.distribution', side_effect=metadata.PackageNotFoundError()):
                        mock_dist = MagicMock()
                        mock_dist.files = [MagicMock()]
                        mock_dist.files[0].side_effect = Exception("Test error")

                        with patch('importlib.metadata.distributions', return_value=[mock_dist]):
                            pypi_name, version = get_pypi_info('test_module')
                            assert pypi_name is None

    def test_importlib_metadata_general_exception(self):
        """Test get_pypi_info handles general exceptions from importlib.metadata."""
        if sys.version_info >= (3, 8):
            with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
                with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                    with patch('importlib.metadata.distributions', side_effect=Exception("Test error")):
                        pypi_name, version = get_pypi_info('test_module')
                        assert pypi_name is None

    def test_module_version_fallback(self):
        """Test get_pypi_info falls back to module.__version__ attribute."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                mock_module = MagicMock()
                mock_module.__version__ = '2.0.0'
                with patch('importlib.import_module', return_value=mock_module):
                    pypi_name, version = get_pypi_info('test_module')
                    assert pypi_name is None
                    assert version == '2.0.0'

    def test_module_version_exception(self):
        """Test get_pypi_info handles exceptions when importing module for version."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                with patch('importlib.import_module', side_effect=Exception("Test error")):
                    pypi_name, version = get_pypi_info('test_module')
                    assert pypi_name is None
                    assert version is None

    def test_no_version_found(self):
        """Test get_pypi_info returns None when no version information is available."""
        with patch('pkg_resources.get_distribution', side_effect=pkg_resources.DistributionNotFound()):
            with patch('pkg_resources.working_set', side_effect=Exception("Test error")):
                mock_module = MagicMock()
                del mock_module.__version__
                with patch('importlib.import_module', return_value=mock_module):
                    pypi_name, version = get_pypi_info('test_module')
                    assert pypi_name is None
                    assert version is None
