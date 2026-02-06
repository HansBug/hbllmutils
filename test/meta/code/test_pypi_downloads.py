"""
Unit tests for PyPI package download statistics and popularity analysis utilities.

This module contains comprehensive tests for the pypi_downloads module, including:
- Loading PyPI download statistics from CSV files
- Querying package popularity
- Determining if packages meet specific download thresholds
- Edge cases and error handling
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from hbllmutils.meta.code.pypi_downloads import (
    get_pypi_downloads,
    _get_pypi_downloads_dict,
    is_hot_pypi_project
)


@pytest.fixture(scope="module")
def sample_csv_content():
    """Provide sample CSV content for testing."""
    return """name,last_month
numpy,10000000
pandas,8000000
requests,15000000
flask,5000000
django,7000000
pytest,3000000
small-package,500000
tiny-package,100
"""


@pytest.fixture(scope="module")
def temp_csv_file(sample_csv_content):
    """Create a temporary CSV file with sample PyPI download data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_csv_content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture(scope="function")
def clear_cache():
    """Clear LRU cache before and after each test to ensure test isolation."""
    get_pypi_downloads.cache_clear()
    _get_pypi_downloads_dict.cache_clear()
    yield
    get_pypi_downloads.cache_clear()
    _get_pypi_downloads_dict.cache_clear()


@pytest.mark.unittest
class TestGetPypiDownloads:
    """Tests for the get_pypi_downloads function."""

    def test_returns_dataframe(self, clear_cache):
        """Test that get_pypi_downloads returns a pandas DataFrame."""
        result = get_pypi_downloads()
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_required_columns(self, clear_cache):
        """Test that the returned DataFrame has 'name' and 'last_month' columns."""
        df = get_pypi_downloads()
        assert 'name' in df.columns
        assert 'last_month' in df.columns

    def test_dataframe_not_empty(self, clear_cache):
        """Test that the DataFrame contains data."""
        df = get_pypi_downloads()
        assert len(df) > 0

    def test_name_column_is_string(self, clear_cache):
        """Test that the 'name' column contains string values."""
        df = get_pypi_downloads()
        assert df['name'].dtype == object

    def test_last_month_column_is_numeric(self, clear_cache):
        """Test that the 'last_month' column contains numeric values."""
        df = get_pypi_downloads()
        assert pd.api.types.is_numeric_dtype(df['last_month'])

    def test_caching_behavior(self, clear_cache):
        """Test that the function uses LRU cache and returns the same object."""
        df1 = get_pypi_downloads()
        df2 = get_pypi_downloads()
        assert df1 is df2

    def test_csv_file_exists(self):
        """Test that the pypi_downloads.csv file exists in the expected location."""
        from hbllmutils.meta.code import pypi_downloads
        csv_file = os.path.join(os.path.dirname(pypi_downloads.__file__), 'pypi_downloads.csv')
        assert os.path.exists(csv_file)

    @patch('hbllmutils.meta.code.pypi_downloads.pd.read_csv')
    def test_file_not_found_error(self, mock_read_csv, clear_cache):
        """Test that FileNotFoundError is raised when CSV file is missing."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        with pytest.raises(FileNotFoundError):
            get_pypi_downloads()

    @patch('hbllmutils.meta.code.pypi_downloads.pd.read_csv')
    def test_empty_data_error(self, mock_read_csv, clear_cache):
        """Test that EmptyDataError is raised when CSV file is empty."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        with pytest.raises(pd.errors.EmptyDataError):
            get_pypi_downloads()

    @patch('hbllmutils.meta.code.pypi_downloads.pd.read_csv')
    def test_parser_error(self, mock_read_csv, clear_cache):
        """Test that ParserError is raised when CSV format is invalid."""
        mock_read_csv.side_effect = pd.errors.ParserError("Invalid format")
        with pytest.raises(pd.errors.ParserError):
            get_pypi_downloads()


@pytest.mark.unittest
class TestGetPypiDownloadsDict:
    """Tests for the _get_pypi_downloads_dict internal function."""

    @patch('hbllmutils.meta.code.pypi_downloads.get_pypi_downloads')
    def test_returns_dictionary(self, mock_get_downloads, clear_cache):
        """Test that _get_pypi_downloads_dict returns a dictionary."""
        mock_df = pd.DataFrame({
            'name': ['numpy', 'pandas', 'requests'],
            'last_month': [10000000, 8000000, 15000000]
        })
        mock_get_downloads.return_value = mock_df

        result = _get_pypi_downloads_dict()
        assert isinstance(result, dict)

    @patch('hbllmutils.meta.code.pypi_downloads.get_pypi_downloads')
    def test_dictionary_mapping_correct(self, mock_get_downloads, clear_cache):
        """Test that the dictionary correctly maps package names to download counts."""
        mock_df = pd.DataFrame({
            'name': ['numpy', 'pandas', 'requests'],
            'last_month': [10000000, 8000000, 15000000]
        })
        mock_get_downloads.return_value = mock_df

        result = _get_pypi_downloads_dict()
        assert result['numpy'] == 10000000
        assert result['pandas'] == 8000000
        assert result['requests'] == 15000000

    @patch('hbllmutils.meta.code.pypi_downloads.get_pypi_downloads')
    def test_caching_behavior(self, mock_get_downloads, clear_cache):
        """Test that _get_pypi_downloads_dict uses LRU cache."""
        mock_df = pd.DataFrame({
            'name': ['numpy'],
            'last_month': [10000000]
        })
        mock_get_downloads.return_value = mock_df

        dict1 = _get_pypi_downloads_dict()
        dict2 = _get_pypi_downloads_dict()
        assert dict1 is dict2
        assert mock_get_downloads.call_count == 1

    @patch('hbllmutils.meta.code.pypi_downloads.get_pypi_downloads')
    def test_empty_dataframe(self, mock_get_downloads, clear_cache):
        """Test handling of empty DataFrame."""
        mock_df = pd.DataFrame({
            'name': [],
            'last_month': []
        })
        mock_get_downloads.return_value = mock_df

        result = _get_pypi_downloads_dict()
        assert isinstance(result, dict)
        assert len(result) == 0


@pytest.mark.unittest
class TestIsHotPypiProject:
    """Tests for the is_hot_pypi_project function."""

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_hot_project_default_threshold(self, mock_get_dict, clear_cache):
        """Test that a package with >= 1M downloads is considered hot (default threshold)."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'pandas': 8000000,
            'small-package': 500000
        }

        assert is_hot_pypi_project('numpy') is True
        assert is_hot_pypi_project('pandas') is True

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_not_hot_project_default_threshold(self, mock_get_dict, clear_cache):
        """Test that a package with < 1M downloads is not considered hot (default threshold)."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'small-package': 500000
        }

        assert is_hot_pypi_project('small-package') is False

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_hot_project_custom_threshold(self, mock_get_dict, clear_cache):
        """Test with custom threshold values."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'pandas': 8000000,
            'small-package': 500000
        }

        assert is_hot_pypi_project('numpy', min_last_month_downloads=5000000) is True
        assert is_hot_pypi_project('pandas', min_last_month_downloads=5000000) is True
        assert is_hot_pypi_project('small-package', min_last_month_downloads=100000) is True

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_not_hot_project_custom_threshold(self, mock_get_dict, clear_cache):
        """Test that packages below custom threshold are not considered hot."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'pandas': 8000000,
            'small-package': 500000
        }

        assert is_hot_pypi_project('pandas', min_last_month_downloads=10000000) is False
        assert is_hot_pypi_project('small-package', min_last_month_downloads=1000000) is False

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_exact_threshold_match(self, mock_get_dict, clear_cache):
        """Test that a package exactly at the threshold is considered hot."""
        mock_get_dict.return_value = {
            'exact-package': 1000000
        }

        assert is_hot_pypi_project('exact-package', min_last_month_downloads=1000000) is True

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_package_not_found(self, mock_get_dict, clear_cache):
        """Test that non-existent packages return False."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'pandas': 8000000
        }

        assert is_hot_pypi_project('non-existent-package') is False
        assert is_hot_pypi_project('this-package-does-not-exist') is False

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    @pytest.mark.parametrize("package_name,downloads,threshold,expected", [
        ('numpy', 10000000, 1000000, True),
        ('pandas', 8000000, 1000000, True),
        ('requests', 15000000, 1000000, True),
        ('small-package', 500000, 1000000, False),
        ('tiny-package', 100, 1000000, False),
        ('numpy', 10000000, 20000000, False),
        ('pandas', 8000000, 8000000, True),
        ('zero-downloads', 0, 1000000, False),
        ('zero-downloads', 0, 0, True),
    ])
    def test_various_scenarios_parametrized(self, mock_get_dict, package_name,
                                            downloads, threshold, expected, clear_cache):
        """Test various package/threshold combinations using parameterization."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'pandas': 8000000,
            'requests': 15000000,
            'small-package': 500000,
            'tiny-package': 100,
            'zero-downloads': 0
        }

        result = is_hot_pypi_project(package_name, min_last_month_downloads=threshold)
        assert result is expected

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_case_sensitivity(self, mock_get_dict, clear_cache):
        """Test that package name matching is case-sensitive."""
        mock_get_dict.return_value = {
            'numpy': 10000000,
            'NumPy': 5000000
        }

        assert is_hot_pypi_project('numpy') is True
        assert is_hot_pypi_project('NumPy', min_last_month_downloads=1000000) is True
        assert is_hot_pypi_project('NUMPY') is False

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_zero_threshold(self, mock_get_dict, clear_cache):
        """Test with zero threshold - any package with downloads should be hot."""
        mock_get_dict.return_value = {
            'tiny-package': 1,
            'zero-package': 0
        }

        assert is_hot_pypi_project('tiny-package', min_last_month_downloads=0) is True
        assert is_hot_pypi_project('zero-package', min_last_month_downloads=0) is True

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_negative_downloads(self, mock_get_dict, clear_cache):
        """Test handling of negative download counts (edge case)."""
        mock_get_dict.return_value = {
            'negative-package': -100
        }

        assert is_hot_pypi_project('negative-package', min_last_month_downloads=1000000) is False
        assert is_hot_pypi_project('negative-package', min_last_month_downloads=-200) is True

    @patch('hbllmutils.meta.code.pypi_downloads._get_pypi_downloads_dict')
    def test_empty_package_name(self, mock_get_dict, clear_cache):
        """Test with empty string as package name."""
        mock_get_dict.return_value = {
            '': 1000000,
            'numpy': 10000000
        }

        assert is_hot_pypi_project('', min_last_month_downloads=500000) is True
        assert is_hot_pypi_project('', min_last_month_downloads=2000000) is False
