"""
PyPI package download statistics and popularity analysis utilities.

This module provides functionality for analyzing PyPI package popularity based on
download statistics. It includes utilities for loading download data, querying
package popularity, and determining if packages meet specific download thresholds.

The module contains the following main components:

* :func:`get_pypi_downloads` - Load PyPI download statistics from CSV data
* :func:`is_hot_pypi_project` - Check if a package meets popularity threshold

.. note::
   Download statistics are cached using LRU cache for performance optimization.
   The data is loaded from a bundled CSV file containing package download counts.

.. warning::
   The download statistics are static and reflect data at the time of package
   installation. For real-time statistics, consider using the PyPI API directly.

Example::

    >>> from hbllmutils.meta.code.pypi_downloads import get_pypi_downloads, is_hot_pypi_project
    >>> 
    >>> # Get all download statistics
    >>> df = get_pypi_downloads()
    >>> print(df.head())
    >>> 
    >>> # Check if a package is popular
    >>> is_popular = is_hot_pypi_project('numpy', min_last_month_downloads=1000000)
    >>> print(f"Is numpy popular? {is_popular}")
    >>> 
    >>> # Check with custom threshold
    >>> is_very_popular = is_hot_pypi_project('requests', min_last_month_downloads=5000000)

"""

import os.path
from functools import lru_cache
from typing import Dict

import pandas as pd


@lru_cache()
def get_pypi_downloads() -> pd.DataFrame:
    """
    Load PyPI package download statistics from bundled CSV file.

    This function reads download statistics for PyPI packages from a CSV file
    bundled with the module. The data includes package names and their download
    counts for the last month. Results are cached for improved performance on
    subsequent calls.

    :return: DataFrame containing package download statistics with columns:
             - 'name': Package name (str)
             - 'last_month': Download count for the last month (int)
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the pypi_downloads.csv file is not found
    :raises pd.errors.EmptyDataError: If the CSV file is empty
    :raises pd.errors.ParserError: If the CSV file format is invalid

    .. note::
       This function uses LRU cache with unlimited size. The data is loaded
       only once per Python session and reused for all subsequent calls.

    .. warning::
       The returned DataFrame should not be modified directly as it is cached.
       Create a copy if modifications are needed.

    Example::

        >>> df = get_pypi_downloads()
        >>> print(df.columns)
        Index(['name', 'last_month'], dtype='object')
        >>> 
        >>> # Get top 5 most downloaded packages
        >>> top_packages = df.nlargest(5, 'last_month')
        >>> print(top_packages)
        >>> 
        >>> # Get specific package statistics
        >>> numpy_stats = df[df['name'] == 'numpy']
        >>> print(numpy_stats['last_month'].values[0])

    """
    csv_file = os.path.join(os.path.dirname(__file__), 'pypi_downloads.csv')
    return pd.read_csv(csv_file)


@lru_cache()
def _get_pypi_downloads_dict() -> Dict[str, int]:
    """
    Get PyPI download statistics as a dictionary mapping package names to download counts.

    This internal function converts the DataFrame from get_pypi_downloads() into a
    dictionary for efficient lookup operations. Results are cached for performance.

    :return: Dictionary mapping package names to their last month download counts
    :rtype: Dict[str, int]

    .. note::
       This is an internal function used by is_hot_pypi_project(). It is cached
       separately from get_pypi_downloads() for optimal performance.

    """
    df = get_pypi_downloads()
    return dict(zip(df['name'], df['last_month']))


def is_hot_pypi_project(pypi_name: str, min_last_month_downloads: int = 1000000) -> bool:
    """
    Check if a PyPI package meets the specified popularity threshold.

    This function determines whether a given PyPI package is considered "hot" or
    popular based on its download count from the last month. A package is considered
    hot if its download count meets or exceeds the specified minimum threshold.

    :param pypi_name: Name of the PyPI package to check
    :type pypi_name: str
    :param min_last_month_downloads: Minimum download count threshold for considering
                                     a package as hot, defaults to 1000000 (1 million)
    :type min_last_month_downloads: int, optional
    :return: True if the package exists and meets the download threshold, False otherwise
    :rtype: bool

    .. note::
       The function returns False if the package name is not found in the statistics,
       even if the package exists on PyPI. This means the package either doesn't exist
       or wasn't included in the statistics dataset.

    .. warning::
       Package name matching is case-sensitive. Ensure the package name matches
       exactly as it appears on PyPI.

    Example::

        >>> # Check if numpy is a hot project (default 1M threshold)
        >>> is_hot_pypi_project('numpy')
        True
        >>> 
        >>> # Check with custom threshold
        >>> is_hot_pypi_project('requests', min_last_month_downloads=5000000)
        True
        >>> 
        >>> # Check a less popular package
        >>> is_hot_pypi_project('obscure-package', min_last_month_downloads=100)
        False
        >>> 
        >>> # Check non-existent package
        >>> is_hot_pypi_project('this-package-does-not-exist')
        False

    """
    d = _get_pypi_downloads_dict()
    return pypi_name in d and d[pypi_name] >= min_last_month_downloads
