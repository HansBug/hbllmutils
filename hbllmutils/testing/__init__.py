"""
Testing framework for binary language model checks.

This package module exposes the core testing interfaces and convenience
functions used to run pass/fail tests against language models. It provides
base classes for implementing custom tests, result containers for single
and multiple runs, and simple "alive" checks for basic responsiveness.

The module contains the following main components:

* :class:`BinaryTest` - Base class for defining binary pass/fail tests
* :class:`BinaryTestResult` - Result container for a single test run
* :class:`MultiBinaryTestResult` - Aggregated results and statistics
* :func:`hello` - Basic greeting test for model responsiveness
* :func:`ping` - Ping-pong response test for model responsiveness

Example::

    >>> from hbllmutils.testing import hello, ping, BinaryTestResult
    >>> result = hello(my_model)
    >>> result.passed
    True
    >>> results = ping(my_model, n=3)
    >>> results.passed_count >= 0
    True

.. note::
   The detailed implementations of the tests and result classes are provided
   by the submodules :mod:`hbllmutils.testing.alive` and
   :mod:`hbllmutils.testing.base`.

"""

from .alive import hello, ping
from .base import BinaryTest, MultiBinaryTestResult, BinaryTestResult
