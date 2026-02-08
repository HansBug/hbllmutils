"""
Binary testing utilities for language model evaluation.

This module provides a small framework for executing *binary* tests on large
language models, where each test yields a pass/fail result. It offers simple
data structures for representing the results of individual tests and aggregated
statistics for repeated runs. A base class is also provided to simplify the
implementation of concrete tests.

The module contains the following main components:

* :class:`BinaryTestResult` - Stores the outcome of a single binary test
* :class:`MultiBinaryTestResult` - Aggregates multiple test results and statistics
* :class:`BinaryTest` - Base class for implementing binary tests

Typical usage involves subclassing :class:`BinaryTest` and implementing
:meth:`BinaryTest._single_test` to define the test logic. The :meth:`BinaryTest.test`
method can then execute the test once or multiple times to produce statistics.

Example::

    >>> class AlwaysPassTest(BinaryTest):
    ...     def _single_test(self, model, **params):
    ...         return BinaryTestResult(passed=True, content="ok")
    ...
    >>> test = AlwaysPassTest()
    >>> result = test.test(model="my-llm", n=3, silent=True)
    >>> result.passed_ratio
    1.0

.. note::
   This module expects a non-empty list of tests when computing aggregate
   statistics. Passing an empty list to :class:`MultiBinaryTestResult` will
   raise a ``ZeroDivisionError`` due to division by zero.

"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from tqdm import tqdm

from ..model import LLMModel, load_llm_model, LLMModelTyping


@dataclass
class BinaryTestResult:
    """
    Data class representing the result of a single binary test.

    :param passed: Whether the test passed or failed.
    :type passed: bool
    :param content: The content or output produced during the test.
    :type content: str

    Example::

        >>> BinaryTestResult(passed=True, content="response text")
        BinaryTestResult(passed=True, content='response text')
    """
    passed: bool
    content: str


@dataclass
class MultiBinaryTestResult:
    """
    Data class representing aggregated results from multiple binary tests.

    This class automatically calculates statistics about the test results,
    including total count, passed/failed counts, and their ratios.

    :param tests: List of individual binary test results.
    :type tests: List[BinaryTestResult]
    :param total_count: Total number of tests (automatically calculated).
    :type total_count: int
    :param passed_count: Number of tests that passed (automatically calculated).
    :type passed_count: int
    :param passed_ratio: Ratio of tests that passed (automatically calculated).
    :type passed_ratio: float
    :param failed_count: Number of tests that failed (automatically calculated).
    :type failed_count: int
    :param failed_ratio: Ratio of tests that failed (automatically calculated).
    :type failed_ratio: float

    :raises ZeroDivisionError: If ``tests`` is an empty list.

    Example::

        >>> results = [
        ...     BinaryTestResult(passed=True, content="test1"),
        ...     BinaryTestResult(passed=False, content="test2"),
        ... ]
        >>> multi_result = MultiBinaryTestResult(tests=results)
        >>> multi_result.passed_ratio
        0.5
    """
    tests: List[BinaryTestResult]
    total_count: int = 0
    passed_count: int = 0
    passed_ratio: float = 0
    failed_count: int = 0
    failed_ratio: float = 0

    def __post_init__(self) -> None:
        """
        Post-initialization method that calculates test statistics.

        This method is automatically called after the dataclass is initialized.
        It computes the total count, passed/failed counts, and their ratios
        based on the provided test results.

        :raises ZeroDivisionError: If ``tests`` is an empty list.
        """
        self.total_count = len(self.tests)
        self.passed_count, self.failed_count = 0, 0
        for test in self.tests:
            if test.passed:
                self.passed_count += 1
            else:
                self.failed_count += 1
        self.passed_ratio = self.passed_count / self.total_count
        self.failed_ratio = self.failed_count / self.total_count


class BinaryTest:
    """
    Base class for implementing binary tests on language models.

    This class provides a framework for running tests that have a pass/fail
    outcome. Tests can be run once or multiple times to gather statistics.
    Subclasses should implement the :meth:`_single_test` method to define
    the specific test logic.

    :ivar __desc_name__: Optional descriptive name for the test, used in
        progress bars.
    :vartype __desc_name__: Optional[str]

    Example::

        >>> class MyBinaryTest(BinaryTest):
        ...     def _single_test(self, model, **params):
        ...         return BinaryTestResult(passed=True, content="ok")
        ...
        >>> test = MyBinaryTest()
        >>> result = test.test(model="my-llm", n=1, silent=True)
        >>> result.passed
        True
    """
    __desc_name__: Optional[str] = None

    def _single_test(self, model: LLMModel, **params: Any) -> BinaryTestResult:
        """
        Execute a single binary test on the given model.

        This is an abstract method that must be implemented by subclasses to
        define the specific test logic.

        :param model: The language model to test.
        :type model: LLMModel
        :param params: Additional parameters for the test.
        :type params: dict

        :return: The result of the single test.
        :rtype: BinaryTestResult
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def test(
            self,
            model: LLMModelTyping,
            n: int = 1,
            silent: bool = False,
            **params: Any,
    ) -> Union[BinaryTestResult, MultiBinaryTestResult]:
        """
        Run the binary test one or multiple times on the given model.

        If ``n == 1``, runs a single test and returns a :class:`BinaryTestResult`.
        If ``n > 1``, runs multiple tests and returns a
        :class:`MultiBinaryTestResult` with aggregated statistics.

        :param model: The language model to test. Can be a model instance or
            a model identifier.
        :type model: LLMModelTyping
        :param n: Number of times to run the test, defaults to 1.
        :type n: int
        :param silent: If True, suppresses the progress bar, defaults to False.
        :type silent: bool
        :param params: Additional parameters to pass to the test.
        :type params: dict

        :return: Single test result if ``n == 1``, otherwise aggregated results.
        :rtype: Union[BinaryTestResult, MultiBinaryTestResult]

        Example::

            >>> test = MyBinaryTest()  # Assuming MyBinaryTest is a subclass
            >>> result = test.test(model="my-llm", n=10, silent=True)
            >>> print(f"Pass rate: {result.passed_ratio}")
            Pass rate: 0.8
        """
        model = load_llm_model(model)
        if n == 1:
            return self._single_test(model=model, **params)
        else:
            tests = []
            for _ in tqdm(range(n), disable=silent, desc=self.__desc_name__):
                tests.append(self._single_test(model=model, **params))
            return MultiBinaryTestResult(tests=tests)
