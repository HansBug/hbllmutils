"""
This module provides a framework for conducting binary tests on language models.

It defines data structures for storing test results and a base class for implementing
binary tests. Binary tests are tests that have a pass/fail outcome, and can be run
multiple times to gather statistics.

Classes:
    BinaryTestResult: Stores the result of a single binary test
    MultiBinaryTestResult: Stores and analyzes results from multiple binary tests
    BinaryTest: Base class for implementing binary tests on language models
"""

from dataclasses import dataclass
from typing import List, Union, Optional

from tqdm import tqdm

from ..model import LLMModel


@dataclass
class BinaryTestResult:
    """
    Data class representing the result of a single binary test.

    :param passed: Whether the test passed or failed.
    :type passed: bool
    :param content: The content or output from the test.
    :type content: str
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

    Example::
        >>> results = [BinaryTestResult(passed=True, content="test1"), 
        ...            BinaryTestResult(passed=False, content="test2")]
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

    def __post_init__(self):
        """
        Post-initialization method that calculates test statistics.

        This method is automatically called after the dataclass is initialized.
        It computes the total count, passed/failed counts, and their ratios
        based on the provided test results.
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

    This class provides a framework for running tests that have a pass/fail outcome.
    Tests can be run once or multiple times to gather statistics. Subclasses should
    implement the _single_test method to define the specific test logic.

    Methods:
        _single_test: Abstract method to implement the test logic (must be overridden)
        test: Run the test one or multiple times and return results
    """
    __desc_name__: Optional[str] = None

    def _single_test(self, model: LLMModel, **params) -> BinaryTestResult:
        """
        Execute a single binary test on the given model.

        This is an abstract method that must be implemented by subclasses to define
        the specific test logic.

        :param model: The language model to test.
        :type model: LLMModel
        :param params: Additional parameters for the test.
        :type params: dict

        :return: The result of the single test.
        :rtype: BinaryTestResult
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def test(self, model: LLMModel, n: int = 1, silent: bool = False, **params) \
            -> Union[BinaryTestResult, MultiBinaryTestResult]:
        """
        Run the binary test one or multiple times on the given model.

        If n=1, runs a single test and returns a BinaryTestResult.
        If n>1, runs multiple tests and returns a MultiBinaryTestResult with
        aggregated statistics.

        :param model: The language model to test.
        :type model: LLMModel
        :param n: Number of times to run the test, defaults to 1.
        :type n: int
        :param silent: If True, suppresses the progress bar, defaults to False.
        :type silent: bool
        :param params: Additional parameters to pass to the test.
        :type params: dict

        :return: Single test result if n=1, otherwise aggregated results.
        :rtype: Union[BinaryTestResult, MultiBinaryTestResult]

        Example::
            >>> test = MyBinaryTest()  # Assuming MyBinaryTest is a subclass
            >>> result = test.test(model, n=10)
            >>> print(f"Pass rate: {result.passed_ratio}")
            Pass rate: 0.8
        """
        if n == 1:
            return self._single_test(model=model, **params)
        else:
            tests = []
            for _ in tqdm(range(n), disable=silent, desc=self.__desc_name__):
                tests.append(self._single_test(model=model, **params))
            return MultiBinaryTestResult(tests=tests)
