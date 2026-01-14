"""
This module provides functionality for testing LLM models with basic binary tests.

The module implements a simple "hello" test that verifies if an LLM model can respond
to a basic greeting. It uses the BinaryTest framework to perform single or multiple
test runs and returns structured results.

Classes:
    _HelloTest: Internal test class that implements a basic greeting test.

Functions:
    hello: Performs a hello test on an LLM model.
"""

from typing import Union

from .base import BinaryTest, BinaryTestResult, MultiBinaryTestResult
from ..history import LLMHistory
from ..model import LLMModel


# @dataclass
# class BinaryTestResult:
#     passed: bool
#     content: str
#     reasoning_content: Optional[str] = None
#
#
# @dataclass
# class MultiBinaryTestResult:
#     tests: List[BinaryTestResult]
#     total_count: int = 0
#     passed_count: int = 0
#     passed_ratio: float = 0
#     failed_count: int = 0
#     failed_ratio: float = 0
#
#     def __post_init__(self):
#         self.total_count = len(self.tests)
#         self.passed_count, self.failed_count = 0, 0
#         for test in self.tests:
#             if test.passed:
#                 self.passed_count += 1
#             else:
#                 self.failed_count += 1
#         self.passed_ratio = self.passed_count / self.total_count
#         self.failed_ratio = self.failed_count / self.total_count


# class BinaryTest:
#     def _single_test(self, model: LLMModel, **params) -> BinaryTestResult:
#         raise NotImplementedError  # pragma: no cover
# 
#     def test(self, model: LLMModel, n: int = 1, silent: bool = False, **params) \
#             -> Union[BinaryTestResult, MultiBinaryTestResult]:
#         if n == 1:
#             return self._single_test(model=model, **params)
#         else:
#             tests = []
#             for _ in tqdm(range(n), disable=silent):
#                 tests.append(self._single_test(model=model, **params))
#             return MultiBinaryTestResult(tests=tests)


class _HelloTest(BinaryTest):
    """
    Internal test class that implements a basic greeting test for LLM models.
    
    This test sends a simple "hello!" message to the model and checks if it
    receives a non-empty response. The test passes if the model returns any
    content in response to the greeting.
    """

    def _single_test(self, model: LLMModel, **params) -> BinaryTestResult:
        """
        Execute a single hello test on the given LLM model.
        
        Sends a "hello!" message to the model and evaluates whether the model
        responds with any content. The test is considered passed if the model
        returns a non-empty response.
        
        :param model: The LLM model to test.
        :type model: LLMModel
        :param params: Additional parameters to pass to the model's ask method.
        :type params: dict
        
        :return: The result of the binary test, including pass/fail status and content.
        :rtype: BinaryTestResult
        
        Example::
            >>> test = _HelloTest()
            >>> result = test._single_test(my_model)
            >>> print(result.passed)
            True
            >>> print(result.content)
            'Hello! How can I help you today?'
        """
        content = model.ask(
            messages=LLMHistory().append_user('hello!').to_json(),
            with_reasoning=False,
            **params,
        )
        return BinaryTestResult(
            passed=bool(content),
            content=content,
            reasoning_content=None,
        )


def hello(model: LLMModel, n: int = 1) -> Union[MultiBinaryTestResult, BinaryTestResult]:
    """
    Perform a hello test on an LLM model.
    
    This function tests whether the given LLM model can respond to a basic
    greeting ("hello!"). It can run the test once or multiple times to gather
    statistical results.
    
    :param model: The LLM model to test.
    :type model: LLMModel
    :param n: The number of times to run the test. Defaults to 1.
    :type n: int
    
    :return: If n=1, returns a single BinaryTestResult. If n>1, returns a
             MultiBinaryTestResult containing all test results and statistics.
    :rtype: Union[MultiBinaryTestResult, BinaryTestResult]
    
    Example::
        >>> # Single test
        >>> result = hello(my_model)
        >>> print(result.passed)
        True
        
        >>> # Multiple tests
        >>> results = hello(my_model, n=10)
        >>> print(results.passed_count)
        10
        >>> print(results.passed_ratio)
        1.0
    """
    return _HelloTest().test(model=model, n=n)
