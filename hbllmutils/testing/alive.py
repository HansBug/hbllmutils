"""
This module provides functionality for testing LLM models with basic binary tests.

The module implements simple binary tests that verify if an LLM model can respond
to basic interactions. It uses the BinaryTest framework to perform single or multiple
test runs and returns structured results.

Classes:
    _HelloTest: Internal test class that implements a basic greeting test.
    _PingTest: Internal test class that implements a ping-pong response test.

Functions:
    hello: Performs a hello test on an LLM model.
    ping: Performs a ping-pong test on an LLM model.
"""

from typing import Union

from .base import BinaryTest, BinaryTestResult, MultiBinaryTestResult
from ..history import LLMHistory
from ..model import LLMModel


class _HelloTest(BinaryTest):
    """
    Internal test class that implements a basic greeting test for LLM models.
    
    This test sends a simple "hello!" message to the model and checks if it
    receives a non-empty response. The test passes if the model returns any
    content in response to the greeting.
    """

    __desc_name__ = 'hello test'

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
            messages=LLMHistory().with_user_message('hello!').to_json(),
            **params,
        )
        return BinaryTestResult(
            passed=bool(content),
            content=content,
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


class _PingTest(BinaryTest):
    """
    Internal test class that implements a ping-pong response test for LLM models.
    
    This test sends a "ping!" message to the model and checks if the response
    contains the word "pong" (case-insensitive). The test passes if the model
    responds with a message containing "pong".
    """

    __desc_name__ = 'ping test'

    def _single_test(self, model: LLMModel, **params) -> BinaryTestResult:
        """
        Execute a single ping test on the given LLM model.
        
        Sends a "ping!" message to the model and evaluates whether the model
        responds with a message containing "pong" (case-insensitive). The test
        is considered passed if "pong" is found in the response.
        
        :param model: The LLM model to test.
        :type model: LLMModel
        :param params: Additional parameters to pass to the model's ask method.
        :type params: dict
        
        :return: The result of the binary test, including pass/fail status and content.
        :rtype: BinaryTestResult
        
        Example::
            >>> test = _PingTest()
            >>> result = test._single_test(my_model)
            >>> print(result.passed)
            True
            >>> print(result.content)
            'Pong!'
        """
        content = model.ask(
            messages=LLMHistory().with_user_message('ping!').to_json(),
            **params,
        )
        return BinaryTestResult(
            passed='pong' in content.lower(),
            content=content,
        )


def ping(model: LLMModel, n: int = 1) -> Union[MultiBinaryTestResult, BinaryTestResult]:
    """
    Perform a ping-pong test on an LLM model.
    
    This function tests whether the given LLM model can respond appropriately
    to a "ping!" message by including "pong" in its response. It can run the
    test once or multiple times to gather statistical results.
    
    :param model: The LLM model to test.
    :type model: LLMModel
    :param n: The number of times to run the test. Defaults to 1.
    :type n: int
    
    :return: If n=1, returns a single BinaryTestResult. If n>1, returns a
             MultiBinaryTestResult containing all test results and statistics.
    :rtype: Union[MultiBinaryTestResult, BinaryTestResult]
    
    Example::
        >>> # Single test
        >>> result = ping(my_model)
        >>> print(result.passed)
        True
        >>> print(result.content)
        'Pong!'
        
        >>> # Multiple tests
        >>> results = ping(my_model, n=5)
        >>> print(results.passed_count)
        5
        >>> print(results.passed_ratio)
        1.0
    """
    return _PingTest().test(model=model, n=n)
