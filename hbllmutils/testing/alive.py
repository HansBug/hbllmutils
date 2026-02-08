"""
Alive tests for verifying basic LLM responsiveness.

This module provides minimal, binary checks that validate whether an LLM model
can respond to simple interactions. The public API offers two convenience
functions, :func:`hello` and :func:`ping`, which use internal
:class:`~hbllmutils.testing.base.BinaryTest` implementations to execute one or
multiple test runs. Results are returned as either a single
:class:`~hbllmutils.testing.base.BinaryTestResult` or a
:class:`~hbllmutils.testing.base.MultiBinaryTestResult` with aggregated
statistics.

The module contains the following public components:

* :func:`hello` - Run a basic greeting test (expects any non-empty response).
* :func:`ping` - Run a ping-pong test (expects a response containing "pong").

.. note::
   These tests are intentionally lightweight and do not validate the semantic
   correctness of responses beyond the basic criteria described.

Example::

    >>> from hbllmutils.testing.alive import hello, ping
    >>> result = hello(my_model)
    >>> result.passed
    True
    >>> results = ping(my_model, n=3)
    >>> results.passed_ratio
    1.0
"""

from typing import Any, Union

from .base import BinaryTest, BinaryTestResult, MultiBinaryTestResult
from ..history import LLMHistory
from ..model import LLMModel, LLMModelTyping


class _HelloTest(BinaryTest):
    """
    Internal test class that implements a basic greeting test for LLM models.

    This test sends a simple ``"hello!"`` message to the model and checks if it
    receives a non-empty response. The test passes if the model returns any
    content in response to the greeting.

    .. warning::
       This class is internal and not part of the public API. Use
       :func:`hello` instead for a stable interface.
    """

    __desc_name__ = 'hello test'

    def _single_test(self, model: LLMModel, **params: Any) -> BinaryTestResult:
        """
        Execute a single hello test on the given LLM model.

        Sends a ``"hello!"`` message to the model and evaluates whether the model
        responds with any content. The test is considered passed if the model
        returns a non-empty response.

        :param model: The LLM model to test.
        :type model: LLMModel
        :param params: Additional parameters to pass to the model's ``ask`` method.
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


def hello(model: LLMModelTyping, n: int = 1) -> Union[MultiBinaryTestResult, BinaryTestResult]:
    """
    Perform a hello test on an LLM model.

    This function tests whether the given LLM model can respond to a basic
    greeting (``"hello!"``). It can run the test once or multiple times to gather
    statistical results.

    :param model: The LLM model to test.
    :type model: LLMModelTyping
    :param n: The number of times to run the test. Defaults to 1.
    :type n: int
    :return: If ``n == 1``, returns a single :class:`BinaryTestResult`. If
        ``n > 1``, returns a :class:`MultiBinaryTestResult` containing all test
        results and statistics.
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

    This test sends a ``"ping!"`` message to the model and checks if the response
    contains the word ``"pong"`` (case-insensitive). The test passes if the model
    responds with a message containing ``"pong"``.

    .. warning::
       This class is internal and not part of the public API. Use
       :func:`ping` instead for a stable interface.
    """

    __desc_name__ = 'ping test'

    def _single_test(self, model: LLMModel, **params: Any) -> BinaryTestResult:
        """
        Execute a single ping test on the given LLM model.

        Sends a ``"ping!"`` message to the model and evaluates whether the model
        responds with a message containing ``"pong"`` (case-insensitive). The test
        is considered passed if ``"pong"`` is found in the response.

        :param model: The LLM model to test.
        :type model: LLMModel
        :param params: Additional parameters to pass to the model's ``ask`` method.
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


def ping(model: LLMModelTyping, n: int = 1) -> Union[MultiBinaryTestResult, BinaryTestResult]:
    """
    Perform a ping-pong test on an LLM model.

    This function tests whether the given LLM model can respond appropriately
    to a ``"ping!"`` message by including ``"pong"`` in its response. It can run
    the test once or multiple times to gather statistical results.

    :param model: The LLM model to test.
    :type model: LLMModelTyping
    :param n: The number of times to run the test. Defaults to 1.
    :type n: int
    :return: If ``n == 1``, returns a single :class:`BinaryTestResult`. If
        ``n > 1``, returns a :class:`MultiBinaryTestResult` containing all test
        results and statistics.
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
