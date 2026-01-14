"""
Fake LLM Model Module

This module provides a fake implementation of an LLM (Large Language Model) for testing and development purposes.
It simulates LLM behavior by returning predefined responses based on configurable rules, supporting both
synchronous and streaming response modes with customizable word-per-second rates.

The module includes:
- FakeResponseStream: A stream handler for fake responses with reasoning and content separation
- FakeLLMModel: A mock LLM model that returns responses based on rule matching
"""

import time
from typing import List, Union, Tuple, Optional, Any, Callable

import jieba

from .base import LLMModel
from .stream import ResponseStream


class FakeResponseStream(ResponseStream):
    """
    A fake response stream that handles streaming responses with reasoning and content.
    
    This class extends ResponseStream to provide a simple implementation for testing purposes,
    where chunks are tuples of (reasoning_content, content).
    """

    def _get_reasoning_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract reasoning content from a chunk.

        :param chunk: The chunk to extract reasoning content from, expected to be a tuple.
        :type chunk: Any
        :return: The reasoning content from the chunk, or None if not present.
        :rtype: Optional[str]
        """
        return chunk[0]

    def _get_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract main content from a chunk.

        :param chunk: The chunk to extract content from, expected to be a tuple.
        :type chunk: Any
        :return: The main content from the chunk, or None if not present.
        :rtype: Optional[str]
        """
        return chunk[1]


FakeResponseTyping = Union[str, Tuple[str, str], Callable]
"""Type alias for fake response types: can be a string, tuple of (reasoning, content), or a callable."""


def _fn_always_true(messages: List[dict], **params) -> bool:
    """
    A rule function that always returns True.

    :param messages: The list of message dictionaries.
    :type messages: List[dict]
    :param params: Additional parameters (unused).
    :type params: dict
    :return: Always returns True.
    :rtype: bool
    """
    _ = messages, params
    return True


class FakeLLMModel(LLMModel):
    """
    A fake LLM model implementation for testing and development.
    
    This class simulates an LLM by returning predefined responses based on configurable rules.
    It supports both synchronous and streaming response modes, with customizable streaming speed.
    Responses can be configured to match specific conditions or keywords in messages.
    
    Example::
        >>> model = FakeLLMModel(stream_wps=50)
        >>> model.response_when_keyword_in_last_message("weather", "It's sunny today!")
        >>> response = model.ask([{"role": "user", "content": "What's the weather?"}])
        >>> print(response)
        It's sunny today!

        >>> model.response_always("Hello, I'm a fake LLM!")
        >>> response = model.ask([{"role": "user", "content": "Hi"}])
        >>> print(response)
        Hello, I'm a fake LLM!
    """

    def __init__(self, stream_wps: float = 50):
        """
        Initialize the fake LLM model.

        :param stream_wps: Words per second for streaming responses (default: 50).
        :type stream_wps: float
        """
        self.stream_fps = stream_wps
        self._rules: List[Tuple[Callable, FakeResponseTyping]] = []

    def _get_response(self, messages: List[dict], **params) -> Tuple[str, str]:
        """
        Get response by matching rules in order.

        :param messages: The list of message dictionaries containing conversation history.
        :type messages: List[dict]
        :param params: Additional parameters to pass to rule checking and response functions.
        :type params: dict
        :return: A tuple of (reasoning_content, content).
        :rtype: Tuple[str, str]
        :raises AssertionError: If no matching rule is found for the message.
        """
        for fn_rule_check, fn_response in self._rules:
            if fn_rule_check(messages=messages, **params):
                if callable(fn_response):
                    retval = fn_response(messages=messages, **params)
                else:
                    retval = fn_response
                if isinstance(retval, (list, tuple)):
                    reasoning_content, content = retval
                else:
                    reasoning_content, content = '', retval
                return reasoning_content, content
        else:
            assert False, 'No response rule found for this message.'

    def response_always(self, response: FakeResponseTyping) -> 'FakeLLMModel':
        """
        Add a rule that always returns the specified response.

        :param response: The response to return, can be a string, tuple of (reasoning, content), or callable.
        :type response: FakeResponseTyping
        :return: Self for method chaining.
        :rtype: FakeLLMModel
        
        Example::
            >>> model = FakeLLMModel()
            >>> model.response_always("Default response")
            >>> model.ask([{"role": "user", "content": "anything"}])
            'Default response'
        """
        self._rules.append((_fn_always_true, response))
        return self

    def response_when(self, fn_when: Callable, response: FakeResponseTyping) -> 'FakeLLMModel':
        """
        Add a conditional rule that returns the specified response when the condition is met.

        :param fn_when: A callable that takes (messages, **params) and returns bool.
        :type fn_when: Callable
        :param response: The response to return when condition is True.
        :type response: FakeResponseTyping
        :return: Self for method chaining.
        :rtype: FakeLLMModel
        
        Example::
            >>> model = FakeLLMModel()
            >>> model.response_when(
            ...     lambda messages, **params: len(messages) > 2,
            ...     "Long conversation response"
            ... )
        """
        self._rules.append((fn_when, response))
        return self

    def response_when_keyword_in_last_message(
            self,
            keywords: Union[str, List[str]],
            response: FakeResponseTyping
    ) -> 'FakeLLMModel':
        """
        Add a rule that returns the specified response when any keyword is found in the last message.

        :param keywords: A keyword or list of keywords to match in the last message content.
        :type keywords: Union[str, List[str]]
        :param response: The response to return when keyword is found.
        :type response: FakeResponseTyping
        :return: Self for method chaining.
        :rtype: FakeLLMModel
        
        Example::
            >>> model = FakeLLMModel()
            >>> model.response_when_keyword_in_last_message(
            ...     ["weather", "temperature"],
            ...     "It's 25 degrees and sunny!"
            ... )
            >>> model.ask([{"role": "user", "content": "What's the weather?"}])
            "It's 25 degrees and sunny!"
        """
        if isinstance(keywords, (list, tuple)):
            keywords = keywords
        else:
            keywords = [keywords]

        def _fn_keyword_check(messages: List[dict], **params) -> bool:
            """Check if any keyword exists in the last message."""
            _ = params
            for keyword in keywords:
                if keyword in messages[-1]['content']:
                    return True
            return False

        self._rules.append((_fn_keyword_check, response))
        return self

    def ask(
            self,
            messages: List[dict],
            with_reasoning: bool = False,
            **params
    ) -> Union[str, Tuple[Optional[str], str]]:
        """
        Send messages and get a synchronous response.

        :param messages: The list of message dictionaries containing conversation history.
        :type messages: List[dict]
        :param with_reasoning: If True, return both reasoning and content as a tuple (default: False).
        :type with_reasoning: bool
        :param params: Additional parameters to pass to response functions.
        :type params: dict
        :return: The response content string, or tuple of (reasoning_content, content) if with_reasoning is True.
        :rtype: Union[str, Tuple[Optional[str], str]]
        
        Example::
            >>> model = FakeLLMModel()
            >>> model.response_always(("thinking...", "final answer"))
            >>> model.ask([{"role": "user", "content": "test"}])
            'final answer'
            >>> model.ask([{"role": "user", "content": "test"}], with_reasoning=True)
            ('thinking...', 'final answer')
        """
        reasoning_content, content = self._get_response(messages=messages, **params)
        if with_reasoning:
            return reasoning_content, content
        else:
            return content

    def _iter_per_words(
            self,
            content: str,
            reasoning_content: Optional[str] = None
    ):
        """
        Generate word-by-word chunks for streaming, with delays between words.

        :param content: The main content to stream.
        :type content: str
        :param reasoning_content: Optional reasoning content to stream first.
        :type reasoning_content: Optional[str]
        :yield: Tuples of (reasoning_word, content_word) where one is None and the other contains a word.
        :rtype: Generator[Tuple[Optional[str], Optional[str]], None, None]
        """
        if reasoning_content:
            for word in jieba.cut(reasoning_content):
                if word:
                    yield word, None
                    time.sleep(1 / self.stream_fps)

        if content:
            for word in jieba.cut(content):
                if word:
                    yield None, word
                    time.sleep(1 / self.stream_fps)

    def ask_stream(
            self,
            messages: List[dict],
            with_reasoning: bool = False,
            **params
    ) -> ResponseStream:
        """
        Send messages and get a streaming response.

        :param messages: The list of message dictionaries containing conversation history.
        :type messages: List[dict]
        :param with_reasoning: If True, include reasoning content in the stream (default: False).
        :type with_reasoning: bool
        :param params: Additional parameters to pass to response functions.
        :type params: dict
        :return: A ResponseStream object that yields word-by-word chunks.
        :rtype: ResponseStream
        
        Example::
            >>> model = FakeLLMModel(stream_wps=10)
            >>> model.response_always("Hello world")
            >>> stream = model.ask_stream([{"role": "user", "content": "Hi"}])
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
            Hello world
        """
        reasoning_content, content = self._get_response(messages=messages, **params)
        return FakeResponseStream(
            session=self._iter_per_words(
                reasoning_content=reasoning_content,
                content=content,
            ),
            with_reasoning=with_reasoning,
        )
