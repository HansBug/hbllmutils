"""
Fake LLM Model Module

This module provides a fake implementation of an LLM (Large Language Model) for testing and development purposes.
It simulates LLM behavior by returning predefined responses based on configurable rules, supporting both
synchronous and streaming response modes with customizable word-per-second rates.

The module includes:
- FakeResponseStream: A stream handler for fake responses with reasoning and content separation
- FakeLLMModel: An immutable mock LLM model that returns responses based on rule matching
- FakeResponseSequence: A sequence-based response handler that returns responses in order
"""

import time
from typing import List, Union, Tuple, Optional, Any, Callable, Generator

import jieba

from .base import LLMModel
from .stream import ResponseStream


class FakeResponseSequence:
    """
    A sequence-based response handler that returns responses in order.

    This class maintains immutability by creating new instances when the index changes,
    ensuring thread safety and compatibility with FakeLLMModel's immutable design.
    """

    def __init__(self, responses: List[Union[str, Tuple[str, str]]], index: int = 0):
        """
        Initialize the response sequence.

        :param responses: List of responses to return in order.
        :type responses: List[Union[str, Tuple[str, str]]]
        :param index: Current index in the sequence (default: 0).
        :type index: int
        """
        self._response_contents = tuple(responses)  # Make immutable
        self._index = index

    @property
    def current_index(self) -> int:
        """
        Get the current index in the sequence.

        :return: The current index position.
        :rtype: int
        """
        return self._index

    @property
    def total_responses(self) -> int:
        """
        Get the total number of responses in the sequence.

        :return: The total count of responses.
        :rtype: int
        """
        return len(self._response_contents)

    @property
    def has_more_responses(self) -> bool:
        """
        Check if there are more responses available.

        :return: True if more responses are available, False otherwise.
        :rtype: bool
        """
        return self._index < len(self._response_contents)

    def rule_check(self, messages: List[dict], **params) -> bool:
        """
        Check if this sequence can provide a response.

        :param messages: The list of message dictionaries.
        :type messages: List[dict]
        :param params: Additional parameters (unused).
        :type params: dict
        :return: True if there are more responses available, False otherwise.
        :rtype: bool
        """
        _ = messages, params  # Unused parameters
        return self.has_more_responses

    def response(self, messages: List[dict], **params) -> Tuple[str, str]:
        """
        Get the next response in the sequence.

        :param messages: The list of message dictionaries.
        :type messages: List[dict]
        :param params: Additional parameters (unused).
        :type params: dict
        :return: A tuple of (reasoning_content, content).
        :rtype: Tuple[str, str]
        :raises IndexError: If no more responses are available.
        """
        _ = messages, params  # Unused parameters

        if not self.has_more_responses:
            raise IndexError(
                f"No more responses available. Current index: {self._index}, Total: {len(self._response_contents)}")

        retval = self._response_contents[self._index]
        if isinstance(retval, (list, tuple)):
            reasoning_content, content = retval
        else:
            reasoning_content, content = '', retval

        return reasoning_content, content

    def advance(self) -> 'FakeResponseSequence':
        """
        Create a new instance with the index advanced by 1.

        :return: A new FakeResponseSequence instance with incremented index.
        :rtype: FakeResponseSequence
        """
        return FakeResponseSequence(list(self._response_contents), self._index + 1)

    def reset(self) -> 'FakeResponseSequence':
        """
        Create a new instance with the index reset to 0.

        :return: A new FakeResponseSequence instance with index reset to 0.
        :rtype: FakeResponseSequence
        """
        return FakeResponseSequence(list(self._response_contents), 0)

    def __eq__(self, other) -> bool:
        """
        Check equality with another FakeResponseSequence instance.

        :param other: The other instance to compare with.
        :type other: Any
        :return: True if instances are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, FakeResponseSequence):
            return False
        return (self._response_contents == other._response_contents and
                self._index == other._index)

    def __hash__(self) -> int:
        """
        Return hash for use in sets and as dict keys.

        :return: Hash value of the instance.
        :rtype: int
        """
        return hash((self._response_contents, self._index))

    def __repr__(self) -> str:
        """
        Return string representation of the sequence.

        :return: String representation showing responses and current index.
        :rtype: str
        """
        return f"FakeResponseSequence(responses={list(self._response_contents)}, index={self._index})"


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
    An immutable fake LLM model implementation for testing and development.

    This class simulates an LLM by returning predefined responses based on configurable rules.
    It supports both synchronous and streaming response modes, with customizable streaming speed.
    Responses can be configured to match specific conditions or keywords in messages.

    All modification operations return new instances, ensuring immutability and thread safety.

    Example::
        >>> model = FakeLLMModel(stream_wps=50)
        >>> model_with_rule = model.response_when_keyword_in_last_message("weather", "It's sunny today!")
        >>> response = model_with_rule.ask([{"role": "user", "content": "What's the weather?"}])
        >>> print(response)
        It's sunny today!

        >>> final_model = model_with_rule.response_always("Hello, I'm a fake LLM!")
        >>> response = final_model.ask([{"role": "user", "content": "Hi"}])
        >>> print(response)
        Hello, I'm a fake LLM!
    """

    def __init__(self, stream_wps: float = 50, rules: Optional[List[Tuple[Callable, FakeResponseTyping]]] = None):
        """
        Initialize the fake LLM model.

        :param stream_wps: Words per second for streaming responses (default: 50).
        :type stream_wps: float
        :param rules: List of (rule_function, response) tuples. Internal parameter, not intended for direct use.
        :type rules: Optional[List[Tuple[Callable, FakeResponseTyping]]]
        """
        self._stream_wps = stream_wps
        # Create a defensive copy to ensure immutability
        self._rules = tuple(rules) if rules is not None else tuple()

        # Make the object immutable by preventing attribute modification
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Prevent attribute modification after initialization to ensure immutability.

        :param name: The attribute name.
        :type name: str
        :param value: The attribute value.
        :type value: Any
        :raises AttributeError: If attempting to modify attributes after initialization.
        """
        if hasattr(self, '_frozen') and self._frozen:
            raise AttributeError(f"Cannot modify attribute '{name}' of immutable {self.__class__.__name__}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """
        Prevent attribute deletion to ensure immutability.

        :param name: The attribute name.
        :type name: str
        :raises AttributeError: Always, as deletion is not allowed.
        """
        raise AttributeError(f"Cannot delete attribute '{name}' of immutable {self.__class__.__name__}")

    @property
    def stream_wps(self) -> float:
        """
        Get the streaming words per second rate.

        :return: The words per second rate for streaming.
        :rtype: float
        """
        return self._stream_wps

    @property
    def _logger_name(self) -> str:
        """
        Get the logger name for this model.

        :return: The logger name string.
        :rtype: str
        """
        return '<faker>'

    @property
    def rules_count(self) -> int:
        """
        Get the number of configured response rules.

        :return: The count of rules currently configured.
        :rtype: int
        """
        return len(self._rules)

    def _create_new_instance(self, **kwargs) -> 'FakeLLMModel':
        """
        Create a new instance with modified parameters.

        :param kwargs: Parameters to override in the new instance.
        :type kwargs: dict
        :return: A new FakeLLMModel instance.
        :rtype: FakeLLMModel
        """
        new_stream_wps = kwargs.get('stream_wps', self._stream_wps)
        new_rules = kwargs.get('rules', self._rules)

        return FakeLLMModel(stream_wps=new_stream_wps, rules=new_rules)

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

    def with_stream_wps(self, stream_wps: float) -> 'FakeLLMModel':
        """
        Create a new instance with a different streaming words per second rate.

        :param stream_wps: The new words per second rate for streaming responses.
        :type stream_wps: float
        :return: A new FakeLLMModel instance with the updated stream rate.
        :rtype: FakeLLMModel

        Example::
            >>> model = FakeLLMModel(stream_wps=50)
            >>> fast_model = model.with_stream_wps(100)
            >>> fast_model.stream_wps
            100
            >>> model.stream_wps  # Original unchanged
            50
        """
        return self._create_new_instance(stream_wps=stream_wps)

    def response_always(self, response: FakeResponseTyping) -> 'FakeLLMModel':
        """
        Create a new instance with a rule that always returns the specified response.

        :param response: The response to return, can be a string, tuple of (reasoning, content), or callable.
        :type response: FakeResponseTyping
        :return: A new FakeLLMModel instance with the added rule.
        :rtype: FakeLLMModel

        Example::
            >>> model = FakeLLMModel()
            >>> new_model = model.response_always("Default response")
            >>> new_model.ask([{"role": "user", "content": "anything"}])
            'Default response'
            >>> model.rules_count  # Original unchanged
            0
            >>> new_model.rules_count
            1
        """
        new_rules = list(self._rules) + [(_fn_always_true, response)]
        return self._create_new_instance(rules=new_rules)

    def response_when(self, fn_when: Callable, response: FakeResponseTyping) -> 'FakeLLMModel':
        """
        Create a new instance with a conditional rule that returns the specified response when the condition is met.

        :param fn_when: A callable that takes (messages, **params) and returns bool.
        :type fn_when: Callable
        :param response: The response to return when condition is True.
        :type response: FakeResponseTyping
        :return: A new FakeLLMModel instance with the added rule.
        :rtype: FakeLLMModel

        Example::
            >>> model = FakeLLMModel()
            >>> new_model = model.response_when(
            ...     lambda messages, **params: len(messages) > 2,
            ...     "Long conversation response"
            ... )
        """
        new_rules = list(self._rules) + [(fn_when, response)]
        return self._create_new_instance(rules=new_rules)

    def response_when_keyword_in_last_message(
            self,
            keywords: Union[str, List[str]],
            response: FakeResponseTyping
    ) -> 'FakeLLMModel':
        """
        Create a new instance with a rule that returns the specified response when any keyword is found in the last message.

        :param keywords: A keyword or list of keywords to match in the last message content.
        :type keywords: Union[str, List[str]]
        :param response: The response to return when keyword is found.
        :type response: FakeResponseTyping
        :return: A new FakeLLMModel instance with the added rule.
        :rtype: FakeLLMModel

        Example::
            >>> model = FakeLLMModel()
            >>> new_model = model.response_when_keyword_in_last_message(
            ...     ["weather", "temperature"],
            ...     "It's 25 degrees and sunny!"
            ... )
            >>> new_model.ask([{"role": "user", "content": "What's the weather?"}])
            "It's 25 degrees and sunny!"
        """
        if isinstance(keywords, (list, tuple)):
            keywords_tuple = tuple(keywords)  # Make immutable
        else:
            keywords_tuple = (keywords,)

        def _fn_keyword_check(messages: List[dict], **params) -> bool:
            """
            Check if any keyword exists in the last message.

            :param messages: The list of message dictionaries.
            :type messages: List[dict]
            :param params: Additional parameters (unused).
            :type params: dict
            :return: True if any keyword is found in the last message content, False otherwise.
            :rtype: bool
            """
            _ = params
            for keyword in keywords_tuple:
                if keyword in messages[-1]['content']:
                    return True
            return False

        new_rules = list(self._rules) + [(_fn_keyword_check, response)]
        return self._create_new_instance(rules=new_rules)

    def response_sequence(self, responses: List[Union[str, Tuple[str, str]]]) -> 'FakeLLMModel':
        """
        Create a new instance with a rule that returns responses in sequence.

        Each call to ask() or ask_stream() will return the next response in the sequence.
        Once all responses are exhausted, the rule will no longer match.

        :param responses: List of responses to return in order. Each can be a string or tuple of (reasoning, content).
        :type responses: List[Union[str, Tuple[str, str]]]
        :return: A new FakeLLMModel instance with the sequence rule added.
        :rtype: FakeLLMModel
        :raises ValueError: If the response list is empty.

        Example::
            >>> model = FakeLLMModel()
            >>> seq_model = model.response_sequence([
            ...     "First response",
            ...     ("thinking about second", "Second response"),
            ...     "Third response"
            ... ])
            >>> seq_model.ask([{"role": "user", "content": "test1"}])
            'First response'
            >>> seq_model.ask([{"role": "user", "content": "test2"}])
            'Second response'
            >>> seq_model.ask([{"role": "user", "content": "test3"}])
            'Third response'
        """
        if not responses:
            raise ValueError("Response sequence cannot be empty")

        sequence = FakeResponseSequence(responses)

        # Create a stateful wrapper that maintains the sequence state
        class _SequenceWrapper:
            """
            Internal wrapper class for managing sequence state.

            This class maintains the current position in the response sequence
            and advances it after each response is retrieved.
            """

            def __init__(self, initial_sequence: FakeResponseSequence):
                """
                Initialize the sequence wrapper.

                :param initial_sequence: The initial response sequence.
                :type initial_sequence: FakeResponseSequence
                """
                self._sequence = initial_sequence

            def rule_check(self, messages: List[dict], **params) -> bool:
                """
                Check if the sequence has more responses available.

                :param messages: The list of message dictionaries.
                :type messages: List[dict]
                :param params: Additional parameters.
                :type params: dict
                :return: True if more responses are available, False otherwise.
                :rtype: bool
                """
                return self._sequence.rule_check(messages, **params)

            def response(self, messages: List[dict], **params) -> Tuple[str, str]:
                """
                Get the next response and advance the sequence.

                :param messages: The list of message dictionaries.
                :type messages: List[dict]
                :param params: Additional parameters.
                :type params: dict
                :return: A tuple of (reasoning_content, content).
                :rtype: Tuple[str, str]
                """
                result = self._sequence.response(messages, **params)
                # Advance the sequence for next call
                self._sequence = self._sequence.advance()
                return result

            def __repr__(self) -> str:
                """
                Return string representation of the wrapper.

                :return: String representation.
                :rtype: str
                """
                return f"_SequenceWrapper({self._sequence})"

        wrapper = _SequenceWrapper(sequence)
        return self.response_when(wrapper.rule_check, wrapper.response)

    def clear_rules(self) -> 'FakeLLMModel':
        """
        Create a new instance with all rules removed.

        :return: A new FakeLLMModel instance with no rules.
        :rtype: FakeLLMModel

        Example::
            >>> model = FakeLLMModel().response_always("Hello")
            >>> model.rules_count
            1
            >>> clean_model = model.clear_rules()
            >>> clean_model.rules_count
            0
        """
        return self._create_new_instance(rules=[])

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
            >>> model = FakeLLMModel().response_always(("thinking...", "final answer"))
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
    ) -> Generator[Tuple[Optional[str], Optional[str]], None, None]:
        """
        Generate word-by-word chunks for streaming, with delays between words.

        This method uses jieba to segment text into words and yields them one at a time,
        with a delay calculated based on the stream_wps (words per second) setting.
        Reasoning content is yielded first if provided, followed by the main content.

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
                    time.sleep(1 / self._stream_wps)

        if content:
            for word in jieba.cut(content):
                if word:
                    yield None, word
                    time.sleep(1 / self._stream_wps)

    def ask_stream(
            self,
            messages: List[dict],
            with_reasoning: bool = False,
            **params
    ) -> ResponseStream:
        """
        Send messages and get a streaming response.

        This method returns a ResponseStream that yields the response word-by-word,
        simulating the streaming behavior of a real LLM. The streaming speed is
        controlled by the stream_wps parameter set during initialization.

        :param messages: The list of message dictionaries containing conversation history.
        :type messages: List[dict]
        :param with_reasoning: If True, include reasoning content in the stream (default: False).
        :type with_reasoning: bool
        :param params: Additional parameters to pass to response functions.
        :type params: dict
        :return: A ResponseStream object that yields word-by-word chunks.
        :rtype: ResponseStream

        Example::
            >>> model = FakeLLMModel(stream_wps=10).response_always("Hello world")
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

    def __repr__(self) -> str:
        """
        Return a string representation of the FakeLLMModel instance.

        Shows the stream_wps parameter and the number of configured rules.

        :return: String representation of the instance.
        :rtype: str

        Example::
            >>> model = FakeLLMModel(stream_wps=100).response_always("Hello")
            >>> repr(model)
            'FakeLLMModel(stream_wps=100, rules_count=1)'
        """
        # Collect all parameters
        params = {
            'stream_wps': self._stream_wps,
            'rules_count': len(self._rules),
        }

        # Build parameter string list
        param_strings = []
        for key, value in params.items():
            param_strings.append(f"{key}={value!r}")

        params_str = ', '.join(param_strings)
        return f"{self.__class__.__name__}({params_str})"

    def _params(self) -> tuple:
        """
        Get the parameters that define this model instance.

        This method returns a stable and hashable representation of the model's
        parameters, including the streaming rate and rules configuration.
        Since rules contain functions which are not directly hashable in a stable way,
        we use their string representation and memory addresses for comparison.

        :return: A hashable tuple representation of the model's parameters.
        :rtype: tuple
        """
        # Convert rules to a hashable format
        # Each rule is (function, response), we need to make this hashable
        hashable_rules = []
        for fn_rule, response in self._rules:
            # For functions, use their string representation and id for uniqueness
            # This ensures that the same function object will have the same hash
            rule_key = (
                id(fn_rule),  # Memory address for uniqueness
                str(fn_rule),  # String representation for readability
            )

            # Handle different response types
            if callable(response):
                response_key = (
                    'callable',
                    id(response),
                    str(response)
                )
            elif isinstance(response, (list, tuple)):
                # Convert to tuple to make it hashable
                response_key = ('tuple', tuple(response))
            else:
                # String or other hashable type
                response_key = ('value', response)

            hashable_rules.append((rule_key, response_key))

        return (
            self._stream_wps,
            tuple(hashable_rules)  # Convert list to tuple for hashability
        )
