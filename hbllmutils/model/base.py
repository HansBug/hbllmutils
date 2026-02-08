"""
Abstract base interfaces for Large Language Model (LLM) implementations.

This module defines the :class:`LLMModel` abstract base class, which serves as the
contract for LLM backends in the ``hbllmutils`` package. Implementations are
expected to provide synchronous and streaming query methods while supporting
optional reasoning output.

The module contains the following main components:

* :class:`LLMModel` - Abstract interface for LLM model implementations

Example::

    >>> from hbllmutils.model.base import LLMModel
    >>> class MyLLM(LLMModel):
    ...     @property
    ...     def _logger_name(self) -> str:
    ...         return "my-llm"
    ...
    ...     def ask(self, messages, with_reasoning=False, **params):
    ...         return "Hello"
    ...
    ...     def ask_stream(self, messages, with_reasoning=False, **params):
    ...         raise NotImplementedError
    ...
    ...     def _params(self):
    ...         return ("my-llm",)
    ...
    >>> model = MyLLM()
    >>> model.ask([{"role": "user", "content": "Hi"}])
    'Hello'

"""
import logging
from abc import ABC
from typing import List, Union, Tuple, Optional, Hashable

from .stream import ResponseStream
from ..utils import get_global_logger


class LLMModel(ABC):
    """
    Abstract base class for Large Language Model implementations.

    This class defines the interface that all LLM model implementations must follow.
    It provides two main methods: :meth:`ask` and :meth:`ask_stream` for different
    interaction patterns with language models. Subclasses must implement both methods
    to provide concrete LLM functionality.

    The class supports both synchronous and streaming responses, as well as optional
    reasoning output for models that support chain-of-thought or similar capabilities.

    Subclasses should also implement :meth:`_params` to provide a stable, hashable
    representation of their configuration, enabling reliable equality checks and
    usage as dictionary keys.

    Example::

        >>> class EchoModel(LLMModel):
        ...     @property
        ...     def _logger_name(self) -> str:
        ...         return "echo"
        ...
        ...     def ask(self, messages, with_reasoning=False, **params):
        ...         return messages[-1]["content"]
        ...
        ...     def ask_stream(self, messages, with_reasoning=False, **params):
        ...         raise NotImplementedError
        ...
        ...     def _params(self):
        ...         return ("echo",)
        ...
        >>> model = EchoModel()
        >>> model.ask([{"role": "user", "content": "Hello"}])
        'Hello'
    """

    @property
    def _logger_name(self) -> str:
        """
        Get the logger name for this LLM model instance.

        This property should be implemented by subclasses to provide a unique
        identifier for logging purposes. The name is used to create a child
        logger under the global logger hierarchy.

        :return: The logger name string.
        :rtype: str

        :raises NotImplementedError: This property must be implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def _logger(self) -> logging.Logger:
        """
        Get the logger instance for this LLM model.

        This property returns a logger that is a child of the global logger,
        with a name that includes ``'LLM:'`` prefix followed by the model's logger name.
        This allows for hierarchical logging and easy filtering of LLM-related logs.

        :return: A logger instance specific to this LLM model.
        :rtype: logging.Logger
        """
        return get_global_logger().getChild(f'LLM:{self._logger_name}')

    def ask(self, messages: List[dict], with_reasoning: bool = False, **params) \
            -> Union[str, Tuple[Optional[str], str]]:
        """
        Ask a question to the language model and get a response.

        This method provides a higher-level interface for querying the model.
        It can optionally return reasoning information along with the answer,
        which is useful for models that support explicit reasoning steps.

        :param messages: A list of message dictionaries containing the conversation history.
                        Each dictionary typically contains ``'role'`` and ``'content'`` keys.
                        Example: ``[{"role": "user", "content": "What is 2+2?"}]``
        :type messages: List[dict]
        :param with_reasoning: If True, return both reasoning and answer as a tuple.
                              If False, return only the answer string.
                              Default is False.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model implementation.
                      These may include temperature, max_tokens, top_p, etc.,
                      depending on the specific model implementation.
        :type params: dict

        :return: If with_reasoning is False, returns the answer as a string.
                If with_reasoning is True, returns a tuple of ``(reasoning, answer)``,
                where reasoning can be None if not available or not supported by the model.
        :rtype: Union[str, Tuple[Optional[str], str]]

        :raises NotImplementedError: This method must be implemented by subclasses.

        Example::
            >>> model = SomeLLMModel()
            >>> messages = [{"role": "user", "content": "What is 2+2?"}]
            >>> model.ask(messages)
            '4'
            >>> model.ask(messages, with_reasoning=True)
            ('Adding 2 and 2', '4')
        """
        raise NotImplementedError  # pragma: no cover

    def ask_stream(self, messages: List[dict], with_reasoning: bool = False, **params) -> ResponseStream:
        """
        Ask a question to the language model and get a streaming response.

        This method allows for real-time streaming of the model's response,
        which is useful for long responses or interactive applications where
        immediate feedback is desired. The response is delivered incrementally
        as it is generated by the model.

        :param messages: A list of message dictionaries containing the conversation history.
                        Each dictionary typically contains ``'role'`` and ``'content'`` keys.
                        Example: ``[{"role": "user", "content": "Tell me a story"}]``
        :type messages: List[dict]
        :param with_reasoning: If True, the stream should include reasoning information.
                              If False, only the answer is streamed.
                              Default is False.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model implementation.
                      These may include temperature, max_tokens, top_p, etc.,
                      depending on the specific model implementation.
        :type params: dict

        :return: A ResponseStream object that can be iterated to receive response chunks.
                The stream yields text chunks as they become available from the model.
        :rtype: ResponseStream

        :raises NotImplementedError: This method must be implemented by subclasses.

        Example::
            >>> model = SomeLLMModel()
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> stream = model.ask_stream(messages)
            >>> for chunk in stream:
            ...     print(chunk, end='')
            # Prints the story as it's generated, chunk by chunk
        """
        raise NotImplementedError  # pragma: no cover

    def _params(self) -> Hashable:
        """
        Get the parameters that define this model instance.

        This method should return a stable and hashable representation of the model's
        parameters. It is used for equality comparison and hashing of model instances.
        The returned value must be hashable (e.g., tuple, frozenset) to support
        the :meth:`__hash__` method.

        :return: A hashable representation of the model's parameters.
        :rtype: Hashable

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def _values(self) -> Tuple[type, Hashable]:
        """
        Get the values that uniquely identify this model instance.

        This method returns a tuple containing the model's class and its parameters,
        which together uniquely identify the model instance. This is used for
        equality comparison and hashing.

        :return: A tuple of ``(class, parameters)`` that uniquely identifies this model.
        :rtype: tuple
        """
        return self.__class__, self._params()

    def __eq__(self, other: object) -> bool:
        """
        Check equality between this model and another object.

        Two :class:`LLMModel` instances are considered equal if they are of the same
        class and have the same parameters as returned by :meth:`_values`.

        :param other: The object to compare with.
        :type other: object

        :return: True if the objects are equal, False otherwise.
        :rtype: bool
        """
        if type(other) != type(self):
            return False
        # noinspection PyProtectedMember,PyUnresolvedReferences
        return self._values() == other._values()

    def __hash__(self) -> int:
        """
        Get the hash value of this model instance.

        The hash is computed from the values returned by :meth:`_values`, which includes
        the model's class and parameters. This allows :class:`LLMModel` instances to be used
        as dictionary keys or in sets.

        :return: The hash value of this model instance.
        :rtype: int
        """
        return hash(self._values())
