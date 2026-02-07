"""
LLM Task Management Module.

This module provides an abstract base class for managing Large Language Model (LLM) tasks.
It serves as a high-level interface that combines LLM models with conversation history,
enabling convenient interaction patterns for both standard and streaming responses.

The module contains the following main components:

* :class:`LLMTask` - Abstract base class for LLM task management and execution

The LLMTask class acts as a wrapper around :class:`~hbllmutils.model.base.LLMModel` and
:class:`~hbllmutils.history.history.LLMHistory`, providing simplified methods for:

- Asking questions with automatic history management
- Streaming responses for real-time output
- Maintaining conversation context across multiple interactions
- Supporting optional reasoning output for chain-of-thought models

.. note::
   This is an abstract base class. While it can be instantiated directly,
   subclasses may provide additional specialized functionality.

.. warning::
   The conversation history is maintained throughout the task lifetime.
   For long-running tasks, consider periodically clearing or truncating history
   to manage memory usage.

Example::

    >>> from hbllmutils.model.task import LLMTask
    >>> from hbllmutils.model.load import load_llm_model
    >>> from hbllmutils.history import LLMHistory
    >>> 
    >>> # Initialize task with model and optional history
    >>> model = load_llm_model('gpt-4')
    >>> history = LLMHistory().with_system_prompt('You are a helpful assistant.')
    >>> task = LLMTask(model, history)
    >>> 
    >>> # Standard question-answer interaction
    >>> response = task.ask("What is the capital of France?")
    >>> print(response)
    'The capital of France is Paris.'
    >>> 
    >>> # Streaming response
    >>> stream = task.ask_stream("Tell me a short story")
    >>> for chunk in stream:
    ...     print(chunk, end='', flush=True)
    Once upon a time...
    >>> 
    >>> # With reasoning output
    >>> reasoning, answer = task.ask("Solve 2+2", with_reasoning=True)
    >>> print(f"Reasoning: {reasoning}")
    >>> print(f"Answer: {answer}")

"""

import logging
from abc import ABC
from typing import Union, Tuple, Optional, Any

from .base import LLMModel
from .load import LLMModelTyping, load_llm_model
from .stream import ResponseStream
from ..history import LLMHistory


class LLMTask(ABC):
    """
    Abstract base class for managing LLM task execution and conversation history.

    This class provides a high-level interface for interacting with language models,
    handling both standard and streaming responses while automatically managing
    conversation context. It wraps an LLM model and maintains a conversation history,
    simplifying the process of multi-turn interactions.

    The class supports:
    
    - Standard question-answer interactions with automatic history updates
    - Streaming responses for real-time output processing
    - Optional reasoning output for models supporting chain-of-thought
    - Flexible model initialization from various input types
    - Conversation history persistence and management

    :param model: The LLM model to use. Can be a model name string, an LLMModel instance,
                 or None to load the default model from configuration.
    :type model: LLMModelTyping
    :param history: Optional conversation history. If not provided, a new empty history
                   is created. The history maintains the context of the conversation.
    :type history: Optional[LLMHistory]

    :ivar model: The initialized LLM model instance used for generating responses.
    :vartype model: LLMModel
    :ivar history: The conversation history tracking all messages in the task.
    :vartype history: LLMHistory

    .. note::
       This is an abstract base class. While it can be instantiated directly,
       subclasses may provide additional specialized functionality.

    Example::

        >>> # Initialize with model name
        >>> task = LLMTask('gpt-4')
        >>> 
        >>> # Initialize with existing model and history
        >>> model = load_llm_model('gpt-4')
        >>> history = LLMHistory().with_system_prompt('You are helpful.')
        >>> task = LLMTask(model, history)
        >>> 
        >>> # Basic usage
        >>> response = task.ask("Hello!")
        >>> print(response)
        'Hello! How can I help you today?'

    """

    def __init__(self, model: LLMModelTyping, history: Optional[LLMHistory] = None):
        """
        Initialize the LLMTask with a model and optional conversation history.

        The model parameter is flexible and can accept various input types:
        
        - A string representing the model name (loaded from configuration)
        - An existing LLMModel instance
        - None to load the default model from configuration

        If no history is provided, a new empty LLMHistory instance is created.

        :param model: The LLM model specification. Can be a model name string,
                     an LLMModel instance, or None for the default model.
        :type model: LLMModelTyping
        :param history: Optional conversation history. If None, creates a new
                       empty history to track the conversation.
        :type history: Optional[LLMHistory]

        :raises TypeError: If model is not a valid type (string, LLMModel, or None).
        :raises ValueError: If model name is invalid or not found in configuration.

        Example::

            >>> # With model name
            >>> task = LLMTask('gpt-4')
            >>> 
            >>> # With existing model
            >>> model = load_llm_model('gpt-4')
            >>> task = LLMTask(model)
            >>> 
            >>> # With model and history
            >>> history = LLMHistory().with_system_prompt('Be concise.')
            >>> task = LLMTask('gpt-4', history)

        """
        self.model: LLMModel = load_llm_model(model)
        self.history: LLMHistory = history or LLMHistory()

    @property
    def _logger(self) -> logging.Logger:
        """
        Get the logger instance from the underlying model.

        This property provides access to the model's logger for debugging and
        monitoring purposes. The logger is inherited from the model to maintain
        consistent logging behavior across the task and model layers.

        :return: The logger instance used by the underlying model.
        :rtype: logging.Logger

        Example::

            >>> task = LLMTask('gpt-4')
            >>> logger = task._logger
            >>> logger.info("Task initialized")

        """
        # noinspection PyProtectedMember
        return self.model._logger

    def ask(self, input_content: Optional[str] = None,
            with_reasoning: bool = False, **params) -> Union[str, Tuple[Optional[str], str]]:
        """
        Ask a question to the LLM model and receive a response.

        This method sends the current conversation history (optionally with new user input)
        to the model and retrieves a response. The conversation history is used as context
        but is not automatically updated - use the returned response to update history manually
        if needed.

        The method supports two response formats:
        
        - Standard mode (with_reasoning=False): Returns only the response text
        - Reasoning mode (with_reasoning=True): Returns a tuple of (reasoning, response)

        :param input_content: Optional user input to add to the history before asking.
                             If None, uses the existing history without modification.
                             The original history is not modified; a temporary copy is used.
        :type input_content: Optional[str]
        :param with_reasoning: If True, returns both reasoning and response as a tuple.
                              If False, returns only the response string.
                              Defaults to False.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model's ask method.
                      May include temperature, max_tokens, top_p, etc., depending
                      on the specific model implementation.
        :type params: dict

        :return: If with_reasoning is False, returns the response string.
                If with_reasoning is True, returns a tuple of (reasoning, response)
                where reasoning may be None if not supported by the model.
        :rtype: Union[str, Tuple[Optional[str], str]]

        .. note::
           This method does not modify the task's history. If you want to maintain
           the conversation context, you need to manually update the history with
           the input and response.

        Example::

            >>> task = LLMTask('gpt-4')
            >>> 
            >>> # Simple question
            >>> response = task.ask("What is 2+2?")
            >>> print(response)
            '4'
            >>> 
            >>> # With reasoning
            >>> reasoning, response = task.ask(
            ...     "Explain quantum entanglement",
            ...     with_reasoning=True
            ... )
            >>> print(f"Reasoning: {reasoning}")
            >>> print(f"Response: {response}")
            >>> 
            >>> # With additional parameters
            >>> response = task.ask(
            ...     "Write a poem",
            ...     temperature=0.9,
            ...     max_tokens=100
            ... )

        """
        history = self.history
        if input_content is not None:
            history = history.with_user_message(input_content)
        return self.model.ask(
            messages=history.to_json(),
            with_reasoning=with_reasoning,
            **params
        )

    def ask_stream(self, input_content: Optional[str] = None,
                   with_reasoning: bool = False, **params) -> ResponseStream:
        """
        Ask a question to the LLM model and receive a streaming response.

        This method sends the current conversation history (optionally with new user input)
        to the model and retrieves a streaming response. This is useful for long responses
        or interactive applications where immediate feedback is desired. The response is
        delivered incrementally as it's generated by the model.

        The stream can optionally include reasoning information when with_reasoning=True,
        which will be separated from the regular content using configurable splitters.

        :param input_content: Optional user input to add to the history before asking.
                             If None, uses the existing history without modification.
                             The original history is not modified; a temporary copy is used.
        :type input_content: Optional[str]
        :param with_reasoning: If True, the stream includes reasoning information
                              separated from the regular content. If False, only the
                              response content is streamed. Defaults to False.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model's ask_stream method.
                      May include temperature, max_tokens, top_p, etc., depending
                      on the specific model implementation.
        :type params: dict

        :return: A ResponseStream object that can be iterated to receive response chunks
                in real-time. The stream yields text chunks as they become available.
        :rtype: ResponseStream

        .. note::
           This method does not modify the task's history. The stream must be fully
           consumed before the response content is available via stream properties.

        .. warning::
           The ResponseStream can only be iterated once. After iteration completes,
           attempting to iterate again will raise a RuntimeError.

        Example::

            >>> task = LLMTask('gpt-4')
            >>> 
            >>> # Basic streaming
            >>> stream = task.ask_stream("Tell me a story")
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
            Once upon a time, there was...
            >>> 
            >>> # With reasoning
            >>> stream = task.ask_stream(
            ...     "Solve this problem",
            ...     with_reasoning=True
            ... )
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
            >>> 
            >>> # Access full content after streaming
            >>> print(stream.reasoning_content)
            >>> print(stream.content)
            >>> 
            >>> # With additional parameters
            >>> stream = task.ask_stream(
            ...     "Write a poem",
            ...     temperature=0.9,
            ...     max_tokens=200
            ... )

        """
        history = self.history
        if input_content is not None:
            history = history.with_user_message(input_content)
        return self.model.ask_stream(
            messages=history.to_json(),
            with_reasoning=with_reasoning,
            **params,
        )

    def _params(self) -> Tuple[LLMModel, LLMHistory]:
        """
        Get the internal parameters of this LLMTask instance.

        This method returns the model and history that define the task's state.
        It is used internally for equality comparison and hashing operations.

        :return: A tuple containing the model and history instances.
        :rtype: Tuple[LLMModel, LLMHistory]

        Example::

            >>> task = LLMTask('gpt-4')
            >>> model, history = task._params()
            >>> isinstance(model, LLMModel)
            True
            >>> isinstance(history, LLMHistory)
            True

        """
        return self.model, self.history

    def _values(self) -> Tuple[type, Any]:
        """
        Get the class type and parameters of this LLMTask instance.

        This method returns a tuple containing the class type and the parameters
        that uniquely identify this task instance. It is used internally for
        equality comparison and hashing operations to ensure proper behavior
        in collections and comparisons.

        :return: A tuple containing the class type and the parameters tuple
                from _params().
        :rtype: Tuple[type, Any]

        Example::

            >>> task = LLMTask('gpt-4')
            >>> cls, params = task._values()
            >>> cls is LLMTask
            True
            >>> isinstance(params, tuple)
            True

        """
        return self.__class__, self._params()

    def __eq__(self, other) -> bool:
        """
        Check equality between this LLMTask and another object.

        Two LLMTask instances are considered equal if they have the same class type
        and the same model and history parameters. This allows for proper comparison
        of task instances in collections and conditional logic.

        The comparison is based on the values returned by _values(), which includes
        both the class type and the internal parameters (model and history).

        :param other: The object to compare with.
        :type other: object

        :return: True if the objects are equal (same class and same parameters),
                False otherwise.
        :rtype: bool

        Example::

            >>> model = load_llm_model('gpt-4')
            >>> history = LLMHistory()
            >>> task1 = LLMTask(model, history)
            >>> task2 = LLMTask(model, history)
            >>> task1 == task2
            True
            >>> 
            >>> task3 = LLMTask(model, history.with_user_message("Hello"))
            >>> task1 == task3
            False

        """
        if type(other) != type(self):
            return False
        # noinspection PyProtectedMember,PyUnresolvedReferences
        return self._values() == other._values()

    def __hash__(self) -> int:
        """
        Get the hash value of this LLMTask instance.

        The hash is computed based on the class type and the model and history
        parameters. This allows LLMTask instances to be used as dictionary keys
        or in sets, provided the underlying model and history are also hashable.

        The hash is derived from the values returned by _values(), ensuring
        consistency with the equality comparison implemented in __eq__.

        :return: The hash value of this task instance.
        :rtype: int

        :raises TypeError: If the underlying model or history is not hashable.

        Example::

            >>> model = load_llm_model('gpt-4')
            >>> history = LLMHistory()
            >>> task = LLMTask(model, history)
            >>> hash_value = hash(task)
            >>> isinstance(hash_value, int)
            True
            >>> 
            >>> # Can be used in sets and as dict keys
            >>> task_set = {task}
            >>> task_dict = {task: "some_value"}

        """
        return hash(self._values())
