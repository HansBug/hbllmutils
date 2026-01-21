"""
This module provides an abstract base class for LLM (Large Language Model) tasks.

It defines the LLMTask class which acts as a wrapper around LLMModel and LLMHistory,
providing convenient methods for asking questions and streaming responses while
maintaining conversation history.
"""
import logging
from abc import ABC
from typing import Union, Tuple, Optional, Any

from .base import LLMModel
from .stream import ResponseStream
from ..history import LLMHistory


class LLMTask(ABC):
    """
    Abstract base class for LLM tasks that manages model interactions and conversation history.

    This class provides a high-level interface for interacting with language models,
    handling both standard and streaming responses while maintaining conversation context.

    :param model: The LLM model instance to use for generating responses.
    :type model: LLMModel
    :param history: Optional conversation history. If not provided, a new empty history is created.
    :type history: Optional[LLMHistory]

    :ivar model: The LLM model instance.
    :vartype model: LLMModel
    :ivar history: The conversation history.
    :vartype history: LLMHistory
    """

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None):
        """
        Initialize the LLMTask with a model and optional history.

        :param model: The LLM model instance to use for generating responses.
        :type model: LLMModel
        :param history: Optional conversation history. If not provided, a new empty history is created.
        :type history: Optional[LLMHistory]
        """
        self.model: LLMModel = model
        self.history: LLMHistory = history or LLMHistory()

    @property
    def _logger(self) -> logging.Logger:
        """
        Get the logger instance from the underlying model.

        :return: The logger instance used by the model.
        :rtype: logging.Logger
        """
        # noinspection PyProtectedMember
        return self.model._logger

    def ask(self, input_content: Optional[str] = None,
            with_reasoning: bool = False, **params) -> Union[str, Tuple[Optional[str], str]]:
        """
        Ask a question to the LLM model and get a response.

        This method sends the current conversation history to the model and retrieves
        a response. The response format depends on the with_reasoning parameter.

        :param input_content: Optional user input content to add to the history before asking.
                             If None, uses the existing history without modification.
        :type input_content: Optional[str]
        :param with_reasoning: If True, returns both reasoning and response as a tuple.
                              If False, returns only the response string.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model's ask method.
        :type params: dict

        :return: If with_reasoning is False, returns the response string.
                If with_reasoning is True, returns a tuple of (reasoning, response).
        :rtype: Union[str, Tuple[Optional[str], str]]

        Example::
            >>> task = LLMTask(model)
            >>> response = task.ask("What is the weather today?")
            >>> print(response)
            'The weather is sunny today.'

            >>> reasoning, response = task.ask("Explain quantum physics", with_reasoning=True)
            >>> print(f"Reasoning: {reasoning}, Response: {response}")
            Reasoning: Let me break this down step by step..., Response: Quantum physics is...
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
        Ask a question to the LLM model and get a streaming response.

        This method sends the current conversation history to the model and retrieves
        a streaming response, allowing for real-time processing of the model's output.

        :param input_content: Optional user input content to add to the history before asking.
                             If None, uses the existing history without modification.
        :type input_content: Optional[str]
        :param with_reasoning: If True, the stream includes reasoning information.
                              If False, only the response is streamed.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model's ask_stream method.
        :type params: dict

        :return: A ResponseStream object that can be iterated to receive response chunks.
        :rtype: ResponseStream

        Example::
            >>> task = LLMTask(model)
            >>> stream = task.ask_stream("Tell me a story")
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
            Once upon a time, there was...
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
        Get the parameters of this LLMTask instance.

        :return: A tuple containing the model and history.
        :rtype: Tuple[LLMModel, LLMHistory]
        """
        return self.model, self.history

    def _values(self) -> Tuple[type, Any]:
        """
        Get the class type and parameters of this LLMTask instance.

        This method is used for equality comparison and hashing.

        :return: A tuple containing the class type and the parameters tuple.
        :rtype: Tuple[type, Any]
        """
        return self.__class__, self._params()

    def __eq__(self, other) -> bool:
        """
        Check equality between this LLMTask and another object.

        Two LLMTask instances are considered equal if they have the same class type
        and the same model and history parameters.

        :param other: The object to compare with.
        :type other: object

        :return: True if the objects are equal, False otherwise.
        :rtype: bool

        Example::
            >>> task1 = LLMTask(model, history)
            >>> task2 = LLMTask(model, history)
            >>> task1 == task2
            True
        """
        if type(other) != type(self):
            return False
        # noinspection PyProtectedMember,PyUnresolvedReferences
        return self._values() == other._values()

    def __hash__(self) -> int:
        """
        Get the hash value of this LLMTask instance.

        The hash is computed based on the class type and the model and history parameters.

        :return: The hash value.
        :rtype: int

        Example::
            >>> task = LLMTask(model, history)
            >>> hash(task)
            1234567890
        """
        return hash(self._values())
