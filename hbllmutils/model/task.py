"""
This module provides an abstract base class for LLM (Large Language Model) tasks.

It defines the LLMTask class which acts as a wrapper around LLMModel and LLMHistory,
providing convenient methods for asking questions and streaming responses while
maintaining conversation history.
"""
import logging
from abc import ABC
from typing import Union, Tuple, Optional

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
        self.model = model
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

    def ask(self, with_reasoning: bool = False, **params) -> Union[str, Tuple[Optional[str], str]]:
        """
        Ask a question to the LLM model and get a response.

        This method sends the current conversation history to the model and retrieves
        a response. The response format depends on the with_reasoning parameter.

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
            >>> response = task.ask()
            >>> print(response)
            'This is the model response'

            >>> reasoning, response = task.ask(with_reasoning=True)
            >>> print(f"Reasoning: {reasoning}, Response: {response}")
            Reasoning: None, Response: This is the model response
        """
        return self.model.ask(
            messages=self.history.to_json(),
            with_reasoning=with_reasoning,
            **params
        )

    def ask_stream(self, with_reasoning: bool = False, **params) -> ResponseStream:
        """
        Ask a question to the LLM model and get a streaming response.

        This method sends the current conversation history to the model and retrieves
        a streaming response, allowing for real-time processing of the model's output.

        :param with_reasoning: If True, the stream includes reasoning information.
                              If False, only the response is streamed.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model's ask_stream method.
        :type params: dict

        :return: A ResponseStream object that can be iterated to receive response chunks.
        :rtype: ResponseStream

        Example::
            >>> task = LLMTask(model)
            >>> stream = task.ask_stream()
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
            This is the streaming response...
        """
        return self.model.ask_stream(
            messages=self.history.to_json(),
            with_reasoning=with_reasoning,
            **params,
        )
