"""
This module defines the abstract base class for Large Language Model (LLM) implementations.

The module provides a common interface that all LLM implementations should follow,
including methods for chat interactions, question-answering, and streaming responses.
"""

from typing import List, Union, Tuple, Optional

from .stream import ResponseStream


class LLMAbstractModel:
    """
    Abstract base class for Large Language Model implementations.
    
    This class defines the interface that all LLM model implementations must follow.
    It provides three main methods: chat, ask, and ask_stream for different interaction patterns.
    """
    
    def chat(self, messages: List[dict], **params):
        """
        Perform a chat interaction with the language model.
        
        :param messages: A list of message dictionaries containing the conversation history.
                        Each dictionary typically contains 'role' and 'content' keys.
        :type messages: List[dict]
        :param params: Additional parameters to pass to the model implementation.
        :type params: dict
        
        :raises NotImplementedError: This method must be implemented by subclasses.
        
        Example::
            >>> model = SomeLLMModel()
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> model.chat(messages)
            # Returns model-specific response
        """
        raise NotImplementedError  # pragma: no cover

    def ask(self, messages: List[dict], with_reasoning: bool = False, **params) \
            -> Union[str, Tuple[Optional[str], str]]:
        """
        Ask a question to the language model and get a response.
        
        :param messages: A list of message dictionaries containing the conversation history.
                        Each dictionary typically contains 'role' and 'content' keys.
        :type messages: List[dict]
        :param with_reasoning: If True, return both reasoning and answer as a tuple.
                              If False, return only the answer string.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model implementation.
        :type params: dict
        
        :return: If with_reasoning is False, returns the answer as a string.
                If with_reasoning is True, returns a tuple of (reasoning, answer),
                where reasoning can be None if not available.
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
        which is useful for long responses or interactive applications.
        
        :param messages: A list of message dictionaries containing the conversation history.
                        Each dictionary typically contains 'role' and 'content' keys.
        :type messages: List[dict]
        :param with_reasoning: If True, the stream should include reasoning information.
                              If False, only the answer is streamed.
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the model implementation.
        :type params: dict
        
        :return: A ResponseStream object that can be iterated to receive response chunks.
        :rtype: ResponseStream
        
        :raises NotImplementedError: This method must be implemented by subclasses.
        
        Example::
            >>> model = SomeLLMModel()
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> stream = model.ask_stream(messages)
            >>> for chunk in stream:
            ...     print(chunk, end='')
            # Prints the story as it's generated
        """
        raise NotImplementedError  # pragma: no cover
