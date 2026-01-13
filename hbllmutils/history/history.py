"""
This module provides utilities for creating and managing Large Language Model (LLM) message histories.

It includes functionality for:
- Creating LLM messages with various content types (text, images, or mixed)
- Managing conversation history with role-based messages
- Converting between different message formats

The module supports multiple content types including strings, PIL Images, and lists of mixed content.
"""
import copy
from collections.abc import Sequence
from typing import Union, List, Optional

from PIL import Image

from .image import to_blob_url

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

LLMContentTyping = Union[str, Image.Image, List[Union[str, Image.Image]]]
LLMRoleTyping = Literal["system", "user", "assistant", "tool", "function"]


def create_llm_message(message: LLMContentTyping, role: LLMRoleTyping = 'user') -> dict:
    """
    Create a structured LLM message from various content types.

    This function converts different types of message content (text, images, or mixed)
    into a standardized dictionary format suitable for LLM APIs.

    :param message: The message content, which can be a string, PIL Image, or list of strings/images.
    :type message: LLMContentTyping
    :param role: The role of the message sender (default is 'user').
    :type role: LLMRoleTyping

    :return: A dictionary containing the role and formatted content.
    :rtype: dict

    :raises TypeError: If the message type is unsupported or if a list item has an unsupported type.

    Example::
        >>> create_llm_message("Hello, world!")
        {'role': 'user', 'content': 'Hello, world!'}

        >>> create_llm_message(["Text message", image_obj], role='assistant')
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Text message'}, {'type': 'image_url', 'image_url': '...'}]}
    """
    if isinstance(message, str):
        content = message
    elif isinstance(message, Image.Image):
        content = [{"type": "image_url", "image_url": to_blob_url(message)}]
    elif isinstance(message, (list, tuple)):
        content = []
        for i, item in enumerate(message):
            if isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                content.append({"type": "image_url", "image_url": to_blob_url(item)})
            else:
                raise TypeError(f'Unsupported type for message content item at #{i!r} - {item!r}')
    else:
        raise TypeError(f'Unsupported content type - {message!r}')

    return {
        "role": role,
        "content": content
    }


class LLMHistory(Sequence):
    """
    A sequence-like container for managing LLM conversation history.

    This class provides methods to build and maintain a conversation history
    with different roles (user, assistant, system, etc.). It implements the
    Sequence protocol, allowing indexing and iteration.

    :param history: Optional initial history as a list of message dictionaries.
    :type history: Optional[List[dict]]

    Example::
        >>> history = LLMHistory()
        >>> history.append_user("Hello!")
        >>> history.append_assistant("Hi there!")
        >>> len(history)
        2
        >>> history[0]
        {'role': 'user', 'content': 'Hello!'}
    """

    def __init__(self, history: Optional[List[dict]] = None):
        """
        Initialize the LLMHistory instance.

        :param history: Optional initial history as a list of message dictionaries.
        :type history: Optional[List[dict]]
        """
        self._history = list(history or [])

    def __getitem__(self, index):
        """
        Get an item or slice from the history.

        :param index: The index or slice to retrieve.
        :type index: int or slice

        :return: A single message dict or a new LLMHistory instance for slices.
        :rtype: dict or LLMHistory
        """
        result = self._history[index]
        if isinstance(result, list):
            return LLMHistory(result)
        else:
            return result

    def __len__(self) -> int:
        """
        Get the number of messages in the history.

        :return: The number of messages.
        :rtype: int
        """
        return len(self._history)

    def append(self, role: LLMRoleTyping, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append a message with a specific role to the history.

        :param role: The role of the message sender.
        :type role: LLMRoleTyping
        :param message: The message content.
        :type message: LLMContentTyping

        :return: The LLMHistory instance for method chaining.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> history.append('user', 'Hello!')
        """
        self._history.append(create_llm_message(message=message, role=role))
        return self

    def append_user(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append a user message to the history.

        This is a convenience method equivalent to calling append with role='user'.

        :param message: The message content.
        :type message: LLMContentTyping

        :return: The LLMHistory instance for method chaining.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> history.append_user('Hello!')
        """
        return self.append(role='user', message=message)

    def append_assistant(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append an assistant message to the history.

        This is a convenience method equivalent to calling append with role='assistant'.

        :param message: The message content.
        :type message: LLMContentTyping

        :return: The LLMHistory instance for method chaining.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> history.append_assistant('How can I help you?')
        """
        return self.append(role='assistant', message=message)

    def set_system_prompt(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Set or update the system prompt.

        If a system message already exists at the beginning of the history,
        it will be replaced. Otherwise, the new system message will be inserted
        at the start of the history.

        :param message: The system prompt content.
        :type message: LLMContentTyping

        :return: The LLMHistory instance for method chaining.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> history.set_system_prompt('You are a helpful assistant.')
            >>> history[0]['role']
            'system'
        """
        message = create_llm_message(message=message, role='system')
        if self._history and self._history[0]['role'] == 'system':
            self._history[0] = message
        else:
            self._history.insert(0, message)
        return self

    def to_json(self) -> List[dict]:
        """
        Convert the history to a JSON-serializable list of dictionaries.

        :return: A list of message dictionaries.
        :rtype: List[dict]

        Example::
            >>> history = LLMHistory()
            >>> history.append_user('Hello!')
            >>> history.to_json()
            [{'role': 'user', 'content': 'Hello!'}]
        """
        return self._history

    def clone(self) -> 'LLMHistory':
        """
        Create a deep copy of the current LLMHistory instance.

        This method creates a new LLMHistory object with a deep copy of the
        internal message history, ensuring that modifications to the clone
        do not affect the original instance.

        :return: A new LLMHistory instance with copied message history.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> history.append_user('Hello!')
            >>> cloned = history.clone()
            >>> cloned.append_user('Another message')
            >>> len(history)
            1
            >>> len(cloned)
            2
        """
        return LLMHistory(history=copy.deepcopy(self._history))
