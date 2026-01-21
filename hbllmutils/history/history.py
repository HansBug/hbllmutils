"""
This module provides utilities for creating and managing Large Language Model (LLM) message histories.

It includes functionality for:
- Creating LLM messages with various content types (text, images, or mixed)
- Managing conversation history with role-based messages
- Converting between different message formats

The module supports multiple content types including strings, PIL Images, and lists of mixed content.
"""
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Union, List, Optional

import yaml
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

    .. note::
        LLMHistory is an immutable object. Any operation will cause a new object creation.

    :param history: Optional initial history as a list of message dictionaries.
    :type history: Optional[List[dict]]

    Example::
        >>> history = LLMHistory()
        >>> history = history.with_user_message("Hello!")
        >>> history = history.with_assistant_message("Hi there!")
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
            return copy.deepcopy(result)

    def __len__(self) -> int:
        """
        Get the number of messages in the history.

        :return: The number of messages.
        :rtype: int
        """
        return len(self._history)

    def with_message(self, role: LLMRoleTyping, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append a message with a specific role to the history.

        This method creates a new LLMHistory instance with the appended message,
        leaving the original instance unchanged.

        :param role: The role of the message sender.
        :type role: LLMRoleTyping
        :param message: The message content.
        :type message: LLMContentTyping

        :return: A new LLMHistory instance with the appended message.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> new_history = history.with_message('user', 'Hello!')
            >>> len(history)
            0
            >>> len(new_history)
            1
        """
        return LLMHistory(history=[*self._history, create_llm_message(message=message, role=role)])

    def with_user_message(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append a user message to the history.

        This is a convenience method equivalent to calling with_message with role='user'.
        Creates a new LLMHistory instance with the appended message.

        :param message: The message content.
        :type message: LLMContentTyping

        :return: A new LLMHistory instance with the appended user message.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> new_history = history.with_user_message('Hello!')
            >>> new_history[0]['role']
            'user'
        """
        return self.with_message(role='user', message=message)

    def with_assistant_message(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append an assistant message to the history.

        This is a convenience method equivalent to calling with_message with role='assistant'.
        Creates a new LLMHistory instance with the appended message.

        :param message: The message content.
        :type message: LLMContentTyping

        :return: A new LLMHistory instance with the appended assistant message.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> new_history = history.with_assistant_message('How can I help you?')
            >>> new_history[0]['role']
            'assistant'
        """
        return self.with_message(role='assistant', message=message)

    def with_system_prompt(self, message: LLMContentTyping) -> 'LLMHistory':
        """
        Set or update the system prompt.

        If a system message already exists at the beginning of the history,
        it will be replaced. Otherwise, the new system message will be inserted
        at the start of the history. This method creates a new LLMHistory instance.

        :param message: The system prompt content.
        :type message: LLMContentTyping

        :return: A new LLMHistory instance with the system prompt set or updated.
        :rtype: LLMHistory

        Example::
            >>> history = LLMHistory()
            >>> new_history = history.with_system_prompt('You are a helpful assistant.')
            >>> new_history[0]['role']
            'system'
            >>> new_history[0]['content']
            'You are a helpful assistant.'
        """
        message = create_llm_message(message=message, role='system')
        if self._history and self._history[0]['role'] == 'system':
            return LLMHistory(history=[message, *self._history[1:]])
        else:
            return LLMHistory(history=[message, *self._history])

    def to_json(self) -> List[dict]:
        """
        Convert the history to a JSON-serializable list of dictionaries.

        :return: A list of message dictionaries.
        :rtype: List[dict]

        Example::
            >>> history = LLMHistory()
            >>> history = history.with_user_message('Hello!')
            >>> history.to_json()
            [{'role': 'user', 'content': 'Hello!'}]
        """
        return copy.deepcopy(self._history)

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
            >>> history = history.with_user_message('Hello!')
            >>> cloned = history.clone()
            >>> cloned = cloned.with_user_message('Another message')
            >>> len(history)
            1
            >>> len(cloned)
            2
        """
        return LLMHistory(history=copy.deepcopy(self._history))

    def __hash__(self) -> int:
        """
        Generate a hash value for the LLMHistory instance.

        The hash is computed based on the message history content, allowing
        LLMHistory instances to be used as dictionary keys or in sets.

        :return: Hash value of the history.
        :rtype: int

        Example::
            >>> history1 = LLMHistory().with_user_message('Hello!')
            >>> history2 = LLMHistory().with_user_message('Hello!')
            >>> hash(history1) == hash(history2)
            True
            >>> history_set = {history1, history2}
            >>> len(history_set)
            1
        """

        def _make_hashable(obj):
            """
            Recursively convert nested data structures to hashable types.

            :param obj: Object to convert (dict, list, or primitive type)
            :return: Hashable representation of the object
            """
            if isinstance(obj, dict):
                # Convert dict to sorted tuple of key-value pairs
                return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, (list, tuple)):
                # Convert list/tuple to tuple of hashable elements
                return tuple(_make_hashable(item) for item in obj)
            elif isinstance(obj, (str, int, float, bool, type(None))):
                # Primitive types are already hashable
                return obj
            else:
                # For other types (like custom objects), convert to string
                return str(obj)

        # Convert the entire history to a hashable structure
        hashable_history = _make_hashable(self._history)
        return hash(hashable_history)

    def __eq__(self, other) -> bool:
        """
        Check equality between LLMHistory instances.

        Two LLMHistory instances are considered equal if they have the same
        message history content.

        :param other: Another LLMHistory instance to compare with.
        :type other: LLMHistory
        :return: True if histories are equal, False otherwise.
        :rtype: bool

        Example::
            >>> history1 = LLMHistory().with_user_message('Hello!')
            >>> history2 = LLMHistory().with_user_message('Hello!')
            >>> history1 == history2
            True
        """
        if not isinstance(other, LLMHistory):
            return False
        return self._history == other._history

    def dump_json(self, file: str, **params) -> None:
        """
        Export the history to a JSON file.

        :param file: The file path to save the JSON data.
        :type file: str
        :param params: Additional parameters to pass to json.dump (e.g., indent, ensure_ascii).

        :raises IOError: If the file cannot be written.

        Example::
            >>> history = LLMHistory()
            >>> history = history.with_user_message('Hello!')
            >>> history.dump_json('conversation.json', indent=2)
        """
        # Set default parameters for better formatting
        default_params = {'indent': 2, 'ensure_ascii': False, 'sort_keys': True}
        default_params.update(params)

        file_path = Path(file)
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, **default_params)

    @classmethod
    def load_json(cls, file: str) -> 'LLMHistory':
        """
        Load history from a JSON file.

        :param file: The file path to load the JSON data from.
        :type file: str

        :return: A new LLMHistory instance loaded from the file.
        :rtype: LLMHistory

        :raises FileNotFoundError: If the file does not exist.
        :raises json.JSONDecodeError: If the file contains invalid JSON.
        :raises ValueError: If the JSON structure is invalid for LLMHistory.

        Example::
            >>> history = LLMHistory.load_json('conversation.json')
            >>> len(history)
            1
        """
        file_path = Path(file)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate the loaded data
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of messages")

        # Validate each message structure
        for i, message in enumerate(data):
            if not isinstance(message, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if 'role' not in message or 'content' not in message:
                raise ValueError(f"Message at index {i} must have 'role' and 'content' fields")

        return cls(history=data)

    def dump_yaml(self, file: str, **params) -> None:
        """
        Export the history to a YAML file.

        :param file: The file path to save the YAML data.
        :type file: str
        :param params: Additional parameters to pass to yaml.dump (e.g., default_flow_style, indent).

        :raises IOError: If the file cannot be written.
        :raises ImportError: If PyYAML is not installed.

        Example::
            >>> history = LLMHistory()
            >>> history = history.with_user_message('Hello!')
            >>> history.dump_yaml('conversation.yaml', default_flow_style=False)
        """
        # Set default parameters for better formatting
        default_params = {'default_flow_style': False, 'allow_unicode': True, 'indent': 2, 'sort_keys': True}
        default_params.update(params)

        file_path = Path(file)
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_json(), f, **default_params)

    @classmethod
    def load_yaml(cls, file: str) -> 'LLMHistory':
        """
        Load history from a YAML file.

        :param file: The file path to load the YAML data from.
        :type file: str

        :return: A new LLMHistory instance loaded from the file.
        :rtype: LLMHistory

        :raises FileNotFoundError: If the file does not exist.
        :raises yaml.YAMLError: If the file contains invalid YAML.
        :raises ValueError: If the YAML structure is invalid for LLMHistory.
        :raises ImportError: If PyYAML is not installed.

        Example::
            >>> history = LLMHistory.load_yaml('conversation.yaml')
            >>> len(history)
            1
        """
        file_path = Path(file)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Validate the loaded data
        if not isinstance(data, list):
            raise ValueError("YAML file must contain a list of messages")

        # Validate each message structure
        for i, message in enumerate(data):
            if not isinstance(message, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if 'role' not in message or 'content' not in message:
                raise ValueError(f"Message at index {i} must have 'role' and 'content' fields")

        return cls(history=data)
