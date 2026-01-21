"""
This module provides utilities for creating and managing Large Language Model (LLM) message histories.

It includes functionality for:
- Creating LLM messages with various content types (text, images, or mixed)
- Managing conversation history with role-based messages
- Converting between different message formats
- Serializing and deserializing conversation histories to/from JSON and YAML formats

The module supports multiple content types including strings, PIL Images, and lists of mixed content.
It provides an immutable sequence-like container for managing conversation history with support for
different roles (user, assistant, system, tool, function).
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
    into a standardized dictionary format suitable for LLM APIs. The function handles:
    
    - Plain text strings: Returned as-is in the content field
    - PIL Images: Converted to blob URLs and wrapped in image_url format
    - Lists of mixed content: Each item is converted to appropriate format (text or image_url)

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
    Sequence protocol, allowing indexing and iteration. The class is designed
    to be immutable - all modification operations return new instances rather
    than modifying the existing one.

    The class supports:
    
    - Adding messages with specific roles (user, assistant, system, etc.)
    - Setting and updating system prompts
    - Serialization to JSON and YAML formats
    - Deserialization from JSON and YAML files
    - Sequence operations (indexing, slicing, iteration, length)
    - Hashing and equality comparison

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
                       Each dictionary should contain 'role' and 'content' keys.
        :type history: Optional[List[dict]]
        """
        self._history = list(history or [])

    def __getitem__(self, index):
        """
        Get an item or slice from the history.

        When indexing with a single integer, returns a deep copy of the message
        dictionary at that position. When slicing, returns a new LLMHistory instance
        containing the sliced messages.

        :param index: The index or slice to retrieve.
        :type index: int or slice

        :return: A single message dict (deep copy) or a new LLMHistory instance for slices.
        :rtype: dict or LLMHistory

        Example::
            >>> history = LLMHistory().with_user_message("Hello!")
            >>> history[0]
            {'role': 'user', 'content': 'Hello!'}
            >>> history[0:1]
            <LLMHistory object with 1 message>
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

        Example::
            >>> history = LLMHistory()
            >>> len(history)
            0
            >>> history = history.with_user_message("Hello!")
            >>> len(history)
            1
        """
        return len(self._history)

    def with_message(self, role: LLMRoleTyping, message: LLMContentTyping) -> 'LLMHistory':
        """
        Append a message with a specific role to the history.

        This method creates a new LLMHistory instance with the appended message,
        leaving the original instance unchanged. The message content is processed
        through create_llm_message() to ensure proper formatting.

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

        System prompts are typically used to set the behavior or context for the
        LLM at the start of a conversation.

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

        Returns a deep copy of the internal message history, ensuring that
        modifications to the returned list do not affect the original history.

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
        LLMHistory instances to be used as dictionary keys or in sets. The
        hash is computed by recursively converting nested data structures
        (dicts, lists) to hashable types (tuples).

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

            Converts dictionaries to sorted tuples of key-value pairs,
            lists/tuples to tuples of hashable elements, and handles
            primitive types directly. For other types, converts to string.

            :param obj: Object to convert (dict, list, or primitive type)
            :type obj: Any

            :return: Hashable representation of the object
            :rtype: tuple or str or int or float or bool or None
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
        message history content. Returns False if the other object is not
        an LLMHistory instance.

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

        Saves the conversation history to a JSON file with configurable formatting.
        The parent directory will be created if it doesn't exist. Default parameters
        provide pretty-printed output with 2-space indentation, UTF-8 encoding,
        and sorted keys.

        :param file: The file path to save the JSON data.
        :type file: str
        :param params: Additional parameters to pass to json.dump (e.g., indent, ensure_ascii).
                      Default parameters: indent=2, ensure_ascii=False, sort_keys=True

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

        Reads and validates a JSON file containing conversation history.
        The file must contain a list of message dictionaries, where each
        dictionary has 'role' and 'content' fields.

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

        Saves the conversation history to a YAML file with configurable formatting.
        The parent directory will be created if it doesn't exist. Default parameters
        provide human-readable output with block style, UTF-8 encoding, 2-space
        indentation, and sorted keys.

        :param file: The file path to save the YAML data.
        :type file: str
        :param params: Additional parameters to pass to yaml.dump (e.g., default_flow_style, indent).
                      Default parameters: default_flow_style=False, allow_unicode=True,
                      indent=2, sort_keys=True

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

        Reads and validates a YAML file containing conversation history.
        The file must contain a list of message dictionaries, where each
        dictionary has 'role' and 'content' fields.

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
