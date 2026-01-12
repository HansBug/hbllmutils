"""
This module provides utilities for handling LLM (Large Language Model) message histories and image processing.

It exports key components for:

- Creating and managing LLM message histories with role-based messages
- Handling various content types (text, images, or mixed content)
- Converting images to blob URLs for use in LLM messages

The module serves as the main entry point for the history package, providing convenient access
to message creation, history management, and image processing utilities.

Exported Components:
    - :class:`LLMHistory`: Class for managing LLM conversation history
    - :func:`create_llm_message`: Function for creating individual LLM messages
    - :type:`LLMContentTyping`: Type hint for LLM message content
    - :type:`LLMRoleTyping`: Type hint for LLM message roles
    - :func:`to_blob_url`: Function for converting images to blob URLs

Example::
    >>> from hbllmutils.history import LLMHistory, create_llm_message, to_blob_url
    >>> # Create a message history
    >>> history = LLMHistory()
    >>> # Add a user message
    >>> history.add_message(create_llm_message('user', 'Hello!'))
    >>> # Convert an image to blob URL
    >>> blob_url = to_blob_url(image)
"""

from .history import LLMContentTyping, LLMRoleTyping, create_llm_message, LLMHistory
from .image import to_blob_url
