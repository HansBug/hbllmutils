"""
Public entry point for the ``hbllmutils.history`` package.

This module re-exports the most frequently used components for working with
LLM (Large Language Model) message histories and image-to-blob URL conversion.
It offers a single import location for constructing role-based messages,
managing immutable conversation histories, and embedding images as data URLs.

The module contains the following public components:

* :class:`LLMHistory` - Immutable sequence for LLM conversation history
* :func:`create_llm_message` - Build a role-based message with text or images
* :data:`LLMContentTyping` - Type alias for accepted content payloads
* :data:`LLMRoleTyping` - Type alias for allowed message roles
* :func:`to_blob_url` - Convert images into base64 ``data:`` URLs

Example::

    >>> from hbllmutils.history import LLMHistory, create_llm_message, to_blob_url
    >>> history = LLMHistory()
    >>> history = history.with_user_message("Hello!")
    >>> history[0]["role"]
    'user'
    >>> # Create a standalone message
    >>> message = create_llm_message("Hi there!", role="assistant")
    >>> message["role"]
    'assistant'

.. note::
   The :class:`LLMHistory` container is immutable. Methods that append or update
   messages return new instances instead of modifying the existing history.

"""

from .history import LLMContentTyping, LLMRoleTyping, create_llm_message, LLMHistory
from .image import to_blob_url
