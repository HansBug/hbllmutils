"""
Large Language Model (LLM) client package initialization.

This module defines the public API for the :mod:`hbllmutils.model` package by
exposing the core LLM abstractions, remote model client, streaming handlers,
task helper, and configuration loaders. It provides a single import surface for
applications that need to interact with OpenAI-compatible endpoints, simulate
responses for testing, or manage conversation tasks.

The module contains the following main components:

* :class:`LLMModel` - Abstract interface for LLM implementations
* :class:`RemoteLLMModel` - OpenAI-compatible remote client
* :class:`FakeLLMModel` - Fake model for testing and development
* :class:`ResponseStream` - Streaming response handler
* :class:`OpenAIResponseStream` - OpenAI-specific streaming handler
* :class:`LLMTask` - Task wrapper with conversation history management
* :func:`load_llm_model_from_config` - Load model from configuration
* :func:`load_llm_model` - Load model by name, instance, or default
* :data:`LLMModelTyping` - Type alias for model inputs

Example::

    >>> from hbllmutils.model import RemoteLLMModel, LLMTask
    >>> model = RemoteLLMModel(
    ...     base_url="https://api.openai.com/v1",
    ...     api_token="your-key",
    ...     model_name="gpt-3.5-turbo"
    ... )
    >>> task = LLMTask(model)
    >>> response = task.ask("Hello!")
    >>> print(response)

"""

from .base import LLMModel
from .fake import FakeResponseStream, FakeLLMModel
from .load import load_llm_model_from_config, load_llm_model, LLMModelTyping
from .remote import RemoteLLMModel
from .stream import ResponseStream, OpenAIResponseStream
from .task import LLMTask
