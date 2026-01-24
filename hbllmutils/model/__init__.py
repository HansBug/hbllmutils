"""
This module provides remote LLM (Large Language Model) client implementation and related utilities.

The module offers a unified interface for interacting with OpenAI-compatible API endpoints,
supporting both synchronous and asynchronous operations, streaming responses, and
customizable parameters. It includes base classes, streaming handlers, and remote model
implementations.

Key Components:
    - Base LLM interface definitions
    - Streaming response handlers with reasoning support
    - Remote model client for OpenAI-compatible APIs
    - Asynchronous and synchronous operation support

Example::
    >>> from hbllmutils.model import RemoteLLMModel
    >>> model = RemoteLLMModel(base_url="https://api.openai.com/v1", api_key="your-key")
    >>> response = model.create_message([{"role": "user", "content": "Hello!"}])
    >>> print(response)
"""

from .base import LLMModel
from .fake import FakeResponseStream, FakeLLMModel
from .load import load_llm_model_from_config
from .remote import RemoteLLMModel
from .stream import ResponseStream, OpenAIResponseStream
from .task import LLMTask
