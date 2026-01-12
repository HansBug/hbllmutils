"""
This module provides a remote LLM (Large Language Model) client implementation.

It offers a unified interface for interacting with OpenAI-compatible API endpoints,
supporting both synchronous and asynchronous operations, streaming responses, and
customizable parameters.

Classes:
    LLMRemoteModel: Main class for managing remote LLM API interactions.
"""

from typing import Dict, Optional, Union, Any, List, Tuple
from urllib.parse import urlparse

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from .base import LLMAbstractModel
from .stream import OpenAIResponseStream, ResponseStream


class LLMRemoteModel(LLMAbstractModel):
    """
    A client for interacting with remote Large Language Model APIs.

    This class provides a unified interface for communicating with OpenAI-compatible
    API endpoints. It supports both synchronous and asynchronous operations, streaming
    responses, and allows customization of request parameters.

    :ivar base_url: API base URL (e.g., "https://api.openai.com/v1")
    :vartype base_url: str
    :ivar api_token: API access token for authentication
    :vartype api_token: str
    :ivar model_name: Name of the model to use (e.g., "gpt-3.5-turbo", "claude-3-opus")
    :vartype model_name: str
    :ivar organization_id: Organization ID (required by some APIs)
    :vartype organization_id: Optional[str]
    :ivar timeout: Request timeout in seconds
    :vartype timeout: int
    :ivar max_retries: Maximum number of retry attempts
    :vartype max_retries: int
    :ivar headers: Custom request headers
    :vartype headers: Dict[str, str]
    :ivar default_params: Default parameters for API requests
    :vartype default_params: Dict[str, Any]
    """

    def __init__(self, base_url: str, api_token: str, model_name: str,
                 organization_id: Optional[str] = None, timeout: int = 30, max_retries: int = 3,
                 headers: Optional[Dict[str, str]] = None, **default_params):
        """
        Initialize the LLMRemoteModel instance.

        :param base_url: API base URL (e.g., "https://api.openai.com/v1")
        :type base_url: str
        :param api_token: API access token for authentication
        :type api_token: str
        :param model_name: Name of the model to use (e.g., "gpt-3.5-turbo")
        :type model_name: str
        :param organization_id: Organization ID (optional, required by some APIs)
        :type organization_id: Optional[str]
        :param timeout: Request timeout in seconds (default: 30)
        :type timeout: int
        :param max_retries: Maximum number of retry attempts (default: 3)
        :type max_retries: int
        :param headers: Custom request headers (optional)
        :type headers: Optional[Dict[str, str]]
        :param default_params: Default parameters for API requests (optional)

        :raises ValueError: If base_url format is invalid
        :raises ValueError: If api_token is empty
        :raises ValueError: If model_name is empty
        :raises ValueError: If timeout is not positive
        :raises ValueError: If max_retries is negative

        Example::
            >>> model = LLMRemoteModel(
            ...     base_url="https://api.openai.com/v1",
            ...     api_token="sk-xxx",
            ...     model_name="gpt-3.5-turbo"
            ... )
        """
        self.base_url = base_url
        # Validate URL format
        try:
            result = urlparse(self.base_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError(f"Invalid base_url format - {self.base_url!r}")
        except Exception as e:
            raise ValueError(f"Invalid base_url - {self.base_url!r}: {e}")

        self.api_token = api_token
        if not self.api_token.strip():
            raise ValueError(f"api_token cannot be empty, but {self.api_token!r} found")

        self.model_name = model_name
        if not self.model_name.strip():
            raise ValueError(f"model_name cannot be empty, but {self.model_name!r} found")

        self.organization_id = organization_id
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, but {self.timeout!r} found")

        self.max_retries = max_retries
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, but {self.max_retries!r} found")

        self.headers = dict(headers or {})
        self.default_params: Dict[str, Any] = default_params

        self._client_non_async = None

    def _create_openai_client(self, use_async: bool = False) -> Union[OpenAI, AsyncOpenAI]:
        """
        Create an OpenAI client instance (synchronous or asynchronous).

        :param use_async: Whether to create an asynchronous client (default: False)
        :type use_async: bool

        :return: OpenAI client instance (synchronous or asynchronous)
        :rtype: Union[OpenAI, AsyncOpenAI]

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> sync_client = model._create_openai_client(use_async=False)
            >>> async_client = model._create_openai_client(use_async=True)
        """
        return (AsyncOpenAI if use_async else OpenAI)(
            api_key=self.api_token,
            base_url=self.base_url,
            organization=self.organization_id,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.headers
        )

    @property
    def _client(self) -> OpenAI:
        """
        Get the synchronous OpenAI client instance.

        Creates a new client on first access and caches it for subsequent calls.

        :return: Synchronous OpenAI client instance
        :rtype: OpenAI

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> client = model._client
        """
        self._client_non_async = self._client_non_async or self._create_openai_client(use_async=False)
        return self._client_non_async

    def _get_non_async_session(self, messages: List[dict], stream: bool = False, **params):
        """
        Create a synchronous chat completion session.

        This is an internal method used by other public methods to create API sessions.

        :param messages: List of message dictionaries for the conversation
        :type messages: List[dict]
        :param stream: Whether to enable streaming mode (default: False)
        :type stream: bool
        :param params: Additional parameters to pass to the API
        :type params: Any

        :return: Chat completion response or stream
        :rtype: Any

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> session = model._get_non_async_session(messages, stream=False)
        """
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            **{
                **self.default_params,
                **params,
            }
        )

    def create_message(self, messages: List[dict], **params) -> ChatCompletionMessage:
        """
        Send a chat request and get the complete message response.

        :param messages: List of message dictionaries for the conversation
        :type messages: List[dict]
        :param params: Additional parameters to pass to the API
        :type params: Any

        :return: The message object from the first choice in the response
        :rtype: ChatCompletionMessage

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> messages = [{"role": "user", "content": "What is AI?"}]
            >>> response = model.create_message(messages)
            >>> print(response.content)
        """
        session = self._get_non_async_session(messages=messages, stream=False, **params)
        return session.choices[0].message

    def ask(self, messages: List[dict], with_reasoning: bool = False, **params) \
            -> Union[str, Tuple[Optional[str], str]]:
        """
        Send a chat request and get the text response.

        :param messages: List of message dictionaries for the conversation
        :type messages: List[dict]
        :param with_reasoning: Whether to return reasoning content along with the response (default: False)
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the API
        :type params: Any

        :return: If with_reasoning is False, returns the content string.
                 If with_reasoning is True, returns a tuple of (reasoning_content, content).
        :rtype: Union[str, Tuple[Optional[str], str]]

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> messages = [{"role": "user", "content": "Explain quantum computing"}]
            >>> # Get only the response content
            >>> response = model.ask(messages)
            >>> print(response)
            >>> # Get both reasoning and response content
            >>> reasoning, response = model.ask(messages, with_reasoning=True)
            >>> print(f"Reasoning: {reasoning}")
            >>> print(f"Response: {response}")
        """
        message = self.create_message(messages=messages, **params)
        if with_reasoning:
            return getattr(message, 'reasoning_content', None), message.content
        else:
            return message.content

    def ask_stream(self, messages: List[dict], with_reasoning: bool = False, **params) -> ResponseStream:
        """
        Send a chat request and get a streaming response.

        :param messages: List of message dictionaries for the conversation
        :type messages: List[dict]
        :param with_reasoning: Whether to include reasoning content in the stream (default: False)
        :type with_reasoning: bool
        :param params: Additional parameters to pass to the API
        :type params: Any

        :return: A ResponseStream object for iterating over the streaming response
        :rtype: ResponseStream

        Example::
            >>> model = LLMRemoteModel(base_url="...", api_token="...", model_name="...")
            >>> messages = [{"role": "user", "content": "Write a story"}]
            >>> stream = model.ask_stream(messages)
            >>> for chunk in stream:
            ...     print(chunk, end='', flush=True)
        """
        session = self._get_non_async_session(messages=messages, stream=True, **params)
        return OpenAIResponseStream(session, with_reasoning=with_reasoning)

    def __repr__(self) -> str:
        """
        Return a string representation of the LLMRemoteModel instance.

        All constructor parameters including default_params are displayed at the same level.
        The API token is masked for security purposes.

        :return: String representation of the instance
        :rtype: str

        Example::
            >>> model = LLMRemoteModel(
            ...     base_url="https://api.openai.com/v1",
            ...     api_token="sk-xxx",
            ...     model_name="gpt-3.5-turbo",
            ...     max_tokens=1000
            ... )
            >>> repr(model)
            'LLMRemoteModel(base_url=..., api_token=..., max_tokens=1000, ...)'
        """
        # Collect all parameters
        params = {
            'base_url': self.base_url,
            'api_token': self.api_token,
            'model_name': self.model_name,
            'organization_id': self.organization_id,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'headers': self.headers,

            # Expand default_params content to the same level
            **self.default_params
        }

        # Build parameter string list
        param_strings = []
        for key, value in params.items():
            if key == 'api_token' and value:
                # Mask API token for security
                if len(value) > 12:
                    masked_value = f"'{value[:6]}***{value[-6:]}'"
                elif len(value) > 8:
                    masked_value = f"'{value[:4]}***{value[-4:]}'"
                else:
                    masked_value = "'***'"
                param_strings.append(f"{key}={masked_value}")
            else:
                param_strings.append(f"{key}={value!r}")

        params_str = ', '.join(param_strings)
        return f"{self.__class__.__name__}({params_str})"
