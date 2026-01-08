import io
from textwrap import indent
from typing import Dict, Optional, Union, Any, List
from urllib.parse import urlparse

from openai import OpenAI, AsyncOpenAI


class LLMRemoteModel:
    # base_url: str  # API基础URL，如 "https://api.openai.com/v1"
    # api_token: str  # API访问令牌
    # model_name: str  # 模型名称，如 "gpt-3.5-turbo", "claude-3-opus"
    #
    # organization_id: Optional[str] = None  # 组织ID（某些API需要）
    # timeout: int = 30  # 请求超时时间（秒）
    # max_retries: int = 3  # 最大重试次数
    # headers: Dict[str, str] = field(default_factory=dict)  # 自定义请求头

    def __init__(self, base_url: str, api_token: str, model_name: str,
                 organization_id: Optional[str] = None, timeout: int = 30, max_retries: int = 3,
                 headers: Optional[Dict[str, str]] = None, default_params: Optional[Dict[str, Any]] = None):
        self.base_url = base_url
        # 验证URL格式
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
        self.default_params = dict(default_params or {})

        self._client_non_async = None
        self._client_async = None

    def create_openai_client(self, use_async: bool = False) -> Union[OpenAI, AsyncOpenAI]:
        return (AsyncOpenAI if use_async else OpenAI)(
            api_key=self.api_token,
            base_url=self.base_url,
            organization=self.organization_id,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.headers
        )

    @property
    def client(self) -> OpenAI:
        self._client_non_async = self._client_non_async or self.create_openai_client(use_async=False)
        return self._client_non_async

    @property
    def async_client(self) -> AsyncOpenAI:
        self._client_async = self._client_async or self.create_openai_client(use_async=True)
        return self._client_async

    def _get_non_async_session(self, messages: List[dict], stream=False, **params):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            **{
                **self.default_params,
                **params,
            }
        )

    def chat(self, messages: List[dict], **params):
        session = self._get_non_async_session(messages=messages, stream=False, **params)
        return session.choices[0].message

    def ask(self, messages: List[dict], with_reasoning: bool = False, **params):
        message = self.chat(messages=messages, **params)
        with io.StringIO() as sio:
            if with_reasoning and getattr(message, 'reasoning_content'):
                print(indent(message.reasoning_content.rstrip(), prefix='> '), file=sio)
                print('', file=sio)
            print(message.content.rstrip(), file=sio)
            return sio.getvalue()

    def ask_simple(self, prompt: str, system_prompt: Optional[str] = None, with_reasoning: bool = False, **params):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.ask(
            messages=messages,
            with_reasoning=with_reasoning,
            **params,
        )

    def ask_stream(self, messages: List[dict], with_reasoning: bool = False, **params):
        session = self._get_non_async_session(messages=messages, stream=True, **params)
        status = 'none'
        for chunk in session:
            delta = chunk.choices[0].delta
            if with_reasoning and getattr(delta, 'reasoning_content'):
                if with_reasoning and status != 'reasoning':
                    if status != 'none':
                        yield '\n\n'
                    yield '---------------------------reasoning---------------------------\n\n'
                    status = 'reasoning'
                yield delta.reasoning_content
            if delta.content is not None:
                if with_reasoning and status != 'content':
                    if status != 'none':
                        yield '\n\n'
                    yield '---------------------------content---------------------------\n\n'
                    status = 'content'
                yield delta.content
