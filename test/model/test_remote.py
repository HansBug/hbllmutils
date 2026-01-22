from unittest.mock import Mock, patch

import pytest
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from hbllmutils.model.remote import RemoteLLMModel  # Replace with actual import path


@pytest.fixture
def valid_base_url():
    return "https://api.openai.com/v1"


@pytest.fixture
def valid_api_token():
    return "sk-test123456789"


@pytest.fixture
def valid_model_name():
    return "gpt-3.5-turbo"


@pytest.fixture
def valid_organization_id():
    return "org-123"


@pytest.fixture
def valid_headers():
    return {"Custom-Header": "value"}


@pytest.fixture
def valid_default_params():
    return {"temperature": 0.7, "max_tokens": 1000}


@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "Hello, world!"}]


@pytest.fixture
def mock_openai_client():
    client = Mock(spec=OpenAI)
    client.chat.completions.create.return_value = Mock()
    return client


@pytest.fixture
def mock_async_openai_client():
    client = Mock(spec=AsyncOpenAI)
    client.chat.completions.create.return_value = Mock()
    return client


@pytest.fixture
def mock_chat_completion_message():
    message = Mock(spec=ChatCompletionMessage)
    message.content = "Test response"
    message.reasoning_content = "Test reasoning"
    return message


@pytest.fixture
def mock_chat_completion_response():
    response = Mock()
    message = Mock(spec=ChatCompletionMessage)
    message.content = "Test response"
    message.reasoning_content = "Test reasoning"
    choice = Mock()
    choice.message = message
    response.choices = [choice]
    return response


@pytest.fixture
def mock_stream_response():
    stream = Mock()
    return stream


@pytest.mark.unittest
class TestRemoteLLMModel:

    def test_init_valid_parameters(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        assert model.base_url == valid_base_url
        assert model.api_token == valid_api_token
        assert model.model_name == valid_model_name
        assert model.organization_id is None
        assert model.timeout == 30
        assert model.max_retries == 3
        assert model.headers == {}
        assert model.default_params == {}
        assert model._client_non_async is None

    def test_init_with_all_parameters(self, valid_base_url, valid_api_token, valid_model_name,
                                      valid_organization_id, valid_headers, valid_default_params):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id=valid_organization_id,
            timeout=60,
            max_retries=5,
            headers=valid_headers,
            **valid_default_params
        )
        assert model.base_url == valid_base_url
        assert model.api_token == valid_api_token
        assert model.model_name == valid_model_name
        assert model.organization_id == valid_organization_id
        assert model.timeout == 60
        assert model.max_retries == 5
        assert model.headers == valid_headers
        assert model.default_params == valid_default_params

    def test_init_invalid_base_url_format(self, valid_api_token, valid_model_name):
        with pytest.raises(ValueError, match="Invalid base_url format"):
            RemoteLLMModel(
                base_url="invalid-url",
                api_token=valid_api_token,
                model_name=valid_model_name
            )

    def test_init_invalid_base_url_no_scheme(self, valid_api_token, valid_model_name):
        with pytest.raises(ValueError, match="Invalid base_url format"):
            RemoteLLMModel(
                base_url="api.openai.com/v1",
                api_token=valid_api_token,
                model_name=valid_model_name
            )

    def test_init_empty_api_token(self, valid_base_url, valid_model_name):
        with pytest.raises(ValueError, match="api_token cannot be empty"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token="",
                model_name=valid_model_name
            )

    def test_init_whitespace_api_token(self, valid_base_url, valid_model_name):
        with pytest.raises(ValueError, match="api_token cannot be empty"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token="   ",
                model_name=valid_model_name
            )

    def test_init_empty_model_name(self, valid_base_url, valid_api_token):
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token=valid_api_token,
                model_name=""
            )

    def test_init_whitespace_model_name(self, valid_base_url, valid_api_token):
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token=valid_api_token,
                model_name="   "
            )

    def test_init_zero_timeout(self, valid_base_url, valid_api_token, valid_model_name):
        with pytest.raises(ValueError, match="timeout must be positive"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token=valid_api_token,
                model_name=valid_model_name,
                timeout=0
            )

    def test_init_negative_timeout(self, valid_base_url, valid_api_token, valid_model_name):
        with pytest.raises(ValueError, match="timeout must be positive"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token=valid_api_token,
                model_name=valid_model_name,
                timeout=-1
            )

    def test_init_negative_max_retries(self, valid_base_url, valid_api_token, valid_model_name):
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            RemoteLLMModel(
                base_url=valid_base_url,
                api_token=valid_api_token,
                model_name=valid_model_name,
                max_retries=-1
            )

    def test_init_zero_max_retries(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            max_retries=0
        )
        assert model.max_retries == 0

    def test_init_none_headers(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            headers=None
        )
        assert model.headers == {}

    def test_init_invalid_base_url_exception(self, valid_api_token, valid_model_name):
        with patch('hbllmutils.model.remote.urlparse') as mock_urlparse:
            mock_urlparse.side_effect = Exception("Parse error")
            with pytest.raises(ValueError, match="Invalid base_url"):
                RemoteLLMModel(
                    base_url="https://api.openai.com/v1",
                    api_token=valid_api_token,
                    model_name=valid_model_name
                )

    def test_logger_name_property(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        assert model._logger_name == valid_model_name

    @patch('hbllmutils.model.remote.OpenAI')  # Replace with actual import path
    def test_create_openai_client_sync(self, mock_openai_class, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id="org-123",
            timeout=60,
            max_retries=5,
            headers={"Custom": "Header"}
        )

        client = model._create_openai_client(use_async=False)

        mock_openai_class.assert_called_once_with(
            api_key=valid_api_token,
            base_url=valid_base_url,
            organization="org-123",
            timeout=60,
            max_retries=5,
            default_headers={"Custom": "Header"}
        )

    @patch('hbllmutils.model.remote.AsyncOpenAI')  # Replace with actual import path
    def test_create_openai_client_async(self, mock_async_openai_class, valid_base_url, valid_api_token,
                                        valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        client = model._create_openai_client(use_async=True)

        mock_async_openai_class.assert_called_once_with(
            api_key=valid_api_token,
            base_url=valid_base_url,
            organization=None,
            timeout=30,
            max_retries=3,
            default_headers={}
        )

    @patch('hbllmutils.model.remote.OpenAI')  # Replace with actual import path
    def test_client_property_creates_and_caches(self, mock_openai_class, valid_base_url, valid_api_token,
                                                valid_model_name):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        # First access
        client1 = model._client
        assert client1 == mock_client
        assert model._client_non_async == mock_client

        # Second access should return cached client
        client2 = model._client
        assert client2 == mock_client
        assert client1 is client2

        # OpenAI should only be called once
        mock_openai_class.assert_called_once()

    def test_get_non_async_session(self, valid_base_url, valid_api_token, valid_model_name, sample_messages):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            temperature=0.5
        )

        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        model._client_non_async = mock_client

        result = model._get_non_async_session(sample_messages, stream=False, max_tokens=100)

        mock_client.chat.completions.create.assert_called_once_with(
            model=valid_model_name,
            messages=sample_messages,
            stream=False,
            temperature=0.5,
            max_tokens=100
        )
        assert result == mock_response

    def test_get_non_async_session_with_stream(self, valid_base_url, valid_api_token, valid_model_name,
                                               sample_messages):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        model._client_non_async = mock_client

        result = model._get_non_async_session(sample_messages, stream=True)

        mock_client.chat.completions.create.assert_called_once_with(
            model=valid_model_name,
            messages=sample_messages,
            stream=True
        )
        assert result == mock_response

    def test_create_message(self, valid_base_url, valid_api_token, valid_model_name, sample_messages,
                            mock_chat_completion_response):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        with patch.object(model, '_get_non_async_session', return_value=mock_chat_completion_response):
            result = model.create_message(sample_messages, temperature=0.7)

            model._get_non_async_session.assert_called_once_with(
                messages=sample_messages,
                stream=False,
                temperature=0.7
            )
            assert result == mock_chat_completion_response.choices[0].message

    def test_ask_without_reasoning(self, valid_base_url, valid_api_token, valid_model_name, sample_messages,
                                   mock_chat_completion_message):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        with patch.object(model, 'create_message', return_value=mock_chat_completion_message):
            result = model.ask(sample_messages, temperature=0.8)

            model.create_message.assert_called_once_with(
                messages=sample_messages,
                temperature=0.8
            )
            assert result == "Test response"

    def test_ask_with_reasoning(self, valid_base_url, valid_api_token, valid_model_name, sample_messages,
                                mock_chat_completion_message):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        with patch.object(model, 'create_message', return_value=mock_chat_completion_message):
            reasoning, content = model.ask(sample_messages, with_reasoning=True, temperature=0.8)

            model.create_message.assert_called_once_with(
                messages=sample_messages,
                temperature=0.8
            )
            assert reasoning == "Test reasoning"
            assert content == "Test response"

    def test_ask_with_reasoning_no_reasoning_content(self, valid_base_url, valid_api_token, valid_model_name,
                                                     sample_messages):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        message = Mock(spec=ChatCompletionMessage)
        message.content = "Test response"
        # No reasoning_content attribute

        with patch.object(model, 'create_message', return_value=message):
            reasoning, content = model.ask(sample_messages, with_reasoning=True)

            assert reasoning is None
            assert content == "Test response"

    @patch('hbllmutils.model.remote.OpenAIResponseStream')  # Replace with actual import path
    def test_ask_stream(self, mock_stream_class, valid_base_url, valid_api_token, valid_model_name, sample_messages,
                        mock_stream_response):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        mock_session = Mock()
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance

        with patch.object(model, '_get_non_async_session', return_value=mock_session):
            result = model.ask_stream(sample_messages, with_reasoning=True, temperature=0.9)

            model._get_non_async_session.assert_called_once_with(
                messages=sample_messages,
                stream=True,
                temperature=0.9
            )
            mock_stream_class.assert_called_once_with(mock_session, with_reasoning=True)
            assert result == mock_stream_instance

    @patch('hbllmutils.model.remote.OpenAIResponseStream')  # Replace with actual import path
    def test_ask_stream_without_reasoning(self, mock_stream_class, valid_base_url, valid_api_token, valid_model_name,
                                          sample_messages):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        mock_session = Mock()
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance

        with patch.object(model, '_get_non_async_session', return_value=mock_session):
            result = model.ask_stream(sample_messages)

            mock_stream_class.assert_called_once_with(mock_session, with_reasoning=False)
            assert result == mock_stream_instance

    def test_repr_basic(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        repr_str = repr(model)

        assert "RemoteLLMModel(" in repr_str
        assert f"base_url='{valid_base_url}'" in repr_str
        assert f"model_name='{valid_model_name}'" in repr_str
        assert "organization_id=None" in repr_str
        assert "timeout=30" in repr_str
        assert "max_retries=3" in repr_str
        assert "headers={}" in repr_str
        # API token should be masked
        assert "sk-tes***456789" in repr_str

    def test_repr_with_all_params(self, valid_base_url, valid_api_token, valid_model_name, valid_organization_id,
                                  valid_headers, valid_default_params):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id=valid_organization_id,
            timeout=60,
            max_retries=5,
            headers=valid_headers,
            **valid_default_params
        )

        repr_str = repr(model)

        assert "RemoteLLMModel(" in repr_str
        assert f"organization_id='{valid_organization_id}'" in repr_str
        assert "timeout=60" in repr_str
        assert "max_retries=5" in repr_str
        assert "temperature=0.7" in repr_str
        assert "max_tokens=1000" in repr_str

    def test_repr_short_token_masking(self, valid_base_url, valid_model_name):
        short_token = "sk-123"
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=short_token,
            model_name=valid_model_name
        )

        repr_str = repr(model)
        assert "api_token='***'" in repr_str

    def test_repr_medium_token_masking(self, valid_base_url, valid_model_name):
        medium_token = "sk-123456789"
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=medium_token,
            model_name=valid_model_name
        )

        repr_str = repr(model)
        assert "api_token='sk-1***6789'" in repr_str

    def test_repr_long_token_masking(self, valid_base_url, valid_model_name):
        long_token = "sk-1234567890abcdef"
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=long_token,
            model_name=valid_model_name
        )

        repr_str = repr(model)
        assert "api_token='sk-123***abcdef'" in repr_str

    def test_params_method(self, valid_base_url, valid_api_token, valid_model_name, valid_organization_id,
                           valid_headers, valid_default_params):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id=valid_organization_id,
            timeout=60,
            max_retries=5,
            headers=valid_headers,
            **valid_default_params
        )

        params = model._params()

        expected_headers_tuple = tuple(sorted(valid_headers.items()))
        expected_default_params_tuple = tuple(sorted(valid_default_params.items()))

        expected_params = (
            valid_base_url,
            valid_api_token,
            valid_model_name,
            valid_organization_id,
            60,
            5,
            expected_headers_tuple,
            expected_default_params_tuple
        )

        assert params == expected_params
        assert isinstance(params, tuple)

    def test_params_method_empty_headers_and_default_params(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        params = model._params()

        expected_params = (
            valid_base_url,
            valid_api_token,
            valid_model_name,
            None,
            30,
            3,
            (),
            ()
        )

        assert params == expected_params

    def test_params_method_none_headers(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            headers=None
        )

        params = model._params()
        assert params[6] == ()  # headers should be empty tuple

    def test_eq_same_instances(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        assert model1 == model2

    def test_eq_different_parameters(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            timeout=30
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            timeout=60
        )

        assert model1 != model2

    def test_eq_different_types(self, valid_base_url, valid_api_token, valid_model_name):
        model = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        assert model != "not a model"
        assert model != 123
        assert model != None

    def test_eq_with_all_parameters(self, valid_base_url, valid_api_token, valid_model_name, valid_organization_id,
                                    valid_headers, valid_default_params):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id=valid_organization_id,
            timeout=60,
            max_retries=5,
            headers=valid_headers,
            **valid_default_params
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            organization_id=valid_organization_id,
            timeout=60,
            max_retries=5,
            headers=valid_headers,
            **valid_default_params
        )

        assert model1 == model2

    def test_hash_same_instances(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        assert hash(model1) == hash(model2)

    def test_hash_different_parameters(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            timeout=30
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            timeout=60
        )

        assert hash(model1) != hash(model2)

    def test_hash_can_be_used_in_set(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        model3 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name,
            timeout=60
        )

        model_set = {model1, model2, model3}
        assert len(model_set) == 2  # model1 and model2 are equal, so only 2 unique models

    def test_hash_can_be_used_as_dict_key(self, valid_base_url, valid_api_token, valid_model_name):
        model1 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )
        model2 = RemoteLLMModel(
            base_url=valid_base_url,
            api_token=valid_api_token,
            model_name=valid_model_name
        )

        model_dict = {model1: "value1"}
        model_dict[model2] = "value2"

        assert len(model_dict) == 1  # model1 and model2 are equal, so only 1 key
        assert model_dict[model1] == "value2"  # model2 overwrote model1's value
