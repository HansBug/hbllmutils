from unittest.mock import patch

import pytest

from hbllmutils.model import FakeResponseStream, FakeLLMModel
from hbllmutils.model.fake import _fn_always_true


@pytest.fixture
def fake_model():
    """Create a FakeLLMModel instance for testing."""
    return FakeLLMModel(stream_wps=100)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How's the weather today?"}
    ]


@pytest.fixture
def empty_messages():
    """Create empty messages list for testing."""
    return []


@pytest.fixture
def single_message():
    """Create a single message for testing."""
    return [{"role": "user", "content": "test message"}]


@pytest.fixture
def weather_keywords():
    """Create weather-related keywords for testing."""
    return ["weather", "temperature", "sunny", "rain"]


@pytest.fixture
def mock_jieba_cut():
    """Mock jieba.cut function."""
    with patch('jieba.cut') as mock_cut:
        yield mock_cut


@pytest.mark.unittest
class TestFakeResponseStream:

    def test_get_reasoning_content_from_chunk(self):
        """Test extracting reasoning content from chunk."""
        stream = FakeResponseStream(session=iter([]), with_reasoning=False)
        chunk = ("reasoning text", "content text")

        result = stream._get_reasoning_content_from_chunk(chunk)

        assert result == "reasoning text"

    def test_get_content_from_chunk(self):
        """Test extracting content from chunk."""
        stream = FakeResponseStream(session=iter([]), with_reasoning=False)
        chunk = ("reasoning text", "content text")

        result = stream._get_content_from_chunk(chunk)

        assert result == "content text"


@pytest.mark.unittest
class TestFnAlwaysTrue:

    def test_fn_always_true_returns_true(self, sample_messages):
        """Test that _fn_always_true always returns True."""
        result = _fn_always_true(sample_messages, param1="value1", param2="value2")
        assert result is True

    def test_fn_always_true_with_empty_messages(self, empty_messages):
        """Test _fn_always_true with empty messages."""
        result = _fn_always_true(empty_messages)
        assert result is True


@pytest.mark.unittest
class TestFakeLLMModel:

    def test_init_default_stream_wps(self):
        """Test FakeLLMModel initialization with default stream_wps."""
        model = FakeLLMModel()
        assert model.stream_fps == 50
        assert model._rules == []

    def test_init_custom_stream_wps(self):
        """Test FakeLLMModel initialization with custom stream_wps."""
        model = FakeLLMModel(stream_wps=100)
        assert model.stream_fps == 100
        assert model._rules == []

    def test_get_response_with_string_response(self, fake_model, sample_messages):
        """Test _get_response with string response."""
        fake_model.response_always("test response")

        reasoning, content = fake_model._get_response(sample_messages)

        assert reasoning == ""
        assert content == "test response"

    def test_get_response_with_tuple_response(self, fake_model, sample_messages):
        """Test _get_response with tuple response."""
        fake_model.response_always(("reasoning", "content"))

        reasoning, content = fake_model._get_response(sample_messages)

        assert reasoning == "reasoning"
        assert content == "content"

    def test_get_response_with_list_response(self, fake_model, sample_messages):
        """Test _get_response with list response."""
        fake_model.response_always(["reasoning", "content"])

        reasoning, content = fake_model._get_response(sample_messages)

        assert reasoning == "reasoning"
        assert content == "content"

    def test_get_response_with_callable_string_response(self, fake_model, sample_messages):
        """Test _get_response with callable returning string."""

        def response_func(messages, **params):
            return "callable response"

        fake_model.response_always(response_func)

        reasoning, content = fake_model._get_response(sample_messages)

        assert reasoning == ""
        assert content == "callable response"

    def test_get_response_with_callable_tuple_response(self, fake_model, sample_messages):
        """Test _get_response with callable returning tuple."""

        def response_func(messages, **params):
            return ("callable reasoning", "callable content")

        fake_model.response_always(response_func)

        reasoning, content = fake_model._get_response(sample_messages)

        assert reasoning == "callable reasoning"
        assert content == "callable content"

    def test_get_response_no_matching_rule(self, sample_messages):
        """Test _get_response when no rule matches."""
        model = FakeLLMModel()

        with pytest.raises(AssertionError, match="No response rule found for this message."):
            model._get_response(sample_messages)

    def test_get_response_first_matching_rule(self, fake_model, sample_messages):
        """Test _get_response returns first matching rule."""
        fake_model.response_always("first response")
        fake_model.response_always("second response")

        reasoning, content = fake_model._get_response(sample_messages)

        assert content == "first response"

    def test_response_always_returns_self(self, fake_model):
        """Test response_always returns self for chaining."""
        result = fake_model.response_always("test")
        assert result is fake_model

    def test_response_always_adds_rule(self, fake_model):
        """Test response_always adds rule to _rules."""
        fake_model.response_always("test response")

        assert len(fake_model._rules) == 1
        rule_func, response = fake_model._rules[0]
        assert rule_func is _fn_always_true
        assert response == "test response"

    def test_response_when_returns_self(self, fake_model):
        """Test response_when returns self for chaining."""

        def condition(messages, **params):
            return True

        result = fake_model.response_when(condition, "test")
        assert result is fake_model

    def test_response_when_adds_rule(self, fake_model):
        """Test response_when adds rule to _rules."""

        def condition(messages, **params):
            return len(messages) > 1

        fake_model.response_when(condition, "conditional response")

        assert len(fake_model._rules) == 1
        rule_func, response = fake_model._rules[0]
        assert rule_func is condition
        assert response == "conditional response"

    def test_response_when_keyword_in_last_message_string_keyword(self, fake_model):
        """Test response_when_keyword_in_last_message with string keyword."""
        result = fake_model.response_when_keyword_in_last_message("weather", "weather response")

        assert result is fake_model
        assert len(fake_model._rules) == 1

    def test_response_when_keyword_in_last_message_list_keywords(self, fake_model, weather_keywords):
        """Test response_when_keyword_in_last_message with list of keywords."""
        fake_model.response_when_keyword_in_last_message(weather_keywords, "weather response")

        assert len(fake_model._rules) == 1

    def test_response_when_keyword_in_last_message_tuple_keywords(self, fake_model):
        """Test response_when_keyword_in_last_message with tuple of keywords."""
        keywords = ("weather", "temperature")
        fake_model.response_when_keyword_in_last_message(keywords, "weather response")

        assert len(fake_model._rules) == 1

    def test_keyword_check_function_match(self, fake_model):
        """Test keyword check function matches keyword in last message."""
        fake_model.response_when_keyword_in_last_message("weather", "weather response")
        rule_func, _ = fake_model._rules[0]

        messages = [{"role": "user", "content": "What's the weather like?"}]
        result = rule_func(messages)

        assert result is True

    def test_keyword_check_function_no_match(self, fake_model):
        """Test keyword check function doesn't match when keyword absent."""
        fake_model.response_when_keyword_in_last_message("weather", "weather response")
        rule_func, _ = fake_model._rules[0]

        messages = [{"role": "user", "content": "Hello there!"}]
        result = rule_func(messages)

        assert result is False

    def test_keyword_check_function_multiple_keywords_match(self, fake_model, weather_keywords):
        """Test keyword check function with multiple keywords, one matches."""
        fake_model.response_when_keyword_in_last_message(weather_keywords, "weather response")
        rule_func, _ = fake_model._rules[0]

        messages = [{"role": "user", "content": "Is it sunny today?"}]
        result = rule_func(messages)

        assert result is True

    def test_keyword_check_function_multiple_keywords_no_match(self, fake_model, weather_keywords):
        """Test keyword check function with multiple keywords, none match."""
        fake_model.response_when_keyword_in_last_message(weather_keywords, "weather response")
        rule_func, _ = fake_model._rules[0]

        messages = [{"role": "user", "content": "Hello world!"}]
        result = rule_func(messages)

        assert result is False

    def test_ask_without_reasoning(self, fake_model, sample_messages):
        """Test ask method without reasoning."""
        fake_model.response_always("test response")

        result = fake_model.ask(sample_messages)

        assert result == "test response"

    def test_ask_with_reasoning(self, fake_model, sample_messages):
        """Test ask method with reasoning."""
        fake_model.response_always(("reasoning", "content"))

        result = fake_model.ask(sample_messages, with_reasoning=True)

        assert result == ("reasoning", "content")

    def test_ask_with_params(self, fake_model, sample_messages):
        """Test ask method passes params to _get_response."""

        def response_func(messages, **params):
            return f"param value: {params.get('test_param')}"

        fake_model.response_always(response_func)

        result = fake_model.ask(sample_messages, test_param="test_value")

        assert result == "param value: test_value"

    def test_iter_per_words_content_only(self, fake_model, mock_jieba_cut):
        """Test _iter_per_words with content only."""
        mock_jieba_cut.return_value = ["Hello", "world"]

        with patch('time.sleep') as mock_sleep:
            chunks = list(fake_model._iter_per_words("Hello world"))

        expected_chunks = [(None, "Hello"), (None, "world")]
        assert chunks == expected_chunks
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1 / 100)  # stream_fps = 100

    def test_iter_per_words_reasoning_and_content(self, fake_model, mock_jieba_cut):
        """Test _iter_per_words with both reasoning and content."""
        mock_jieba_cut.side_effect = [["Think", "about"], ["Hello", "world"]]

        with patch('time.sleep') as mock_sleep:
            chunks = list(fake_model._iter_per_words("Hello world", "Think about"))

        expected_chunks = [("Think", None), ("about", None), (None, "Hello"), (None, "world")]
        assert chunks == expected_chunks
        assert mock_sleep.call_count == 4

    def test_iter_per_words_empty_words_filtered(self, fake_model, mock_jieba_cut):
        """Test _iter_per_words filters out empty words."""
        mock_jieba_cut.return_value = ["Hello", "", "world", ""]

        with patch('time.sleep'):
            chunks = list(fake_model._iter_per_words("Hello world"))

        expected_chunks = [(None, "Hello"), (None, "world")]
        assert chunks == expected_chunks

    def test_iter_per_words_empty_content(self, fake_model, mock_jieba_cut):
        """Test _iter_per_words with empty content."""
        chunks = list(fake_model._iter_per_words(""))
        assert chunks == []

    def test_iter_per_words_empty_reasoning(self, fake_model, mock_jieba_cut):
        """Test _iter_per_words with empty reasoning content."""
        mock_jieba_cut.return_value = ["Hello"]

        with patch('time.sleep'):
            chunks = list(fake_model._iter_per_words("Hello", ""))

        expected_chunks = [(None, "Hello")]
        assert chunks == expected_chunks

    def test_ask_stream_returns_fake_response_stream(self, fake_model, sample_messages):
        """Test ask_stream returns FakeResponseStream."""
        fake_model.response_always("test response")

        stream = fake_model.ask_stream(sample_messages)

        assert isinstance(stream, FakeResponseStream)

    def test_ask_stream_with_reasoning_true(self, fake_model, sample_messages):
        """Test ask_stream with with_reasoning=True."""
        fake_model.response_always(("reasoning", "content"))

        stream = fake_model.ask_stream(sample_messages, with_reasoning=True)

        assert stream._with_reasoning is True

    def test_ask_stream_with_reasoning_false(self, fake_model, sample_messages):
        """Test ask_stream with with_reasoning=False."""
        fake_model.response_always("content")

        stream = fake_model.ask_stream(sample_messages, with_reasoning=False)

        assert stream._with_reasoning is False

    def test_ask_stream_passes_params(self, fake_model, sample_messages):
        """Test ask_stream passes params to _get_response."""

        def response_func(messages, **params):
            return f"param: {params.get('test_param')}"

        fake_model.response_always(response_func)

        with patch.object(fake_model, '_get_response', wraps=fake_model._get_response) as mock_get_response:
            fake_model.ask_stream(sample_messages, test_param="test_value")
            mock_get_response.assert_called_once_with(messages=sample_messages, test_param="test_value")

    def test_method_chaining(self, fake_model):
        """Test method chaining works correctly."""
        result = (fake_model
                  .response_when_keyword_in_last_message("hello", "hi response")
                  .response_when_keyword_in_last_message("weather", "weather response")
                  .response_always("default response"))

        assert result is fake_model
        assert len(fake_model._rules) == 3

    def test_rule_priority_order(self, fake_model):
        """Test that rules are checked in the order they were added."""
        fake_model.response_when_keyword_in_last_message("test", "first match")
        fake_model.response_always("second match")

        messages = [{"role": "user", "content": "this is a test message"}]
        result = fake_model.ask(messages)

        assert result == "first match"
