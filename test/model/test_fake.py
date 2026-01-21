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
        assert model.stream_wps == 50
        assert model.rules_count == 0

    def test_init_custom_stream_wps(self):
        """Test FakeLLMModel initialization with custom stream_wps."""
        model = FakeLLMModel(stream_wps=100)
        assert model.stream_wps == 100
        assert model.rules_count == 0

    def test_init_with_rules(self):
        """Test FakeLLMModel initialization with rules parameter."""
        rules = [(_fn_always_true, "response")]
        model = FakeLLMModel(stream_wps=50, rules=rules)
        assert model.rules_count == 1
        assert model._rules == tuple(rules)

    def test_init_with_none_rules(self):
        """Test FakeLLMModel initialization with None rules."""
        model = FakeLLMModel(stream_wps=50, rules=None)
        assert model.rules_count == 0
        assert model._rules == tuple()

    def test_immutability_setattr_after_frozen(self):
        """Test that attributes cannot be modified after initialization."""
        model = FakeLLMModel()
        with pytest.raises(AttributeError, match="Cannot modify attribute '_stream_wps' of immutable FakeLLMModel"):
            model._stream_wps = 200

    def test_immutability_setattr_new_attribute(self):
        """Test that new attributes cannot be added after initialization."""
        model = FakeLLMModel()
        with pytest.raises(AttributeError, match="Cannot modify attribute 'new_attr' of immutable FakeLLMModel"):
            model.new_attr = "value"

    def test_immutability_delattr(self):
        """Test that attributes cannot be deleted."""
        model = FakeLLMModel()
        with pytest.raises(AttributeError, match="Cannot delete attribute '_stream_wps' of immutable FakeLLMModel"):
            del model._stream_wps

    def test_logger_name_property(self):
        """Test _logger_name property."""
        model = FakeLLMModel()
        assert model._logger_name == '<faker>'

    def test_rules_count_property_empty(self):
        """Test rules_count property with no rules."""
        model = FakeLLMModel()
        assert model.rules_count == 0

    def test_rules_count_property_with_rules(self):
        """Test rules_count property with multiple rules."""
        model = FakeLLMModel()
        model = model.response_always("response1")
        model = model.response_always("response2")
        model = model.response_always("response3")
        assert model.rules_count == 3

    def test_create_new_instance_default_params(self):
        """Test _create_new_instance with default parameters."""
        model = FakeLLMModel(stream_wps=100)
        new_model = model._create_new_instance()

        assert new_model is not model
        assert new_model.stream_wps == 100
        assert new_model.rules_count == 0

    def test_create_new_instance_override_stream_wps(self):
        """Test _create_new_instance with overridden stream_wps."""
        model = FakeLLMModel(stream_wps=100)
        new_model = model._create_new_instance(stream_wps=200)

        assert new_model.stream_wps == 200

    def test_create_new_instance_override_rules(self):
        """Test _create_new_instance with overridden rules."""
        model = FakeLLMModel()
        rules = [(_fn_always_true, "response")]
        new_model = model._create_new_instance(rules=rules)

        assert new_model.rules_count == 1

    def test_get_response_with_string_response(self, sample_messages):
        """Test _get_response with string response."""
        model = FakeLLMModel().response_always("test response")

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == ""
        assert content == "test response"

    def test_get_response_with_tuple_response(self, sample_messages):
        """Test _get_response with tuple response."""
        model = FakeLLMModel().response_always(("reasoning", "content"))

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == "reasoning"
        assert content == "content"

    def test_get_response_with_list_response(self, sample_messages):
        """Test _get_response with list response."""
        model = FakeLLMModel().response_always(["reasoning", "content"])

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == "reasoning"
        assert content == "content"

    def test_get_response_with_callable_string_response(self, sample_messages):
        """Test _get_response with callable returning string."""

        def response_func(messages, **params):
            return "callable response"

        model = FakeLLMModel().response_always(response_func)

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == ""
        assert content == "callable response"

    def test_get_response_with_callable_tuple_response(self, sample_messages):
        """Test _get_response with callable returning tuple."""

        def response_func(messages, **params):
            return ("callable reasoning", "callable content")

        model = FakeLLMModel().response_always(response_func)

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == "callable reasoning"
        assert content == "callable content"

    def test_get_response_with_callable_list_response(self, sample_messages):
        """Test _get_response with callable returning list."""

        def response_func(messages, **params):
            return ["callable reasoning", "callable content"]

        model = FakeLLMModel().response_always(response_func)

        reasoning, content = model._get_response(sample_messages)

        assert reasoning == "callable reasoning"
        assert content == "callable content"

    def test_get_response_no_matching_rule(self, sample_messages):
        """Test _get_response when no rule matches."""
        model = FakeLLMModel()

        with pytest.raises(AssertionError, match="No response rule found for this message."):
            model._get_response(sample_messages)

    def test_get_response_first_matching_rule(self, sample_messages):
        """Test _get_response returns first matching rule."""
        model = FakeLLMModel().response_always("first response").response_always("second response")

        reasoning, content = model._get_response(sample_messages)

        assert content == "first response"

    def test_get_response_passes_params(self, sample_messages):
        """Test _get_response passes params to callable response."""

        def response_func(messages, **params):
            return f"param: {params.get('test_param')}"

        model = FakeLLMModel().response_always(response_func)
        reasoning, content = model._get_response(sample_messages, test_param="test_value")

        assert content == "param: test_value"

    def test_with_stream_wps_returns_new_instance(self):
        """Test with_stream_wps returns new instance."""
        model = FakeLLMModel(stream_wps=100)
        new_model = model.with_stream_wps(200)

        assert new_model is not model
        assert new_model.stream_wps == 200
        assert model.stream_wps == 100

    def test_with_stream_wps_preserves_rules(self):
        """Test with_stream_wps preserves existing rules."""
        model = FakeLLMModel(stream_wps=100).response_always("test")
        new_model = model.with_stream_wps(200)

        assert new_model.rules_count == 1
        assert model.rules_count == 1

    def test_response_always_returns_new_instance(self):
        """Test response_always returns new instance."""
        model = FakeLLMModel()
        new_model = model.response_always("test response")

        assert new_model is not model
        assert new_model.rules_count == 1
        assert model.rules_count == 0

    def test_response_always_adds_rule(self):
        """Test response_always adds rule to _rules."""
        model = FakeLLMModel().response_always("test response")

        assert len(model._rules) == 1
        rule_func, response = model._rules[0]
        assert rule_func is _fn_always_true
        assert response == "test response"

    def test_response_when_returns_new_instance(self):
        """Test response_when returns new instance."""

        def condition(messages, **params):
            return True

        model = FakeLLMModel()
        new_model = model.response_when(condition, "test")

        assert new_model is not model
        assert new_model.rules_count == 1
        assert model.rules_count == 0

    def test_response_when_adds_rule(self):
        """Test response_when adds rule to _rules."""

        def condition(messages, **params):
            return len(messages) > 1

        model = FakeLLMModel().response_when(condition, "conditional response")

        assert len(model._rules) == 1
        rule_func, response = model._rules[0]
        assert rule_func is condition
        assert response == "conditional response"

    def test_response_when_keyword_in_last_message_string_keyword(self):
        """Test response_when_keyword_in_last_message with string keyword."""
        model = FakeLLMModel()
        new_model = model.response_when_keyword_in_last_message("weather", "weather response")

        assert new_model is not model
        assert new_model.rules_count == 1
        assert model.rules_count == 0

    def test_response_when_keyword_in_last_message_list_keywords(self, weather_keywords):
        """Test response_when_keyword_in_last_message with list of keywords."""
        model = FakeLLMModel().response_when_keyword_in_last_message(weather_keywords, "weather response")

        assert len(model._rules) == 1

    def test_response_when_keyword_in_last_message_tuple_keywords(self):
        """Test response_when_keyword_in_last_message with tuple of keywords."""
        keywords = ("weather", "temperature")
        model = FakeLLMModel().response_when_keyword_in_last_message(keywords, "weather response")

        assert len(model._rules) == 1

    def test_keyword_check_function_match(self):
        """Test keyword check function matches keyword in last message."""
        model = FakeLLMModel().response_when_keyword_in_last_message("weather", "weather response")
        rule_func, _ = model._rules[0]

        messages = [{"role": "user", "content": "What's the weather like?"}]
        result = rule_func(messages)

        assert result is True

    def test_keyword_check_function_no_match(self):
        """Test keyword check function doesn't match when keyword absent."""
        model = FakeLLMModel().response_when_keyword_in_last_message("weather", "weather response")
        rule_func, _ = model._rules[0]

        messages = [{"role": "user", "content": "Hello there!"}]
        result = rule_func(messages)

        assert result is False

    def test_keyword_check_function_multiple_keywords_match(self, weather_keywords):
        """Test keyword check function with multiple keywords, one matches."""
        model = FakeLLMModel().response_when_keyword_in_last_message(weather_keywords, "weather response")
        rule_func, _ = model._rules[0]

        messages = [{"role": "user", "content": "Is it sunny today?"}]
        result = rule_func(messages)

        assert result is True

    def test_keyword_check_function_multiple_keywords_no_match(self, weather_keywords):
        """Test keyword check function with multiple keywords, none match."""
        model = FakeLLMModel().response_when_keyword_in_last_message(weather_keywords, "weather response")
        rule_func, _ = model._rules[0]

        messages = [{"role": "user", "content": "Hello world!"}]
        result = rule_func(messages)

        assert result is False

    def test_keyword_check_function_ignores_params(self):
        """Test keyword check function ignores additional params."""
        model = FakeLLMModel().response_when_keyword_in_last_message("test", "response")
        rule_func, _ = model._rules[0]

        messages = [{"role": "user", "content": "test message"}]
        result = rule_func(messages, extra_param="value")

        assert result is True

    def test_clear_rules_returns_new_instance(self):
        """Test clear_rules returns new instance."""
        model = FakeLLMModel().response_always("test")
        new_model = model.clear_rules()

        assert new_model is not model
        assert new_model.rules_count == 0
        assert model.rules_count == 1

    def test_clear_rules_removes_all_rules(self):
        """Test clear_rules removes all rules."""
        model = (FakeLLMModel()
                 .response_always("test1")
                 .response_always("test2")
                 .response_always("test3"))

        clean_model = model.clear_rules()

        assert clean_model.rules_count == 0

    def test_ask_without_reasoning(self, sample_messages):
        """Test ask method without reasoning."""
        model = FakeLLMModel().response_always("test response")

        result = model.ask(sample_messages)

        assert result == "test response"

    def test_ask_with_reasoning(self, sample_messages):
        """Test ask method with reasoning."""
        model = FakeLLMModel().response_always(("reasoning", "content"))

        result = model.ask(sample_messages, with_reasoning=True)

        assert result == ("reasoning", "content")

    def test_ask_with_params(self, sample_messages):
        """Test ask method passes params to _get_response."""

        def response_func(messages, **params):
            return f"param value: {params.get('test_param')}"

        model = FakeLLMModel().response_always(response_func)

        result = model.ask(sample_messages, test_param="test_value")

        assert result == "param value: test_value"

    def test_iter_per_words_content_only(self, mock_jieba_cut):
        """Test _iter_per_words with content only."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.return_value = ["Hello", "world"]

        with patch('time.sleep') as mock_sleep:
            chunks = list(model._iter_per_words("Hello world"))

        expected_chunks = [(None, "Hello"), (None, "world")]
        assert chunks == expected_chunks
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1 / 100)  # stream_wps = 100

    def test_iter_per_words_reasoning_and_content(self, mock_jieba_cut):
        """Test _iter_per_words with both reasoning and content."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.side_effect = [["Think", "about"], ["Hello", "world"]]

        with patch('time.sleep') as mock_sleep:
            chunks = list(model._iter_per_words("Hello world", "Think about"))

        expected_chunks = [("Think", None), ("about", None), (None, "Hello"), (None, "world")]
        assert chunks == expected_chunks
        assert mock_sleep.call_count == 4

    def test_iter_per_words_empty_words_filtered(self, mock_jieba_cut):
        """Test _iter_per_words filters out empty words."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.return_value = ["Hello", "", "world", ""]

        with patch('time.sleep'):
            chunks = list(model._iter_per_words("Hello world"))

        expected_chunks = [(None, "Hello"), (None, "world")]
        assert chunks == expected_chunks

    def test_iter_per_words_empty_content(self, mock_jieba_cut):
        """Test _iter_per_words with empty content."""
        model = FakeLLMModel(stream_wps=100)
        chunks = list(model._iter_per_words(""))
        assert chunks == []

    def test_iter_per_words_empty_reasoning(self, mock_jieba_cut):
        """Test _iter_per_words with empty reasoning content."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.return_value = ["Hello"]

        with patch('time.sleep'):
            chunks = list(model._iter_per_words("Hello", ""))

        expected_chunks = [(None, "Hello")]
        assert chunks == expected_chunks

    def test_iter_per_words_none_reasoning(self, mock_jieba_cut):
        """Test _iter_per_words with None reasoning content."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.return_value = ["Hello"]

        with patch('time.sleep'):
            chunks = list(model._iter_per_words("Hello", None))

        expected_chunks = [(None, "Hello")]
        assert chunks == expected_chunks

    def test_iter_per_words_none_content(self, mock_jieba_cut):
        """Test _iter_per_words with None content."""
        model = FakeLLMModel(stream_wps=100)
        mock_jieba_cut.return_value = ["Think"]

        with patch('time.sleep'):
            chunks = list(model._iter_per_words(None, "Think"))

        expected_chunks = [("Think", None)]
        assert chunks == expected_chunks

    def test_iter_per_words_stream_wps_timing(self, mock_jieba_cut):
        """Test _iter_per_words uses correct timing based on stream_wps."""
        model = FakeLLMModel(stream_wps=50)
        mock_jieba_cut.return_value = ["Hello"]

        with patch('time.sleep') as mock_sleep:
            list(model._iter_per_words("Hello"))

        mock_sleep.assert_called_with(1 / 50)

    def test_ask_stream_returns_fake_response_stream(self, sample_messages):
        """Test ask_stream returns FakeResponseStream."""
        model = FakeLLMModel().response_always("test response")

        stream = model.ask_stream(sample_messages)

        assert isinstance(stream, FakeResponseStream)

    def test_ask_stream_with_reasoning_true(self, sample_messages):
        """Test ask_stream with with_reasoning=True."""
        model = FakeLLMModel().response_always(("reasoning", "content"))

        stream = model.ask_stream(sample_messages, with_reasoning=True)

        assert stream._with_reasoning is True

    def test_ask_stream_with_reasoning_false(self, sample_messages):
        """Test ask_stream with with_reasoning=False."""
        model = FakeLLMModel().response_always("content")

        stream = model.ask_stream(sample_messages, with_reasoning=False)

        assert stream._with_reasoning is False

    def test_method_chaining(self):
        """Test method chaining works correctly."""
        result = (FakeLLMModel()
                  .response_when_keyword_in_last_message("hello", "hi response")
                  .response_when_keyword_in_last_message("weather", "weather response")
                  .response_always("default response"))

        assert result.rules_count == 3

    def test_rule_priority_order(self):
        """Test that rules are checked in the order they were added."""
        model = (FakeLLMModel()
                 .response_when_keyword_in_last_message("test", "first match")
                 .response_always("second match"))

        messages = [{"role": "user", "content": "this is a test message"}]
        result = model.ask(messages)

        assert result == "first match"

    def test_repr_default_stream_wps(self):
        """Test __repr__ with default stream_wps."""
        model = FakeLLMModel()
        result = repr(model)
        assert result == "FakeLLMModel(stream_wps=50, rules_count=0)"

    def test_repr_custom_stream_wps(self):
        """Test __repr__ with custom stream_wps."""
        model = FakeLLMModel(stream_wps=100)
        result = repr(model)
        assert result == "FakeLLMModel(stream_wps=100, rules_count=0)"

    def test_repr_with_rules(self):
        """Test __repr__ with rules added."""
        model = FakeLLMModel(stream_wps=100).response_always("response1").response_always("response2")
        result = repr(model)
        assert result == "FakeLLMModel(stream_wps=100, rules_count=2)"

    def test_params_method_basic(self):
        """Test _params method returns correct parameters."""
        model = FakeLLMModel(stream_wps=100)
        params = model._params()

        assert isinstance(params, tuple)
        assert params[0] == 100  # stream_wps
        assert params[1] == tuple()  # empty rules

    def test_params_method_with_rules(self):
        """Test _params method with rules."""

        def custom_rule(messages, **params):
            return True

        model = FakeLLMModel(stream_wps=100).response_when(custom_rule, "response")
        params = model._params()

        assert params[0] == 100
        assert len(params[1]) == 1  # one rule

        rule_key, response_key = params[1][0]
        assert rule_key[0] == id(custom_rule)  # function id
        assert rule_key[1] == str(custom_rule)  # function string
        assert response_key == ('value', 'response')

    def test_params_method_with_callable_response(self):
        """Test _params method with callable response."""

        def response_func(messages, **params):
            return "response"

        model = FakeLLMModel().response_always(response_func)
        params = model._params()

        rule_key, response_key = params[1][0]
        assert response_key[0] == 'callable'
        assert response_key[1] == id(response_func)
        assert response_key[2] == str(response_func)

    def test_params_method_with_tuple_response(self):
        """Test _params method with tuple response."""
        model = FakeLLMModel().response_always(("reasoning", "content"))
        params = model._params()

        rule_key, response_key = params[1][0]
        assert response_key == ('tuple', ('reasoning', 'content'))

    def test_params_method_with_list_response(self):
        """Test _params method with list response."""
        model = FakeLLMModel().response_always(["reasoning", "content"])
        params = model._params()

        rule_key, response_key = params[1][0]
        assert response_key == ('tuple', ('reasoning', 'content'))

    def test_params_method_multiple_rules(self):
        """Test _params method with multiple rules."""

        def rule1(messages, **params):
            return True

        def rule2(messages, **params):
            return False

        model = (FakeLLMModel(stream_wps=50)
                 .response_when(rule1, "response1")
                 .response_when(rule2, ("reasoning", "response2")))
        params = model._params()

        assert params[0] == 50
        assert len(params[1]) == 2

    def test_equality_same_instance(self):
        """Test equality with same instance."""
        model = FakeLLMModel(stream_wps=100)
        assert model == model

    def test_equality_different_instances_same_params(self):
        """Test equality with different instances but same parameters."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=100)
        assert model1 == model2

    def test_equality_different_stream_wps(self):
        """Test inequality with different stream_wps."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=200)
        assert model1 != model2

    def test_equality_different_rules(self):
        """Test inequality with different rules."""
        model1 = FakeLLMModel().response_always("response1")
        model2 = FakeLLMModel().response_always("response2")
        assert model1 != model2

    def test_equality_same_rules_different_order(self):
        """Test inequality with same rules in different order."""
        model1 = (FakeLLMModel()
                  .response_always("response1")
                  .response_always("response2"))
        model2 = (FakeLLMModel()
                  .response_always("response2")
                  .response_always("response1"))
        assert model1 != model2

    def test_equality_with_non_llm_model(self):
        """Test inequality with non-LLMModel object."""
        model = FakeLLMModel()
        assert model != "not a model"
        assert model != 42
        assert model != None

    def test_equality_with_different_llm_model_type(self):
        """Test inequality with different LLMModel subclass."""
        from hbllmutils.model.base import LLMModel

        class OtherLLMModel(LLMModel):
            @property
            def _logger_name(self):
                return "other"

            def ask(self, messages, with_reasoning=False, **params):
                return "response"

            def ask_stream(self, messages, with_reasoning=False, **params):
                pass

            def _params(self):
                return (100,)

        fake_model = FakeLLMModel(stream_wps=100)
        other_model = OtherLLMModel()
        assert fake_model != other_model

    def test_hash_same_instances(self):
        """Test hash is same for instances with same parameters."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=100)
        assert hash(model1) == hash(model2)

    def test_hash_different_instances(self):
        """Test hash is different for instances with different parameters."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=200)
        assert hash(model1) != hash(model2)

    def test_hash_with_rules(self):
        """Test hash with rules."""
        model1 = FakeLLMModel().response_always("response")
        model2 = FakeLLMModel().response_always("response")
        # Note: These may not be equal due to different function objects
        # but they should be hashable
        assert isinstance(hash(model1), int)
        assert isinstance(hash(model2), int)

    def test_hash_consistency(self):
        """Test hash consistency - same object should have same hash."""
        model = FakeLLMModel(stream_wps=100).response_always("test")
        hash1 = hash(model)
        hash2 = hash(model)
        assert hash1 == hash2

    def test_hashable_in_set(self):
        """Test that FakeLLMModel instances can be used in sets."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=200)
        model3 = FakeLLMModel(stream_wps=100)

        model_set = {model1, model2, model3}
        # model1 and model3 should be considered equal, so set should have 2 elements
        assert len(model_set) == 2

    def test_hashable_in_dict(self):
        """Test that FakeLLMModel instances can be used as dict keys."""
        model1 = FakeLLMModel(stream_wps=100)
        model2 = FakeLLMModel(stream_wps=200)

        model_dict = {model1: "value1", model2: "value2"}
        assert len(model_dict) == 2
        assert model_dict[model1] == "value1"
        assert model_dict[model2] == "value2"
