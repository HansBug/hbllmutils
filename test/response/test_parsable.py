import logging
from unittest.mock import Mock

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.model import FakeLLMModel, LLMTask
from hbllmutils.response import ParsableLLMTask, OutputParseWithException, OutputParseFailed


@pytest.fixture
def fake_model():
    """Create a basic fake LLM model for testing."""
    return FakeLLMModel()


@pytest.fixture
def history():
    """Create an empty LLM history for testing."""
    return LLMHistory()


@pytest.fixture
def model_with_responses():
    """Create a fake model with predefined responses."""
    return FakeLLMModel().response_sequence([
        "invalid_json",
        "still_invalid",
        '{"valid": "json"}'
    ])


@pytest.fixture
def model_always_invalid():
    """Create a fake model that always returns invalid responses."""
    return FakeLLMModel().response_always("invalid_response")


@pytest.fixture
def model_always_valid():
    """Create a fake model that always returns valid responses."""
    return FakeLLMModel().response_always('{"valid": "json"}')


@pytest.fixture
def logger_mock():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


class TestParsableTask(ParsableLLMTask):
    """Test implementation of ParsableLLMTask for JSON parsing."""

    def _parse_and_validate(self, content: str):
        import json
        return json.loads(content)


class TestParsableTaskWithCustomExceptions(ParsableLLMTask):
    """Test implementation with custom exception types."""
    __exceptions__ = (ValueError, KeyError)

    def _parse_and_validate(self, content: str):
        if content == "value_error":
            raise ValueError("Value error")
        elif content == "key_error":
            raise KeyError("Key error")
        elif content == "type_error":
            raise TypeError("Type error")
        else:
            return content


class TestParsableTaskAlwaysFails(ParsableLLMTask):
    """Test implementation that always fails parsing."""

    def _parse_and_validate(self, content: str):
        raise ValueError("Always fails")


@pytest.mark.unittest
class TestOutputParseWithException:
    """Test cases for OutputParseWithException data class."""

    def test_initialization(self):
        """Test OutputParseWithException initialization."""
        exception = ValueError("Test error")
        output = "test output"

        parse_attempt = OutputParseWithException(output=output, exception=exception)

        assert parse_attempt.output == output
        assert parse_attempt.exception == exception

    def test_attributes_access(self):
        """Test accessing attributes of OutputParseWithException."""
        exception = RuntimeError("Runtime error")
        output = "runtime output"

        parse_attempt = OutputParseWithException(output=output, exception=exception)

        assert parse_attempt.output == "runtime output"
        assert isinstance(parse_attempt.exception, RuntimeError)
        assert str(parse_attempt.exception) == "Runtime error"


@pytest.mark.unittest
class TestOutputParseFailed:
    """Test cases for OutputParseFailed exception."""

    def test_initialization(self):
        """Test OutputParseFailed initialization."""
        tries = [
            OutputParseWithException("output1", ValueError("error1")),
            OutputParseWithException("output2", ValueError("error2"))
        ]
        message = "Parsing failed"

        exception = OutputParseFailed(message, tries)

        assert str(exception) == message
        assert exception.tries == tries

    def test_empty_tries_list(self):
        """Test OutputParseFailed with empty tries list."""
        tries = []
        message = "No attempts made"

        exception = OutputParseFailed(message, tries)

        assert str(exception) == message
        assert exception.tries == []

    def test_inheritance(self):
        """Test that OutputParseFailed inherits from Exception."""
        tries = [OutputParseWithException("output", ValueError("error"))]
        exception = OutputParseFailed("test", tries)

        assert isinstance(exception, Exception)


@pytest.mark.unittest
class TestParsableLLMTask:
    """Test cases for ParsableLLMTask class."""

    def test_initialization_default_params(self, fake_model):
        """Test ParsableLLMTask initialization with default parameters."""
        task = TestParsableTask(fake_model)

        assert task.model == fake_model
        assert isinstance(task.history, LLMHistory)
        assert task.default_max_retries == 5

    def test_initialization_with_history(self, fake_model, history):
        """Test ParsableLLMTask initialization with custom history."""
        task = TestParsableTask(fake_model, history)

        assert task.model == fake_model
        assert task.history == history
        assert task.default_max_retries == 5

    def test_initialization_with_custom_retries(self, fake_model):
        """Test ParsableLLMTask initialization with custom max retries."""
        task = TestParsableTask(fake_model, default_max_retries=3)

        assert task.default_max_retries == 3

    def test_parse_and_validate_not_implemented(self, fake_model):
        """Test that _parse_and_validate raises NotImplementedError in base class."""
        task = ParsableLLMTask(fake_model)

        with pytest.raises(NotImplementedError):
            task._parse_and_validate("test content")

    def test_ask_then_parse_success_first_try(self, model_always_valid):
        """Test successful parsing on first attempt."""
        task = TestParsableTask(model_always_valid)

        result = task.ask_then_parse(input_content="test", prompt="test prompt")

        assert result == {"valid": "json"}

    def test_ask_then_parse_success_after_retries(self, model_with_responses):
        """Test successful parsing after several failed attempts."""
        task = TestParsableTask(model_with_responses)

        result = task.ask_then_parse(input_content="test", max_retries=5)

        assert result == {"valid": "json"}

    def test_ask_then_parse_failure_all_retries(self, model_always_invalid):
        """Test failure after exhausting all retry attempts."""
        task = TestParsableTask(model_always_invalid)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=3)

        exception = exc_info.value
        assert "Output parse failed after 4 tries" in str(exception)
        assert len(exception.tries) == 4

        for attempt in exception.tries:
            assert attempt.output == "invalid_response"
            assert isinstance(attempt.exception, Exception)

    def test_ask_then_parse_with_custom_max_retries(self, model_always_invalid):
        """Test ask_then_parse with custom max_retries parameter."""
        task = TestParsableTask(model_always_invalid)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=2)

        exception = exc_info.value
        assert len(exception.tries) == 3

    def test_ask_then_parse_uses_default_max_retries(self, model_always_invalid):
        """Test that ask_then_parse uses default_max_retries when max_retries is None."""
        task = TestParsableTask(model_always_invalid, default_max_retries=4)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test")

        exception = exc_info.value
        assert len(exception.tries) == 5

    def test_ask_then_parse_without_input_content(self, model_always_valid):
        """Test ask_then_parse without input_content parameter."""
        task = TestParsableTask(model_always_valid)

        result = task.ask_then_parse(prompt="test prompt")

        assert result == {"valid": "json"}

    def test_ask_then_parse_with_additional_params(self, model_always_valid):
        """Test ask_then_parse with additional parameters passed to ask method."""
        task = TestParsableTask(model_always_valid)

        result = task.ask_then_parse(
            input_content="test",
            prompt="test prompt",
            temperature=0.7,
            max_tokens=100
        )

        assert result == {"valid": "json"}

    def test_custom_exceptions_handling(self):
        """Test handling of custom exception types."""
        model = FakeLLMModel().response_sequence([
            "value_error",
            "key_error",
            "valid_response"
        ])
        task = TestParsableTaskWithCustomExceptions(model)

        result = task.ask_then_parse(input_content="test", max_retries=5)

        assert result == "valid_response"

    def test_non_matching_exception_propagates(self):
        """Test that exceptions not matching __exceptions__ propagate immediately."""
        model = FakeLLMModel().response_always("type_error")
        task = TestParsableTaskWithCustomExceptions(model)

        with pytest.raises(TypeError) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=5)

        assert "Type error" in str(exc_info.value)

    def test_default_exceptions_catches_all(self):
        """Test that default __exceptions__ = Exception catches all exceptions."""
        model = FakeLLMModel().response_always("any_error")
        task = TestParsableTaskAlwaysFails(model)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=2)

        exception = exc_info.value
        assert len(exception.tries) == 3

    def test_output_parse_failed_message_singular(self, model_always_invalid):
        """Test OutputParseFailed message with singular 'try'."""
        task = TestParsableTask(model_always_invalid)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=1)

        exception = exc_info.value
        assert "Output parse failed after 2 tries." in str(exception)

    def test_output_parse_failed_message_plural(self, model_always_invalid):
        """Test OutputParseFailed message with plural 'tries'."""
        task = TestParsableTask(model_always_invalid)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=3)

        exception = exc_info.value
        assert "Output parse failed after 4 tries." in str(exception)

    def test_inheritance_from_llm_task(self, fake_model):
        """Test that ParsableLLMTask inherits from LLMTask."""

        task = TestParsableTask(fake_model)

        assert isinstance(task, LLMTask)

    def test_sequence_responses_with_retries(self):
        """Test sequence responses with multiple retries and eventual success."""
        responses = [
            "invalid1",
            "invalid2",
            "invalid3",
            '{"success": true}'
        ]
        model = FakeLLMModel().response_sequence(responses)
        task = TestParsableTask(model)

        result = task.ask_then_parse(input_content="test", max_retries=5)

        assert result == {"success": True}

    def test_sequence_responses_exhausted_before_success(self):
        """Test when sequence responses are exhausted before finding valid response."""
        responses = ["invalid1", "invalid2"]
        model = FakeLLMModel().response_sequence(responses)
        task = TestParsableTask(model)

        with pytest.raises(AssertionError):  # No response rule found
            task.ask_then_parse(input_content="test", max_retries=5)

    def test_zero_max_retries(self, model_always_invalid):
        """Test behavior with zero max retries."""
        task = TestParsableTask(model_always_invalid)

        with pytest.raises(OutputParseFailed) as exc_info:
            task.ask_then_parse(input_content="test", max_retries=0)

        exception = exc_info.value
        assert len(exception.tries) == 1
        assert "Output parse failed after 1 try." in str(exception)

    def test_inheritance_methods_available(self, fake_model):
        """Test that inherited methods from LLMTask are available."""
        task = TestParsableTask(fake_model)

        # Test that we can access inherited methods
        assert hasattr(task, 'ask')
        assert hasattr(task, 'ask_stream')
        assert hasattr(task, '_logger')
        assert hasattr(task, 'model')
        assert hasattr(task, 'history')
