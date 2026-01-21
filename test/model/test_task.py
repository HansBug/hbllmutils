import logging

import pytest

from hbllmutils.history import LLMHistory
from hbllmutils.model import FakeLLMModel, LLMTask, ResponseStream


@pytest.fixture
def mock_model():
    """Create a mock FakeLLMModel for testing."""
    return FakeLLMModel().response_always("Test response")


@pytest.fixture
def mock_model_with_reasoning():
    """Create a mock FakeLLMModel that returns reasoning and response."""
    return FakeLLMModel().response_always(("Test reasoning", "Test response"))


@pytest.fixture
def empty_history():
    """Create an empty LLMHistory for testing."""
    return LLMHistory()


@pytest.fixture
def history_with_messages():
    """Create an LLMHistory with some messages for testing."""
    history = LLMHistory()
    history = history.with_user_message("Hello")
    history = history.with_assistant_message("Hi there")
    return history


@pytest.fixture
def mock_stream_model():
    """Create a mock FakeLLMModel for stream testing."""
    return FakeLLMModel(stream_wps=100).response_always("Stream response")


@pytest.mark.unittest
class TestLLMTask:

    def test_init_with_model_only(self, mock_model):
        """Test initialization with model only."""

        task = LLMTask(mock_model)
        assert task.model is mock_model
        assert isinstance(task.history, LLMHistory)
        assert len(task.history) == 0

    def test_init_with_model_and_history(self, mock_model, history_with_messages):
        """Test initialization with model and history."""

        task = LLMTask(mock_model, history_with_messages)
        assert task.model is mock_model
        assert task.history is history_with_messages
        assert len(task.history) == 2

    def test_logger_property(self, mock_model):
        """Test _logger property returns model's logger."""

        task = LLMTask(mock_model)
        logger = task._logger
        assert isinstance(logger, logging.Logger)
        assert logger is mock_model._logger

    def test_ask_without_reasoning(self, mock_model, history_with_messages):
        """Test ask method without reasoning."""

        task = LLMTask(mock_model, history_with_messages)
        response = task.ask()

        assert response == "Test response"
        assert isinstance(response, str)

    def test_ask_with_reasoning_false(self, mock_model_with_reasoning, history_with_messages):
        """Test ask method with reasoning=False."""

        task = LLMTask(mock_model_with_reasoning, history_with_messages)
        response = task.ask(with_reasoning=False)

        assert response == "Test response"
        assert isinstance(response, str)

    def test_ask_with_reasoning_true(self, mock_model_with_reasoning, history_with_messages):
        """Test ask method with reasoning=True."""

        task = LLMTask(mock_model_with_reasoning, history_with_messages)
        response = task.ask(with_reasoning=True)

        assert isinstance(response, tuple)
        assert len(response) == 2
        assert response[0] == "Test reasoning"
        assert response[1] == "Test response"

    def test_ask_with_params(self, mock_model, history_with_messages):
        """Test ask method with additional parameters."""

        # Create a model that can handle parameters
        def response_func(messages, **params):
            if params.get('temperature') == 0.5:
                return "Custom response"
            return "Default response"

        custom_model = FakeLLMModel().response_always(response_func)
        task = LLMTask(custom_model, history_with_messages)

        response = task.ask(temperature=0.5)
        assert response == "Custom response"

    def test_ask_stream_without_reasoning(self, mock_stream_model, history_with_messages):
        """Test ask_stream method without reasoning."""

        task = LLMTask(mock_stream_model, history_with_messages)
        stream = task.ask_stream()

        assert isinstance(stream, ResponseStream)
        # Collect all chunks
        chunks = list(stream)
        full_response = ''.join(chunks)
        assert "Stream response" in full_response

    def test_ask_stream_with_reasoning_false(self, mock_stream_model, history_with_messages):
        """Test ask_stream method with reasoning=False."""

        task = LLMTask(mock_stream_model, history_with_messages)
        stream = task.ask_stream(with_reasoning=False)

        assert isinstance(stream, ResponseStream)
        chunks = list(stream)
        full_response = ''.join(chunks)
        assert "Stream response" in full_response

    def test_ask_stream_with_reasoning_true(self, mock_stream_model, history_with_messages):
        """Test ask_stream method with reasoning=True."""

        # Create model with reasoning
        reasoning_model = FakeLLMModel(stream_wps=100).response_always(("Stream reasoning", "Stream response"))
        task = LLMTask(reasoning_model, history_with_messages)

        stream = task.ask_stream(with_reasoning=True)
        assert isinstance(stream, ResponseStream)

        # Check that we can iterate through the stream
        chunks = list(stream)
        assert len(chunks) > 0

    def test_ask_stream_with_params(self, mock_stream_model, history_with_messages):
        """Test ask_stream method with additional parameters."""

        # Create a model that can handle parameters
        def response_func(messages, **params):
            if params.get('max_tokens') == 100:
                return "Limited response"
            return "Full response"

        custom_model = FakeLLMModel(stream_wps=100).response_always(response_func)
        task = LLMTask(custom_model, history_with_messages)

        stream = task.ask_stream(max_tokens=100)
        chunks = list(stream)
        full_response = ''.join(chunks)
        assert "Limited response" in full_response

    def test_params_method(self, mock_model, history_with_messages):
        """Test _params method returns model and history."""

        task = LLMTask(mock_model, history_with_messages)
        params = task._params()

        assert isinstance(params, tuple)
        assert len(params) == 2
        assert params[0] is mock_model
        assert params[1] is history_with_messages

    def test_values_method(self, mock_model, history_with_messages):
        """Test _values method returns class and params."""

        task = LLMTask(mock_model, history_with_messages)
        values = task._values()

        assert isinstance(values, tuple)
        assert len(values) == 2
        assert values[0] is LLMTask
        assert values[1] == task._params()

    def test_equality_same_instances(self, mock_model, history_with_messages):
        """Test equality with same model and history instances."""

        task1 = LLMTask(mock_model, history_with_messages)
        task2 = LLMTask(mock_model, history_with_messages)

        assert task1 == task2

    def test_equality_different_models(self, mock_model, history_with_messages):
        """Test equality with different models."""

        different_model = FakeLLMModel().response_always("Different response")
        task1 = LLMTask(mock_model, history_with_messages)
        task2 = LLMTask(different_model, history_with_messages)

        assert task1 != task2

    def test_equality_different_histories(self, mock_model, history_with_messages, empty_history):
        """Test equality with different histories."""

        task1 = LLMTask(mock_model, history_with_messages)
        task2 = LLMTask(mock_model, empty_history)

        assert task1 != task2

    def test_equality_different_types(self, mock_model, history_with_messages):
        """Test equality with different object types."""

        task = LLMTask(mock_model, history_with_messages)

        assert task != "not a task"
        assert task != 123
        assert task != None
        assert task != mock_model

    def test_equality_same_values_different_instances(self, history_with_messages):
        """Test equality with equivalent but different instances."""

        model1 = FakeLLMModel().response_always("Same response")
        model2 = FakeLLMModel().response_always("Same response")

        # Create equivalent histories
        history1 = LLMHistory().with_user_message("Hello").with_assistant_message("Hi there")
        history2 = LLMHistory().with_user_message("Hello").with_assistant_message("Hi there")

        task1 = LLMTask(model1, history1)
        task2 = LLMTask(model2, history2)

        # They should be equal if the models and histories are equal
        assert task1 == task2

    def test_hash_consistency(self, mock_model, history_with_messages):
        """Test hash consistency."""

        task = LLMTask(mock_model, history_with_messages)
        hash1 = hash(task)
        hash2 = hash(task)

        assert hash1 == hash2

    def test_hash_equality_implies_same_hash(self, history_with_messages):
        """Test that equal objects have the same hash."""
        model1 = FakeLLMModel().response_always("Same response")
        model2 = FakeLLMModel().response_always("Same response")

        # Create equivalent histories
        history1 = LLMHistory().with_user_message("Hello").with_assistant_message("Hi there")
        history2 = LLMHistory().with_user_message("Hello").with_assistant_message("Hi there")

        task1 = LLMTask(model1, history1)
        task2 = LLMTask(model2, history2)

        if task1 == task2:
            assert hash(task1) == hash(task2)

    def test_hash_different_objects(self, mock_model, history_with_messages, empty_history):
        """Test that different objects typically have different hashes."""

        task1 = LLMTask(mock_model, history_with_messages)
        task2 = LLMTask(mock_model, empty_history)

        # Different objects should typically have different hashes
        # Note: Hash collisions are possible but unlikely
        assert hash(task1) != hash(task2)

    def test_ask_passes_history_to_model(self, history_with_messages):
        """Test that ask method passes history.to_json() to model."""

        # Create a model that captures the messages parameter
        captured_messages = None

        def capture_messages(messages, **params):
            nonlocal captured_messages
            captured_messages = messages
            return "Response"

        model = FakeLLMModel().response_always(capture_messages)
        task = LLMTask(model, history_with_messages)

        task.ask()

        assert captured_messages is not None
        assert captured_messages == history_with_messages.to_json()

    def test_ask_stream_passes_history_to_model(self, history_with_messages):
        """Test that ask_stream method passes history.to_json() to model."""

        # Create a model that captures the messages parameter
        captured_messages = None

        def capture_messages(messages, **params):
            nonlocal captured_messages
            captured_messages = messages
            return "Stream response"

        model = FakeLLMModel(stream_wps=100).response_always(capture_messages)
        task = LLMTask(model, history_with_messages)

        stream = task.ask_stream()
        # Consume the stream to trigger the model call
        list(stream)

        assert captured_messages is not None
        assert captured_messages == history_with_messages.to_json()

    def test_empty_history_initialization(self, mock_model):
        """Test that empty history is created when none provided."""

        task = LLMTask(mock_model)

        assert isinstance(task.history, LLMHistory)
        assert len(task.history) == 0
        assert task.history.to_json() == []

    def test_ask_with_empty_history(self, mock_model, empty_history):
        """Test ask method with empty history."""

        task = LLMTask(mock_model, empty_history)
        response = task.ask()

        assert response == "Test response"

    def test_ask_stream_with_empty_history(self, mock_stream_model, empty_history):
        """Test ask_stream method with empty history."""

        task = LLMTask(mock_stream_model, empty_history)
        stream = task.ask_stream()

        assert isinstance(stream, ResponseStream)
        chunks = list(stream)
        full_response = ''.join(chunks)
        assert "Stream response" in full_response
