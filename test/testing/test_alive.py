import pytest

from hbllmutils.model import FakeLLMModel
from hbllmutils.testing import hello, BinaryTestResult, MultiBinaryTestResult, ping


@pytest.fixture
def mock_model():
    """Create a mock LLM model for testing."""
    model = FakeLLMModel()
    return model


@pytest.fixture
def hello_model():
    """Create a FakeLLMModel that responds to hello messages."""
    return FakeLLMModel().response_when_keyword_in_last_message('hello', 'Hello! How can I help you?')


@pytest.fixture
def empty_response_model():
    """Create a FakeLLMModel that returns empty responses."""
    return FakeLLMModel().response_always('')


@pytest.fixture
def ping_pong_model():
    """Create a FakeLLMModel that responds to ping with pong."""
    return FakeLLMModel().response_when_keyword_in_last_message('ping', 'Pong!')


@pytest.fixture
def no_pong_model():
    """Create a FakeLLMModel that doesn't respond with pong."""
    return FakeLLMModel().response_always('Hello there!')


@pytest.fixture
def case_insensitive_pong_model():
    """Create a FakeLLMModel that responds with uppercase PONG."""
    return FakeLLMModel().response_when_keyword_in_last_message('ping', 'PONG!')


@pytest.fixture
def pong_in_sentence_model():
    """Create a FakeLLMModel that has pong within a sentence."""
    return FakeLLMModel().response_when_keyword_in_last_message('ping', 'I will respond with pong to your ping!')


@pytest.mark.unittest
class TestHelloFunction:
    def test_hello_single_test_pass(self, hello_model):
        """Test hello function with single test that passes."""

        result = hello(hello_model, n=1)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is True
        assert result.content == 'Hello! How can I help you?'

    def test_hello_single_test_fail(self, empty_response_model):
        """Test hello function with single test that fails."""

        result = hello(empty_response_model, n=1)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is False
        assert result.content == ''

    def test_hello_default_n_parameter(self, hello_model):
        """Test hello function with default n parameter."""

        result = hello(hello_model)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is True

    def test_hello_multiple_tests(self, hello_model):
        """Test hello function with multiple test runs."""

        result = hello(hello_model, n=5)

        assert isinstance(result, MultiBinaryTestResult)
        assert result.passed_count == 5
        assert result.passed_ratio == 1.0

    def test_hello_failed_tests(self, empty_response_model):
        result = hello(empty_response_model, n=5)

        assert isinstance(result, MultiBinaryTestResult)
        assert result.passed_count == 0
        assert result.passed_ratio == 0.0


@pytest.mark.unittest
class TestPingFunction:
    def test_ping_single_test_pass(self, ping_pong_model):
        """Test ping function with single test that passes."""

        result = ping(ping_pong_model, n=1)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is True
        assert result.content == 'Pong!'

    def test_ping_single_test_fail(self, no_pong_model):
        """Test ping function with single test that fails."""

        result = ping(no_pong_model, n=1)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is False
        assert result.content == 'Hello there!'

    def test_ping_default_n_parameter(self, ping_pong_model):
        """Test ping function with default n parameter."""

        result = ping(ping_pong_model)

        assert isinstance(result, BinaryTestResult)
        assert result.passed is True

    def test_ping_multiple_tests(self, ping_pong_model):
        """Test ping function with multiple test runs."""

        result = ping(ping_pong_model, n=3)

        assert isinstance(result, MultiBinaryTestResult)
        assert result.passed_count == 3
        assert result.passed_ratio == 1.0

    def test_ping_case_insensitive_matching(self):
        """Test ping function with various case combinations of pong."""

        test_cases = ['pong', 'PONG', 'Pong', 'PoNg', 'pOnG']

        for case in test_cases:
            model = FakeLLMModel().response_when_keyword_in_last_message('ping', case)

            result = ping(model, n=1)

            assert isinstance(result, BinaryTestResult)
            assert result.passed is True
            assert result.content == case
