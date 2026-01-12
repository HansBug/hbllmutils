from typing import Any, Optional
from unittest.mock import Mock

import pytest

from hbllmutils.model import ResponseStream
from hbllmutils.model.stream import DEFAULT_REASONING_SPLITTER, DEFAULT_CONTENT_SPLITTER, OpenAIResponseStream


@pytest.fixture
def mock_session():
    """Mock session that yields chunks."""
    session = Mock()
    session.__iter__ = Mock(return_value=iter([]))
    return session


@pytest.fixture
def mock_chunk_with_reasoning():
    """Mock chunk with reasoning content."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.reasoning_content = "reasoning text"
    chunk.choices[0].delta.content = "regular text"
    return chunk


@pytest.fixture
def mock_chunk_without_reasoning():
    """Mock chunk without reasoning content."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].delta.content = "regular text"
    return chunk


@pytest.fixture
def mock_chunk_empty_choices():
    """Mock chunk with empty choices."""
    chunk = Mock()
    chunk.choices = []
    return chunk


@pytest.fixture
def mock_chunk_no_content():
    """Mock chunk with no content."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].delta.content = None
    return chunk


@pytest.fixture
def session_with_chunks(mock_chunk_with_reasoning, mock_chunk_without_reasoning):
    """Mock session with multiple chunks."""
    session = Mock()
    session.__iter__ = Mock(return_value=iter([mock_chunk_with_reasoning, mock_chunk_without_reasoning]))
    return session


@pytest.fixture
def session_with_mixed_chunks():
    """Mock session with mixed chunk types."""
    chunks = []

    # Chunk with only reasoning
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.reasoning_content = "thinking..."
    chunk1.choices[0].delta.content = None
    chunks.append(chunk1)

    # Chunk with only content
    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.reasoning_content = None
    chunk2.choices[0].delta.content = "hello"
    chunks.append(chunk2)

    # Chunk with both
    chunk3 = Mock()
    chunk3.choices = [Mock()]
    chunk3.choices[0].delta = Mock()
    chunk3.choices[0].delta.reasoning_content = "more thinking"
    chunk3.choices[0].delta.content = "world"
    chunks.append(chunk3)

    session = Mock()
    session.__iter__ = Mock(return_value=iter(chunks))
    return session


class ConcreteResponseStream(ResponseStream):
    """Concrete implementation for testing."""

    def _get_reasoning_content_from_chunk(self, chunk: Any) -> Optional[str]:
        return getattr(chunk, 'reasoning_content', None)

    def _get_content_from_chunk(self, chunk: Any) -> Optional[str]:
        return getattr(chunk, 'content', None)


@pytest.mark.unittest
class TestResponseStream:

    def test_init_default_params(self, mock_session):
        """Test initialization with default parameters."""
        stream = ConcreteResponseStream(mock_session)
        assert stream.session == mock_session
        assert stream._with_reasoning is False
        assert stream._reasoning_splitter == DEFAULT_REASONING_SPLITTER
        assert stream._content_splitter == DEFAULT_CONTENT_SPLITTER
        assert stream._reasoning_content is None
        assert stream._content is None
        assert stream._iter_status == 'none'

    def test_init_custom_params(self, mock_session):
        """Test initialization with custom parameters."""
        custom_reasoning_splitter = "---reasoning---"
        custom_content_splitter = "---content---"

        stream = ConcreteResponseStream(
            mock_session,
            with_reasoning=True,
            reasoning_splitter=custom_reasoning_splitter,
            content_splitter=custom_content_splitter
        )

        assert stream.session == mock_session
        assert stream._with_reasoning is True
        assert stream._reasoning_splitter == custom_reasoning_splitter
        assert stream._content_splitter == custom_content_splitter

    def test_get_reasoning_content_from_chunk_not_implemented(self, mock_session):
        """Test that base class raises NotImplementedError for reasoning content."""
        stream = ResponseStream(mock_session)
        with pytest.raises(NotImplementedError):
            stream._get_reasoning_content_from_chunk(Mock())

    def test_get_content_from_chunk_not_implemented(self, mock_session):
        """Test that base class raises NotImplementedError for content."""
        stream = ResponseStream(mock_session)
        with pytest.raises(NotImplementedError):
            stream._get_content_from_chunk(Mock())

    def test_iter_empty_session(self, mock_session):
        """Test iteration with empty session."""
        stream = ConcreteResponseStream(mock_session)
        chunks = list(stream)
        assert chunks == []
        assert stream.is_ended
        assert stream._reasoning_content == ""
        assert stream._content == ""

    def test_iter_already_entered(self, mock_session):
        """Test that entering stream twice raises RuntimeError."""
        stream = ConcreteResponseStream(mock_session)
        stream._iter_status = 'entered'

        with pytest.raises(RuntimeError, match='Stream already entered or ended.'):
            list(stream)

    def test_iter_already_ended(self, mock_session):
        """Test that entering ended stream raises RuntimeError."""
        stream = ConcreteResponseStream(mock_session)
        stream._iter_status = 'ended'

        with pytest.raises(RuntimeError, match='Stream already entered or ended.'):
            list(stream)

    def test_iter_with_content_only(self, mock_session):
        """Test iteration with content only."""
        chunk = Mock()
        chunk.reasoning_content = None
        chunk.content = "hello world"

        mock_session.__iter__ = Mock(return_value=iter([chunk]))
        stream = ConcreteResponseStream(mock_session)

        chunks = list(stream)
        assert chunks == ["hello world"]
        assert stream._content == "hello world"
        assert stream._reasoning_content == ""

    def test_iter_with_reasoning_disabled(self, mock_session):
        """Test iteration with reasoning content but reasoning disabled."""
        chunk = Mock()
        chunk.reasoning_content = "thinking"
        chunk.content = "hello"

        mock_session.__iter__ = Mock(return_value=iter([chunk]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=False)

        chunks = list(stream)
        assert chunks == ["hello"]
        assert stream._content == "hello"
        assert stream._reasoning_content == "thinking"

    def test_iter_with_reasoning_enabled(self, mock_session):
        """Test iteration with reasoning enabled."""
        chunk = Mock()
        chunk.reasoning_content = "thinking"
        chunk.content = "hello"

        mock_session.__iter__ = Mock(return_value=iter([chunk]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        expected = [
            f"{DEFAULT_REASONING_SPLITTER}\n\n",
            "thinking",
            "\n\n",
            f"{DEFAULT_CONTENT_SPLITTER}\n\n",
            "hello"
        ]
        assert chunks == expected
        assert stream._content == "hello"
        assert stream._reasoning_content == "thinking"

    def test_iter_reasoning_to_content_transition(self, mock_session):
        """Test transition from reasoning to content."""
        chunk1 = Mock()
        chunk1.reasoning_content = "thinking"
        chunk1.content = None

        chunk2 = Mock()
        chunk2.reasoning_content = None
        chunk2.content = "hello"

        mock_session.__iter__ = Mock(return_value=iter([chunk1, chunk2]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        expected = [
            f"{DEFAULT_REASONING_SPLITTER}\n\n",
            "thinking",
            "\n\n",
            f"{DEFAULT_CONTENT_SPLITTER}\n\n",
            "hello"
        ]
        assert chunks == expected

    def test_iter_content_to_reasoning_transition(self, mock_session):
        """Test transition from content to reasoning."""
        chunk1 = Mock()
        chunk1.reasoning_content = None
        chunk1.content = "hello"

        chunk2 = Mock()
        chunk2.reasoning_content = "thinking"
        chunk2.content = None

        mock_session.__iter__ = Mock(return_value=iter([chunk1, chunk2]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        expected = [
            f"{DEFAULT_CONTENT_SPLITTER}\n\n",
            "hello",
            "\n\n",
            f"{DEFAULT_REASONING_SPLITTER}\n\n",
            "thinking"
        ]
        assert chunks == expected

    def test_iter_multiple_reasoning_chunks(self, mock_session):
        """Test multiple consecutive reasoning chunks."""
        chunk1 = Mock()
        chunk1.reasoning_content = "thinking1"
        chunk1.content = None

        chunk2 = Mock()
        chunk2.reasoning_content = "thinking2"
        chunk2.content = None

        mock_session.__iter__ = Mock(return_value=iter([chunk1, chunk2]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        expected = [
            f"{DEFAULT_REASONING_SPLITTER}\n\n",
            "thinking1",
            "thinking2"
        ]
        assert chunks == expected
        assert stream._reasoning_content == "thinking1thinking2"

    def test_iter_multiple_content_chunks(self, mock_session):
        """Test multiple consecutive content chunks."""
        chunk1 = Mock()
        chunk1.reasoning_content = None
        chunk1.content = "hello"

        chunk2 = Mock()
        chunk2.reasoning_content = None
        chunk2.content = " world"

        mock_session.__iter__ = Mock(return_value=iter([chunk1, chunk2]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        expected = [
            f"{DEFAULT_CONTENT_SPLITTER}\n\n",
            "hello",
            " world"
        ]
        assert chunks == expected
        assert stream._content == "hello world"

    def test_iter_no_content_chunks(self, mock_session):
        """Test chunks with no content."""
        chunk = Mock()
        chunk.reasoning_content = None
        chunk.content = None

        mock_session.__iter__ = Mock(return_value=iter([chunk]))
        stream = ConcreteResponseStream(mock_session, with_reasoning=True)

        chunks = list(stream)
        assert chunks == []
        assert stream._content == ""
        assert stream._reasoning_content == ""

    def test_is_entered_property(self, mock_session):
        """Test is_entered property."""
        stream = ConcreteResponseStream(mock_session)
        assert not stream.is_entered

        stream._iter_status = 'entered'
        assert stream.is_entered

        stream._iter_status = 'ended'
        assert not stream.is_entered

    def test_is_ended_property(self, mock_session):
        """Test is_ended property."""
        stream = ConcreteResponseStream(mock_session)
        assert not stream.is_ended

        stream._iter_status = 'entered'
        assert not stream.is_ended

        stream._iter_status = 'ended'
        assert stream.is_ended

    def test_reasoning_content_property(self, mock_session):
        """Test reasoning_content property."""
        stream = ConcreteResponseStream(mock_session)
        assert stream.reasoning_content is None

        stream._reasoning_content = "test reasoning"
        assert stream.reasoning_content == "test reasoning"

    def test_content_property(self, mock_session):
        """Test content property."""
        stream = ConcreteResponseStream(mock_session)
        assert stream.content is None

        stream._content = "test content"
        assert stream.content == "test content"

    def test_custom_splitters(self, mock_session):
        """Test custom splitters."""
        chunk = Mock()
        chunk.reasoning_content = "thinking"
        chunk.content = "hello"

        mock_session.__iter__ = Mock(return_value=iter([chunk]))
        stream = ConcreteResponseStream(
            mock_session,
            with_reasoning=True,
            reasoning_splitter="---R---",
            content_splitter="---C---"
        )

        chunks = list(stream)
        expected = [
            "---R---\n\n",
            "thinking",
            "\n\n",
            "---C---\n\n",
            "hello"
        ]
        assert chunks == expected


@pytest.mark.unittest
class TestOpenAIResponseStream:

    def test_init(self, mock_session):
        """Test OpenAI stream initialization."""
        stream = OpenAIResponseStream(mock_session)
        assert stream.session == mock_session
        assert isinstance(stream, ResponseStream)

    def test_get_reasoning_content_from_chunk_with_content(self, mock_chunk_with_reasoning):
        """Test extracting reasoning content from chunk with content."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_reasoning_content_from_chunk(mock_chunk_with_reasoning)
        assert content == "reasoning text"

    def test_get_reasoning_content_from_chunk_without_content(self, mock_chunk_without_reasoning):
        """Test extracting reasoning content from chunk without reasoning content."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_reasoning_content_from_chunk(mock_chunk_without_reasoning)
        assert content is None

    def test_get_reasoning_content_from_chunk_empty_choices(self, mock_chunk_empty_choices):
        """Test extracting reasoning content from chunk with empty choices."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_reasoning_content_from_chunk(mock_chunk_empty_choices)
        assert content is None

    def test_get_reasoning_content_from_chunk_no_attribute(self):
        """Test extracting reasoning content when attribute doesn't exist."""
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta = Mock(spec=[])  # No reasoning_content attribute

        stream = OpenAIResponseStream(Mock())
        content = stream._get_reasoning_content_from_chunk(chunk)
        assert content is None

    def test_get_content_from_chunk_with_content(self, mock_chunk_with_reasoning):
        """Test extracting regular content from chunk."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_content_from_chunk(mock_chunk_with_reasoning)
        assert content == "regular text"

    def test_get_content_from_chunk_no_content(self, mock_chunk_no_content):
        """Test extracting content from chunk with no content."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_content_from_chunk(mock_chunk_no_content)
        assert content is None

    def test_get_content_from_chunk_empty_choices(self, mock_chunk_empty_choices):
        """Test extracting content from chunk with empty choices."""
        stream = OpenAIResponseStream(Mock())
        content = stream._get_content_from_chunk(mock_chunk_empty_choices)
        assert content is None

    def test_full_iteration_with_openai_chunks(self, session_with_chunks):
        """Test full iteration with OpenAI-style chunks."""
        stream = OpenAIResponseStream(session_with_chunks, with_reasoning=True)
        chunks = list(stream)

        expected = [
            f"{DEFAULT_REASONING_SPLITTER}\n\n",
            "reasoning text",
            "\n\n",
            f"{DEFAULT_CONTENT_SPLITTER}\n\n",
            "regular text",
            "regular text"
        ]
        assert chunks == expected
        assert stream.reasoning_content == "reasoning text"
        assert stream.content == "regular textregular text"

    def test_openai_stream_without_reasoning(self, session_with_chunks):
        """Test OpenAI stream without reasoning enabled."""
        stream = OpenAIResponseStream(session_with_chunks, with_reasoning=False)
        chunks = list(stream)

        assert chunks == ["regular text", "regular text"]
        assert stream.reasoning_content == "reasoning text"
        assert stream.content == "regular textregular text"
