"""
This module provides a streaming response handler for processing session outputs with optional reasoning content.

The ResponseStream class manages the iteration over session responses, separating reasoning content
from regular content, and providing access to both after streaming is complete.

Key Features:
    - Stream processing of session responses
    - Optional separation of reasoning and regular content
    - Configurable content splitters
    - Post-stream access to accumulated content

Example::
    >>> import sys
    >>> stream = ResponseStream(session, with_reasoning=True)
    >>> for chunk in stream:
    ...     print(chunk, end='')
    ...     sys.stdout.flush()
    >>> print(f"Reasoning: {stream.reasoning_content}")
    >>> print(f"Content: {stream.content}")
"""

import io
from typing import Iterator

DEFAULT_REASONING_SPLITTER: str = '---------------------------reasoning---------------------------'
DEFAULT_CONTENT_SPLITTER: str = '---------------------------content---------------------------'


class ResponseStream:
    """
    A stream handler for processing session responses with optional reasoning content separation.
    
    This class wraps a session object and provides an iterator interface to stream response chunks,
    optionally separating reasoning content from regular content with configurable splitters.
    
    The stream maintains internal state to track iteration progress and accumulates both reasoning
    and regular content for post-iteration access through properties.
    """

    def __init__(self, session, with_reasoning: bool = False,
                 reasoning_splitter: str = DEFAULT_REASONING_SPLITTER,
                 content_splitter: str = DEFAULT_CONTENT_SPLITTER):
        """
        Initialize the ResponseStream.
        
        :param session: The session object to stream responses from. Must support iteration
                       and yield chunks with choices[0].delta attributes.
        :param with_reasoning: Whether to include reasoning content in the stream output, defaults to False.
                              When True, reasoning content will be prefixed with the reasoning_splitter.
        :type with_reasoning: bool
        :param reasoning_splitter: The separator string for reasoning content sections, defaults to a dashed line.
        :type reasoning_splitter: str
        :param content_splitter: The separator string for regular content sections, defaults to a dashed line.
        :type content_splitter: str
        
        Example::
            >>> stream = ResponseStream(session)
            >>> # Stream without reasoning
            >>> for chunk in stream:
            ...     print(chunk, end='')
            
            >>> stream_with_reasoning = ResponseStream(session, with_reasoning=True)
            >>> # Stream with reasoning separated by splitters
            >>> for chunk in stream_with_reasoning:
            ...     print(chunk, end='')
        """
        self.session = session
        self._with_reasoning = with_reasoning
        self._reasoning_splitter = reasoning_splitter
        self._content_splitter = content_splitter

        self._reasoning_content = None
        self._content = None
        self._iter_status = 'none'

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the session responses, yielding content chunks.
        
        This method streams response chunks from the session, separating reasoning content
        from regular content when applicable. It accumulates both types of content internally
        for later access via properties.
        
        The iteration process:
        1. Checks if stream has already been used
        2. Iterates through session chunks
        3. Extracts reasoning_content and content from delta objects
        4. Yields content with appropriate splitters when transitioning between content types
        5. Accumulates all content for post-iteration access
        
        :return: An iterator yielding string chunks of content.
        :rtype: Iterator[str]
        :raises RuntimeError: If the stream has already been entered or ended.
        
        Example::
            >>> import sys
            >>> stream = ResponseStream(session, with_reasoning=True)
            >>> for chunk in stream:
            ...     print(chunk, end='')
            ...     sys.stdout.flush()  # need to flush it
            ---------------------------reasoning---------------------------
            
            This is reasoning content...
            
            ---------------------------content---------------------------
            
            This is regular content...
            
            >>> # After iteration, access accumulated content
            >>> print(stream.reasoning_content)
            This is reasoning content...
            >>> print(stream.content)
            This is regular content...
        """
        if self._iter_status != 'none':
            raise RuntimeError('Stream already entered or ended.')
        else:
            self._iter_status = 'entered'

        status = 'none'
        with io.StringIO() as _s_reasoning, io.StringIO() as _s_content:
            for chunk in self.session:
                delta = chunk.choices[0].delta
                # Handle reasoning content
                if self._with_reasoning and getattr(delta, 'reasoning_content', None):
                    if self._with_reasoning and status != 'reasoning':
                        if status != 'none':
                            yield '\n\n'
                        yield f'{self._reasoning_splitter}\n\n'
                        status = 'reasoning'
                    yield delta.reasoning_content
                if getattr(delta, 'reasoning_content', None):
                    _s_reasoning.write(delta.reasoning_content)

                # Handle regular content
                if delta.content is not None:
                    if self._with_reasoning and status != 'content':
                        if status != 'none':
                            yield '\n\n'
                        yield f'{self._content_splitter}\n\n'
                        status = 'content'
                    yield delta.content
                    _s_content.write(delta.content)

            self._reasoning_content = _s_reasoning.getvalue()
            self._content = _s_content.getvalue()
            self._iter_status = 'ended'

    @property
    def is_entered(self) -> bool:
        """
        Check if the stream has been entered (iteration started).
        
        :return: True if iteration has started, False otherwise.
        :rtype: bool
        
        Example::
            >>> stream = ResponseStream(session)
            >>> stream.is_entered
            False
            >>> iter(stream)
            >>> stream.is_entered
            True
        """
        return self._iter_status == 'entered'

    @property
    def is_ended(self) -> bool:
        """
        Check if the stream has ended (iteration completed).
        
        :return: True if iteration has completed, False otherwise.
        :rtype: bool
        
        Example::
            >>> stream = ResponseStream(session)
            >>> stream.is_ended
            False
            >>> for chunk in stream:
            ...     pass
            >>> stream.is_ended
            True
        """
        return self._iter_status == 'ended'

    @property
    def reasoning_content(self) -> str:
        """
        Get the accumulated reasoning content from the stream.
        
        This property is only available after the stream has been fully consumed.
        The reasoning content includes all text that was marked as reasoning_content
        in the session's delta objects.
        
        :return: The complete reasoning content, or None if not yet available.
        :rtype: str
        
        Example::
            >>> stream = ResponseStream(session, with_reasoning=True)
            >>> for chunk in stream:
            ...     pass
            >>> reasoning = stream.reasoning_content
            >>> print(reasoning)
            This is the reasoning content...
        """
        return self._reasoning_content

    @property
    def content(self) -> str:
        """
        Get the accumulated regular content from the stream.
        
        This property is only available after the stream has been fully consumed.
        The content includes all text that was marked as regular content
        in the session's delta objects.
        
        :return: The complete regular content, or None if not yet available.
        :rtype: str
        
        Example::
            >>> stream = ResponseStream(session)
            >>> for chunk in stream:
            ...     pass
            >>> regular_content = stream.content
            >>> print(regular_content)
            This is the regular content...
        """
        return self._content
