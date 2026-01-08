"""
This module provides a streaming response handler for processing session outputs with optional reasoning content.

The ResponseStream class manages the iteration over session responses, separating reasoning content
from regular content, and providing access to both after streaming is complete.
"""

import io


class ResponseStream:
    """
    A stream handler for processing session responses with optional reasoning content separation.
    
    This class wraps a session object and provides an iterator interface to stream response chunks,
    optionally separating reasoning content from regular content with configurable splitters.
    """

    def __init__(self, session, with_reasoning: bool = False,
                 reasoning_splitter: str = '---------------------------reasoning---------------------------',
                 content_splitter: str = '---------------------------content---------------------------'):
        """
        Initialize the ResponseStream.
        
        :param session: The session object to stream responses from.
        :param with_reasoning: Whether to include reasoning content in the stream output, defaults to False.
        :type with_reasoning: bool
        :param reasoning_splitter: The separator string for reasoning content sections, defaults to a dashed line.
        :type reasoning_splitter: str
        :param content_splitter: The separator string for regular content sections, defaults to a dashed line.
        :type content_splitter: str
        """
        self.session = session
        self._with_reasoning = with_reasoning
        self._reasoning_splitter = reasoning_splitter
        self._content_splitter = content_splitter

        self._reasoning_content = None
        self._content = None
        self._iter_status = 'none'

    def __iter__(self):
        """
        Iterate over the session responses, yielding content chunks.
        
        This method streams response chunks from the session, separating reasoning content
        from regular content when applicable. It accumulates both types of content internally
        for later access via properties.
        
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
        """
        if self._iter_status != 'none':
            raise RuntimeError('Stream already entered or ended.')
        else:
            self._iter_status = 'entered'

        status = 'none'
        with io.StringIO() as _s_reasoning, io.StringIO() as _s_content:
            for chunk in self.session:
                delta = chunk.choices[0].delta
                if self._with_reasoning and getattr(delta, 'reasoning_content', None):
                    if self._with_reasoning and status != 'reasoning':
                        if status != 'none':
                            yield '\n\n'
                        yield f'{self._reasoning_splitter}\n\n'
                        status = 'reasoning'
                    yield delta.reasoning_content
                if getattr(delta, 'reasoning_content', None):
                    _s_reasoning.write(delta.reasoning_content)

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
        """
        return self._iter_status == 'entered'

    @property
    def is_ended(self) -> bool:
        """
        Check if the stream has ended (iteration completed).
        
        :return: True if iteration has completed, False otherwise.
        :rtype: bool
        """
        return self._iter_status == 'ended'

    @property
    def reasoning_content(self) -> str:
        """
        Get the accumulated reasoning content from the stream.
        
        This property is only available after the stream has been fully consumed.
        
        :return: The complete reasoning content, or None if not yet available.
        :rtype: str
        """
        return self._reasoning_content

    @property
    def content(self) -> str:
        """
        Get the accumulated regular content from the stream.
        
        This property is only available after the stream has been fully consumed.
        
        :return: The complete regular content, or None if not yet available.
        :rtype: str
        """
        return self._content
