import io


class ResponseStream:
    def __init__(self, session, with_reasoning: bool = False,
                 reasoning_splitter: str = '---------------------------reasoning---------------------------',
                 content_splitter: str = '---------------------------content---------------------------'):
        self.session = session
        self._with_reasoning = with_reasoning
        self._reasoning_splitter = reasoning_splitter
        self._content_splitter = content_splitter

        self._reasoning_content = None
        self._content = None
        self._iter_status = 'none'

    def __iter__(self):
        if self._iter_status != 'none':
            raise RuntimeError('Stream already entered or ended.')
        else:
            self._iter_status = 'entered'

        status = 'none'
        with io.StringIO() as _s_reasoning, io.StringIO() as _s_content:
            for chunk in self.session:
                delta = chunk.choices[0].delta
                if self._with_reasoning and getattr(delta, 'reasoning_content'):
                    if self._with_reasoning and status != 'reasoning':
                        if status != 'none':
                            yield '\n\n'
                        yield f'{self._reasoning_splitter}\n\n'
                        status = 'reasoning'
                    yield delta.reasoning_content
                if getattr(delta, 'reasoning_content'):
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
        return self._iter_status == 'entered'

    @property
    def is_ended(self) -> bool:
        return self._iter_status == 'ended'

    @property
    def reasoning_content(self) -> str:
        return self._reasoning_content

    @property
    def content(self) -> str:
        return self._content
