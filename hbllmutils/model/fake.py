import time
from typing import List, Union, Tuple, Optional, Any, Callable

import jieba

from .base import LLMModel
from .stream import ResponseStream


class FakeResponseStream(ResponseStream):
    def _get_reasoning_content_from_chunk(self, chunk: Any) -> Optional[str]:
        return chunk[0]

    def _get_content_from_chunk(self, chunk: Any) -> Optional[str]:
        return chunk[1]


FakeResponseTyping = Union[str, Tuple[str, str], Callable]


def _fn_always_true(messages: List[dict], **params):
    _ = messages, params
    return True


class FakeLLMModel(LLMModel):
    def __init__(self, stream_wps: float = 50):
        self.stream_fps = stream_wps
        self._rules: List[Tuple[Callable, FakeResponseTyping]] = []

    def _get_response(self, messages: List[dict], **params) -> Tuple[str, str]:
        for fn_rule_check, fn_response in self._rules:
            if fn_rule_check(messages=messages, **params):
                if callable(fn_response):
                    retval = fn_response(messages=messages, **params)
                else:
                    retval = fn_response
                if isinstance(retval, (list, tuple)):
                    reasoning_content, content = retval
                else:
                    reasoning_content, content = '', retval
                return reasoning_content, content
        else:
            assert False, 'No response rule found for this message.'

    def response_always(self, response: FakeResponseTyping):
        self._rules.append((_fn_always_true, response))
        return self

    def response_when(self, fn_when: Callable, response: FakeResponseTyping):
        self._rules.append((fn_when, response))
        return self

    def response_when_keyword_in_last_message(self, keywords: Union[str, List[str]], response: FakeResponseTyping):
        if isinstance(keywords, (list, tuple)):
            keywords = keywords
        else:
            keywords = [keywords]

        def _fn_keyword_check(messages: List[dict], **params):
            _ = params
            for keyword in keywords:
                if keyword in messages[-1]['content']:
                    return True
            return False

        self._rules.append((_fn_keyword_check, response))
        return self

    def ask(self, messages: List[dict], with_reasoning: bool = False, **params) \
            -> Union[str, Tuple[Optional[str], str]]:
        reasoning_content, content = self._get_response(messages=messages, **params)
        if with_reasoning:
            return reasoning_content, content
        else:
            return content

    def _iter_per_words(self, content: str, reasoning_content: Optional[str] = None):
        if reasoning_content:
            for word in jieba.cut(reasoning_content):
                if word:
                    yield word, None
                    time.sleep(1 / self.stream_fps)

        if content:
            for word in jieba.cut(content):
                if word:
                    yield None, word
                    time.sleep(1 / self.stream_fps)

    def ask_stream(self, messages: List[dict], with_reasoning: bool = False, **params) -> ResponseStream:
        reasoning_content, content = self._get_response(messages=messages, **params)
        return FakeResponseStream(
            session=self._iter_per_words(
                reasoning_content=reasoning_content,
                content=content,
            ),
            with_reasoning=with_reasoning,
        )
