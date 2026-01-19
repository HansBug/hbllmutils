from abc import ABC
from typing import Union, Tuple, Optional

from .base import LLMModel
from .stream import ResponseStream
from ..history import LLMHistory


class LLMTask(ABC):
    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None):
        self.model = model
        self.history = history or LLMHistory()

    def ask(self, with_reasoning: bool = False, **params) -> Union[str, Tuple[Optional[str], str]]:
        return self.model.ask(
            messages=self.history.to_json(),
            with_reasoning=with_reasoning,
            **params
        )

    def ask_stream(self, with_reasoning: bool = False, **params) -> ResponseStream:
        return self.model.ask_stream(
            messages=self.history.to_json(),
            with_reasoning=with_reasoning,
            **params,
        )
