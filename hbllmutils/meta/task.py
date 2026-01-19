from abc import ABC
from typing import Union, Tuple, Optional

from hbllmutils.history import LLMHistory
from hbllmutils.model import ResponseStream, LLMModel


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
