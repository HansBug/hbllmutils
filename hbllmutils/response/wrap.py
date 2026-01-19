import logging
from typing import Optional

from ..history import LLMHistory
from ..model import LLMTask, LLMModel


class RobustLLMTask(LLMTask):
    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_retries: int = 5):
        super().__init__(model, history)
        self.default_retries = default_retries

    def ask_robust(self, with_reasoning: bool = False, retries: Optional[int] = None, **params):
        if retries is None:
            retries = self.default_retries

        tries = 0
        while tries < retries:
            if with_reasoning:
                reasoning, content = self.ask(with_reasoning=with_reasoning, **params)
            else:
                content = self.ask(with_reasoning=with_reasoning, **params)

            tries += 1
