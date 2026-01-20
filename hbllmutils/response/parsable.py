from dataclasses import dataclass
from typing import Optional, Union, Type, Tuple, List

from hbutils.string import plural_word

from ..history import LLMHistory
from ..model import LLMTask, LLMModel


@dataclass
class OutputParseWithException:
    output: str
    exception: Exception


class OutputParseFailed(Exception):
    def __init__(self, message: str, tries: List[OutputParseWithException]):
        super().__init__(message)
        self.tries = tries


class ParsableLLMTask(LLMTask):
    __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_retries: int = 5):
        super().__init__(model, history)
        self.default_retries = default_retries

    def _parse_output(self, content: str):
        pass

    def ask_then_parse(self, with_reasoning: bool = False, retries: Optional[int] = None, **params):
        if retries is None:
            retries = self.default_retries

        tries = 0
        err_tries = []
        while tries < retries:
            if with_reasoning:
                _, content = self.ask(with_reasoning=with_reasoning, **params)
            else:
                content = self.ask(with_reasoning=with_reasoning, **params)

            try:
                parsed_output = self._parse_output(content)
            except self.__exceptions__ as err:
                self._logger.error(f'Error when parse output of model - {err!r}')
                tries += 1
                err_tries.append((content, err))
            else:
                return parsed_output

        raise OutputParseFailed(
            message=f'Output parse failed after {plural_word(len(err_tries), "try")}.',
            tries=[OutputParseWithException(output=content, exception=err) for content, err in err_tries]
        )
