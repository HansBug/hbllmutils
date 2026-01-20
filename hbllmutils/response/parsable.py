"""
This module provides parsable LLM task functionality with automatic retry mechanism for output parsing.

It extends the base LLM task to support parsing of model outputs with configurable retry logic
when parsing fails. The module includes exception handling for parse failures and tracking of
all retry attempts.
"""

from dataclasses import dataclass
from typing import Optional, Union, Type, Tuple, List

from hbutils.string import plural_word

from ..history import LLMHistory
from ..model import LLMTask, LLMModel


@dataclass
class OutputParseWithException:
    """
    Data class to store a failed parse attempt with its output and exception.

    :ivar output: The raw output string that failed to parse.
    :vartype output: str
    :ivar exception: The exception that occurred during parsing.
    :vartype exception: Exception
    """
    output: str
    exception: Exception


class OutputParseFailed(Exception):
    """
    Exception raised when output parsing fails after all retry attempts.

    :ivar tries: List of all failed parse attempts with their outputs and exceptions.
    :vartype tries: List[OutputParseWithException]
    """

    def __init__(self, message: str, tries: List[OutputParseWithException]):
        """
        Initialize the OutputParseFailed exception.

        :param message: The error message describing the failure.
        :type message: str
        :param tries: List of all failed parse attempts.
        :type tries: List[OutputParseWithException]
        """
        super().__init__(message)
        self.tries = tries


class ParsableLLMTask(LLMTask):
    """
    An LLM task that supports automatic parsing of model outputs with retry mechanism.

    This class extends LLMTask to provide automatic parsing of model outputs with configurable
    retry logic. When parsing fails, it will retry up to a maximum number of times before
    raising an OutputParseFailed exception.

    :cvar __exceptions__: Exception types to catch during parsing attempts.
    :vartype __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]]
    """
    __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5):
        """
        Initialize the ParsableLLMTask.

        :param model: The LLM model to use for generating responses.
        :type model: LLMModel
        :param history: Optional conversation history. Defaults to None.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Default maximum number of retry attempts for parsing. Defaults to 5.
        :type default_max_retries: int
        """
        super().__init__(model, history)
        self.default_max_retries = default_max_retries

    def _parse_output(self, content: str):
        """
        Parse the raw output content from the model.

        This method should be implemented by subclasses to define how to parse
        the model's output into the desired format.

        :param content: The raw output string from the model.
        :type content: str
        :return: The parsed output in the desired format.
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def ask_then_parse(self, with_reasoning: bool = False, max_retries: Optional[int] = None, **params):
        """
        Ask the model a question and parse the response with automatic retry on parse failure.

        This method will repeatedly ask the model and attempt to parse the output until
        either parsing succeeds or the maximum number of retries is reached. All failed
        attempts are tracked and included in the exception if all retries fail.

        :param with_reasoning: Whether to include reasoning in the response. Defaults to False.
        :type with_reasoning: bool
        :param max_retries: Maximum number of retry attempts. If None, uses default_max_retries.
        :type max_retries: Optional[int]
        :param params: Additional parameters to pass to the ask method.
        :return: The successfully parsed output.
        :raises OutputParseFailed: If parsing fails after all retry attempts.

        Example::
            >>> task = ParsableLLMTask(model)
            >>> result = task.ask_then_parse(prompt="What is 2+2?", max_retries=3)
            >>> print(result)
            4
        """
        if max_retries is None:
            max_retries = self.default_max_retries

        tries = 0
        err_tries = []
        while tries < max_retries:
            if with_reasoning:
                _, content = self.ask(with_reasoning=with_reasoning, **params)
            else:
                content = self.ask(with_reasoning=with_reasoning, **params)

            try:
                parsed_output = self._parse_output(content)
            except self.__exceptions__ as err:
                tries += 1
                self._logger.warning(f'Error when parsing output of model ({tries}/{max_retries}) - {err!r}')
                err_tries.append((content, err))
            else:
                return parsed_output

        raise OutputParseFailed(
            message=f'Output parse failed after {plural_word(len(err_tries), "try")}.',
            tries=[OutputParseWithException(output=content, exception=err) for content, err in err_tries]
        )
