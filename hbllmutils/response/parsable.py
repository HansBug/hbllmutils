"""
This module provides parsable LLM task functionality with automatic retry mechanism for output parsing.

It extends the base LLM task to support parsing of model outputs with configurable retry logic
when parsing fails. The module includes exception handling for parse failures and tracking of
all retry attempts.

Key Components:
    - OutputParseWithException: Data class for storing failed parse attempts
    - OutputParseFailed: Exception raised when all parsing attempts fail
    - ParsableLLMTask: LLM task with automatic output parsing and retry logic

The module is designed to handle scenarios where LLM outputs need to be parsed into specific
formats, with automatic retry when parsing fails due to malformed or unexpected output.
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

    This class encapsulates information about a single failed parsing attempt,
    including both the raw output that failed to parse and the exception that
    was raised during the parsing process.

    :ivar output: The raw output string that failed to parse.
    :vartype output: str
    :ivar exception: The exception that occurred during parsing.
    :vartype exception: Exception

    Example::
        >>> attempt = OutputParseWithException(
        ...     output="invalid json",
        ...     exception=ValueError("Invalid format")
        ... )
        >>> print(attempt.output)
        invalid json
    """
    output: str
    exception: Exception


class OutputParseFailed(Exception):
    """
    Exception raised when output parsing fails after all retry attempts.

    This exception is raised when the ParsableLLMTask exhausts all retry attempts
    without successfully parsing the model's output. It contains information about
    all failed attempts for debugging purposes.

    :ivar tries: List of all failed parse attempts with their outputs and exceptions.
    :vartype tries: List[OutputParseWithException]

    Example::
        >>> tries = [
        ...     OutputParseWithException("bad output 1", ValueError("error 1")),
        ...     OutputParseWithException("bad output 2", ValueError("error 2"))
        ... ]
        >>> raise OutputParseFailed("Parsing failed", tries)
        Traceback (most recent call last):
        ...
        OutputParseFailed: Parsing failed
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
    raising an OutputParseFailed exception. This is useful when the model's output needs to
    be parsed into a specific format (e.g., JSON, structured data) and the model may
    occasionally produce malformed output.

    The class uses a template method pattern where subclasses implement the _parse_and_validate
    method to define their specific parsing logic.

    :cvar __exceptions__: Exception types to catch during parsing attempts. Can be a single
                          exception type or a tuple of exception types. Defaults to Exception.
    :vartype __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]]

    Example::
        >>> class JSONParsableTask(ParsableLLMTask):
        ...     __exceptions__ = (ValueError, KeyError)
        ...     
        ...     def _parse_and_validate(self, content: str):
        ...         import json
        ...         return json.loads(content)
        >>> 
        >>> task = JSONParsableTask(model)
        >>> result = task.ask_then_parse(prompt="Return JSON with key 'answer'")
    """
    __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5):
        """
        Initialize the ParsableLLMTask.

        :param model: The LLM model to use for generating responses.
        :type model: LLMModel
        :param history: Optional conversation history. If None, a new history will be created.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Default maximum number of retry attempts for parsing.
                                   Must be a positive integer. Defaults to 5.
        :type default_max_retries: int
        """
        super().__init__(model, history)
        self.default_max_retries = default_max_retries

    def _parse_and_validate(self, content: str):
        """
        Parse and validate the raw output content from the model.

        This method should be implemented by subclasses to define how to parse
        the model's output into the desired format. The method should raise an
        exception (matching __exceptions__) if the content cannot be parsed or
        validated successfully.

        :param content: The raw output string from the model.
        :type content: str
        :return: The parsed output in the desired format. The return type depends
                on the specific implementation.
        :raises NotImplementedError: This method must be implemented by subclasses.

        Example::
            >>> class MyParsableTask(ParsableLLMTask):
            ...     def _parse_and_validate(self, content: str):
            ...         # Parse content as integer
            ...         return int(content.strip())
        """
        raise NotImplementedError  # pragma: no cover

    def ask_then_parse(self, input_content: Optional[str] = None, max_retries: Optional[int] = None, **params):
        """
        Ask the model a question and parse the response with automatic retry on parse failure.

        This method will repeatedly ask the model and attempt to parse the output until
        either parsing succeeds or the maximum number of retries is reached. Each failed
        attempt is logged and tracked. If all retries fail, an OutputParseFailed exception
        is raised containing all failed attempts for debugging.

        The method uses the _parse_and_validate method to parse outputs and will catch
        exceptions specified in __exceptions__. Other exceptions will propagate immediately.

        :param input_content: Optional user input content to add to the history before asking.
                             If None, uses the existing history without modification.
        :type input_content: Optional[str]
        :param max_retries: Maximum number of retry attempts. If None, uses default_max_retries.
                           Must be a positive integer if provided.
        :type max_retries: Optional[int]
        :param params: Additional parameters to pass to the ask method (e.g., prompt, temperature).
        :return: The successfully parsed output from the model.
        :raises OutputParseFailed: If parsing fails after all retry attempts. The exception
                                  contains all failed attempts in its tries attribute.
        :raises Exception: Any exception not matching __exceptions__ will propagate immediately.

        Example::
            >>> task = ParsableLLMTask(model)
            >>> # Simple usage
            >>> result = task.ask_then_parse(input_content="What is 2+2?", max_retries=3)
            >>> print(result)
            4
        """
        if max_retries is None:
            max_retries = self.default_max_retries

        tries = 0
        err_tries = []
        while tries < max_retries:
            content = self.ask(input_content=input_content, **params)
            try:
                parsed_output = self._parse_and_validate(content)
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
