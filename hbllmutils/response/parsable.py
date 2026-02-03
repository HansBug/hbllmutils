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

Example::
    >>> import json
    >>> from hbllmutils.model import LLMModel
    >>> from hbllmutils.response import ParsableLLMTask, extract_code, parse_json
    >>> 
    >>> class JSONParsableTask(ParsableLLMTask):
    ...     '''A task that parses JSON responses from the model.'''
    ...     __exceptions__ = (json.JSONDecodeError, KeyError)
    ...     
    ...     def _parse_and_validate(self, content: str):
    ...         data = parse_json(extract_code(content))
    ...         if 'answer' not in data:
    ...             raise KeyError("Missing 'answer' field")
    ...         return data
    >>> 
    >>> model = LLMModel(...)
    >>> task = JSONParsableTask(model, default_max_retries=3)
    >>> result = task.ask_then_parse(
    ...     input_content="What is the capital of France? Answer me in format of JSON with a key of \\"answer\\"",
    ...     max_retries=5
    ... )
    >>> print(result['answer'])
    Paris
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
        >>> print(type(attempt.exception))
        <class 'ValueError'>
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
        >>> exc = OutputParseFailed("Parsing failed after 2 tries", tries)
        >>> print(len(exc.tries))
        2
        >>> print(exc.tries[0].output)
        bad output 1
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
        >>> import json
        >>> from hbllmutils.model import LLMModel
        >>> from hbllmutils.response import ParsableLLMTask, extract_code, parse_json
        >>>
        >>> class JSONParsableTask(ParsableLLMTask):
        ...     __exceptions__ = (json.JSONDecodeError, KeyError)
        ...     
        ...     def _parse_and_validate(self, content: str):
        ...         data = parse_json(extract_code(content))
        ...         if 'result' not in data:
        ...             raise KeyError("Missing 'result' field")
        ...         return data['result']
        >>> 
        >>> model = LLMModel(...)
        >>> task = JSONParsableTask(model)
        >>> result = task.ask_then_parse(input_content="Calculate 2+2")
        >>> print(result)
        4
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

        Example::
            >>> model = LLMModel(...)
            >>> task = ParsableLLMTask(model, default_max_retries=10)
            >>> print(task.default_max_retries)
            10
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
            >>> class IntegerParsableTask(ParsableLLMTask):
            ...     __exceptions__ = (ValueError,)
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         value = int(content.strip())
            ...         if value < 0:
            ...             raise ValueError("Value must be non-negative")
            ...         return value
            >>> 
            >>> task = IntegerParsableTask(model)
            >>> result = task._parse_and_validate("42")
            >>> print(result)
            42
        """
        raise NotImplementedError  # pragma: no cover

    def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
        """
        Preprocess the input content before sending to the model.

        This method can be overridden by subclasses to modify or transform the input
        content before it is sent to the model. The default implementation returns
        the input unchanged.

        :param input_content: The original input content.
        :type input_content: Optional[str]
        :return: The preprocessed input content.
        :rtype: Optional[str]

        Example::
            >>> class CustomTask(ParsableLLMTask):
            ...     def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
            ...         if input_content:
            ...             return input_content.strip().lower()
            ...         return input_content
            >>> 
            >>> task = CustomTask(model)
            >>> result = task._preprocess_input_content("  HELLO  ")
            >>> print(result)
            hello
        """
        return input_content

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
            >>> class NumberTask(ParsableLLMTask):
            ...     __exceptions__ = (ValueError,)
            ...     def _parse_and_validate(self, content: str):
            ...         return int(content.strip())
            >>> 
            >>> task = NumberTask(model)
            >>> # Simple usage with default retries
            >>> result = task.ask_then_parse(input_content="What is 2+2?")
            >>> print(result)
            4
            >>> 
            >>> # Usage with custom max_retries
            >>> result = task.ask_then_parse(
            ...     input_content="Calculate 10*5",
            ...     max_retries=3,
            ...     temperature=0.7
            ... )
            >>> print(result)
            50
        """
        if max_retries is None:
            max_retries = self.default_max_retries
        input_content = self._preprocess_input_content(input_content)

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
