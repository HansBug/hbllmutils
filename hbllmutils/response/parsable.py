"""
Parsable LLM task functionality with automatic retry mechanism for output parsing.

This module provides parsable LLM task functionality with automatic retry mechanism for output parsing.
It extends the base LLM task to support parsing of model outputs with configurable retry logic
when parsing fails. The module includes exception handling for parse failures and tracking of
all retry attempts.

The module is designed to handle scenarios where LLM outputs need to be parsed into specific
formats (such as JSON, XML, structured data), with automatic retry when parsing fails due to 
malformed or unexpected output. This is particularly useful when working with LLMs that may
occasionally produce outputs that don't conform to the expected format.

Key Components:
    * :class:`OutputParseWithException` - Data class for storing failed parse attempts
    * :class:`OutputParseFailed` - Exception raised when all parsing attempts fail
    * :class:`ParsableLLMTask` - LLM task with automatic output parsing and retry logic

Architecture:
    The module uses a template method pattern where subclasses implement the 
    :meth:`ParsableLLMTask._parse_and_validate` method to define their specific parsing logic.
    The base class handles the retry mechanism, exception tracking, and logging automatically.

Retry Mechanism:
    When parsing fails, the task will:
    
    1. Log the parsing error with attempt count
    2. Store the failed output and exception for debugging
    3. Request a new response from the model
    4. Attempt to parse the new response
    5. Repeat until success or max retries reached
    
    If all retries are exhausted, an :exc:`OutputParseFailed` exception is raised containing
    all failed attempts for comprehensive debugging.

.. note::
   The retry mechanism sends new requests to the LLM for each failed parse attempt,
   which may incur additional API costs and latency. Consider setting appropriate
   max_retries values based on your use case.

.. warning::
   Ensure that your parsing logic in :meth:`_parse_and_validate` raises exceptions
   that match the types specified in :attr:`__exceptions__`. Other exception types
   will propagate immediately without retry.

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
    ...         # Extract code block and parse JSON
    ...         data = parse_json(extract_code(content))
    ...         
    ...         # Validate required fields
    ...         if 'answer' not in data:
    ...             raise KeyError("Missing 'answer' field")
    ...         
    ...         return data
    >>> 
    >>> # Initialize model and task
    >>> model = LLMModel(...)
    >>> task = JSONParsableTask(model, default_max_retries=3)
    >>> 
    >>> # Ask question with automatic parsing and retry
    >>> result = task.ask_then_parse(
    ...     input_content="What is the capital of France? Answer in JSON with 'answer' key",
    ...     max_retries=5
    ... )
    >>> print(result['answer'])
    Paris
    >>> 
    >>> # Handle parsing failures
    >>> try:
    ...     result = task.ask_then_parse("Invalid request")
    ... except OutputParseFailed as e:
    ...     print(f"Failed after {len(e.tries)} attempts")
    ...     for i, attempt in enumerate(e.tries):
    ...         print(f"Attempt {i+1}: {attempt.exception}")
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
    was raised during the parsing process. It is used by :exc:`OutputParseFailed`
    to provide comprehensive debugging information about all failed attempts.

    The class is immutable (frozen dataclass) to ensure that stored failure
    information cannot be accidentally modified.

    :ivar output: The raw output string from the model that failed to parse.
    :vartype output: str
    :ivar exception: The exception that occurred during parsing attempt.
    :vartype exception: Exception

    Example::

        >>> import json
        >>> attempt = OutputParseWithException(
        ...     output='{"invalid": json}',
        ...     exception=json.JSONDecodeError("Expecting ',' delimiter", "", 0)
        ... )
        >>> print(attempt.output)
        {"invalid": json}
        >>> print(type(attempt.exception))
        <class 'json.decoder.JSONDecodeError'>
        >>> 
        >>> # Used in debugging failed parsing attempts
        >>> for attempt in failed_attempts:
        ...     print(f"Output: {attempt.output[:50]}...")
        ...     print(f"Error: {attempt.exception}")
    """
    output: str
    exception: Exception


class OutputParseFailed(Exception):
    """
    Exception raised when output parsing fails after all retry attempts.

    This exception is raised by :meth:`ParsableLLMTask.ask_then_parse` when the task
    exhausts all retry attempts without successfully parsing the model's output. It
    contains comprehensive information about all failed attempts, including the raw
    outputs and exceptions, which is invaluable for debugging parsing issues.

    The exception message includes the total number of failed attempts, and the
    :attr:`tries` attribute provides detailed information about each individual
    failure for in-depth analysis.

    :ivar tries: List of all failed parse attempts with their outputs and exceptions.
                Each element is an :class:`OutputParseWithException` instance containing
                the raw output and the exception that occurred.
    :vartype tries: List[OutputParseWithException]

    Example::

        >>> import json
        >>> tries = [
        ...     OutputParseWithException(
        ...         output='{"incomplete": ',
        ...         exception=json.JSONDecodeError("Expecting value", "", 15)
        ...     ),
        ...     OutputParseWithException(
        ...         output='not json at all',
        ...         exception=json.JSONDecodeError("Expecting value", "", 0)
        ...     )
        ... ]
        >>> exc = OutputParseFailed("Parsing failed after 2 tries", tries)
        >>> print(str(exc))
        Parsing failed after 2 tries
        >>> print(len(exc.tries))
        2
        >>> 
        >>> # Accessing detailed failure information
        >>> for i, attempt in enumerate(exc.tries, 1):
        ...     print(f"Attempt {i}:")
        ...     print(f"  Output: {attempt.output}")
        ...     print(f"  Error: {attempt.exception}")
        Attempt 1:
          Output: {"incomplete": 
          Error: Expecting value: line 1 column 15 (char 15)
        Attempt 2:
          Output: not json at all
          Error: Expecting value: line 1 column 0 (char 0)
        >>> 
        >>> # Handling in try-except block
        >>> try:
        ...     result = task.ask_then_parse("some input")
        ... except OutputParseFailed as e:
        ...     print(f"All {len(e.tries)} parsing attempts failed")
        ...     # Log or analyze failures
        ...     for attempt in e.tries:
        ...         logger.error(f"Failed output: {attempt.output}")
    """

    def __init__(self, message: str, tries: List[OutputParseWithException]):
        """
        Initialize the OutputParseFailed exception.

        :param message: The error message describing the failure. Typically includes
                       the total number of failed attempts.
        :type message: str
        :param tries: List of all failed parse attempts. Each element should be an
                     :class:`OutputParseWithException` instance containing the output
                     and exception from that attempt.
        :type tries: List[OutputParseWithException]

        Example::

            >>> tries = [
            ...     OutputParseWithException("bad output 1", ValueError("error 1")),
            ...     OutputParseWithException("bad output 2", ValueError("error 2"))
            ... ]
            >>> exc = OutputParseFailed("Parsing failed after 2 tries", tries)
            >>> raise exc
            Traceback (most recent call last):
                ...
            OutputParseFailed: Parsing failed after 2 tries
        """
        super().__init__(message)
        self.tries = tries


class ParsableLLMTask(LLMTask):
    """
    An LLM task that supports automatic parsing of model outputs with retry mechanism.

    This class extends :class:`LLMTask` to provide automatic parsing of model outputs with 
    configurable retry logic. When parsing fails, it will retry up to a maximum number of 
    times before raising an :exc:`OutputParseFailed` exception. This is useful when the 
    model's output needs to be parsed into a specific format (e.g., JSON, XML, structured 
    data) and the model may occasionally produce malformed output.

    The class uses a template method pattern where subclasses implement the 
    :meth:`_parse_and_validate` method to define their specific parsing logic. The base 
    class handles the retry mechanism, exception tracking, and logging automatically.

    Workflow:
        1. Send request to model (optionally with new input content)
        2. Receive raw text response from model
        3. Attempt to parse response using :meth:`_parse_and_validate`
        4. If parsing succeeds, return parsed result
        5. If parsing fails with an exception in :attr:`__exceptions__`:
           
           - Log the failure with attempt count
           - Store the failed output and exception
           - Send a new request to the model
           - Repeat from step 2
        
        6. If max retries reached, raise :exc:`OutputParseFailed` with all attempts

    :cvar __exceptions__: Exception types to catch during parsing attempts. Can be a single
                          exception type or a tuple of exception types. Only exceptions
                          matching these types will trigger a retry; other exceptions will
                          propagate immediately. Defaults to :exc:`Exception` (catches all).
    :vartype __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]]

    :ivar default_max_retries: Default maximum number of retry attempts for parsing.
                               Used when max_retries is not specified in :meth:`ask_then_parse`.
    :vartype default_max_retries: int

    .. note::
       Each retry sends a new request to the LLM, which may incur additional API costs
       and increase response time. Set appropriate max_retries values based on your
       use case and budget constraints.

    .. warning::
       Ensure that :meth:`_parse_and_validate` raises exceptions that match the types
       specified in :attr:`__exceptions__`. Other exception types will not trigger
       retries and will propagate immediately.

    Example::

        >>> import json
        >>> from hbllmutils.model import LLMModel
        >>> from hbllmutils.response import ParsableLLMTask, extract_code, parse_json
        >>>
        >>> class JSONParsableTask(ParsableLLMTask):
        ...     '''Task that parses JSON responses with validation.'''
        ...     __exceptions__ = (json.JSONDecodeError, KeyError, ValueError)
        ...     
        ...     def _parse_and_validate(self, content: str):
        ...         # Extract code block if present
        ...         data = parse_json(extract_code(content))
        ...         
        ...         # Validate structure
        ...         if 'result' not in data:
        ...             raise KeyError("Missing 'result' field")
        ...         if not isinstance(data['result'], (int, float)):
        ...             raise ValueError("Result must be numeric")
        ...         
        ...         return data['result']
        >>> 
        >>> # Initialize with custom default retries
        >>> model = LLMModel(...)
        >>> task = JSONParsableTask(model, default_max_retries=3)
        >>> 
        >>> # Simple usage with default retries
        >>> result = task.ask_then_parse(input_content="Calculate 2+2")
        >>> print(result)
        4
        >>> 
        >>> # Usage with custom max_retries for specific request
        >>> result = task.ask_then_parse(
        ...     input_content="What is 10*5?",
        ...     max_retries=10,
        ...     temperature=0.7
        ... )
        >>> print(result)
        50
        >>> 
        >>> # Handling parse failures
        >>> try:
        ...     result = task.ask_then_parse("Invalid request")
        ... except OutputParseFailed as e:
        ...     print(f"Failed after {len(e.tries)} attempts")
        ...     for i, attempt in enumerate(e.tries, 1):
        ...         print(f"Attempt {i}: {attempt.exception}")
    """
    __exceptions__: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception

    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5):
        """
        Initialize the ParsableLLMTask.

        :param model: The LLM model to use for generating responses.
        :type model: LLMModel
        :param history: Optional conversation history. If None, a new empty history will be created.
                       The history tracks the conversation context across multiple interactions.
        :type history: Optional[LLMHistory]
        :param default_max_retries: Default maximum number of retry attempts for parsing.
                                   Must be a positive integer. This value is used when
                                   max_retries is not specified in :meth:`ask_then_parse`.
                                   Defaults to 5.
        :type default_max_retries: int

        :raises ValueError: If default_max_retries is not a positive integer.

        Example::

            >>> from hbllmutils.model import LLMModel
            >>> from hbllmutils.history import LLMHistory
            >>> 
            >>> # Simple initialization with defaults
            >>> model = LLMModel(...)
            >>> task = ParsableLLMTask(model)
            >>> print(task.default_max_retries)
            5
            >>> 
            >>> # Initialize with custom default retries
            >>> task = ParsableLLMTask(model, default_max_retries=10)
            >>> print(task.default_max_retries)
            10
            >>> 
            >>> # Initialize with existing history
            >>> history = LLMHistory().with_system_prompt("You are a helpful assistant.")
            >>> task = ParsableLLMTask(model, history=history, default_max_retries=3)
            >>> len(task.history)
            1
        """
        super().__init__(model, history)
        self.default_max_retries = default_max_retries

    def _parse_and_validate(self, content: str):
        """
        Parse and validate the raw output content from the model.

        This method should be implemented by subclasses to define how to parse
        the model's output into the desired format. The method should raise an
        exception (matching :attr:`__exceptions__`) if the content cannot be parsed 
        or validated successfully.

        The method is called automatically by :meth:`ask_then_parse` after receiving
        a response from the model. If this method raises an exception that matches
        :attr:`__exceptions__`, the task will retry with a new request. Other exceptions
        will propagate immediately without retry.

        Implementation Guidelines:
            - Raise specific exceptions that match :attr:`__exceptions__`
            - Provide clear error messages for debugging
            - Validate both format and content of the parsed data
            - Return the parsed data in the desired format
            - Keep parsing logic focused and testable

        :param content: The raw output string from the model to parse and validate.
        :type content: str

        :return: The parsed output in the desired format. The return type depends
                on the specific implementation and use case.

        :raises NotImplementedError: This method must be implemented by subclasses.
        :raises Exception: Subclasses should raise appropriate exceptions (matching
                          :attr:`__exceptions__`) when parsing or validation fails.

        Example::

            >>> class IntegerParsableTask(ParsableLLMTask):
            ...     '''Task that parses integer responses.'''
            ...     __exceptions__ = (ValueError,)
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         # Remove whitespace and parse
            ...         value = int(content.strip())
            ...         
            ...         # Validate range
            ...         if value < 0:
            ...             raise ValueError("Value must be non-negative")
            ...         
            ...         return value
            >>> 
            >>> task = IntegerParsableTask(model)
            >>> result = task._parse_and_validate("42")
            >>> print(result)
            42
            >>> 
            >>> # Example with JSON parsing
            >>> import json
            >>> class JSONParsableTask(ParsableLLMTask):
            ...     __exceptions__ = (json.JSONDecodeError, KeyError)
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         # Parse JSON
            ...         data = json.loads(content)
            ...         
            ...         # Validate required fields
            ...         if 'answer' not in data:
            ...             raise KeyError("Missing 'answer' field")
            ...         
            ...         return data['answer']
            >>> 
            >>> task = JSONParsableTask(model)
            >>> result = task._parse_and_validate('{"answer": "Paris"}')
            >>> print(result)
            Paris
        """
        raise NotImplementedError  # pragma: no cover

    def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
        """
        Preprocess the input content before sending to the model.

        This method can be overridden by subclasses to modify or transform the input
        content before it is sent to the model. The default implementation returns
        the input unchanged. Common use cases include:
        
        - Normalizing text (trimming, case conversion)
        - Adding format instructions or templates
        - Sanitizing or validating input
        - Injecting context or metadata

        The method is called automatically by :meth:`ask_then_parse` before adding
        the input to the conversation history and sending it to the model.

        :param input_content: The original input content from the user.
        :type input_content: Optional[str]

        :return: The preprocessed input content. Can be None if input should be omitted.
        :rtype: Optional[str]

        Example::

            >>> class CustomTask(ParsableLLMTask):
            ...     '''Task with input preprocessing.'''
            ...     
            ...     def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
            ...         if input_content:
            ...             # Normalize whitespace and case
            ...             content = input_content.strip().lower()
            ...             
            ...             # Add format instruction
            ...             content += "\\n\\nPlease respond in JSON format."
            ...             
            ...             return content
            ...         return input_content
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         return json.loads(content)
            >>> 
            >>> task = CustomTask(model)
            >>> result = task._preprocess_input_content("  HELLO  ")
            >>> print(result)
            hello
            
            Please respond in JSON format.
            >>> 
            >>> # Example with template injection
            >>> class TemplateTask(ParsableLLMTask):
            ...     def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
            ...         if input_content:
            ...             template = "Question: {question}\\n\\nAnswer in format: {{'answer': '...'}}"
            ...             return template.format(question=input_content)
            ...         return input_content
            >>> 
            >>> task = TemplateTask(model)
            >>> result = task._preprocess_input_content("What is 2+2?")
            >>> print(result)
            Question: What is 2+2?
            
            Answer in format: {'answer': '...'}
        """
        return input_content

    def ask_then_parse(self, input_content: Optional[str] = None, max_retries: Optional[int] = None, **params):
        """
        Ask the model a question and parse the response with automatic retry on parse failure.

        This method will repeatedly ask the model and attempt to parse the output until
        either parsing succeeds or the maximum number of retries is reached. Each failed
        attempt is logged and tracked. If all retries fail, an :exc:`OutputParseFailed` 
        exception is raised containing all failed attempts for debugging.

        The method uses :meth:`_parse_and_validate` to parse outputs and will catch
        exceptions specified in :attr:`__exceptions__`. Other exceptions will propagate 
        immediately without retry.

        Workflow:
            1. Preprocess input content using :meth:`_preprocess_input_content`
            2. Send request to model using :meth:`ask`
            3. Attempt to parse response using :meth:`_parse_and_validate`
            4. On success: return parsed result
            5. On failure (matching :attr:`__exceptions__`):
               
               - Log warning with attempt count
               - Store failed output and exception
               - Increment retry counter
               - Repeat from step 2 if retries remain
            
            6. If max retries exhausted: raise :exc:`OutputParseFailed`

        :param input_content: Optional user input content to add to the history before asking.
                             If None, uses the existing history without modification.
                             The content will be preprocessed by :meth:`_preprocess_input_content`.
        :type input_content: Optional[str]
        :param max_retries: Maximum number of retry attempts. If None, uses :attr:`default_max_retries`.
                           Must be a positive integer if provided. Each retry sends a new
                           request to the model.
        :type max_retries: Optional[int]
        :param params: Additional parameters to pass to the :meth:`ask` method. Common parameters
                      include temperature, max_tokens, top_p, etc., depending on the model.
        :type params: dict

        :return: The successfully parsed output from the model. The return type depends on
                the implementation of :meth:`_parse_and_validate`.

        :raises OutputParseFailed: If parsing fails after all retry attempts. The exception
                                  contains all failed attempts in its :attr:`tries` attribute,
                                  each with the raw output and exception for debugging.
        :raises Exception: Any exception not matching :attr:`__exceptions__` will propagate
                          immediately without retry.

        .. note::
           Each retry sends a new request to the LLM, which may incur additional API costs
           and increase total response time. Monitor your usage and set appropriate retry limits.

        .. warning::
           The method does not modify the conversation history with failed attempts. Only
           successful interactions are recorded in the history.

        Example::

            >>> import json
            >>> class NumberTask(ParsableLLMTask):
            ...     '''Task that parses numeric responses.'''
            ...     __exceptions__ = (ValueError, TypeError)
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         value = float(content.strip())
            ...         if value < 0:
            ...             raise ValueError("Value must be non-negative")
            ...         return value
            >>> 
            >>> task = NumberTask(model)
            >>> 
            >>> # Simple usage with default retries
            >>> result = task.ask_then_parse(input_content="What is 2+2?")
            >>> print(result)
            4.0
            >>> 
            >>> # Usage with custom max_retries and model parameters
            >>> result = task.ask_then_parse(
            ...     input_content="Calculate 10*5",
            ...     max_retries=3,
            ...     temperature=0.7,
            ...     max_tokens=100
            ... )
            >>> print(result)
            50.0
            >>> 
            >>> # Handling parse failures
            >>> try:
            ...     result = task.ask_then_parse(
            ...         input_content="Invalid request",
            ...         max_retries=2
            ...     )
            ... except OutputParseFailed as e:
            ...     print(f"Failed after {len(e.tries)} attempts")
            ...     for i, attempt in enumerate(e.tries, 1):
            ...         print(f"Attempt {i}:")
            ...         print(f"  Output: {attempt.output[:50]}...")
            ...         print(f"  Error: {attempt.exception}")
            Failed after 2 attempts
            Attempt 1:
              Output: I'm not sure what you're asking...
              Error: could not convert string to float: "I'm not sure what you're asking"
            Attempt 2:
              Output: Please clarify your question...
              Error: could not convert string to float: "Please clarify your question"
            >>> 
            >>> # Example with JSON parsing
            >>> class JSONTask(ParsableLLMTask):
            ...     __exceptions__ = (json.JSONDecodeError, KeyError)
            ...     
            ...     def _parse_and_validate(self, content: str):
            ...         data = json.loads(content)
            ...         if 'answer' not in data:
            ...             raise KeyError("Missing 'answer' field")
            ...         return data['answer']
            >>> 
            >>> task = JSONTask(model)
            >>> result = task.ask_then_parse(
            ...     input_content="What is the capital of France? Answer in JSON",
            ...     max_retries=5
            ... )
            >>> print(result)
            Paris
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
