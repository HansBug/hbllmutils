"""
Data model-based LLM task utilities.

This module provides functionality for creating and managing LLM tasks that parse and
validate responses against structured data models. It supports both Pydantic models
and dataclasses, with automatic prompt generation and validation capabilities.

The module serves as a bridge between LLM outputs and structured data validation,
enabling type-safe parsing of LLM responses into well-defined data structures. It
handles the complete workflow from prompt generation to response validation, with
built-in retry mechanisms for handling parsing failures.

Key Features:
    - Structured data validation using Pydantic or dataclasses
    - Automatic format prompt generation from data model schemas
    - Sample-based learning support for few-shot prompting
    - Retry mechanism for failed validations with configurable attempts
    - JSON parsing and validation with error recovery
    - Support for related data models to provide context
    - Customizable parsing and serialization functions

Main Components:
    * :class:`DataModelLLMTask` - Core task class for model-based validation
    * :func:`create_datamodel_task` - Factory function for creating configured tasks

Architecture:
    The module follows a layered architecture:

    1. **Task Layer**: :class:`DataModelLLMTask` handles the high-level workflow
    2. **Prompt Generation**: Automatic creation of format instructions
    3. **Parsing Layer**: Extraction and validation of structured data
    4. **Retry Logic**: Automatic retry on validation failures

Performance Considerations:
    - Format prompts are cached using LRU cache to avoid regeneration
    - Sample serialization is performed once during task creation
    - Validation functions can be customized for optimal performance

.. note::
   This module requires either Pydantic BaseModel or Python dataclasses for
   data model definitions. Custom types require explicit parsing functions.

.. warning::
   Large numbers of samples or complex data models may result in very long
   prompts, which could impact token usage and response time.

Example::

    >>> from pydantic import BaseModel
    >>> from hbllmutils.model import load_llm_model
    >>> from hbllmutils.response import create_datamodel_task
    >>>
    >>> class Person(BaseModel):
    ...     gender: str  # male or female
    ...     age: int
    ...     hair_color: str  # use hex color
    ...     skin_color: str  # use readable color
    ...     appearance_desc: str  # a line of text for description of this guy
    >>>
    >>> model = load_llm_model('gpt-4o')
    >>> print(f"Loaded Model: {model}")
    >>>
    >>> task = create_datamodel_task(
    ...     model=model,
    ...     datamodel_class=Person,
    ...     task_requirements=\"\"\"
    ... You are a bot to tell me the information of a celebrity.
    ...
    ... I will give you his/her name, and you should tell me about his/her appearance information.
    ...
    ...     \"\"\",
    ...     samples=[
    ...         # European female
    ...         ("Taylor Swift", Person(
    ...             gender="female",
    ...             age=34,
    ...             hair_color="#F5DEB3",  # blonde
    ...             skin_color="fair",
    ...             appearance_desc="Tall blonde singer with blue eyes, known for her elegant and graceful appearance"
    ...         )),
    ...
    ...         # African male
    ...         ("Will Smith", Person(
    ...             gender="male",
    ...             age=55,
    ...             hair_color="#2F1B14",  # dark brown
    ...             skin_color="dark brown",
    ...             appearance_desc="Charismatic actor with a bright smile, athletic build and confident demeanor"
    ...         )),
    ...     ]
    ... )
    >>> print(task.ask_then_parse('Jackie Chan'))
    gender='male' age=69 hair_color='#1C1C1C' skin_color='light brown' appearance_desc='Martial arts action star with a lively personality, known for his agile physique and distinctive smile'
    >>> print(task.ask_then_parse('Donald Trump'))
    gender='male' age=77 hair_color='#FFD700' skin_color='light' appearance_desc='Notable public figure known for his distinct hairstyle and fair complexion, often seen in formal suits'
    >>> print(task.ask_then_parse('Tohsaka Rin'))
    gender='female' age=17 hair_color='#2F1B14' skin_color='fair' appearance_desc='A young woman with twin-tailed brown hair and aqua eyes, usually seen wearing a red sweater and black skirt, exuding both elegance and a strong-willed demeanor'

"""

import dataclasses
import io
import json
import textwrap
from functools import lru_cache
from typing import Optional, List, Callable, Any, Tuple

from hbutils.string import plural_word
from pydantic import BaseModel

from .code import extract_code, parse_json
from .parsable import ParsableLLMTask
from ..history import LLMHistory
from ..meta import create_datamodel_prompt_generation_task
from ..model import LLMModel, LLMTask, LLMModelTyping, load_llm_model


class DataModelLLMTask(ParsableLLMTask):
    """
    A specialized LLM task that parses and validates responses against a data model.

    This class extends :class:`ParsableLLMTask` to provide structured data validation
    using a custom parsing and validation function. It handles the complete workflow
    of sending prompts to an LLM, receiving responses, and validating them against
    a predefined data model structure.

    The class is designed to work with any data model that can be validated through
    a callable function, making it flexible enough to support Pydantic models,
    dataclasses, or custom validation logic.

    The workflow consists of:

    1. Sending a request to the LLM with conversation history
    2. Receiving the raw text response
    3. Extracting code blocks from the response
    4. Parsing the extracted code as JSON
    5. Validating the parsed data against the data model
    6. Retrying on validation failure (up to max_retries times)

    :param model: The LLM model to use for generating responses.
    :type model: LLMModelTyping
    :param history: The conversation history to maintain context.
    :type history: LLMHistory
    :param fn_parse_and_validate: Function to parse and validate the response data.
                                  Should accept the parsed JSON data and return a
                                  validated instance of the data model. Must raise
                                  an exception on validation failure.
    :type fn_parse_and_validate: Callable[[Any], Any]
    :param default_max_retries: Maximum number of retries for failed attempts, defaults to 5.
    :type default_max_retries: int

    :ivar _fn_parse_and_validate: The validation function used for parsing responses.
    :vartype _fn_parse_and_validate: Callable[[Any], Any]

    .. note::
       The validation function should raise an exception on invalid data to trigger
       the retry mechanism. The exception type should match the __exceptions__ class
       variable defined in :class:`ParsableLLMTask`.

    .. warning::
       Each retry sends a new request to the LLM, which may incur additional API costs.
       Set appropriate max_retries values based on your use case and budget.

    Example::

        >>> from pydantic import BaseModel
        >>> from hbllmutils.model import load_llm_model
        >>> from hbllmutils.history import LLMHistory
        >>>
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> model = load_llm_model('gpt-4')
        >>> history = LLMHistory().with_system_prompt("Extract person info")
        >>> task = DataModelLLMTask(
        ...     model=model,
        ...     history=history,
        ...     fn_parse_and_validate=MyModel.model_validate
        ... )
        >>> result = task.ask_then_parse("Extract info: John is 30 years old")
        >>> isinstance(result, MyModel)
        True
        >>> result.name
        'John'
        >>> result.age
        30
    """

    def __init__(self, model: LLMModelTyping, history: LLMHistory,
                 fn_parse_and_validate: Callable[[Any], Any], default_max_retries: int = 5):
        """
        Initialize a DataModelLLMTask instance.

        Sets up the task with a model, conversation history, and validation function.
        The validation function will be called on each response to ensure it conforms
        to the expected data model structure.

        :param model: The LLM model to use for generating responses. Can be a model
                     name string, an LLMModel instance, or None for the default model.
        :type model: LLMModelTyping
        :param history: The conversation history to maintain context. Should include
                       system prompts and any previous conversation turns.
        :type history: LLMHistory
        :param fn_parse_and_validate: Function to parse and validate the response data.
                                      Should accept the parsed JSON data and return a
                                      validated instance of the data model. Must raise
                                      an exception (matching __exceptions__) on failure.
        :type fn_parse_and_validate: Callable[[Any], Any]
        :param default_max_retries: Maximum number of retries for failed attempts, defaults to 5.
                                   Must be a positive integer.
        :type default_max_retries: int

        :raises ValueError: If default_max_retries is not a positive integer.

        Example::

            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> task = DataModelLLMTask(
            ...     model='gpt-4',
            ...     history=LLMHistory(),
            ...     fn_parse_and_validate=Person.model_validate,
            ...     default_max_retries=3
            ... )
        """
        super().__init__(
            model=model,
            history=history,
            default_max_retries=default_max_retries,
        )
        self._fn_parse_and_validate = fn_parse_and_validate

    def _parse_and_validate(self, content: str) -> Any:
        """
        Parse and validate the content from LLM response.

        This method extracts code from the content, parses it as JSON,
        and validates it using the configured validation function. It handles
        the complete parsing pipeline including code extraction, JSON parsing,
        and data model validation.

        The method follows these steps:

        1. Extract code blocks from the response using :func:`extract_code`
        2. Parse the extracted code as JSON using :func:`parse_json`
        3. Validate the parsed data using the configured validation function

        If any step fails, an exception is raised which will trigger the retry
        mechanism in the parent :class:`ParsableLLMTask` class.

        :param content: The raw content string from LLM response. May contain
                       Markdown-formatted code blocks or plain JSON.
        :type content: str

        :return: The validated data object as returned by the validation function.
                The type depends on the data model being validated.
        :rtype: Any

        :raises ValueError: If code extraction fails (no code blocks found or
                          multiple ambiguous code blocks).
        :raises json.JSONDecodeError: If the content cannot be parsed as JSON.
        :raises ValidationError: If the parsed data fails validation against the
                               data model (for Pydantic models).
        :raises Exception: Any other exception raised by the validation function.

        Example::

            >>> task = DataModelLLMTask(...)
            >>> # Parse valid JSON in code block
            >>> result = task._parse_and_validate('```json\\n{"name": "test", "age": 25}\\n```')
            >>> result.name
            'test'
            >>> result.age
            25
            >>>
            >>> # Parse plain JSON
            >>> result = task._parse_and_validate('{"name": "Alice", "age": 30}')
            >>> result.name
            'Alice'
        """
        return self._fn_parse_and_validate(parse_json(extract_code(content)))


@lru_cache()
def _ask_for_format_prompt(pg_task: LLMTask) -> str:
    """
    Get the format prompt from a prompt generation task with caching.

    This function is cached to avoid regenerating the same format prompt
    multiple times for the same task. The cache is based on the task object
    identity (using its hash), so the same task instance will always return
    the cached result without re-executing the LLM request.

    The caching mechanism significantly improves performance when creating
    multiple tasks with the same data model, as format prompt generation
    can be computationally expensive and involves LLM API calls.

    The function uses Python's built-in :func:`functools.lru_cache` decorator
    with unlimited cache size, meaning all unique task instances will have
    their results cached indefinitely during the program's lifetime.

    :param pg_task: The prompt generation task to execute. This should be an
                   :class:`LLMTask` instance configured to generate format prompts.
    :type pg_task: LLMTask

    :return: The generated format prompt string describing the expected output format.
    :rtype: str

    .. note::
       The cache uses LRU (Least Recently Used) eviction policy with unlimited
       size by default. Consider the memory implications if generating prompts
       for many different data models in long-running applications.

    .. warning::
       The cache is based on task object identity. If you create multiple task
       instances with the same configuration, they will be treated as different
       cache entries. Reuse task instances when possible for optimal caching.

    Example::

        >>> from hbllmutils.meta import create_datamodel_prompt_generation_task
        >>> pg_task = create_datamodel_prompt_generation_task(model, MyModel)
        >>>
        >>> # First call executes the LLM request
        >>> prompt = _ask_for_format_prompt(pg_task)
        >>>
        >>> # Subsequent calls with the same task return cached result
        >>> prompt2 = _ask_for_format_prompt(pg_task)
        >>> prompt == prompt2
        True
        >>>
        >>> # Different task instance is not cached
        >>> pg_task2 = create_datamodel_prompt_generation_task(model, MyModel)
        >>> prompt3 = _ask_for_format_prompt(pg_task2)  # Executes new LLM request
    """
    return pg_task.ask()


def _get_format_prompt(
        datamodel_class: type,
        prompt_generation_model: LLMModel,
        related_datamodel_classes: Optional[List[type]] = None,
) -> str:
    """
    Generate a format prompt for a given data model class.

    This function creates a prompt generation task and retrieves the format
    prompt that describes how to structure data according to the model. The
    generated prompt includes information about the data model fields, types,
    constraints, and any related data models that provide additional context.

    The function leverages the meta-prompt generation system to create
    comprehensive format instructions that guide the LLM in producing
    correctly structured output. The generated prompt is automatically
    cached by :func:`_ask_for_format_prompt`, so repeated calls with the
    same parameters will not regenerate the prompt.

    The format prompt typically includes:

    - Field names and their types
    - Field descriptions and constraints
    - Example values or formats
    - Related model structures for context
    - JSON schema or similar structural information

    :param datamodel_class: The data model class to generate format prompt for.
                           Can be a Pydantic BaseModel, dataclass, or other
                           structured type supported by the meta-prompt system.
    :type datamodel_class: type
    :param prompt_generation_model: The LLM model to use for prompt generation.
                                   This model generates the format instructions
                                   that will guide the main task model.
    :type prompt_generation_model: LLMModel
    :param related_datamodel_classes: Optional list of related data model classes
                                     to include in the prompt for context. These
                                     models provide additional structural information
                                     that may be referenced in the main model.
                                     Defaults to None.
    :type related_datamodel_classes: Optional[List[type]]

    :return: The generated format prompt string with detailed structural instructions.
    :rtype: str

    .. note::
       The generated prompt is cached by :func:`_ask_for_format_prompt`, so repeated
       calls with the same parameters will not regenerate the prompt or make
       additional LLM API calls.

    .. warning::
       Including many related data models may result in very long prompts, which
       could impact token usage and response time. Include only the models that
       are directly relevant to the main task.

    Example::

        >>> from pydantic import BaseModel
        >>> from hbllmutils.model import load_llm_model
        >>>
        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        ...     country: str
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     address: Address
        >>>
        >>> model = load_llm_model('gpt-4')
        >>> format_prompt = _get_format_prompt(
        ...     datamodel_class=Person,
        ...     prompt_generation_model=model,
        ...     related_datamodel_classes=[Address]
        ... )
        >>> "Person" in format_prompt
        True
        >>> "Address" in format_prompt
        True
    """
    pg_task = create_datamodel_prompt_generation_task(
        model=prompt_generation_model,
        datamodel_class=datamodel_class,
        related_datamodel_classes=related_datamodel_classes,
    )
    return _ask_for_format_prompt(pg_task)


def create_datamodel_task(
        model: LLMModelTyping,
        datamodel_class: type,
        task_requirements: str,
        samples: Optional[List[Tuple[str, Any]]] = None,
        related_datamodel_classes: Optional[List[type]] = None,
        prompt_generation_model: Optional[LLMModelTyping] = None,
        fn_parse_and_validate: Optional[Callable[[Any], Any]] = None,
        fn_dump_json: Optional[Callable[[Any], Any]] = None,
) -> DataModelLLMTask:
    """
    Create a DataModelLLMTask with configured prompts and validation.

    This factory function sets up a complete LLM task that:

    - Generates format prompts based on the data model structure
    - Configures task requirements describing the expected behavior
    - Sets up parsing and validation logic for response processing
    - Optionally includes sample inputs and outputs for few-shot learning
    - Handles related data models to provide additional context

    The function automatically handles Pydantic :class:`BaseModel` and dataclass types,
    providing default parsing and serialization functions. For custom types,
    you can provide your own parsing and serialization functions.

    The generated task uses a structured prompt that includes:

    1. **Requirements Section**: Description of what the task should accomplish
    2. **Samples Section** (optional): Input-output examples for few-shot learning
    3. **Output Guide Section**: Format instructions generated from the data model

    The complete system prompt is printed to stdout for debugging and verification
    purposes before the task is created.

    :param model: The LLM model to use for the main task. Can be a model name string,
                 an LLMModel instance, or None for the default model.
    :type model: LLMModelTyping
    :param datamodel_class: The data model class that defines the expected output structure.
                           Must be a Pydantic BaseModel subclass or dataclass, unless
                           custom parsing/serialization functions are provided.
    :type datamodel_class: type
    :param task_requirements: Description of what the task should accomplish. This text
                             is included in the system prompt to guide the LLM's behavior.
                             Can include multiple lines and will be dedented automatically.
    :type task_requirements: str
    :param samples: Optional list of (input, output) tuples to provide as examples for
                   few-shot learning. Each tuple contains a sample input string and
                   the corresponding data model instance. Defaults to None.
    :type samples: Optional[List[Tuple[str, Any]]]
    :param related_datamodel_classes: Optional list of related data model classes for context.
                                     These models are included in the format prompt to provide
                                     additional structural information. Defaults to None.
    :type related_datamodel_classes: Optional[List[type]]
    :param prompt_generation_model: Optional separate model for prompt generation. If None,
                                   uses the main model. Can be useful to use a more capable
                                   model for prompt generation. Defaults to None.
    :type prompt_generation_model: Optional[LLMModelTyping]
    :param fn_parse_and_validate: Optional custom parsing and validation function. Should
                                 accept parsed JSON data and return a validated instance.
                                 If None, uses the default for Pydantic BaseModel
                                 (model_validate). Defaults to None.
    :type fn_parse_and_validate: Optional[Callable[[Any], Any]]
    :param fn_dump_json: Optional custom function to convert data model instances to
                        JSON-serializable dicts. Used for serializing samples. If None,
                        uses the default for Pydantic BaseModel (model_dump) or dataclass
                        (dataclasses.asdict). Defaults to None.
    :type fn_dump_json: Optional[Callable[[Any], Any]]

    :return: A configured DataModelLLMTask instance ready for use.
    :rtype: DataModelLLMTask

    :raises ValueError: If datamodel_class is not a Pydantic BaseModel subclass and
                       fn_parse_and_validate is not provided.
    :raises ValueError: If samples are provided but datamodel_class is not a Pydantic
                       BaseModel or dataclass and fn_dump_json is not provided.

    .. note::
       The function prints the generated system prompt to stdout for debugging purposes.
       This can be useful for understanding what instructions are being sent to the LLM
       and for verifying the prompt structure.

    .. warning::
       Large numbers of samples or complex data models may result in very long prompts,
       which could impact token usage, response time, and API costs. Monitor your
       prompt lengths and adjust accordingly.

    Example::

        >>> from pydantic import BaseModel
        >>> from hbllmutils.model import load_llm_model
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     occupation: str
        >>>
        >>> model = load_llm_model('gpt-4')
        >>> task = create_datamodel_task(
        ...     model=model,
        ...     datamodel_class=Person,
        ...     task_requirements=\"\"\"
        ...         Extract person information from the given text.
        ...         Parse the name, age, and occupation if available.
        ...     \"\"\",
        ...     samples=[
        ...         ("John Doe, 30, software engineer",
        ...          Person(name="John Doe", age=30, occupation="software engineer")),
        ...         ("Alice Smith is 25 and works as a teacher",
        ...          Person(name="Alice Smith", age=25, occupation="teacher")),
        ...     ]
        ... )
        >>> result = task.ask_then_parse("Bob Johnson, age 35, doctor")
        >>> isinstance(result, Person)
        True
        >>> result.name
        'Bob Johnson'
        >>> result.age
        35
        >>> result.occupation
        'doctor'
    """
    if fn_parse_and_validate is None:
        if isinstance(datamodel_class, type) and issubclass(datamodel_class, BaseModel):
            fn_parse_and_validate = datamodel_class.model_validate
        else:
            raise ValueError(
                f"datamodel_class must be a subclass of pydantic.BaseModel when fn_parse_and_validate is not provided. "
                f"Got {datamodel_class.__name__ if hasattr(datamodel_class, '__name__') else datamodel_class}"
            )
    if samples and fn_dump_json is None:
        if isinstance(datamodel_class, type) and issubclass(datamodel_class, BaseModel):
            fn_dump_json = datamodel_class.model_dump
        elif isinstance(datamodel_class, type) and dataclasses.is_dataclass(datamodel_class):
            fn_dump_json = dataclasses.asdict
        else:
            raise ValueError(
                f"datamodel_class must be a subclass of pydantic.BaseModel or a dataclass when fn_dump_json is not provided. "
                f"Got {datamodel_class.__name__ if hasattr(datamodel_class, '__name__') else datamodel_class}"
            )

    format_prompt = textwrap.dedent(_get_format_prompt(
        datamodel_class=datamodel_class,
        related_datamodel_classes=related_datamodel_classes,
        prompt_generation_model=load_llm_model(prompt_generation_model or model),
    )).strip()
    task_requirements = textwrap.dedent(task_requirements).strip()

    with io.StringIO() as sio:
        print(f'# Requirements', file=sio)
        print(f'', file=sio)
        print(task_requirements, file=sio)
        print(f'', file=sio)

        if samples:
            print(f'# Samples', file=sio)
            print(f'', file=sio)
            print(f'Here are {plural_word(len(samples), "sample")} for reference.', file=sio)
            print(f'', file=sio)
            for i, (sample_input, sample_obj) in enumerate(samples, start=1):
                print(f'## Sample #{i}', file=sio)
                print(f'', file=sio)
                print(f'Sample Input:', file=sio)
                print(f'', file=sio)
                print(f'```', file=sio)
                print(textwrap.dedent(sample_input).strip(), file=sio)
                print(f'```', file=sio)
                print(f'', file=sio)
                print(f'Sample Output:', file=sio)
                print(f'', file=sio)
                print(f'```json', file=sio)
                print(json.dumps(fn_dump_json(sample_obj), indent=4, ensure_ascii=False), file=sio)
                print(f'```', file=sio)
                print(f'', file=sio)

        print(f'# Output guide', file=sio)
        print(f'', file=sio)
        print(format_prompt, file=sio)

        system_prompt = textwrap.dedent(sio.getvalue()).strip()

    print(system_prompt)

    history = LLMHistory().with_system_prompt(system_prompt)
    return DataModelLLMTask(
        model=load_llm_model(model),
        history=history,
        fn_parse_and_validate=fn_parse_and_validate
    )
