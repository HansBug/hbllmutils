"""
Data model-based LLM task module.

This module provides functionality for creating and managing LLM tasks that parse and validate
responses against structured data models. It supports both Pydantic models and dataclasses,
with automatic prompt generation and validation capabilities.

Key Features:
    - Structured data validation using Pydantic or dataclasses
    - Automatic format prompt generation
    - Sample-based learning support
    - Retry mechanism for failed validations
    - JSON parsing and validation

Main Components:
    - DataModelLLMTask: Core task class for model-based validation
    - create_datamodel_task: Factory function for creating configured tasks

Example::
    >>> from pydantic import BaseModel
    >>> from hbllmutils.model import load_llm_model_from_config
    >>> from hbllmutils.response import create_datamodel_task
    >>> 
    >>> class Person(BaseModel):
    ...     gender: str  # male or female
    ...     age: int
    ...     hair_color: str  # use hex color
    ...     skin_color: str  # use readable color
    ...     appearance_desc: str  # a line of text for description of this guy
    >>> 
    >>> model = load_llm_model_from_config(model_name='gpt-4o')
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
    
    This class extends ParsableLLMTask to provide structured data validation
    using a custom parsing and validation function. It handles the complete workflow
    of sending prompts to an LLM, receiving responses, and validating them against
    a predefined data model structure.
    
    :param model: The LLM model to use for generating responses.
    :type model: LLMModel
    :param history: The conversation history to maintain context.
    :type history: LLMHistory
    :param fn_parse_and_validate: Function to parse and validate the response data.
    :type fn_parse_and_validate: Callable[[Any], Any]
    :param default_max_retries: Maximum number of retries for failed attempts, defaults to 5.
    :type default_max_retries: int
    
    Example::
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: int
        >>> task = DataModelLLMTask(
        ...     model=my_model,
        ...     history=my_history,
        ...     fn_parse_and_validate=MyModel.model_validate
        ... )
        >>> result = task.ask_then_parse("Extract info: John is 30 years old")
        >>> isinstance(result, MyModel)
        True
    """

    def __init__(self, model: LLMModel, history: LLMHistory,
                 fn_parse_and_validate: Callable[[Any], Any], default_max_retries: int = 5):
        """
        Initialize a DataModelLLMTask instance.
        
        :param model: The LLM model to use for generating responses.
        :type model: LLMModel
        :param history: The conversation history to maintain context.
        :type history: LLMHistory
        :param fn_parse_and_validate: Function to parse and validate the response data.
        :type fn_parse_and_validate: Callable[[Any], Any]
        :param default_max_retries: Maximum number of retries for failed attempts, defaults to 5.
        :type default_max_retries: int
        
        Example::
            >>> task = DataModelLLMTask(
            ...     model=my_model,
            ...     history=my_history,
            ...     fn_parse_and_validate=MyModel.model_validate
            ... )
        """
        super().__init__(
            model=model,
            history=history,
            default_max_retries=default_max_retries,
        )
        self._fn_parse_and_validate = fn_parse_and_validate

    def _parse_and_validate(self, content: str):
        """
        Parse and validate the content from LLM response.
        
        This method extracts code from the content, parses it as JSON,
        and validates it using the configured validation function. It handles
        the complete parsing pipeline including code extraction, JSON parsing,
        and data model validation.
        
        :param content: The raw content string from LLM response.
        :type content: str
        
        :return: The validated data object.
        :rtype: Any
        :raises json.JSONDecodeError: If the content cannot be parsed as JSON.
        :raises ValidationError: If the parsed data fails validation.
        
        Example::
            >>> task._parse_and_validate('```json\\n{"name": "test", "age": 25}\\n```')
            MyModel(name='test', age=25)
        """
        return self._fn_parse_and_validate(parse_json(extract_code(content)))


@lru_cache()
def _ask_for_format_prompt(pg_task: LLMTask) -> str:
    """
    Get the format prompt from a prompt generation task with caching.
    
    This function is cached to avoid regenerating the same format prompt
    multiple times for the same task. The cache is based on the task object
    identity, so the same task instance will always return the cached result.
    
    :param pg_task: The prompt generation task to execute.
    :type pg_task: LLMTask
    
    :return: The generated format prompt string.
    :rtype: str
    
    Example::
        >>> prompt = _ask_for_format_prompt(my_pg_task)
        >>> # Subsequent calls with the same task return cached result
        >>> prompt2 = _ask_for_format_prompt(my_pg_task)
        >>> prompt == prompt2
        True
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
    and any related data models that provide additional context.
    
    :param datamodel_class: The data model class to generate format prompt for.
    :type datamodel_class: type
    :param prompt_generation_model: The LLM model to use for prompt generation.
    :type prompt_generation_model: LLMModel
    :param related_datamodel_classes: Optional list of related data model classes, defaults to None.
    :type related_datamodel_classes: Optional[List[type]]
    
    :return: The generated format prompt string.
    :rtype: str
    
    Example::
        >>> format_prompt = _get_format_prompt(
        ...     datamodel_class=MyModel,
        ...     prompt_generation_model=my_model
        ... )
        >>> "MyModel" in format_prompt
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
    - Generates format prompts based on the data model
    - Configures task requirements
    - Sets up parsing and validation logic
    - Optionally includes sample inputs and outputs for reference
    
    The function automatically handles Pydantic BaseModel and dataclass types,
    providing default parsing and serialization functions. For custom types,
    you can provide your own parsing and serialization functions.
    
    :param model: The LLM model to use for the main task.
    :type model: ModelTyping
    :param datamodel_class: The data model class that defines the expected output structure.
    :type datamodel_class: type
    :param task_requirements: Description of what the task should accomplish.
    :type task_requirements: str
    :param samples: Optional list of (input, output) tuples to provide as examples, defaults to None.
    :type samples: Optional[List[Tuple[str, Any]]]
    :param related_datamodel_classes: Optional list of related data model classes for context, defaults to None.
    :type related_datamodel_classes: Optional[List[type]]
    :param prompt_generation_model: Optional separate model for prompt generation, defaults to None (uses main model).
    :type prompt_generation_model: Optional[ModelTyping]
    :param fn_parse_and_validate: Optional custom parsing and validation function, defaults to None.
    :type fn_parse_and_validate: Optional[Callable[[Any], Any]]
    :param fn_dump_json: Optional custom function to convert data model instances to JSON-serializable dicts, defaults to None.
    :type fn_dump_json: Optional[Callable[[Any], Any]]
    
    :return: A configured DataModelLLMTask instance.
    :rtype: DataModelLLMTask
    :raises ValueError: If datamodel_class is not a pydantic BaseModel subclass and fn_parse_and_validate is not provided.
    :raises ValueError: If samples are provided but datamodel_class is not a pydantic BaseModel or dataclass and fn_dump_json is not provided.
    
    Example::
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: int
        >>> task = create_datamodel_task(
        ...     model=my_llm_model,
        ...     datamodel_class=MyModel,
        ...     task_requirements="Extract user information from the text",
        ...     samples=[
        ...         ("John Doe, age 30", MyModel(name="John Doe", age=30)),
        ...     ],
        ...     related_datamodel_classes=[AddressModel]
        ... )
        >>> result = task.ask_then_parse("Jane Smith is 25 years old")
        >>> isinstance(result, MyModel)
        True
        >>> result.name
        'Jane Smith'
        >>> result.age
        25
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
