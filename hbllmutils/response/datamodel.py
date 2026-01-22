"""
This module provides functionality for creating and managing LLM tasks that work with structured data models.

It includes classes and functions for:
- Creating LLM tasks that parse and validate responses against data models
- Generating format prompts for data models
- Managing task requirements and validation logic

The main components are:
- DataModelLLMTask: A task class that handles parsing and validation of LLM responses
- create_datamodel_task: Factory function for creating data model tasks with proper configuration
"""
import dataclasses
import io
import json
import textwrap
from functools import lru_cache
from typing import Optional, List, Callable, Any, Tuple

from hbutils.string import plural_word
from pydantic import BaseModel

from .code import extract_code
from .parsable import ParsableLLMTask
from ..history import LLMHistory
from ..meta import create_datamodel_prompt_generation_task
from ..model import LLMModel, LLMTask


class DataModelLLMTask(ParsableLLMTask):
    """
    A specialized LLM task that parses and validates responses against a data model.
    
    This class extends ParsableLLMTask to provide structured data validation
    using a custom parsing and validation function.
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
        and validates it using the configured validation function.
        
        :param content: The raw content string from LLM response.
        :type content: str
        
        :return: The validated data object.
        :rtype: Any
        :raises json.JSONDecodeError: If the content cannot be parsed as JSON.
        :raises ValidationError: If the parsed data fails validation.
        
        Example::
            >>> task._parse_and_validate('```json\\n{"name": "test"}\\n```')
            MyModel(name='test')
        """
        return self._fn_parse_and_validate(json.loads(extract_code(content)))


@lru_cache()
def _ask_for_format_prompt(pg_task: LLMTask):
    """
    Get the format prompt from a prompt generation task with caching.
    
    This function is cached to avoid regenerating the same format prompt
    multiple times for the same task.
    
    :param pg_task: The prompt generation task to execute.
    :type pg_task: LLMTask
    
    :return: The generated format prompt string.
    :rtype: str
    
    Example::
        >>> prompt = _ask_for_format_prompt(my_pg_task)
        >>> # Subsequent calls with the same task return cached result
    """
    return pg_task.ask()


def _get_format_prompt(
        datamodel_class: type,
        prompt_generation_model: LLMModel,
        related_datamodel_classes: Optional[List[type]] = None,
):
    """
    Generate a format prompt for a given data model class.
    
    This function creates a prompt generation task and retrieves the format
    prompt that describes how to structure data according to the model.
    
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
    """
    pg_task = create_datamodel_prompt_generation_task(
        model=prompt_generation_model,
        datamodel_class=datamodel_class,
        related_datamodel_classes=related_datamodel_classes,
    )
    return _ask_for_format_prompt(pg_task)


def create_datamodel_task(
        model: LLMModel,
        datamodel_class: type,
        task_requirements: str,
        samples: Optional[List[Tuple[str, Any]]] = None,
        related_datamodel_classes: Optional[List[type]] = None,
        prompt_generation_model: Optional[LLMModel] = None,
        fn_parse_and_validate: Optional[Callable[[Any], Any]] = None,
        fn_dump_json: Optional[Callable[[Any], Any]] = None,
):
    """
    Create a DataModelLLMTask with configured prompts and validation.
    
    This factory function sets up a complete LLM task that:
    - Generates format prompts based on the data model
    - Configures task requirements
    - Sets up parsing and validation logic
    - Optionally includes sample inputs and outputs for reference
    
    :param model: The LLM model to use for the main task.
    :type model: LLMModel
    :param datamodel_class: The data model class that defines the expected output structure.
    :type datamodel_class: type
    :param task_requirements: Description of what the task should accomplish.
    :type task_requirements: str
    :param samples: Optional list of (input, output) tuples to provide as examples, defaults to None.
    :type samples: Optional[List[Tuple[str, Any]]]
    :param related_datamodel_classes: Optional list of related data model classes for context, defaults to None.
    :type related_datamodel_classes: Optional[List[type]]
    :param prompt_generation_model: Optional separate model for prompt generation, defaults to None (uses main model).
    :type prompt_generation_model: Optional[LLMModel]
    :param fn_parse_and_validate: Optional custom parsing and validation function, defaults to None.
    :type fn_parse_and_validate: Optional[Callable[[Any], Any]]
    :param fn_dump_json: Optional custom function to convert data model instances to JSON-serializable dicts, defaults to None.
    :type fn_dump_json: Optional[Callable[[Any], Any]]
    
    :return: A configured DataModelLLMTask instance.
    :rtype: DataModelLLMTask
    :raises ValueError: If datamodel_class is not a pydantic BaseModel subclass and fn_parse_and_validate is not provided.
    :raises ValueError: If samples are provided but datamodel_class is not a pydantic BaseModel or dataclass and fn_dump_json is not provided.
    
    Example::
        >>> task = create_datamodel_task(
        ...     model=my_llm_model,
        ...     datamodel_class=MyModel,
        ...     task_requirements="Extract user information from the text",
        ...     samples=[
        ...         ("John Doe, age 30", MyModel(name="John Doe", age=30)),
        ...     ],
        ...     related_datamodel_classes=[AddressModel]
        ... )
        >>> result = task.ask("John Doe lives at 123 Main St")
        >>> isinstance(result, MyModel)
        True
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
        prompt_generation_model=prompt_generation_model or model,
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
        model=model,
        history=history,
        fn_parse_and_validate=fn_parse_and_validate
    )
