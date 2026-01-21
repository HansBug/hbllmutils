import json
import textwrap
from functools import lru_cache
from typing import Optional, List, Callable, Any

from pydantic import BaseModel

from .code import extract_code
from .parsable import ParsableLLMTask
from ..history import LLMHistory
from ..meta import create_datamodel_prompt_generation_task
from ..model import LLMModel, LLMTask


class DataModelLLMTask(ParsableLLMTask):
    def __init__(self, model: LLMModel, history: LLMHistory,
                 fn_parse_and_validate: Callable[[Any], Any], default_max_retries: int = 5):
        super().__init__(
            model=model,
            history=history,
            default_max_retries=default_max_retries,
        )
        self._fn_parse_and_validate = fn_parse_and_validate

    def _parse_and_validate(self, content: str):
        return self._fn_parse_and_validate(json.loads(extract_code(content)))


@lru_cache()
def _ask_for_format_prompt(pg_task: LLMTask):
    return pg_task.ask()


def _get_format_prompt(
        datamodel_class: type,
        prompt_generation_model: LLMModel,
        related_datamodel_classes: Optional[List[type]] = None,
):
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
        related_datamodel_classes: Optional[List[type]] = None,
        prompt_generation_model: Optional[LLMModel] = None,
        fn_parse_and_validate: Optional[Callable[[Any], Any]] = None,
):
    format_prompt = textwrap.dedent(_get_format_prompt(
        datamodel_class=datamodel_class,
        related_datamodel_classes=related_datamodel_classes,
        prompt_generation_model=prompt_generation_model or model,
    )).strip()
    task_requirements = textwrap.dedent(task_requirements).strip()

    system_prompt = textwrap.dedent(f"""
{task_requirements}

# Output guide

{format_prompt}
    """).strip()

    history = LLMHistory().with_system_prompt(system_prompt)
    if fn_parse_and_validate is None:
        if isinstance(datamodel_class, type) and issubclass(datamodel_class, BaseModel):
            fn_parse_and_validate = datamodel_class.model_validate
        else:
            raise ValueError(
                f"datamodel_class must be a subclass of pydantic.BaseModel when fn_parse_and_validate is not provided. "
                f"Got {datamodel_class.__name__ if hasattr(datamodel_class, '__name__') else datamodel_class}"
            )

    return DataModelLLMTask(
        model=model,
        history=history,
        fn_parse_and_validate=fn_parse_and_validate
    )
