import textwrap
from functools import lru_cache
from typing import Optional, List

from ..history import LLMHistory
from ..meta import create_datamodel_prompt_generation_task
from ..model import LLMModel, LLMTask


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
):
    format_prompt = _get_format_prompt(
        datamodel_class=datamodel_class,
        related_datamodel_classes=related_datamodel_classes,
        prompt_generation_model=prompt_generation_model or model,
    )

    system_prompt = textwrap.dedent(f"""
{task_requirements}

# Output guide

{format_prompt}
    """).strip()

    history = LLMHistory().with_system_prompt(system_prompt)
    return LLMTask(
        model=model,
        history=history,
    )
