from typing import Optional, List

from .prompt import create_meta_prompt_for_datamodel
from ..task import LLMTask
from ...history import LLMHistory
from ...model import LLMModel


def create_datamodel_prompt_generation_task(
        model: LLMModel,
        datamodel_class: type,
        related_datamodel_classes: Optional[List[type]] = None,
) -> LLMTask:
    return LLMTask(
        model=model,
        history=LLMHistory().append_user(
            create_meta_prompt_for_datamodel(
                datamodel_class=datamodel_class,
                related_datamodel_classes=related_datamodel_classes,
            )
        )
    )
