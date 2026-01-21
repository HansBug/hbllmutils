"""
This module provides functionality for creating LLM tasks that generate prompts for datamodels.

The module facilitates the creation of specialized LLM tasks designed to generate prompts
based on datamodel classes and their related classes. It integrates with the LLM history
and model components to produce structured prompt generation tasks.
"""

from typing import Optional, List

from .prompt import create_meta_prompt_for_datamodel
from ...history import LLMHistory
from ...model import LLMModel, LLMTask


def create_datamodel_prompt_generation_task(
        model: LLMModel,
        datamodel_class: type,
        related_datamodel_classes: Optional[List[type]] = None,
) -> LLMTask:
    """
    Create an LLM task for generating prompts based on a datamodel class.

    This function creates a specialized LLM task that generates prompts for a given
    datamodel class. It constructs a meta-prompt based on the datamodel class and
    optionally related datamodel classes, then wraps it in an LLM task with appropriate
    history context.

    :param model: The LLM model to use for the task.
    :type model: LLMModel
    :param datamodel_class: The datamodel class for which to generate prompts.
    :type datamodel_class: type
    :param related_datamodel_classes: Optional list of related datamodel classes to
        include in the prompt generation context. Defaults to None.
    :type related_datamodel_classes: Optional[List[type]]

    :return: An LLM task configured for datamodel prompt generation.
    :rtype: LLMTask

    Example::
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class MyDataModel:
        ...     pass
        >>> task = create_datamodel_prompt_generation_task(model, MyDataModel)
        >>> # task is now ready to be executed
    """
    return LLMTask(
        model=model,
        history=LLMHistory().with_user_message(
            create_meta_prompt_for_datamodel(
                datamodel_class=datamodel_class,
                related_datamodel_classes=related_datamodel_classes,
            )
        )
    )
