import os

from .task import PythonDetailedCodeGenerationLLMTask, PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model
from ...template import PromptTemplate


def create_pydoc_generation_task(model: LLMModelTyping, show_module_directory_tree: bool = False,
                                 skip_when_error: bool = True) -> PythonCodeGenerationLLMTask:
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'rst-doc-req.md')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render()

    return PythonDetailedCodeGenerationLLMTask(
        model=load_llm_model(model),
        code_name='Code For Task',
        description_text='This is the source code for you to generate new code with pydoc',
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
    )
