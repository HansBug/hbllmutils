import io
import os
from typing import Optional

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

from .prompt import get_prompt_for_source_file
from .task import PythonCodeGenerationLLMTask
from ...history import LLMHistory
from ...model import LLMModelTyping, load_llm_model, LLMModel
from ...template import PromptTemplate


class UnittestCodeGenerationLLmTask(PythonCodeGenerationLLMTask):
    def __init__(self, model: LLMModel, history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 show_module_directory_tree: bool = False, skip_when_error: bool = True,
                 force_ast_check: bool = True):
        super().__init__(model, history, default_max_retries, force_ast_check)
        self.show_module_directory_tree = show_module_directory_tree
        self.skip_when_error = skip_when_error

    def generate(self, source_file: str, test_file: Optional[str] = None, max_retries: Optional[int] = None, **params):
        with io.StringIO() as sf:
            source_prompt, imported_items = get_prompt_for_source_file(
                source_file=source_file,
                level=1,
                code_name='Code For Unittest Generation',
                description_text='This is the source code for you to generate unittest code',
                show_module_directory_tree=self.show_module_directory_tree,
                skip_when_error=self.skip_when_error,
                return_imported_items=True,
            )
            print(source_prompt, file=sf)
            print(f'', file=sf)

            if test_file:
                test_prompt = get_prompt_for_source_file(
                    source_file=test_file,
                    level=1,
                    code_name='Code Of Existing Unittest',
                    description_text='This is the source code of existing unittest',
                    show_module_directory_tree=self.show_module_directory_tree,
                    skip_when_error=self.skip_when_error,
                    ignore_modules=imported_items,
                )
                print(test_prompt, file=sf)
                print(f'', file=sf)

            prompt = sf.getvalue()

        return self.ask_then_parse(
            input_content=prompt,
            max_retries=max_retries,
            **params,
        )


def create_unittest_generation_task(
        model: LLMModelTyping,
        show_module_directory_tree: bool = False,
        skip_when_error: bool = True,
        force_ast_check: bool = True,
        test_framework_name: Literal['pytest', 'unittest', 'nose2'] = "pytest",
        # decide which system prompt mode will be enabled
        mark_name: Optional[str] = 'unittest',  # will not use @pytest.mark.mark_name when empty
) -> UnittestCodeGenerationLLmTask:
    system_prompt_file = os.path.join(os.path.dirname(__file__), 'unittest_generation.j2')
    system_prompt_template = PromptTemplate.from_file(system_prompt_file)
    system_prompt = system_prompt_template.render(
        test_framework_name=test_framework_name,
        mark_name=mark_name,
    )

    return UnittestCodeGenerationLLmTask(
        model=load_llm_model(model),
        history=LLMHistory().with_system_prompt(system_prompt),
        show_module_directory_tree=show_module_directory_tree,
        skip_when_error=skip_when_error,
        force_ast_check=force_ast_check,
    )
