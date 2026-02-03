import ast
from typing import Optional

from .prompt import get_prompt_for_source_file
from ...history import LLMHistory
from ...model import LLMModel
from ...response import ParsableLLMTask, extract_code


class PythonCodeGenerationLLMTask(ParsableLLMTask):
    def _parse_and_validate(self, content: str):
        code = extract_code(content)
        ast.parse(code)
        return code


class PythonDetailedCodeGenerationLLMTask(PythonCodeGenerationLLMTask):
    def __init__(self, model: LLMModel, code_name: str, description_text: str,
                 history: Optional[LLMHistory] = None, default_max_retries: int = 5,
                 show_module_directory_tree: bool = False, skip_when_error: bool = True):
        super().__init__(model, history, default_max_retries)
        self.code_name = code_name
        self.description_text = description_text
        self.show_module_directory_tree = show_module_directory_tree
        self.skip_when_error = skip_when_error

    def _preprocess_input_content(self, input_content: Optional[str]) -> Optional[str]:
        if input_content:
            return get_prompt_for_source_file(
                source_file=input_content,
                level=1,
                code_name=self.code_name,
                description_text=self.description_text,
                show_module_directory_tree=self.show_module_directory_tree,
                skip_when_error=self.skip_when_error,
            )
        else:
            raise ValueError('Empty content is not supported.')
