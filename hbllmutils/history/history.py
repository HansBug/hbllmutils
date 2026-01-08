from collections.abc import Sequence
from typing import Union, List, Literal, Optional

from PIL import Image

from .image import to_blob_url

LLMContentTyping = Union[str, Image.Image, List[Union[str, Image.Image]]]
LLMRoleTyping = Literal["system", "user", "assistant", "tool", "function"]


def create_llm_message(message: LLMContentTyping, role: LLMRoleTyping = 'user') -> dict:
    if isinstance(message, str):
        content = message
    elif isinstance(message, Image.Image):
        content = [{"type": "image_url", "image_url": to_blob_url(message)}]
    elif isinstance(message, (list, tuple)):
        content = []
        for i, item in enumerate(message):
            if isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                content.append({"type": "image_url", "image_url": to_blob_url(item)})
            else:
                raise TypeError(f'Unsupported type for message content item at #{i!r} - {item!r}')
    else:
        raise TypeError(f'Unsupported content type - {message!r}')

    return {
        "role": role,
        "content": content
    }


class LLMHistory(Sequence):
    def __init__(self, history: Optional[List[dict]] = None):
        self._history = list(history or [])

    def __getitem__(self, index):
        result = self._history[index]
        if isinstance(result, list):
            return LLMHistory(result)
        else:
            return result

    def __len__(self):
        return len(self._history)

    def append(self, role: LLMRoleTyping, message: LLMContentTyping):
        self._history.append(create_llm_message(message=message, role=role))

    def append_user(self, message: LLMContentTyping):
        return self.append(role='user', message=message)

    def append_assistant(self, message: LLMContentTyping):
        return self.append(role='assistant', message=message)

    def set_system_prompt(self, message: LLMContentTyping):
        message = create_llm_message(message=message, role='system')
        if self._history and self._history[0]['role'] == 'system':
            self._history[0] = message
        else:
            self._history.insert(0, message)

    def to_json(self) -> List[dict]:
        return self._history
