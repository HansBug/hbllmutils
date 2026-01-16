import pathlib

import jinja2

from .decode import auto_decode
from .env import create_env


class PromptTemplate:
    def __init__(self, template_text: str):
        env = create_env()
        env = self._preprocess_env(env)
        self._template = env.from_string(template_text)

    def _preprocess_env(self, env: jinja2.Environment) -> jinja2.Environment:
        return env

    def render(self, **kwargs) -> str:
        return self._template.render(**kwargs)

    @classmethod
    def from_file(cls, template_file):
        return cls(template_text=auto_decode(pathlib.Path(template_file).read_bytes()))
