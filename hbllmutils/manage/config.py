import os.path
from typing import Dict, Any, Optional

import yaml

from ..model import LLMRemoteModel


class LLMConfig:
    def __init__(self, config):
        self.config = config

    @property
    def models(self):
        return self.config.get('models') or {}

    def get_model_params(self, model_name: Optional[str] = None, **params: Dict[str, Any]):
        models = self.models
        if not model_name:
            model_params = models['__default__']
        elif model_name in models:
            model_params = models[model_name]
        elif '__fallback__' in models:
            model_params = {**models['__fallback__'], 'model_name': model_name}
        else:
            raise KeyError(f'Model {model_name!r} not found, and no __fallback__ is provided.')
        return {**model_params, **params}

    def get_model(self, model_name: Optional[str] = None, **params: Dict[str, Any]):
        model_params = self.get_model_params(model_name=model_name, **params)
        if 'base_url' in model_params:
            return LLMRemoteModel(**model_params)
        else:
            raise ValueError(f'Unknown params for model {model_name!r} - {model_params!r}.')

    @classmethod
    def open_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            return LLMConfig(config=yaml.safe_load(f))

    @classmethod
    def open_from_directory(cls, directory: str):
        return cls.open_from_yaml(os.path.join(directory, '.llmconfig.yaml'))

    @classmethod
    def open(cls, file_or_dir: str = '.'):
        if os.path.isdir(file_or_dir):
            return cls.open_from_directory(file_or_dir)
        else:
            return cls.open_from_yaml(file_or_dir)
