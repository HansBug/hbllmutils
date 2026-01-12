"""
Configuration management module for Language Learning Models (LLM).

This module provides functionality to load and manage LLM configuration from YAML files.
It supports model-specific parameters with fallback and default configurations.

The configuration file should contain a 'models' section with model definitions,
and can include special keys '__default__' and '__fallback__' for default behavior.

Example::
    >>> config = LLMConfig.open('config.yaml')
    >>> params = config.get_model_params('gpt-4')
    >>> # Use params for model initialization

An example configuration file (``.llmconfig.yaml``):

.. code-block:: yaml

    deepseek: &deepseek
      base_url: https://api.deepseek.com/v1
      api_token: sk-457***af74

    aihubmix: &aihubmix
      base_url: https://aihubmix.com/v1
      api_token: sk-6B9***F0Ad

    aigcbest: &aigcbest
      base_url: https://api2.aigcbest.top/v1
      api_token: sk-tbK***49kA

    openroute: &openroute
      base_url: https://openrouter.ai/api/v1
      api_token: sk-or-v1-9bf***a3d4

    models:
      __default__:
        <<: *deepseek
        model_name: deepseek-chat

      deepseek-R1:
        <<: *deepseek
        model_name: deepseek-reasoner

      deepseek-V3:
        <<: *deepseek
        model_name: deepseek-chat

      __fallback__:
        <<: *aihubmix

"""

import os.path
from typing import Dict, Any, Optional

import yaml


class LLMConfig:
    """
    Configuration manager for Language Learning Models.
    
    This class handles loading and accessing LLM configuration from YAML files,
    providing methods to retrieve model-specific parameters with support for
    default and fallback configurations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLMConfig with a configuration dictionary.
        
        :param config: The configuration dictionary loaded from YAML.
        :type config: Dict[str, Any]
        """
        self.config = config

    @property
    def models(self) -> Dict[str, Any]:
        """
        Get the models configuration dictionary.
        
        :return: Dictionary containing model configurations, or empty dict if not found.
        :rtype: Dict[str, Any]
        """
        return self.config.get('models') or {}

    def get_model_params(self, model_name: Optional[str] = None, **params: Any) -> Dict[str, Any]:
        """
        Retrieve parameters for a specific model.
        
        This method looks up model parameters in the following order:
        
        1. If model_name is None, returns '__default__' configuration
        2. If model_name exists in models, returns its configuration
        3. If '__fallback__' exists, returns fallback config with the model_name
        4. Otherwise, raises KeyError
        
        Additional parameters passed as kwargs will override the base configuration.
        
        :param model_name: Name of the model to retrieve parameters for. If None, uses '__default__'.
        :type model_name: Optional[str]
        :param params: Additional parameters to override the base configuration.
        :type params: Any
        
        :return: Dictionary containing the merged model parameters.
        :rtype: Dict[str, Any]
        :raises KeyError: If the model is not found and no __fallback__ is provided.
        
        Example::
            >>> config = LLMConfig({'models': {'gpt-4': {'api_key': 'xxx'}}})
            >>> config.get_model_params('gpt-4', temperature=0.7)
            {'api_key': 'xxx', 'temperature': 0.7}
        """
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

    @classmethod
    def open_from_yaml(cls, yaml_file: str) -> 'LLMConfig':
        """
        Load LLM configuration from a YAML file.
        
        :param yaml_file: Path to the YAML configuration file.
        :type yaml_file: str
        
        :return: A new LLMConfig instance with the loaded configuration.
        :rtype: LLMConfig
        :raises FileNotFoundError: If the YAML file does not exist.
        :raises yaml.YAMLError: If the YAML file is malformed.
        
        Example::
            >>> config = LLMConfig.open_from_yaml('config.yaml')
        """
        with open(yaml_file, 'r') as f:
            return LLMConfig(config=yaml.safe_load(f))

    @classmethod
    def open_from_directory(cls, directory: str) -> 'LLMConfig':
        """
        Load LLM configuration from a directory by looking for '.llmconfig.yaml'.
        
        :param directory: Path to the directory containing '.llmconfig.yaml'.
        :type directory: str
        
        :return: A new LLMConfig instance with the loaded configuration.
        :rtype: LLMConfig
        :raises FileNotFoundError: If '.llmconfig.yaml' does not exist in the directory.
        
        Example::
            >>> config = LLMConfig.open_from_directory('/path/to/project')
        """
        return cls.open_from_yaml(os.path.join(directory, '.llmconfig.yaml'))

    @classmethod
    def open(cls, file_or_dir: str = '.') -> 'LLMConfig':
        """
        Load LLM configuration from a file or directory.
        
        This method automatically detects whether the provided path is a file or directory:
        
        - If it's a directory, looks for '.llmconfig.yaml' inside it
        - If it's a file, loads it directly as a YAML configuration
        
        :param file_or_dir: Path to a configuration file or directory. Defaults to current directory.
        :type file_or_dir: str
        
        :return: A new LLMConfig instance with the loaded configuration.
        :rtype: LLMConfig
        :raises FileNotFoundError: If no valid configuration file or directory is found.
        
        Example::
            >>> config = LLMConfig.open('.')  # Load from current directory
            >>> config = LLMConfig.open('config.yaml')  # Load from specific file
        """
        if os.path.isdir(file_or_dir):
            return cls.open_from_directory(file_or_dir)
        elif os.path.isfile(file_or_dir):
            return cls.open_from_yaml(file_or_dir)
        else:
            raise FileNotFoundError(f'No LLM config file or directory found at {file_or_dir!r}.')
