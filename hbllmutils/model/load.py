"""
This module provides functionality for loading Large Language Model (LLM) configurations and creating remote model instances.

The module handles loading LLM configurations from config files or directories, and creates remote model instances
with appropriate parameters. It supports both pre-configured models and dynamically specified configurations.
"""

from typing import Optional

from .remote import LLMRemoteModel


def load_llm_model(
        config_file_or_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None,
        **params,
) -> LLMRemoteModel:
    """
    Load a Large Language Model with specified configuration.

    This function attempts to load LLM configuration from a config file or directory,
    and creates a remote model instance. It supports both pre-configured models from
    config files and dynamically specified configurations.

    :param config_file_or_dir: Path to the configuration file or directory. If None, defaults to current directory.
    :type config_file_or_dir: Optional[str]
    :param base_url: Base URL for the LLM API endpoint. If provided, overrides config file settings.
    :type base_url: Optional[str]
    :param api_token: API token for authentication. Required when base_url is provided without config file.
    :type api_token: Optional[str]
    :param model_name: Name of the model to load. Required when base_url is provided without config file.
    :type model_name: Optional[str]
    :param params: Additional parameters to pass to the model.
    :type params: dict

    :return: An initialized LLM remote model instance.
    :rtype: LLMRemoteModel

    :raises FileNotFoundError: When config file is not found (handled internally).
    :raises KeyError: When specified model is not found in config (handled internally).
    :raises ValueError: When api_token is not specified but required, or when model_name is empty but required.
    :raises RuntimeError: When no model parameters are specified and no local configuration is available.

    Example::
        >>> # Load model from config file
        >>> model = load_llm_model(config_file_or_dir='./config')
        
        >>> # Load model with explicit parameters
        >>> model = load_llm_model(
        ...     base_url='https://api.example.com',
        ...     api_token='your-token',
        ...     model_name='gpt-4'
        ... )
        
        >>> # Load model from config with overrides
        >>> model = load_llm_model(
        ...     config_file_or_dir='./config',
        ...     model_name='gpt-4',
        ...     base_url='https://custom-api.example.com'
        ... )
    """
    from ..manage import LLMConfig

    try:
        llm_config = LLMConfig.open(config_file_or_dir or '.')
    except FileNotFoundError:
        llm_config = None

    if llm_config:
        try:
            llm_params = llm_config.get_model_params(model_name=model_name, **params)
        except KeyError:
            llm_params = None
    else:
        llm_params = None

    if llm_params is not None:
        # known model is found or generated from the config file
        if base_url:
            llm_params['base_url'] = base_url
        if api_token:
            llm_params['api_token'] = api_token
        llm_params.update(**params)

    elif base_url:
        # newly generated llm config
        llm_params = {'base_url': base_url}
        if api_token is None:
            raise ValueError(f'API token must be specified, but {api_token!r} found.')
        llm_params['api_token'] = api_token
        if not model_name:
            raise ValueError(f'Model name must be non-empty, but {model_name!r} found.')
        llm_params['model_name'] = model_name
        llm_params.update(**params)

    else:
        raise RuntimeError('No model parameters specified and no local configuration for falling back.')

    return LLMRemoteModel(**llm_params)
