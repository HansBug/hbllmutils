"""
Utilities for loading Large Language Model (LLM) configurations and creating model instances.

This module focuses on creating :class:`~hbllmutils.model.remote.RemoteLLMModel` instances
from configuration files/directories or explicit runtime parameters. It also provides a
convenient dispatcher for loading a model by name, reusing an existing instance, or
falling back to the default configuration.

The module contains the following main components:

* :func:`load_llm_model_from_config` - Load a remote LLM model using configuration and overrides
* :func:`load_llm_model` - Convenience loader that accepts names, instances, or ``None``
* :data:`LLMModelTyping` - Type alias for model identifiers or instances

.. note::
   Configuration loading is delegated to :class:`hbllmutils.manage.LLMConfig`.
   If configuration is missing and no explicit parameters are provided,
   a :class:`RuntimeError` is raised.

Example::

    >>> from hbllmutils.model.load import load_llm_model, load_llm_model_from_config
    >>> # Load a configured model by name (from config files)
    >>> model = load_llm_model("gpt-4")
    >>> # Load the default model (from config files)
    >>> default_model = load_llm_model()
    >>> # Create a model directly with explicit parameters
    >>> direct_model = load_llm_model_from_config(
    ...     base_url="https://api.example.com/v1",
    ...     api_token="your-token",
    ...     model_name="gpt-4"
    ... )

"""

from typing import Any, Dict, Optional, Union

from .base import LLMModel
from .remote import RemoteLLMModel


def load_llm_model_from_config(
        config_file_or_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None,
        **params: Any,
) -> RemoteLLMModel:
    """
    Load a Large Language Model with specified configuration.

    This function attempts to load LLM configuration from a config file or directory,
    and creates a remote model instance. It supports both pre-configured models from
    config files and dynamically specified configurations.

    :param config_file_or_dir: Path to the configuration file or directory. If ``None``,
                               defaults to the current directory.
    :type config_file_or_dir: Optional[str]
    :param base_url: Base URL for the LLM API endpoint. If provided, overrides config
                     file settings.
    :type base_url: Optional[str]
    :param api_token: API token for authentication. Required when ``base_url`` is
                      provided without a config file.
    :type api_token: Optional[str]
    :param model_name: Name of the model to load. Required when ``base_url`` is
                       provided without a config file.
    :type model_name: Optional[str]
    :param params: Additional parameters to pass to the model.
    :type params: Any

    :return: An initialized LLM remote model instance.
    :rtype: RemoteLLMModel

    :raises FileNotFoundError: When config file is not found (handled internally).
    :raises KeyError: When specified model is not found in config (handled internally).
    :raises ValueError: When ``api_token`` is not specified but required, or when
                        ``model_name`` is empty but required.
    :raises RuntimeError: When no model parameters are specified and no local
                          configuration is available.

    Example::
        >>> # Load model from config file
        >>> model = load_llm_model_from_config(config_file_or_dir='./config')
        >>>
        >>> # Load model with explicit parameters
        >>> model = load_llm_model_from_config(
        ...     base_url='https://api.example.com',
        ...     api_token='your-token',
        ...     model_name='gpt-4'
        ... )
        >>>
        >>> # Load model from config with overrides
        >>> model = load_llm_model_from_config(
        ...     config_file_or_dir='./config',
        ...     model_name='gpt-4',
        ...     base_url='https://custom-api.example.com'
        ... )
    """
    from ..manage import LLMConfig
    params: Dict[str, Any]

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
        # Known model is found or generated from the config file.
        if base_url:
            llm_params['base_url'] = base_url
        if api_token:
            llm_params['api_token'] = api_token
        llm_params.update(**params)

    elif base_url:
        # Newly generated LLM config.
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

    return RemoteLLMModel(**llm_params)


#: Type alias for model input, which can be either a string (model name) or an LLMModel instance.
LLMModelTyping = Union[str, LLMModel]


def load_llm_model(model: Optional[LLMModelTyping] = None) -> LLMModel:
    """
    Load a Large Language Model from various input types.

    This is a convenience function that handles different types of model specifications.
    It can load a model by name from configuration, use an existing model instance,
    or load the default model from configuration.

    :param model: The model specification. Can be:
        - A string representing the model name to load from configuration
        - An :class:`~hbllmutils.model.base.LLMModel` instance to use directly
        - ``None`` to load the default model from configuration
    :type model: Optional[LLMModelTyping]

    :return: An initialized LLM model instance.
    :rtype: LLMModel

    :raises TypeError: When model is not a string, :class:`LLMModel` instance, or ``None``.
    :raises ValueError: When model name is invalid or not found in configuration.
    :raises RuntimeError: When no model parameters are specified and no local configuration is available.

    Example::
        >>> # Load model by name from configuration
        >>> model = load_llm_model('gpt-4')
        >>>
        >>> # Use an existing model instance
        >>> existing_model = RemoteLLMModel(base_url='...', api_token='...', model_name='gpt-4')
        >>> model = load_llm_model(existing_model)
        >>>
        >>> # Load default model from configuration
        >>> model = load_llm_model()
    """
    model = model or None
    if isinstance(model, str):
        return load_llm_model_from_config(model_name=model)
    elif isinstance(model, LLMModel):
        return model
    elif model is None:
        return load_llm_model_from_config()
    else:
        raise TypeError(
            f'Model must be a string, LLMModel instance, or None, but got {type(model).__name__}: {model!r}'
        )
