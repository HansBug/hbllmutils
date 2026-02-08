"""
Configuration management for Language Learning Model (LLM) providers.

This module centralizes configuration loading and model parameter retrieval for
LLM-based applications. It provides a single public class, :class:`LLMConfig`,
which can parse YAML configuration files that define providers and model presets.
The configuration format supports default and fallback model definitions, enabling
projects to share common settings while still allowing model-specific overrides.

The module contains the following main components:

* :class:`LLMConfig` - Main configuration manager for LLM settings

.. note::
   Configuration files should be named ``.llmconfig.yaml`` when placed in project
   directories for automatic discovery by :meth:`LLMConfig.open` and
   :meth:`LLMConfig.open_from_directory`.

.. warning::
   API tokens and other sensitive credentials must be kept secure and should never
   be committed to version control systems.

Example::

    >>> from hbllmutils.manage.config import LLMConfig
    >>> 
    >>> # Load configuration from current directory
    >>> config = LLMConfig.open('.')
    >>> 
    >>> # Get parameters for a specific model
    >>> gpt4_params = config.get_model_params('gpt-4o')
    >>> print(gpt4_params['base_url'])
    https://api.openai.com/v1
    >>> 
    >>> # Override parameters
    >>> custom_params = config.get_model_params('gpt-4o', temperature=0.7)
    >>> 
    >>> # Use default model
    >>> default_params = config.get_model_params()

An example configuration file (``.llmconfig.yaml``):

.. code-block:: yaml

    providers:
      # --- International Providers ---
      openai: &openai
        base_url: https://api.openai.com/v1
        api_token: sk-*** # Replace with your API Key

      anthropic_proxy: &anthropic_proxy
        # Anthropic native interface is not OpenAI format
        base_url: https://api.anthropic.com/v1
        api_token: sk-ant-***

      google_gemini: &google_gemini
        base_url: https://generativelanguage.googleapis.com/v1beta/openai/
        api_token: AIza***

      groq: &groq
        base_url: https://api.groq.com/openai/v1
        api_token: gsk_***

      mistral: &mistral
        base_url: https://api.mistral.ai/v1
        api_token: ***

      perplexity: &perplexity
        base_url: https://api.perplexity.ai
        api_token: pplx-***

      openrouter: &openrouter
        base_url: https://openrouter.ai/api/v1
        api_token: sk-or-v1-***

      # --- Chinese Providers ---
      aliyun_qwen: &aliyun_qwen
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        api_token: sk-***

      baidu_qianfan: &baidu_qianfan
        base_url: https://qianfan.baidubce.com/v2
        api_token: *** # Qianfan V2 uses Access Token or API Key

      zhipu_ai: &zhipu_ai
        base_url: https://open.bigmodel.cn/api/paas/v4/
        api_token: ***

      moonshot_kimi: &moonshot_kimi
        base_url: https://api.moonshot.cn/v1
        api_token: sk-***

      volcengine_doubao: &volcengine_doubao
        # Note: Volcengine usually requires endpoint ID as model name
        base_url: https://ark.cn-beijing.volces.com/api/v3
        api_token: ***

      deepseek: &deepseek
        base_url: https://api.deepseek.com/v1
        api_token: sk-***

      yi_01: &yi_01
        base_url: https://api.yi.ai/v1
        api_token: ***

      tencent_hunyuan: &tencent_hunyuan
        base_url: https://api.hunyuan.cloud.tencent.com/v1
        api_token: ***

      aihubmix: &aihubmix
        base_url: https://aihubmix.com/v1
        api_token: sk-6B9***F0Ad

      aigcbest: &aigcbest
        base_url: https://api2.aigcbest.top/v1
        api_token: sk-tbK***49kA

    models:
      # Default model configuration
      __default__:
        <<: *deepseek
        model_name: deepseek-chat

      # OpenAI series
      gpt-4o:
        <<: *openai
        model_name: gpt-4o
      gpt-4o-mini:
        <<: *openai
        model_name: gpt-4o-mini
      o1-preview:
        <<: *openai
        model_name: o1-preview

      # Anthropic series (via adapter)
      claude-3-5-sonnet:
        <<: *anthropic_proxy
        model_name: claude-3-5-sonnet-20240620

      # Google Gemini series
      gemini-1.5-pro:
        <<: *google_gemini
        model_name: gemini-1.5-pro
      gemini-1.5-flash:
        <<: *google_gemini
        model_name: gemini-1.5-flash

      # Groq series (ultra-fast)
      llama-3.3-70b:
        <<: *groq
        model_name: llama-3.3-70b-versatile
      mixtral-8x7b:
        <<: *groq
        model_name: mixtral-8x7b-32768

      # Alibaba Qwen
      qwen-max:
        <<: *aliyun_qwen
        model_name: qwen-max
      qwen-plus:
        <<: *aliyun_qwen
        model_name: qwen-plus
      qwen-turbo:
        <<: *aliyun_qwen
        model_name: qwen-turbo

      # Baidu Wenxin
      ernie-4.0:
        <<: *baidu_qianfan
        model_name: ernie-4.0-turbo-8k

      # Zhipu GLM series
      glm-4-plus:
        <<: *zhipu_ai
        model_name: glm-4-plus
      glm-4-flash:
        <<: *zhipu_ai
        model_name: glm-4-flash

      # Moonshot Kimi
      kimi-v1:
        <<: *moonshot_kimi
        model_name: moonshot-v1-128k

      # Bytedance Doubao
      doubao-pro:
        <<: *volcengine_doubao
        model_name: ep-*** # Replace with your endpoint ID

      # Yi AI
      yi-lightning:
        <<: *yi_01
        model_name: yi-lightning

      # Tencent Hunyuan
      hunyuan-pro:
        <<: *tencent_hunyuan
        model_name: hunyuan-pro

      # Fallback configuration
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
    default and fallback configurations. It enables centralized management of
    multiple LLM providers and models with their respective API endpoints and
    authentication credentials.

    The configuration supports a hierarchical structure with provider definitions
    that can be referenced by multiple models, reducing duplication and improving
    maintainability.

    :param config: The configuration dictionary loaded from YAML
    :type config: Dict[str, Any]

    :ivar config: The complete configuration dictionary
    :vartype config: Dict[str, Any]

    .. note::
       The configuration dictionary should contain a 'models' key with model
       definitions. Special keys '__default__' and '__fallback__' provide
       default behavior when specific models are not found.

    Example::

        >>> config_dict = {
        ...     'models': {
        ...         '__default__': {'base_url': 'https://api.example.com', 'model_name': 'default-model'},
        ...         'gpt-4': {'base_url': 'https://api.openai.com/v1', 'model_name': 'gpt-4'}
        ...     }
        ... }
        >>> config = LLMConfig(config_dict)
        >>> params = config.get_model_params('gpt-4')
        >>> print(params['model_name'])
        gpt-4
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLMConfig with a configuration dictionary.

        :param config: The configuration dictionary loaded from YAML, should contain
                      a 'models' section with model definitions
        :type config: Dict[str, Any]

        Example::

            >>> config_data = {'models': {'gpt-4': {'api_key': 'xxx'}}}
            >>> llm_config = LLMConfig(config_data)
        """
        self.config = config

    @property
    def models(self) -> Dict[str, Any]:
        """
        Get the models configuration dictionary.

        This property provides access to the 'models' section of the configuration,
        which contains all model definitions including special keys like '__default__'
        and '__fallback__'.

        :return: Dictionary containing model configurations, or empty dict if 'models'
                section is not found in the configuration
        :rtype: Dict[str, Any]

        Example::

            >>> config = LLMConfig({'models': {'gpt-4': {'api_key': 'xxx'}}})
            >>> models = config.models
            >>> print('gpt-4' in models)
            True
            >>> 
            >>> # Empty config returns empty dict
            >>> empty_config = LLMConfig({})
            >>> print(empty_config.models)
            {}
        """
        return self.config.get('models') or {}

    def get_model_params(self, model_name: Optional[str] = None, **params: Any) -> Dict[str, Any]:
        """
        Retrieve parameters for a specific model with fallback support.

        This method looks up model parameters in the following priority order:

        1. If model_name is None, returns '__default__' configuration
        2. If model_name exists in models, returns its configuration
        3. If '__fallback__' exists, returns fallback config with the model_name
        4. Otherwise, raises KeyError

        Additional parameters passed as kwargs will override the base configuration,
        allowing runtime customization of model parameters.

        :param model_name: Name of the model to retrieve parameters for. If None,
                          uses '__default__' configuration
        :type model_name: Optional[str]
        :param params: Additional parameters to override the base configuration.
                      Common parameters include temperature, max_tokens, etc.
        :type params: Any

        :return: Dictionary containing the merged model parameters including base
                configuration and any overrides
        :rtype: Dict[str, Any]

        :raises KeyError: If the model is not found in configuration and no
                         '__fallback__' is provided

        .. note::
           The method creates a new dictionary for each call, so modifications
           to the returned dictionary will not affect the original configuration.

        .. warning::
           If both the base configuration and params contain the same key,
           the value from params will take precedence.

        Example::

            >>> config_dict = {
            ...     'models': {
            ...         '__default__': {'base_url': 'https://api.default.com', 'model_name': 'default'},
            ...         'gpt-4': {'base_url': 'https://api.openai.com/v1', 'api_key': 'xxx'},
            ...         '__fallback__': {'base_url': 'https://api.fallback.com'}
            ...     }
            ... }
            >>> config = LLMConfig(config_dict)
            >>> 
            >>> # Get default model
            >>> default_params = config.get_model_params()
            >>> print(default_params['model_name'])
            default
            >>> 
            >>> # Get specific model with overrides
            >>> gpt4_params = config.get_model_params('gpt-4', temperature=0.7, max_tokens=1000)
            >>> print(gpt4_params['temperature'])
            0.7
            >>> 
            >>> # Use fallback for unknown model
            >>> unknown_params = config.get_model_params('unknown-model')
            >>> print(unknown_params['model_name'])
            unknown-model
            >>> 
            >>> # Raises KeyError if model not found and no fallback
            >>> config_no_fallback = LLMConfig({'models': {'gpt-4': {}}})
            >>> try:
            ...     config_no_fallback.get_model_params('gpt-5')
            ... except KeyError as e:
            ...     print(f"Error: {e}")
            Error: 'Model 'gpt-5' not found, and no __fallback__ is provided.'
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

        This class method creates a new LLMConfig instance by loading and parsing
        a YAML configuration file. The YAML file should follow the expected
        configuration structure with a 'models' section.

        :param yaml_file: Path to the YAML configuration file. Can be absolute or
                         relative to the current working directory
        :type yaml_file: str

        :return: A new LLMConfig instance with the loaded configuration
        :rtype: LLMConfig

        :raises FileNotFoundError: If the YAML file does not exist at the specified path
        :raises yaml.YAMLError: If the YAML file is malformed or cannot be parsed
        :raises PermissionError: If the file cannot be read due to permission issues

        .. note::
           This method uses yaml.safe_load() for security, which only constructs
           simple Python objects and prevents arbitrary code execution.

        Example::

            >>> # Load from absolute path
            >>> config = LLMConfig.open_from_yaml('/etc/llm/config.yaml')
            >>> 
            >>> # Load from relative path
            >>> config = LLMConfig.open_from_yaml('configs/llm.yaml')
            >>> 
            >>> # Handle file not found
            >>> try:
            ...     config = LLMConfig.open_from_yaml('nonexistent.yaml')
            ... except FileNotFoundError:
            ...     print("Configuration file not found")
            Configuration file not found
        """
        with open(yaml_file, 'r') as f:
            return LLMConfig(config=yaml.safe_load(f))

    @classmethod
    def open_from_directory(cls, directory: str) -> 'LLMConfig':
        """
        Load LLM configuration from a directory by looking for '.llmconfig.yaml'.

        This class method searches for a file named '.llmconfig.yaml' in the specified
        directory and loads the configuration from it. This is useful for project-based
        configurations where the config file is stored in the project root.

        :param directory: Path to the directory containing '.llmconfig.yaml'. Can be
                         absolute or relative to the current working directory
        :type directory: str

        :return: A new LLMConfig instance with the loaded configuration
        :rtype: LLMConfig

        :raises FileNotFoundError: If '.llmconfig.yaml' does not exist in the
                                  specified directory
        :raises yaml.YAMLError: If the YAML file is malformed or cannot be parsed
        :raises NotADirectoryError: If the provided path is not a directory

        .. note::
           The configuration file must be named exactly '.llmconfig.yaml' (with
           the leading dot) for this method to find it.

        Example::

            >>> # Load from project root
            >>> config = LLMConfig.open_from_directory('/path/to/project')
            >>> 
            >>> # Load from current directory
            >>> config = LLMConfig.open_from_directory('.')
            >>> 
            >>> # Load from parent directory
            >>> config = LLMConfig.open_from_directory('..')
            >>> 
            >>> # Handle missing config file
            >>> try:
            ...     config = LLMConfig.open_from_directory('/tmp')
            ... except FileNotFoundError:
            ...     print("No .llmconfig.yaml found in directory")
            No .llmconfig.yaml found in directory
        """
        return cls.open_from_yaml(os.path.join(directory, '.llmconfig.yaml'))

    @classmethod
    def open(cls, file_or_dir: str = '.') -> 'LLMConfig':
        """
        Load LLM configuration from a file or directory with automatic detection.

        This is a convenience method that automatically detects whether the provided
        path is a file or directory and loads the configuration accordingly:

        - If it's a directory, looks for '.llmconfig.yaml' inside it
        - If it's a file, loads it directly as a YAML configuration

        This method provides the most flexible way to load configurations and is
        recommended for most use cases.

        :param file_or_dir: Path to a configuration file or directory. Defaults to
                           current directory ('.'). Can be absolute or relative
        :type file_or_dir: str

        :return: A new LLMConfig instance with the loaded configuration
        :rtype: LLMConfig

        :raises FileNotFoundError: If the path does not exist, or if it's a directory
                                  without '.llmconfig.yaml', or if it's neither a
                                  file nor a directory
        :raises yaml.YAMLError: If the YAML file is malformed or cannot be parsed

        .. note::
           When using the default parameter ('.'), the method will look for
           '.llmconfig.yaml' in the current working directory.

        Example::

            >>> # Load from current directory (looks for .llmconfig.yaml)
            >>> config = LLMConfig.open()
            >>> 
            >>> # Load from specific directory
            >>> config = LLMConfig.open('/path/to/project')
            >>> 
            >>> # Load from specific file
            >>> config = LLMConfig.open('custom-config.yaml')
            >>> 
            >>> # Load from parent directory
            >>> config = LLMConfig.open('..')
            >>> 
            >>> # Handle various error cases
            >>> try:
            ...     config = LLMConfig.open('/nonexistent/path')
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: No LLM config file or directory found at '/nonexistent/path'.
            >>> 
            >>> # Typical usage in a project
            >>> import os
            >>> project_root = os.path.dirname(os.path.abspath(__file__))
            >>> config = LLMConfig.open(project_root)
            >>> model_params = config.get_model_params('gpt-4o')
        """
        if os.path.isdir(file_or_dir):
            return cls.open_from_directory(file_or_dir)
        elif os.path.isfile(file_or_dir):
            return cls.open_from_yaml(file_or_dir)
        else:
            raise FileNotFoundError(f'No LLM config file or directory found at {file_or_dir!r}.')
