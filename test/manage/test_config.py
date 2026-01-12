import os

import pytest
import yaml
from hbutils.testing import isolated_directory

from hbllmutils.manage import LLMConfig


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        'deepseek': {
            'base_url': 'https://api.deepseek.com/v1',
            'api_token': 'sk-457***af74'
        },
        'aihubmix': {
            'base_url': 'https://aihubmix.com/v1',
            'api_token': 'sk-6B9***F0Ad'
        },
        'aigcbest': {
            'base_url': 'https://api2.aigcbest.top/v1',
            'api_token': 'sk-tbK***49kA'
        },
        'openroute': {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_token': 'sk-or-v1-9bf***a3d4'
        },
        'models': {
            '__default__': {
                'base_url': 'https://api.deepseek.com/v1',
                'api_token': 'sk-457***af74',
                'model_name': 'deepseek-chat'
            },
            'deepseek-R1': {
                'base_url': 'https://api.deepseek.com/v1',
                'api_token': 'sk-457***af74',
                'model_name': 'deepseek-reasoner'
            },
            'deepseek-V3': {
                'base_url': 'https://api.deepseek.com/v1',
                'api_token': 'sk-457***af74',
                'model_name': 'deepseek-chat'
            },
            '__fallback__': {
                'base_url': 'https://aihubmix.com/v1',
                'api_token': 'sk-6B9***F0Ad'
            }
        }
    }


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
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


@pytest.fixture
def empty_config():
    """Empty configuration for testing edge cases."""
    return {}


@pytest.fixture
def config_without_models():
    """Configuration without models section."""
    return {
        'some_other_key': 'value'
    }


@pytest.fixture
def config_with_models_no_default():
    """Configuration with models but no __default__."""
    return {
        'models': {
            'gpt-4': {
                'api_key': 'test-key',
                'model_name': 'gpt-4'
            }
        }
    }


@pytest.fixture
def config_with_models_no_fallback():
    """Configuration with models but no __fallback__."""
    return {
        'models': {
            '__default__': {
                'api_key': 'default-key',
                'model_name': 'default-model'
            },
            'gpt-4': {
                'api_key': 'test-key',
                'model_name': 'gpt-4'
            }
        }
    }


@pytest.fixture
def malformed_yaml_content():
    """Malformed YAML content for testing error cases."""
    return """
models:
  __default__:
    api_key: test
  invalid_yaml: [unclosed list
"""


@pytest.mark.unittest
class TestLLMConfig:
    def test_init(self, sample_config_dict):
        """Test LLMConfig initialization."""
        config = LLMConfig(sample_config_dict)
        assert config.config == sample_config_dict

    def test_models_property_with_models(self, sample_config_dict):
        """Test models property when models section exists."""
        config = LLMConfig(sample_config_dict)
        assert config.models == sample_config_dict['models']

    def test_models_property_without_models(self, config_without_models):
        """Test models property when models section doesn't exist."""
        config = LLMConfig(config_without_models)
        assert config.models == {}

    def test_models_property_empty_config(self, empty_config):
        """Test models property with empty config."""
        config = LLMConfig(empty_config)
        assert config.models == {}

    def test_get_model_params_none_model_name(self, sample_config_dict):
        """Test get_model_params with None model_name (should return __default__)."""
        config = LLMConfig(sample_config_dict)
        params = config.get_model_params(None)
        expected = {
            'base_url': 'https://api.deepseek.com/v1',
            'api_token': 'sk-457***af74',
            'model_name': 'deepseek-chat'
        }
        assert params == expected

    def test_get_model_params_existing_model(self, sample_config_dict):
        """Test get_model_params with existing model name."""
        config = LLMConfig(sample_config_dict)
        params = config.get_model_params('deepseek-R1')
        expected = {
            'base_url': 'https://api.deepseek.com/v1',
            'api_token': 'sk-457***af74',
            'model_name': 'deepseek-reasoner'
        }
        assert params == expected

    def test_get_model_params_fallback(self, sample_config_dict):
        """Test get_model_params with non-existing model (should use __fallback__)."""
        config = LLMConfig(sample_config_dict)
        params = config.get_model_params('non-existing-model')
        expected = {
            'base_url': 'https://aihubmix.com/v1',
            'api_token': 'sk-6B9***F0Ad',
            'model_name': 'non-existing-model'
        }
        assert params == expected

    def test_get_model_params_with_override_params(self, sample_config_dict):
        """Test get_model_params with additional override parameters."""
        config = LLMConfig(sample_config_dict)
        params = config.get_model_params('deepseek-R1', temperature=0.7, max_tokens=1000)
        expected = {
            'base_url': 'https://api.deepseek.com/v1',
            'api_token': 'sk-457***af74',
            'model_name': 'deepseek-reasoner',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        assert params == expected

    def test_get_model_params_override_existing_param(self, sample_config_dict):
        """Test get_model_params with override of existing parameter."""
        config = LLMConfig(sample_config_dict)
        params = config.get_model_params('deepseek-R1', base_url='https://overridden-model.com/v1')
        expected = {
            'base_url': 'https://overridden-model.com/v1',
            'api_token': 'sk-457***af74',
            'model_name': 'deepseek-reasoner'
        }
        assert params == expected

    def test_get_model_params_no_fallback_raises_error(self, config_with_models_no_fallback):
        """Test get_model_params raises KeyError when model not found and no __fallback__."""
        config = LLMConfig(config_with_models_no_fallback)
        with pytest.raises(KeyError, match="Model 'non-existing-model' not found, and no __fallback__ is provided."):
            config.get_model_params('non-existing-model')

    def test_open_from_yaml_valid_file(self, sample_yaml_content):
        """Test open_from_yaml with valid YAML file."""
        with isolated_directory():
            with open('config.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open_from_yaml('config.yaml')
            assert 'models' in config.config
            assert '__default__' in config.models
            assert config.models['__default__']['model_name'] == 'deepseek-chat'

    def test_open_from_yaml_file_not_found(self):
        """Test open_from_yaml raises FileNotFoundError for non-existing file."""
        with isolated_directory():
            with pytest.raises(FileNotFoundError):
                LLMConfig.open_from_yaml('non-existing.yaml')

    def test_open_from_yaml_malformed_yaml(self, malformed_yaml_content):
        """Test open_from_yaml raises YAMLError for malformed YAML."""
        with isolated_directory():
            with open('malformed.yaml', 'w') as f:
                f.write(malformed_yaml_content)

            with pytest.raises(yaml.YAMLError):
                LLMConfig.open_from_yaml('malformed.yaml')

    def test_open_from_directory_valid_directory(self, sample_yaml_content):
        """Test open_from_directory with valid directory containing .llmconfig.yaml."""
        with isolated_directory():
            with open('.llmconfig.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open_from_directory('.')
            assert 'models' in config.config
            assert '__default__' in config.models

    def test_open_from_directory_no_config_file(self):
        """Test open_from_directory raises FileNotFoundError when .llmconfig.yaml doesn't exist."""
        with isolated_directory():
            with pytest.raises(FileNotFoundError):
                LLMConfig.open_from_directory('.')

    def test_open_from_directory_with_subdirectory(self, sample_yaml_content):
        """Test open_from_directory with subdirectory path."""
        with isolated_directory():
            os.makedirs('subdir')
            with open('subdir/.llmconfig.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open_from_directory('subdir')
            assert 'models' in config.config

    def test_open_with_directory_default(self, sample_yaml_content):
        """Test open with default directory (current directory)."""
        with isolated_directory():
            with open('.llmconfig.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open()
            assert 'models' in config.config

    def test_open_with_directory_path(self, sample_yaml_content):
        """Test open with directory path."""
        with isolated_directory():
            os.makedirs('testdir')
            with open('testdir/.llmconfig.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open('testdir')
            assert 'models' in config.config

    def test_open_with_file_path(self, sample_yaml_content):
        """Test open with file path."""
        with isolated_directory():
            with open('config.yaml', 'w') as f:
                f.write(sample_yaml_content)

            config = LLMConfig.open('config.yaml')
            assert 'models' in config.config

    def test_open_with_non_existing_path(self):
        """Test open raises FileNotFoundError for non-existing path."""
        with isolated_directory():
            with pytest.raises(FileNotFoundError, match="No LLM config file or directory found at 'non-existing'"):
                LLMConfig.open('non-existing')

    def test_open_with_non_existing_directory_default(self):
        """Test open raises FileNotFoundError when default directory has no config."""
        with isolated_directory():
            # Current directory exists but has no .llmconfig.yaml
            with pytest.raises(FileNotFoundError):
                LLMConfig.open()
