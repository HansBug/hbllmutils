from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_llm_config():
    """Mock LLMConfig class"""
    config = Mock()
    config.get_model_params = Mock()
    return config


@pytest.fixture
def mock_llm_remote_model():
    """Mock RemoteLLMModel class"""
    return Mock()


@pytest.fixture
def sample_model_params():
    """Sample model parameters"""
    return {
        'base_url': 'https://api.example.com',
        'api_token': 'test-token',
        'model_name': 'gpt-4',
        'temperature': 0.7
    }


@pytest.mark.unittest
class TestLoadLlmModel:

    def test_load_model_from_config_file_success(self, mock_llm_config, mock_llm_remote_model, sample_model_params):
        """Test loading model from config file successfully"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.return_value = sample_model_params

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(config_file_or_dir='./config')

            mock_config_class.open.assert_called_once_with('./config')
            mock_llm_config.get_model_params.assert_called_once_with(model_name=None)
            mock_model_class.assert_called_once_with(**sample_model_params)
            assert result == mock_llm_remote_model

    def test_load_model_from_config_with_none_path(self, mock_llm_config, mock_llm_remote_model, sample_model_params):
        """Test loading model from config with None path defaults to current directory"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.return_value = sample_model_params

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config()

            mock_config_class.open.assert_called_once_with('.')
            mock_llm_config.get_model_params.assert_called_once_with(model_name=None)
            mock_model_class.assert_called_once_with(**sample_model_params)

    def test_load_model_config_file_not_found_with_base_url(self, mock_llm_remote_model):
        """Test loading model when config file not found but base_url provided"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(
                base_url='https://api.example.com',
                api_token='test-token',
                model_name='gpt-4'
            )

            expected_params = {
                'base_url': 'https://api.example.com',
                'api_token': 'test-token',
                'model_name': 'gpt-4'
            }
            mock_model_class.assert_called_once_with(**expected_params)
            assert result == mock_llm_remote_model

    def test_load_model_config_keyerror_with_base_url(self, mock_llm_config, mock_llm_remote_model):
        """Test loading model when model not found in config but base_url provided"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.side_effect = KeyError('Model not found')

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(
                config_file_or_dir='./config',
                base_url='https://api.example.com',
                api_token='test-token',
                model_name='gpt-4'
            )

            expected_params = {
                'base_url': 'https://api.example.com',
                'api_token': 'test-token',
                'model_name': 'gpt-4'
            }
            mock_model_class.assert_called_once_with(**expected_params)

    def test_load_model_with_overrides(self, mock_llm_config, mock_llm_remote_model, sample_model_params):
        """Test loading model from config with base_url and api_token overrides"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.return_value = sample_model_params.copy()

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(
                config_file_or_dir='./config',
                base_url='https://override.example.com',
                api_token='override-token',
                temperature=0.9
            )

            expected_params = sample_model_params.copy()
            expected_params['base_url'] = 'https://override.example.com'
            expected_params['api_token'] = 'override-token'
            expected_params['temperature'] = 0.9

            mock_model_class.assert_called_once_with(**expected_params)

    def test_load_model_with_additional_params(self, mock_llm_config, mock_llm_remote_model, sample_model_params):
        """Test loading model with additional parameters"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.return_value = sample_model_params.copy()

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(
                config_file_or_dir='./config',
                max_tokens=100,
                temperature=0.5
            )

            mock_llm_config.get_model_params.assert_called_once_with(
                model_name=None, max_tokens=100, temperature=0.5
            )

    def test_load_model_base_url_without_api_token_raises_error(self):
        """Test that providing base_url without api_token raises ValueError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(ValueError, match="API token must be specified"):
                load_llm_model_from_config(base_url='https://api.example.com', model_name='gpt-4')

    def test_load_model_base_url_with_none_api_token_raises_error(self):
        """Test that providing base_url with None api_token raises ValueError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(ValueError, match="API token must be specified"):
                load_llm_model_from_config(base_url='https://api.example.com', api_token=None, model_name='gpt-4')

    def test_load_model_base_url_without_model_name_raises_error(self):
        """Test that providing base_url without model_name raises ValueError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(ValueError, match="Model name must be non-empty"):
                load_llm_model_from_config(base_url='https://api.example.com', api_token='test-token')

    def test_load_model_base_url_with_empty_model_name_raises_error(self):
        """Test that providing base_url with empty model_name raises ValueError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(ValueError, match="Model name must be non-empty"):
                load_llm_model_from_config(base_url='https://api.example.com', api_token='test-token', model_name='')

    def test_load_model_no_config_no_base_url_raises_runtime_error(self):
        """Test that no config and no base_url raises RuntimeError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(RuntimeError, match="No model parameters specified"):
                load_llm_model_from_config()

    def test_load_model_config_keyerror_no_base_url_raises_runtime_error(self, mock_llm_config):
        """Test that config KeyError without base_url raises RuntimeError"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.side_effect = KeyError('Model not found')

            from hbllmutils.model import load_llm_model_from_config
            with pytest.raises(RuntimeError, match="No model parameters specified"):
                load_llm_model_from_config(config_file_or_dir='./config')

    def test_load_model_with_model_name_parameter(self, mock_llm_config, mock_llm_remote_model, sample_model_params):
        """Test loading model with specific model_name parameter"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.return_value = mock_llm_config
            mock_llm_config.get_model_params.return_value = sample_model_params

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(config_file_or_dir='./config', model_name='specific-model')

            mock_llm_config.get_model_params.assert_called_once_with(model_name='specific-model')
            mock_model_class.assert_called_once_with(**sample_model_params)

    def test_load_model_new_config_with_additional_params(self, mock_llm_remote_model):
        """Test loading model with new config and additional parameters"""
        with patch('hbllmutils.manage.LLMConfig') as mock_config_class, \
                patch('hbllmutils.model.load.RemoteLLMModel', return_value=mock_llm_remote_model) as mock_model_class:
            mock_config_class.open.side_effect = FileNotFoundError()

            from hbllmutils.model import load_llm_model_from_config
            result = load_llm_model_from_config(
                base_url='https://api.example.com',
                api_token='test-token',
                model_name='gpt-4',
                temperature=0.8,
                max_tokens=150
            )

            expected_params = {
                'base_url': 'https://api.example.com',
                'api_token': 'test-token',
                'model_name': 'gpt-4',
                'temperature': 0.8,
                'max_tokens': 150
            }
            mock_model_class.assert_called_once_with(**expected_params)
