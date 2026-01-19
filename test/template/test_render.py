import pathlib
import tempfile
from unittest.mock import Mock, patch

import jinja2
import pytest

from hbllmutils.template import PromptTemplate


@pytest.fixture
def sample_template_text():
    return "Hello, {{ name }}!"


@pytest.fixture
def complex_template_text():
    return "Hello, {{ name }}! You are {{ age }} years old."


@pytest.fixture
def mock_env():
    env = Mock(spec=jinja2.Environment)
    template = Mock()
    template.render.return_value = "rendered_result"
    env.from_string.return_value = template
    return env


@pytest.fixture
def temp_template_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, {{ name }} from file!")
        temp_path = f.name
    yield temp_path
    pathlib.Path(temp_path).unlink()


@pytest.mark.unittest
class TestPromptTemplate:

    def test_init_with_template_text(self, sample_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(sample_template_text)

            mock_create_env.assert_called_once()
            mock_env.from_string.assert_called_once_with(sample_template_text)
            assert template._template == mock_template

    def test_preprocess_env_default_behavior(self, sample_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(sample_template_text)
            result = template._preprocess_env(mock_env)

            assert result == mock_env

    def test_preprocess_env_called_during_init(self, sample_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(sample_template_text)

            # Verify that _preprocess_env was called with the environment
            mock_env.from_string.assert_called_once_with(sample_template_text)

    def test_render_with_kwargs(self, sample_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_template.render.return_value = "Hello, World!"
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(sample_template_text)
            result = template.render(name="World")

            mock_template.render.assert_called_once_with(name="World")
            assert result == "Hello, World!"

    def test_render_with_multiple_kwargs(self, complex_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_template.render.return_value = "Hello, Alice! You are 30 years old."
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(complex_template_text)
            result = template.render(name="Alice", age=30)

            mock_template.render.assert_called_once_with(name="Alice", age=30)
            assert result == "Hello, Alice! You are 30 years old."

    def test_render_with_no_kwargs(self, sample_template_text):
        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_template.render.return_value = "rendered without args"
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate(sample_template_text)
            result = template.render()

            mock_template.render.assert_called_once_with()
            assert result == "rendered without args"

    def test_from_file_with_string_path(self, temp_template_file):
        with patch('hbllmutils.template.render.auto_decode') as mock_auto_decode, \
                patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_auto_decode.return_value = "Hello, {{ name }} from file!"
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = PromptTemplate.from_file(temp_template_file)

            # Verify auto_decode was called with the file bytes
            mock_auto_decode.assert_called_once()
            # Verify the template was created with the decoded content
            mock_env.from_string.assert_called_once_with("Hello, {{ name }} from file!")
            assert isinstance(template, PromptTemplate)

    def test_from_file_with_pathlib_path(self, temp_template_file):
        with patch('hbllmutils.template.render.auto_decode') as mock_auto_decode, \
                patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_auto_decode.return_value = "Hello, {{ name }} from pathlib!"
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            path_obj = pathlib.Path(temp_template_file)
            template = PromptTemplate.from_file(path_obj)

            mock_auto_decode.assert_called_once()
            mock_env.from_string.assert_called_once_with("Hello, {{ name }} from pathlib!")
            assert isinstance(template, PromptTemplate)

    def test_from_file_reads_file_bytes(self, temp_template_file):
        with patch('hbllmutils.template.render.auto_decode') as mock_auto_decode, \
                patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_auto_decode.return_value = "decoded content"
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            # Mock pathlib.Path.read_bytes to verify it's called
            with patch('pathlib.Path.read_bytes') as mock_read_bytes:
                mock_read_bytes.return_value = b"file content bytes"

                template = PromptTemplate.from_file(temp_template_file)

                mock_read_bytes.assert_called_once()
                mock_auto_decode.assert_called_once_with(b"file content bytes")

    def test_preprocess_env_can_be_overridden(self, sample_template_text):
        class CustomPromptTemplate(PromptTemplate):
            def _preprocess_env(self, env):
                env.custom_attribute = "custom_value"
                return env

        with patch('hbllmutils.template.render.create_env') as mock_create_env:
            mock_env = Mock(spec=jinja2.Environment)
            mock_template = Mock()
            mock_env.from_string.return_value = mock_template
            mock_create_env.return_value = mock_env

            template = CustomPromptTemplate(sample_template_text)

            # Verify the custom preprocessing was applied
            assert hasattr(mock_env, 'custom_attribute')
            assert mock_env.custom_attribute == "custom_value"
