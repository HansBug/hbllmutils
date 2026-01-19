import os
import pathlib
import tempfile
from unittest.mock import MagicMock

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
def template_file_content():
    return "Welcome, {{ user }}! Today is {{ day }}."


@pytest.fixture
def temp_template_file(template_file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(template_file_content)
        temp_file_path = f.name

    yield temp_file_path

    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)


@pytest.fixture
def mock_jinja_env():
    env = MagicMock(spec=jinja2.Environment)
    template_mock = MagicMock()
    env.from_string.return_value = template_mock
    return env, template_mock


@pytest.mark.unittest
class TestPromptTemplate:

    def test_init_basic(self, sample_template_text):
        """Test basic initialization of PromptTemplate."""
        template = PromptTemplate(sample_template_text)
        assert template._template is not None

    def test_init_with_empty_string(self):
        """Test initialization with empty template string."""
        template = PromptTemplate("")
        assert template._template is not None

    def test_init_with_complex_template(self, complex_template_text):
        """Test initialization with complex template containing multiple variables."""
        template = PromptTemplate(complex_template_text)
        assert template._template is not None

    def test_preprocess_env_default_behavior(self, sample_template_text):
        """Test that _preprocess_env returns the environment unchanged by default."""
        template = PromptTemplate(sample_template_text)

        # Create a mock environment
        mock_env = MagicMock(spec=jinja2.Environment)

        # Test the _preprocess_env method
        result = template._preprocess_env(mock_env)

        # Should return the same environment object
        assert result is mock_env

    def test_render_simple_template(self, sample_template_text):
        """Test rendering a simple template with one variable."""
        template = PromptTemplate(sample_template_text)
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_render_complex_template(self, complex_template_text):
        """Test rendering a template with multiple variables."""
        template = PromptTemplate(complex_template_text)
        result = template.render(name="Alice", age=30)
        assert result == "Hello, Alice! You are 30 years old."

    def test_render_with_no_variables(self):
        """Test rendering a template with no variables."""
        template = PromptTemplate("Hello, World!")
        result = template.render()
        assert result == "Hello, World!"

    def test_render_with_extra_variables(self, sample_template_text):
        """Test rendering with extra variables that aren't used in template."""
        template = PromptTemplate(sample_template_text)
        result = template.render(name="World", extra="unused")
        assert result == "Hello, World!"

    def test_render_with_missing_variables(self, sample_template_text):
        """Test rendering with missing required variables raises an error."""
        template = PromptTemplate(sample_template_text)
        with pytest.raises(jinja2.UndefinedError):
            template.render()

    def test_render_with_various_data_types(self):
        """Test rendering with different data types."""
        template = PromptTemplate("Number: {{ num }}, Boolean: {{ flag }}, List: {{ items }}")
        result = template.render(num=42, flag=True, items=[1, 2, 3])
        assert result == "Number: 42, Boolean: True, List: [1, 2, 3]"

    def test_from_file_with_string_path(self, temp_template_file):
        """Test creating PromptTemplate from file using string path."""
        template = PromptTemplate.from_file(temp_template_file)
        result = template.render(user="John", day="Monday")
        assert result == "Welcome, John! Today is Monday."

    def test_from_file_with_pathlib_path(self, temp_template_file):
        """Test creating PromptTemplate from file using pathlib.Path."""
        path_obj = pathlib.Path(temp_template_file)
        template = PromptTemplate.from_file(path_obj)
        result = template.render(user="Jane", day="Tuesday")
        assert result == "Welcome, Jane! Today is Tuesday."

    def test_from_file_nonexistent_file(self):
        """Test creating PromptTemplate from non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file("nonexistent_file.txt")

    def test_from_file_with_different_encodings(self):
        """Test from_file works with different file encodings."""
        # Create a temporary file with UTF-8 encoding containing special characters
        content = "Héllo, {{ nàme }}! 你好"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_file_path = f.name

        try:
            template = PromptTemplate.from_file(temp_file_path)
            result = template.render(nàme="Wörld")
            assert "Héllo, Wörld! 你好" == result
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_template_with_jinja2_features(self):
        """Test template using various Jinja2 features like filters and conditionals."""
        template_text = """
        {%- if name is defined -%}
        Hello, {{ name | upper }}!
        {%- else -%}
        Hello, Anonymous!
        {%- endif -%}
        """
        template = PromptTemplate(template_text)

        result_with_name = template.render(name="alice")
        assert result_with_name.strip() == "Hello, ALICE!"

        result_without_name = template.render()
        assert result_without_name.strip() == "Hello, Anonymous!"

    def test_template_inheritance_preprocess_env(self):
        """Test that subclasses can override _preprocess_env."""

        class CustomPromptTemplate(PromptTemplate):
            def _preprocess_env(self, env):
                env.globals['custom_var'] = 'custom_value'
                return env

        template = CustomPromptTemplate("Hello {{ custom_var }}!")
        result = template.render()
        assert result == "Hello custom_value!"

    def test_multiple_renders_same_template(self, sample_template_text):
        """Test that the same template can be rendered multiple times with different data."""
        template = PromptTemplate(sample_template_text)

        result1 = template.render(name="Alice")
        result2 = template.render(name="Bob")
        result3 = template.render(name="Charlie")

        assert result1 == "Hello, Alice!"
        assert result2 == "Hello, Bob!"
        assert result3 == "Hello, Charlie!"
