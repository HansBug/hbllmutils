"""
Unit tests for the hbllmutils.template.quick module.

This module contains comprehensive tests for the QuickPromptTemplate class and
quick_render function, covering template rendering, environment preprocessing,
file operations, and error handling.
"""

import pytest
import tempfile
import os
from pathlib import Path
import jinja2

from hbllmutils.template.quick import QuickPromptTemplate, quick_render


@pytest.fixture
def simple_template_text():
    """Provide a simple template string for testing."""
    return "Hello, {{ name }}!"


@pytest.fixture
def complex_template_text():
    """Provide a complex template string with multiple variables."""
    return "{{ greeting }}, {{ name }}! You are {{ age }} years old."


@pytest.fixture
def template_with_filter():
    """Provide a template that uses a custom filter."""
    return "{{ text|reverse }}"


@pytest.fixture
def template_file():
    """Create a temporary template file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello, {{ name }}!")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def complex_template_file():
    """Create a temporary template file with multiple variables."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("{{ greeting }}, {{ name }}! Age: {{ age }}")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def template_with_loop_file():
    """Create a temporary template file with loop structure."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Items:\n{% for item in items %}\n- {{ item }}\n{% endfor %}")
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def add_reverse_filter():
    """Provide an environment preprocessor that adds a reverse filter."""
    def preprocessor(env):
        env.filters['reverse'] = lambda x: x[::-1]
        return env
    return preprocessor


@pytest.fixture
def add_multiple_filters():
    """Provide an environment preprocessor that adds multiple filters."""
    def preprocessor(env):
        env.filters['reverse'] = lambda x: x[::-1]
        env.filters['double'] = lambda x: x * 2
        env.filters['uppercase'] = str.upper
        return env
    return preprocessor


@pytest.mark.unittest
class TestQuickPromptTemplateBasic:
    """Tests for basic QuickPromptTemplate functionality."""

    def test_init_simple_template(self, simple_template_text):
        """Test initialization with a simple template."""
        template = QuickPromptTemplate(simple_template_text)
        assert template is not None
        assert template._template is not None

    def test_init_with_strict_undefined_true(self, simple_template_text):
        """Test initialization with strict_undefined=True."""
        template = QuickPromptTemplate(simple_template_text, strict_undefined=True)
        assert template is not None

    def test_init_with_strict_undefined_false(self, simple_template_text):
        """Test initialization with strict_undefined=False."""
        template = QuickPromptTemplate(simple_template_text, strict_undefined=False)
        assert template is not None

    def test_init_with_env_preprocessor(self, simple_template_text, add_reverse_filter):
        """Test initialization with environment preprocessor."""
        template = QuickPromptTemplate(
            simple_template_text,
            fn_env_preprocess=add_reverse_filter
        )
        assert template is not None
        assert template._fn_env_preprocess is not None

    def test_init_without_env_preprocessor(self, simple_template_text):
        """Test initialization without environment preprocessor."""
        template = QuickPromptTemplate(simple_template_text)
        assert template._fn_env_preprocess is None


@pytest.mark.unittest
class TestQuickPromptTemplateRendering:
    """Tests for QuickPromptTemplate rendering functionality."""

    def test_render_simple_template(self, simple_template_text):
        """Test rendering a simple template with one variable."""
        template = QuickPromptTemplate(simple_template_text)
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_render_complex_template(self, complex_template_text):
        """Test rendering a template with multiple variables."""
        template = QuickPromptTemplate(complex_template_text)
        result = template.render(greeting="Hi", name="Alice", age=30)
        assert result == "Hi, Alice! You are 30 years old."

    @pytest.mark.parametrize("name,expected", [
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
        ("Charlie", "Hello, Charlie!"),
        ("", "Hello, !"),
        ("123", "Hello, 123!"),
    ])
    def test_render_with_various_names(self, simple_template_text, name, expected):
        """Test rendering with various name values."""
        template = QuickPromptTemplate(simple_template_text)
        result = template.render(name=name)
        assert result == expected

    def test_render_with_missing_variable_strict(self, simple_template_text):
        """Test rendering with missing variable in strict mode raises error."""
        template = QuickPromptTemplate(simple_template_text, strict_undefined=True)
        with pytest.raises(jinja2.UndefinedError):
            template.render()

    def test_render_with_missing_variable_non_strict(self, simple_template_text):
        """Test rendering with missing variable in non-strict mode."""
        template = QuickPromptTemplate(simple_template_text, strict_undefined=False)
        result = template.render()
        assert "Hello," in result

    def test_render_with_extra_variables(self, simple_template_text):
        """Test rendering with extra unused variables."""
        template = QuickPromptTemplate(simple_template_text)
        result = template.render(name="World", extra="unused")
        assert result == "Hello, World!"


@pytest.mark.unittest
class TestQuickPromptTemplateWithCustomFilters:
    """Tests for QuickPromptTemplate with custom environment filters."""

    def test_render_with_reverse_filter(self, template_with_filter, add_reverse_filter):
        """Test rendering with a custom reverse filter."""
        template = QuickPromptTemplate(
            template_with_filter,
            fn_env_preprocess=add_reverse_filter
        )
        result = template.render(text="hello")
        assert result == "olleh"

    def test_render_with_multiple_filters(self, add_multiple_filters):
        """Test rendering with multiple custom filters."""
        template_text = "{{ text|reverse }} and {{ number|double }}"
        template = QuickPromptTemplate(
            template_text,
            fn_env_preprocess=add_multiple_filters
        )
        result = template.render(text="hello", number=5)
        assert "olleh" in result
        assert "10" in result

    def test_render_with_uppercase_filter(self, add_multiple_filters):
        """Test rendering with uppercase filter."""
        template_text = "{{ name|uppercase }}"
        template = QuickPromptTemplate(
            template_text,
            fn_env_preprocess=add_multiple_filters
        )
        result = template.render(name="world")
        assert result == "WORLD"

    def test_env_preprocessor_called(self, simple_template_text):
        """Test that environment preprocessor is called during initialization."""
        called = []
        
        def track_calls(env):
            called.append(True)
            return env
        
        template = QuickPromptTemplate(
            simple_template_text,
            fn_env_preprocess=track_calls
        )
        assert len(called) == 1

    def test_env_preprocessor_modifies_environment(self):
        """Test that environment preprocessor can add globals."""
        def add_global(env):
            env.globals['version'] = '1.0.0'
            return env
        
        template_text = "Version: {{ version }}"
        template = QuickPromptTemplate(
            template_text,
            fn_env_preprocess=add_global
        )
        result = template.render()
        assert result == "Version: 1.0.0"


@pytest.mark.unittest
class TestQuickPromptTemplateFromFile:
    """Tests for QuickPromptTemplate.from_file class method."""

    def test_from_file_simple(self, template_file):
        """Test loading a simple template from file."""
        template = QuickPromptTemplate.from_file(template_file)
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_from_file_with_path_object(self, template_file):
        """Test loading template using Path object."""
        template = QuickPromptTemplate.from_file(Path(template_file))
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_from_file_complex(self, complex_template_file):
        """Test loading a complex template from file."""
        template = QuickPromptTemplate.from_file(complex_template_file)
        result = template.render(greeting="Hello", name="Alice", age=25)
        assert "Hello, Alice!" in result
        assert "Age: 25" in result

    def test_from_file_with_strict_undefined(self, template_file):
        """Test loading template with strict_undefined parameter."""
        template = QuickPromptTemplate.from_file(
            template_file,
            strict_undefined=True
        )
        with pytest.raises(jinja2.UndefinedError):
            template.render()

    def test_from_file_with_env_preprocessor(self, add_reverse_filter):
        """Test loading template with environment preprocessor."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("{{ text|reverse }}")
        try:
            template = QuickPromptTemplate.from_file(
                f.name,
                fn_env_preprocess=add_reverse_filter
            )
            result = template.render(text="hello")
            assert result == "olleh"
        finally:
            os.unlink(f.name)

    def test_from_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            QuickPromptTemplate.from_file("/non/existent/file.txt")

    def test_from_file_with_loop(self, template_with_loop_file):
        """Test loading template with loop structure."""
        template = QuickPromptTemplate.from_file(template_with_loop_file)
        result = template.render(items=["apple", "banana", "cherry"])
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result


@pytest.mark.unittest
class TestQuickRenderFunction:
    """Tests for the quick_render convenience function."""

    def test_quick_render_simple(self, template_file):
        """Test quick_render with a simple template."""
        result = quick_render(template_file, name="World")
        assert result == "Hello, World!"

    def test_quick_render_complex(self, complex_template_file):
        """Test quick_render with multiple parameters."""
        result = quick_render(
            complex_template_file,
            greeting="Hi",
            name="Bob",
            age=30
        )
        assert "Hi, Bob!" in result
        assert "Age: 30" in result

    @pytest.mark.parametrize("name,expected_in_result", [
        ("Alice", "Alice"),
        ("Bob", "Bob"),
        ("Charlie", "Charlie"),
    ])
    def test_quick_render_various_names(self, template_file, name, expected_in_result):
        """Test quick_render with various name values."""
        result = quick_render(template_file, name=name)
        assert expected_in_result in result

    def test_quick_render_with_strict_undefined_true(self, template_file):
        """Test quick_render with strict_undefined=True and missing variable."""
        with pytest.raises(jinja2.UndefinedError):
            quick_render(template_file, strict_undefined=True)

    def test_quick_render_with_strict_undefined_false(self, template_file):
        """Test quick_render with strict_undefined=False and missing variable."""
        result = quick_render(template_file, strict_undefined=False)
        assert "Hello," in result

    def test_quick_render_with_env_preprocessor(self, add_reverse_filter):
        """Test quick_render with environment preprocessor."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("{{ text|reverse }}")
        try:
            result = quick_render(
                f.name,
                fn_env_preprocess=add_reverse_filter,
                text="hello"
            )
            assert result == "olleh"
        finally:
            os.unlink(f.name)

    def test_quick_render_with_multiple_filters(self, add_multiple_filters):
        """Test quick_render with multiple custom filters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("{{ text|uppercase }} - {{ number|double }}")
        try:
            result = quick_render(
                f.name,
                fn_env_preprocess=add_multiple_filters,
                text="hello",
                number=5
            )
            assert "HELLO" in result
            assert "10" in result
        finally:
            os.unlink(f.name)

    def test_quick_render_file_not_found(self):
        """Test quick_render with non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            quick_render("/non/existent/file.txt", name="World")

    def test_quick_render_with_loop(self, template_with_loop_file):
        """Test quick_render with template containing loop."""
        result = quick_render(
            template_with_loop_file,
            items=["item1", "item2", "item3"]
        )
        assert "- item1" in result
        assert "- item2" in result
        assert "- item3" in result

    def test_quick_render_empty_params(self, template_file):
        """Test quick_render with no parameters in non-strict mode."""
        result = quick_render(template_file, strict_undefined=False)
        assert "Hello," in result

    def test_quick_render_with_path_object(self, template_file):
        """Test quick_render with Path object."""
        result = quick_render(Path(template_file), name="World")
        assert result == "Hello, World!"


@pytest.mark.unittest
class TestQuickPromptTemplateEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_template(self):
        """Test rendering an empty template."""
        template = QuickPromptTemplate("")
        result = template.render()
        assert result == ""

    def test_template_with_only_text(self):
        """Test template with no variables."""
        template = QuickPromptTemplate("Just plain text")
        result = template.render()
        assert result == "Just plain text"

    def test_template_with_special_characters(self):
        """Test template with special characters."""
        template = QuickPromptTemplate("Hello, {{ name }}! @#$%^&*()")
        result = template.render(name="World")
        assert result == "Hello, World! @#$%^&*()"

    def test_template_with_newlines(self):
        """Test template with newline characters."""
        template_text = "Line 1\nLine 2: {{ value }}\nLine 3"
        template = QuickPromptTemplate(template_text)
        result = template.render(value="test")
        assert "Line 1" in result
        assert "Line 2: test" in result
        assert "Line 3" in result

    def test_template_with_unicode(self):
        """Test template with unicode characters."""
        template = QuickPromptTemplate("Hello, {{ name }}! 你好 🌍")
        result = template.render(name="World")
        assert "Hello, World!" in result
        assert "你好" in result
        assert "🌍" in result

    def test_none_env_preprocessor(self):
        """Test that None env_preprocessor works correctly."""
        template = QuickPromptTemplate("Hello, {{ name }}!", fn_env_preprocess=None)
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_env_preprocessor_returns_same_env(self):
        """Test env preprocessor that returns the same environment."""
        def identity_preprocessor(env):
            return env
        
        template = QuickPromptTemplate(
            "Hello, {{ name }}!",
            fn_env_preprocess=identity_preprocessor
        )
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_template_with_conditional(self):
        """Test template with conditional logic."""
        template_text = "{% if show %}Hello, {{ name }}!{% endif %}"
        template = QuickPromptTemplate(template_text)
        result = template.render(show=True, name="World")
        assert result == "Hello, World!"
        
        result = template.render(show=False, name="World")
        assert result == ""

    def test_template_with_numeric_values(self):
        """Test template with numeric values."""
        template = QuickPromptTemplate("Age: {{ age }}, Count: {{ count }}")
        result = template.render(age=25, count=100)
        assert "Age: 25" in result
        assert "Count: 100" in result

    def test_template_with_boolean_values(self):
        """Test template with boolean values."""
        template = QuickPromptTemplate("Active: {{ active }}, Done: {{ done }}")
        result = template.render(active=True, done=False)
        assert "Active: True" in result
        assert "Done: False" in result


@pytest.mark.unittest
class TestQuickPromptTemplateFileEncoding:
    """Tests for file encoding handling."""

    def test_utf8_encoded_file(self):
        """Test loading UTF-8 encoded template file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Hello, {{ name }}! 你好")
        try:
            template = QuickPromptTemplate.from_file(f.name)
            result = template.render(name="World")
            assert "Hello, World!" in result
            assert "你好" in result
        finally:
            os.unlink(f.name)

    def test_template_file_with_emoji(self):
        """Test loading template file with emoji characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Hello, {{ name }}! 🎉🌟")
        try:
            template = QuickPromptTemplate.from_file(f.name)
            result = template.render(name="World")
            assert "🎉" in result
            assert "🌟" in result
        finally:
            os.unlink(f.name)
