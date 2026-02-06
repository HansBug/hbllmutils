"""
Unit tests for hbllmutils.template.env module.

This module contains comprehensive tests for the Jinja2 environment enhancement utilities,
including tests for adding Python builtins, custom filters, and environment configuration.
"""

import os
import tempfile

import jinja2
import pytest
from jinja2 import StrictUndefined, Undefined, UndefinedError

from hbllmutils.template.env import add_builtins_to_env, add_settings_for_env, create_env


@pytest.fixture
def basic_env():
    """Create a basic Jinja2 environment for testing."""
    return jinja2.Environment()


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with sample content."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Sample file content\n")
        f.write("Line 2\n")
        f.write("Line 3")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def test_env_var():
    """Set up a test environment variable and clean it up after test."""
    test_key = "TEST_JINJA_ENV_VAR_12345"
    test_value = "test_value_xyz"
    os.environ[test_key] = test_value
    yield test_key, test_value
    if test_key in os.environ:
        del os.environ[test_key]


@pytest.mark.unittest
class TestAddBuiltinsToEnv:
    """Tests for the add_builtins_to_env function."""

    def test_adds_builtin_filters(self, basic_env):
        """Test that Python builtin functions are added as filters."""
        env = add_builtins_to_env(basic_env)

        # Test some common builtin filters
        assert 'str' in env.filters
        assert 'len' in env.filters
        assert 'sorted' in env.filters
        assert 'enumerate' in env.filters
        assert 'reversed' in env.filters

        # Test filter functionality
        template = env.from_string("{{ items | len }}")
        assert template.render(items=[1, 2, 3]) == "3"

    def test_adds_builtin_tests(self, basic_env):
        """Test that Python builtin test functions are added."""
        env = add_builtins_to_env(basic_env)

        # Test functions starting with 'is' should be available as tests
        assert 'instance' in env.tests or 'isinstance' in env.globals
        assert 'callable' in env.tests

        # Test test functionality
        template = env.from_string("{% if items is callable %}yes{% else %}no{% endif %}")
        assert template.render(items=lambda: None) == "yes"
        assert template.render(items=[1, 2, 3]) == "no"

    def test_adds_builtin_globals(self, basic_env):
        """Test that Python builtins are added as global functions."""
        env = add_builtins_to_env(basic_env)

        # Test common builtin globals
        assert 'len' in env.globals
        assert 'str' in env.globals
        assert 'int' in env.globals
        assert 'list' in env.globals
        assert 'dict' in env.globals

        # Test global functionality
        template = env.from_string("{{ len(items) }}")
        assert template.render(items=[1, 2, 3]) == "3"

    def test_does_not_override_existing_filters(self, basic_env):
        """Test that existing filters are not overridden."""
        # Add a custom filter before adding builtins
        basic_env.filters['len'] = lambda x: 999
        env = add_builtins_to_env(basic_env)

        # The custom filter should remain
        template = env.from_string("{{ items | len }}")
        assert template.render(items=[1, 2, 3]) == "999"

    def test_custom_filters_added(self, basic_env):
        """Test that custom filters like keys, values, filter are added."""
        env = add_builtins_to_env(basic_env)

        assert 'keys' in env.filters
        assert 'values' in env.filters
        assert 'filter' in env.filters
        assert 'set' in env.filters
        assert 'dict' in env.filters

    def test_keys_filter(self, basic_env):
        """Test the keys filter functionality."""
        env = add_builtins_to_env(basic_env)
        template = env.from_string("{{ data | keys | list | sort }}")
        result = template.render(data={'a': 1, 'b': 2})
        assert result == "['a', 'b']"

    def test_values_filter(self, basic_env):
        """Test the values filter functionality."""
        env = add_builtins_to_env(basic_env)
        template = env.from_string("{{ data | values | list | sort }}")
        result = template.render(data={'a': 1, 'b': 2})
        assert result == "[1, 2]"

    def test_enumerate_filter(self, basic_env):
        """Test the enumerate filter functionality."""
        env = add_builtins_to_env(basic_env)
        template = env.from_string("{% for i, v in items | enumerate %}{{ i }}:{{ v }} {% endfor %}")
        result = template.render(items=['a', 'b', 'c'])
        assert result == "0:a 1:b 2:c "

    def test_reversed_filter(self, basic_env):
        """Test the reversed filter functionality."""
        env = add_builtins_to_env(basic_env)
        template = env.from_string("{{ items | reversed | list }}")
        result = template.render(items=[1, 2, 3])
        assert result == "[3, 2, 1]"

    def test_excludes_private_functions(self, basic_env):
        """Test that functions starting with underscore are excluded."""
        env = add_builtins_to_env(basic_env)

        # Private functions should not be added
        assert '__import__' not in env.globals
        assert '__name__' not in env.globals

    def test_returns_same_environment(self, basic_env):
        """Test that the function returns the same environment instance."""
        env = add_builtins_to_env(basic_env)
        assert env is basic_env


@pytest.mark.unittest
class TestAddSettingsForEnv:
    """Tests for the add_settings_for_env function."""

    def test_includes_builtins(self, basic_env):
        """Test that add_settings_for_env includes builtin functions."""
        env = add_settings_for_env(basic_env)

        # Should have builtins
        assert 'len' in env.filters
        assert 'str' in env.globals

    def test_adds_indent_filter(self, basic_env):
        """Test that indent filter is added."""
        env = add_settings_for_env(basic_env)

        assert 'indent' in env.filters
        assert 'indent' in env.globals

        # Test indent functionality
        template = env.from_string("{{ text | indent(\"    \") }}")
        result = template.render(text="line1\nline2")
        assert result.startswith("    ")

    def test_adds_plural_word_filter(self, basic_env):
        """Test that plural_word filter is added."""
        env = add_settings_for_env(basic_env)

        assert 'plural' in env.filters
        assert 'plural_word' in env.globals

        # Test plural functionality
        template = env.from_string("{{ 1 | plural('word') }}")
        result = template.render()
        assert result == "1 word"

        template = env.from_string("{{ 2 | plural('word') }}")
        result = template.render()
        assert result == "2 words"

    def test_adds_ordinalize_filter(self, basic_env):
        """Test that ordinalize filter is added."""
        env = add_settings_for_env(basic_env)

        assert 'ordinalize' in env.filters
        assert 'ordinalize' in env.globals

        # Test ordinalize functionality
        template = env.from_string("{{ 1 | ordinalize }}")
        assert template.render() == "1st"

        template = env.from_string("{{ 2 | ordinalize }}")
        assert template.render() == "2nd"

        template = env.from_string("{{ 3 | ordinalize }}")
        assert template.render() == "3rd"

    def test_adds_titleize_filter(self, basic_env):
        """Test that titleize filter is added."""
        env = add_settings_for_env(basic_env)

        assert 'titleize' in env.filters
        assert 'titleize' in env.globals

    def test_adds_read_file_text_filter(self, basic_env, temp_text_file):
        """Test that read_file_text filter is added and works."""
        env = add_settings_for_env(basic_env)

        assert 'read_file_text' in env.filters
        assert 'read_file_text' in env.globals

        # Test reading file
        template = env.from_string("{{ path | read_file_text }}")
        result = template.render(path=temp_text_file)
        assert "Sample file content" in result
        assert "Line 2" in result

    def test_adds_environment_variables(self, basic_env, test_env_var):
        """Test that environment variables are added as globals."""
        test_key, test_value = test_env_var
        env = add_settings_for_env(basic_env)

        assert test_key in env.globals
        assert env.globals[test_key] == test_value

        # Test accessing env var in template
        template = env.from_string(f"{{{{ {test_key} }}}}")
        assert template.render() == test_value

    def test_does_not_override_existing_globals_with_env_vars(self, basic_env):
        """Test that existing globals are not overridden by environment variables."""
        # Set an environment variable
        test_key = "TEST_OVERRIDE_VAR"
        os.environ[test_key] = "env_value"

        # Add a global with the same name
        basic_env.globals[test_key] = "original_value"

        try:
            env = add_settings_for_env(basic_env)

            # The original value should remain
            assert env.globals[test_key] == "original_value"
        finally:
            if test_key in os.environ:
                del os.environ[test_key]

    def test_returns_same_environment(self, basic_env):
        """Test that the function returns the same environment instance."""
        env = add_settings_for_env(basic_env)
        assert env is basic_env


@pytest.mark.unittest
class TestCreateEnv:
    """Tests for the create_env function."""

    def test_creates_environment_with_strict_undefined(self):
        """Test that create_env creates an environment with StrictUndefined by default."""
        env = create_env()

        assert isinstance(env, jinja2.Environment)
        assert env.undefined is StrictUndefined

    def test_creates_environment_with_default_undefined(self):
        """Test that create_env can create an environment with default Undefined."""
        env = create_env(strict_undefined=False)

        assert isinstance(env, jinja2.Environment)
        assert env.undefined is Undefined

    def test_strict_undefined_raises_error(self):
        """Test that StrictUndefined raises error on undefined variables."""
        env = create_env(strict_undefined=True)
        template = env.from_string("{{ undefined_var }}")

        with pytest.raises(UndefinedError):
            template.render()

    def test_default_undefined_returns_empty_string(self):
        """Test that default Undefined returns empty string for undefined variables."""
        env = create_env(strict_undefined=False)
        template = env.from_string("{{ undefined_var }}")

        result = template.render()
        assert result == ""

    def test_includes_all_settings(self):
        """Test that created environment includes all enhancements."""
        env = create_env()

        # Should have builtins
        assert 'len' in env.filters
        assert 'str' in env.globals

        # Should have custom filters
        assert 'indent' in env.filters
        assert 'plural' in env.filters
        assert 'ordinalize' in env.filters
        assert 'titleize' in env.filters
        assert 'read_file_text' in env.filters

    def test_can_render_templates_with_builtins(self):
        """Test that templates can use builtin functions."""
        env = create_env()
        template = env.from_string("{{ items | len }}")
        assert template.render(items=[1, 2, 3]) == "3"

    def test_can_render_templates_with_custom_filters(self):
        """Test that templates can use custom filters."""
        env = create_env()
        template = env.from_string("{{ 3 | ordinalize }}")
        assert template.render() == "3rd"

    @pytest.mark.parametrize("strict,expected_error", [
        (True, True),
        (False, False),
    ])
    def test_strict_undefined_parameter(self, strict, expected_error):
        """Test strict_undefined parameter with different values."""
        env = create_env(strict_undefined=strict)
        template = env.from_string("{{ undefined_var }}")

        if expected_error:
            with pytest.raises(UndefinedError):
                template.render()
        else:
            result = template.render()
            assert result == ""


@pytest.mark.unittest
class TestIntegration:
    """Integration tests for the template environment utilities."""

    def test_complete_template_rendering(self):
        """Test rendering a complex template with multiple features."""
        env = create_env()
        template_str = """
Items count: {{ items | len }}
First item: {{ items[0] }}
Sorted: {{ items | sorted }}
Enumerated: {% for i, v in items | enumerate %}{{ i }}:{{ v }} {% endfor %}
Plural: {{ items | len | plural('item') }}
Ordinalized: {{ 1 | ordinalize }}, {{ 2 | ordinalize }}, {{ 3 | ordinalize }}
"""
        template = env.from_string(template_str)
        result = template.render(items=[3, 1, 2])

        assert "Items count: 3" in result
        assert "First item: 3" in result
        assert "Sorted: [1, 2, 3]" in result
        assert "0:3 1:1 2:2" in result
        assert "3 items" in result
        assert "1st, 2nd, 3rd" in result

    def test_file_reading_in_template(self, temp_text_file):
        """Test reading file content in a template."""
        env = create_env()
        template = env.from_string("Content: {{ path | read_file_text }}")
        result = template.render(path=temp_text_file)

        assert "Sample file content" in result

    def test_environment_variable_access(self, test_env_var):
        """Test accessing environment variables in templates."""
        test_key, test_value = test_env_var
        env = create_env()
        template = env.from_string(f"Env: {{{{ {test_key} }}}}")
        result = template.render()

        assert f"Env: {test_value}" in result

    def test_chained_filters(self):
        """Test using multiple filters in a chain."""
        env = create_env()
        template = env.from_string("{{ items | sorted | reversed | list }}")
        result = template.render(items=[3, 1, 2])

        assert result == "[3, 2, 1]"

    def test_dict_operations(self):
        """Test dictionary operations with custom filters."""
        env = create_env()
        template = env.from_string(
            "Keys: {{ data | keys | list | sorted }}, Values: {{ data | values | list | sorted }}")
        result = template.render(data={'b': 2, 'a': 1})

        assert "Keys: ['a', 'b']" in result
        assert "Values: [1, 2]" in result
