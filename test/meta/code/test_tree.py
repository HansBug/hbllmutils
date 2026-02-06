"""
Unit tests for hbllmutils.meta.code.tree module.

This module provides comprehensive unit tests for the file path pattern management
and directory tree building utilities. Tests cover pattern matching, tree building,
text formatting, and various edge cases.

Test classes:
    - TestFileIgnorePatterns: Tests for file ignore pattern matching functionality
    - TestBuildPythonProjectTree: Tests for directory tree building functionality
    - TestGetPythonProjectTreeText: Tests for tree text formatting functionality
    - TestEdgeCases: Tests for edge cases and error conditions
"""

import os
import pathlib
import tempfile

import pytest

from hbllmutils.meta.code.tree import (
    is_file_should_ignore,
    build_python_project_tree,
    get_python_project_tree_text
)


@pytest.fixture
def temp_project_dir():
    """
    Create a temporary directory with a sample project structure.
    
    This fixture creates a realistic Python project structure with various
    files and directories, including some that should be ignored according
    to Python gitignore patterns.
    
    Yields:
        pathlib.Path: Path to the temporary project directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = pathlib.Path(temp_dir) / "test_project"
        project_dir.mkdir()

        # Create regular files
        (project_dir / "main.py").write_text("# Main file")
        (project_dir / "requirements.txt").write_text("pytest>=6.0")
        (project_dir / "README.md").write_text("# Test Project")

        # Create src directory
        src_dir = project_dir / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "module.py").write_text("# Module file")
        (src_dir / "utils.py").write_text("# Utils file")

        # Create tests directory
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_main.py").write_text("# Test file")

        # Create files that should be ignored
        pycache_dir = project_dir / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "main.cpython-39.pyc").write_text("binary")

        vscode_dir = project_dir / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "settings.json").write_text("{}")

        build_dir = project_dir / "build"
        build_dir.mkdir()
        build_lib_dir = build_dir / "lib"
        build_lib_dir.mkdir()
        (build_lib_dir / "package.py").write_text("# Build file")

        # Create empty directory
        (project_dir / "empty_dir").mkdir()

        # Create directory with only ignored files
        ignored_only_dir = project_dir / "ignored_only"
        ignored_only_dir.mkdir()
        (ignored_only_dir / "file.pyc").write_text("binary")

        yield project_dir


@pytest.fixture
def temp_single_file():
    """
    Create a temporary single file for testing.
    
    Yields:
        pathlib.Path: Path to the temporary file
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "single_file.py"
        file_path.write_text("# Single file content")
        yield file_path


@pytest.fixture
def temp_nested_structure():
    """
    Create a temporary directory with deeply nested structure.
    
    Yields:
        pathlib.Path: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = pathlib.Path(temp_dir) / "nested_project"
        base_dir.mkdir()

        # Create deeply nested structure
        deep_dir = base_dir / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep_file.py").write_text("# Deep file")

        # Add some files at intermediate levels
        (base_dir / "level1" / "file1.py").write_text("# Level 1 file")
        (base_dir / "level1" / "level2" / "file2.py").write_text("# Level 2 file")

        yield base_dir


@pytest.mark.unittest
class TestFileIgnorePatterns:
    """Tests for file ignore pattern matching functionality."""

    def test_is_file_should_ignore_with_none_extra_patterns(self):
        """Test is_file_should_ignore with None extra patterns."""
        # Should ignore Python cache files
        assert is_file_should_ignore("__pycache__/test.pyc", None)

        # Should not ignore regular Python files
        assert not is_file_should_ignore("main.py", None)

    def test_is_file_should_ignore_with_empty_extra_patterns(self):
        """Test is_file_should_ignore with empty extra patterns list."""
        # Should ignore Python cache files
        assert is_file_should_ignore("__pycache__/test.pyc", [])

        # Should not ignore regular Python files
        assert not is_file_should_ignore("main.py", [])

    def test_is_file_should_ignore_with_extra_patterns(self):
        """Test is_file_should_ignore with custom extra patterns."""
        extra_patterns = ["*.txt", "temp/", "*.log"]

        # Should ignore files matching extra patterns
        assert is_file_should_ignore("test.txt", extra_patterns)
        assert is_file_should_ignore("temp/file.py", extra_patterns)
        assert is_file_should_ignore("error.log", extra_patterns)

        # Should still ignore default patterns
        assert is_file_should_ignore("__pycache__/test.pyc", extra_patterns)

        # Should not ignore unmatched files
        assert not is_file_should_ignore("main.py", extra_patterns)

    def test_is_file_should_ignore_with_pathlib_path(self):
        """Test is_file_should_ignore with pathlib.Path objects."""
        pathlib_paths = [
            (pathlib.Path("main.py"), False),
            (pathlib.Path("src/module.py"), False),
            (pathlib.Path("__pycache__/test.pyc"), True),
            (pathlib.Path(".vscode/settings.json"), True),
            (pathlib.Path("requirements.txt"), False)
        ]

        for path, should_ignore in pathlib_paths:
            result = is_file_should_ignore(path)
            assert result == should_ignore, f"Expected {path} to {'be' if should_ignore else 'not be'} ignored"

    def test_is_file_should_ignore_pathlib_with_extra_patterns(self):
        """Test pathlib.Path with extra patterns."""
        extra_patterns = ["*.txt", "temp/"]

        assert is_file_should_ignore(pathlib.Path("test.txt"), extra_patterns)
        assert is_file_should_ignore(pathlib.Path("temp/file.py"), extra_patterns)
        assert not is_file_should_ignore(pathlib.Path("main.py"), extra_patterns)

    @pytest.mark.parametrize("file_path", [
        "__pycache__/main.cpython-39.pyc",
        "src/__pycache__/module.cpython-38.pyc",
        "tests/__pycache__/test_utils.cpython-310.pyc",
        "deep/nested/path/__pycache__/helper.cpython-39.pyc",
        "module.pyc",
        "src/core/engine.pyo",
        "utils/helper$py.class",
        "build/lib/package/module.py",
        "dist/package-1.0.0.tar.gz",
        "dist/package-1.0.0-py3-none-any.whl",
        "eggs/package.egg",
        ".eggs/dependency-1.0.egg-info/PKG-INFO",
        "package.egg-info/SOURCES.txt",
        "src/package.egg-info/dependency_links.txt",
        "venv/lib/python3.9/site-packages/requests/__init__.py",
        ".venv/bin/python",
        "env/Scripts/activate.bat",
        "ENV/lib64/python3.8/site-packages/numpy/core.py",
        "project/.venv/pyvenv.cfg",
        ".vscode/settings.json",
        ".vscode/launch.json",
        ".idea/workspace.xml",
        ".idea/misc.xml",
        "src/.vscode/settings.json",
        "tests/.idea/inspectionProfiles/profiles_settings.xml",
        ".pytest_cache/v/cache/nodeids",
        "tests/.pytest_cache/README.md",
        ".coverage",
        ".coverage.xml",
        "htmlcov/index.html",
        "htmlcov/status.json",
        ".tox/py39/lib/python3.9/site-packages/pytest.py",
        ".hypothesis/examples/test_example.py",
        "docs/_build/html/index.html",
        "docs/_build/doctrees/index.doctree",
        "site/index.html",
        "/site/api/reference.html",
        "app.log",
        "debug.log",
        "logs/application.log",
        "logs/error/2023-01-01.log",
        "temp.tmp",
        "backup.bak",
        "src/module.py.bak",
        ".DS_Store",
        "src/.DS_Store",
        "Thumbs.db",
        ".DS_Store?",
        "._hidden_file",
        "Pipfile.lock",
        "poetry.lock",
        ".pdm.toml",
        ".ipynb_checkpoints/notebook-checkpoint.ipynb",
        "notebooks/.ipynb_checkpoints/analysis-checkpoint.ipynb",
        "db.sqlite3",
        "db.sqlite3-journal",
        "media/uploads/image.jpg",
        "staticfiles/css/style.css",
        "local_settings.py",
    ])
    def test_complex_python_project_paths_ignored(self, file_path):
        """Test that complex Python project paths are correctly ignored."""
        assert is_file_should_ignore(file_path), f"Expected {file_path} to be ignored"

    @pytest.mark.parametrize("file_path", [
        "main.py",
        "src/core/engine.py",
        "tests/test_main.py",
        "utils/helpers.py",
        "deep/nested/module/core.py",
        "package/__init__.py",
        "setup.py",
        "conftest.py",
        "manage.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "Makefile",
        "docker-compose.yml",
        "Dockerfile",
    ])
    def test_complex_python_project_paths_not_ignored(self, file_path):
        """Test that valid Python project paths are not ignored."""
        assert not is_file_should_ignore(file_path), f"Expected {file_path} to NOT be ignored"

    @pytest.mark.parametrize("file_path", [
        "level1/__pycache__/file.pyc",
        "level1/level2/__pycache__/file.pyc",
        "level1/level2/level3/__pycache__/file.pyc",
        "src/package/subpackage/__pycache__/module.cpython-39.pyc",
        "tests/unit/integration/__pycache__/test_deep.pyc",
        "deep/very/nested/structure/__pycache__/module.cpython-310.pyc",
        "project/src/core/utils/__pycache__/helper.pyc",
    ])
    def test_nested_directory_structures(self, file_path):
        """Test nested directory structures with ignored patterns."""
        assert is_file_should_ignore(file_path), f"Expected {file_path} to be ignored"

    @pytest.mark.parametrize("file_path", [
        "pycache.py",
        "build.py",
        "dist.py",
        "venv.py",
        "test_cache.py",
        "my_build_script.py",
        "distribution.py",
        "virtual_env.py",
        "cache_utils.py",
        "build_tools.py",
    ])
    def test_edge_cases_with_similar_names(self, file_path):
        """Test edge cases with names similar to ignored patterns."""
        assert not is_file_should_ignore(file_path), f"Expected {file_path} to NOT be ignored"

    @pytest.mark.parametrize("patterns1,patterns2,patterns3", [
        (["*.txt", "*.log", "temp/"], ["temp/", "*.log", "*.txt"], ["*.log", "temp/", "*.txt"]),
        (["*.bak", "cache/"], ["cache/", "*.bak"], ["*.bak", "cache/"]),
        (["debug/", "*.tmp", "*.old"], ["*.old", "debug/", "*.tmp"], ["*.tmp", "*.old", "debug/"]),
    ])
    def test_pattern_sorting_consistency(self, patterns1, patterns2, patterns3):
        """Test that pattern order doesn't affect results (consistency)."""
        test_files = ["test.txt", "debug.log", "temp/file.py", "main.py", "cache/data.json", "backup.bak"]

        for file_path in test_files:
            result1 = is_file_should_ignore(file_path, patterns1)
            result2 = is_file_should_ignore(file_path, patterns2)
            result3 = is_file_should_ignore(file_path, patterns3)

            assert result1 == result2 == result3, f"Inconsistent results for {file_path}"

    @pytest.mark.parametrize("pattern,should_ignore", [
        # Byte-compiled files
        ("__pycache__/", True),
        ("test.pyc", True),
        ("module.pyo", True),
        ("class$py.class", True),

        # Distribution
        ("build/", True),
        ("dist/", True),
        ("package.egg-info/", True),
        ("wheels/", True),

        # Testing
        (".pytest_cache/", True),
        (".coverage", True),
        ("htmlcov/", True),
        (".tox/", True),
        (".hypothesis/", True),

        # Environments
        (".env", True),
        ("venv/", True),
        (".venv", True),
        ("ENV/", True),

        # IDE
        (".vscode/", True),
        (".idea/", True),
        ("*.swp", True),
        ("*.swo", True),

        # OS files
        (".DS_Store", True),
        ("Thumbs.db", True),
        ("thumbs.db", True),

        # Package managers
        ("Pipfile.lock", True),
        ("poetry.lock", True),
        (".pdm.toml", True),

        # Documentation
        ("docs/_build/", True),
        ("/site", True),

        # Logs and temporary
        ("*.log", True),
        ("*.tmp", True),
        ("*.bak", True),
        ("logs/", True),

        # Jupyter
        (".ipynb_checkpoints", True),

        # Django
        ("db.sqlite3", True),
        ("local_settings.py", True),
        ("media/", True),
        ("staticfiles/", True),

        # Valid files that should NOT be ignored
        ("main.py", False),
        ("setup.py", False),
        ("requirements.txt", False),
        ("README.md", False),
        ("pyproject.toml", False),
        ("Makefile", False),
        ("LICENSE", False),
        ("conftest.py", False),
        ("manage.py", False),
        ("__init__.py", False),
        ("config.py", False),
        ("settings.py", False),
        ("urls.py", False),
        ("models.py", False),
        ("views.py", False),
        ("forms.py", False),
        ("admin.py", False),
        ("apps.py", False),
        ("serializers.py", False),
        ("utils.py", False),
        ("constants.py", False),
        ("exceptions.py", False),
    ])
    def test_python_gitignore_patterns_coverage(self, pattern, should_ignore):
        """Test comprehensive coverage of Python gitignore patterns."""
        result = is_file_should_ignore(pattern)
        assert result == should_ignore, f"Pattern {pattern} should {'be' if should_ignore else 'not be'} ignored"

    @pytest.mark.parametrize("file_path,extra_patterns,expected", [
        ("test.txt", ["*.txt"], True),
        ("debug.log", ["*.log"], True),
        ("temp/file.py", ["temp/"], True),
        ("main.py", ["*.txt"], False),
        ("__pycache__/test.pyc", ["*.txt"], True),
        ("custom.xyz", ["*.xyz"], True),
        ("normal.py", ["*.xyz"], False),
        ("backup/data.json", ["backup/"], True),
        ("src/backup.py", ["backup/"], False),
    ])
    def test_extra_patterns_behavior(self, file_path, extra_patterns, expected):
        """Test behavior of extra patterns combined with default patterns."""
        result = is_file_should_ignore(file_path, extra_patterns)
        assert result == expected, f"File {file_path} with patterns {extra_patterns} should {'be' if expected else 'not be'} ignored"


@pytest.mark.unittest
class TestBuildPythonProjectTree:
    """Tests for directory tree building functionality."""

    def test_build_python_project_tree_basic(self, temp_project_dir):
        """Test basic functionality of build_python_project_tree."""
        root, tree = build_python_project_tree(str(temp_project_dir))

        assert root == pathlib.Path(temp_project_dir).name
        assert isinstance(tree, list)

        # Extract names from tree structure
        tree_names = [item[0] for item in tree]

        # Should include regular files and directories
        assert "main.py" in tree_names
        assert "requirements.txt" in tree_names
        assert "README.md" in tree_names
        assert "src" in tree_names
        assert "tests" in tree_names

        # Should not include ignored items
        assert "__pycache__" not in tree_names
        assert ".vscode" not in tree_names
        assert "build" not in tree_names
        assert "empty_dir" not in tree_names
        assert "ignored_only" not in tree_names

    def test_build_python_project_tree_with_extra_patterns(self, temp_project_dir):
        """Test build_python_project_tree with extra ignore patterns."""
        extra_patterns = ["*.txt", "*.md"]
        root, tree = build_python_project_tree(str(temp_project_dir), extra_patterns=extra_patterns)

        tree_names = [item[0] for item in tree]

        # Should exclude files matching extra patterns
        assert "requirements.txt" not in tree_names
        assert "README.md" not in tree_names

        # Should still include other files
        assert "main.py" in tree_names

    def test_build_python_project_tree_with_focus_items_relative_paths(self, temp_project_dir):
        """Test build_python_project_tree with focus items using relative paths."""
        focus_items = {
            "entry": "src/module.py",
            "test": "tests/test_main.py",
            "config": "requirements.txt"
        }

        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

        # Find the focused items in the tree
        def find_focused_items(nodes):
            focused = []
            for name, children in nodes:
                if " <-- (" in name:
                    focused.append(name)
                focused.extend(find_focused_items(children))
            return focused

        focused_items = find_focused_items(tree)

        # Check that focus labels are applied
        assert any("entry" in item for item in focused_items)
        assert any("test" in item for item in focused_items)
        assert any("config" in item for item in focused_items)

    def test_build_python_project_tree_with_focus_items_absolute_paths(self, temp_project_dir):
        """Test build_python_project_tree with focus items using absolute paths."""
        focus_items = {
            "main": str(temp_project_dir / "main.py"),
            "module": str(temp_project_dir / "src" / "module.py")
        }

        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

        def find_focused_items(nodes):
            focused = []
            for name, children in nodes:
                if " <-- (" in name:
                    focused.append(name)
                focused.extend(find_focused_items(children))
            return focused

        focused_items = find_focused_items(tree)

        assert any("main" in item for item in focused_items)
        assert any("module" in item for item in focused_items)

    def test_build_python_project_tree_focus_root_directory(self, temp_project_dir):
        """Test focusing on the root directory itself."""
        focus_items = {"root": str(temp_project_dir)}

        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

        # Should not raise an error
        assert isinstance(tree, list)

    def test_build_python_project_tree_invalid_focus_path(self, temp_project_dir):
        """Test build_python_project_tree with invalid focus path."""
        focus_items = {"invalid": "/completely/different/path/file.py"}

        with pytest.raises(ValueError, match="Focus item .* is not within the root path"):
            build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

    def test_build_python_project_tree_focus_pathlib_path(self, temp_project_dir):
        """Test build_python_project_tree with pathlib.Path focus items."""
        focus_items = {
            "path_obj": pathlib.Path("src/module.py")
        }

        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

        def find_focused_items(nodes):
            focused = []
            for name, children in nodes:
                if " <-- (" in name:
                    focused.append(name)
                focused.extend(find_focused_items(children))
            return focused

        focused_items = find_focused_items(tree)
        assert any("path_obj" in item for item in focused_items)

    def test_build_python_project_tree_single_file(self, temp_single_file):
        """Test build_python_project_tree with a single file."""
        root, tree = build_python_project_tree(str(temp_single_file))

        assert root == pathlib.Path(temp_single_file).name
        assert isinstance(tree, list)
        assert len(tree) == 0

    def test_build_python_project_tree_single_file_with_focus(self, temp_single_file):
        """Test build_python_project_tree with a single file and focus items."""
        focus_items = {"target": str(temp_single_file)}

        root, tree = build_python_project_tree(str(temp_single_file), focus_items=focus_items)

        assert isinstance(tree, list)
        assert len(tree) == 0

    def test_build_python_project_tree_single_ignored_file(self, temp_single_file):
        """Test build_python_project_tree with a single file that should be ignored."""
        ignored_file = temp_single_file.parent / "test.pyc"
        ignored_file.write_text("binary")

        root, tree = build_python_project_tree(str(ignored_file))

        assert isinstance(tree, list)
        assert len(tree) == 0

    def test_build_python_project_tree_nested_structure(self, temp_nested_structure):
        """Test build_python_project_tree with nested directory structure."""
        root, tree = build_python_project_tree(str(temp_nested_structure))

        # Verify nested structure is captured
        def find_deep_file(nodes):
            for name, children in nodes:
                if "deep_file.py" in name:
                    return True
                if find_deep_file(children):
                    return True
            return False

        assert find_deep_file(tree)

    def test_build_python_project_tree_empty_subdirectories_filtered(self, temp_project_dir):
        """Test that empty subdirectories are filtered out."""
        # Create directory with only ignored files
        ignored_dir = temp_project_dir / "only_ignored"
        ignored_dir.mkdir()
        (ignored_dir / "file.pyc").write_text("binary")
        (ignored_dir / "__pycache__").mkdir()

        root, tree = build_python_project_tree(str(temp_project_dir))

        tree_names = [item[0] for item in tree]
        assert "only_ignored" not in tree_names


@pytest.mark.unittest
class TestGetPythonProjectTreeText:
    """Tests for tree text formatting functionality."""

    def test_get_python_project_tree_text_basic(self, temp_project_dir):
        """Test basic functionality of get_python_project_tree_text."""
        result = get_python_project_tree_text(str(temp_project_dir))

        assert isinstance(result, str)
        assert "main.py" in result
        assert "src" in result
        assert "tests" in result

        # Should not contain ignored files
        assert "__pycache__" not in result
        assert ".vscode" not in result

    def test_get_python_project_tree_text_with_extra_patterns(self, temp_project_dir):
        """Test get_python_project_tree_text with extra patterns."""
        extra_patterns = ["*.txt"]
        result = get_python_project_tree_text(str(temp_project_dir), extra_patterns=extra_patterns)

        assert "main.py" in result
        assert "requirements.txt" not in result

    def test_get_python_project_tree_text_with_focus_items(self, temp_project_dir):
        """Test get_python_project_tree_text with focus items."""
        focus_items = {"entry": "main.py", "source": "src/module.py"}
        result = get_python_project_tree_text(str(temp_project_dir), focus_items=focus_items)

        assert "<-- (entry)" in result
        assert "<-- (source)" in result

    def test_get_python_project_tree_text_with_encoding(self, temp_project_dir):
        """Test get_python_project_tree_text with different encodings."""
        # Test with ASCII encoding
        result_ascii = get_python_project_tree_text(str(temp_project_dir), encoding="ascii")
        assert isinstance(result_ascii, str)
        assert "main.py" in result_ascii

        # Test with UTF-8 encoding
        result_utf8 = get_python_project_tree_text(str(temp_project_dir), encoding="utf-8")
        assert isinstance(result_utf8, str)
        assert "main.py" in result_utf8

    def test_get_python_project_tree_text_all_parameters(self, temp_project_dir):
        """Test get_python_project_tree_text with all parameters."""
        extra_patterns = ["*.md"]
        focus_items = {"main": "main.py"}
        encoding = "utf-8"

        result = get_python_project_tree_text(
            str(temp_project_dir),
            extra_patterns=extra_patterns,
            focus_items=focus_items,
            encoding=encoding
        )

        assert isinstance(result, str)
        assert "main.py" in result
        assert "<-- (main)" in result
        assert "README.md" not in result

    def test_get_python_project_tree_text_single_file(self, temp_single_file):
        """Test get_python_project_tree_text with a single file."""
        result = get_python_project_tree_text(str(temp_single_file))

        assert isinstance(result, str)
        assert temp_single_file.name in result or result.strip() == temp_single_file.name

    def test_get_python_project_tree_text_single_file_with_focus(self, temp_single_file):
        """Test get_python_project_tree_text with a single file and focus items."""
        focus_items = {"target": str(temp_single_file)}

        result = get_python_project_tree_text(str(temp_single_file), focus_items=focus_items)

        assert isinstance(result, str)

    def test_get_python_project_tree_text_single_ignored_file(self, temp_single_file):
        """Test get_python_project_tree_text with a single file that should be ignored."""
        ignored_file = temp_single_file.parent / "test.pyc"
        ignored_file.write_text("binary")

        result = get_python_project_tree_text(str(ignored_file))

        assert isinstance(result, str)


@pytest.mark.unittest
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_build_tree_with_multiple_focus_items_same_file(self, temp_project_dir):
        """Test that multiple focus labels on the same file work correctly."""
        # This should work - last one wins or they should be combined
        focus_items = {
            "label1": "main.py",
            "label2": "main.py"
        }

        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items)

        # Should not raise an error
        assert isinstance(tree, list)

    def test_build_tree_with_symlinks(self, temp_project_dir):
        """Test handling of symbolic links in directory structure."""
        # Create a symlink
        link_path = temp_project_dir / "link_to_src"
        src_path = temp_project_dir / "src"

        try:
            link_path.symlink_to(src_path)

            root, tree = build_python_project_tree(str(temp_project_dir))

            # Should handle symlinks without errors
            assert isinstance(tree, list)
        except OSError:
            # Skip test if symlinks are not supported on this platform
            pytest.skip("Symlinks not supported on this platform")

    def test_empty_directory_tree(self):
        """Test building tree for an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root, tree = build_python_project_tree(temp_dir)

            assert isinstance(tree, list)
            assert len(tree) == 0

    def test_directory_with_only_ignored_files(self):
        """Test building tree for directory containing only ignored files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            # Create only ignored files
            (temp_path / "test.pyc").write_text("binary")
            pycache = temp_path / "__pycache__"
            pycache.mkdir()
            (pycache / "module.pyc").write_text("binary")

            root, tree = build_python_project_tree(temp_dir)

            assert isinstance(tree, list)
            assert len(tree) == 0

    def test_very_long_path_names(self, temp_project_dir):
        """Test handling of very long path names."""
        # Create a file with a very long name
        long_name = "a" * 200 + ".py"
        long_file = temp_project_dir / long_name
        long_file.write_text("# Long name file")

        root, tree = build_python_project_tree(str(temp_project_dir))

        # Should handle long names without errors
        assert isinstance(tree, list)

    def test_special_characters_in_filenames(self, temp_project_dir):
        """Test handling of special characters in filenames."""
        # Create files with special characters
        special_files = [
            "file with spaces.py",
            "file-with-dashes.py",
            "file_with_underscores.py",
            "file.multiple.dots.py"
        ]

        for filename in special_files:
            (temp_project_dir / filename).write_text("# Special file")

        root, tree = build_python_project_tree(str(temp_project_dir))

        # Should handle special characters without errors
        assert isinstance(tree, list)

        tree_names = [item[0] for item in tree]
        for filename in special_files:
            assert filename in tree_names

    def test_unicode_filenames(self, temp_project_dir):
        """Test handling of Unicode characters in filenames."""
        unicode_files = [
            "文件.py",
            "файл.py",
            "αρχείο.py"
        ]

        for filename in unicode_files:
            try:
                (temp_project_dir / filename).write_text("# Unicode file")
            except (OSError, UnicodeEncodeError):
                # Skip if filesystem doesn't support Unicode
                continue

        root, tree = build_python_project_tree(str(temp_project_dir))

        # Should handle Unicode without errors
        assert isinstance(tree, list)

    def test_case_sensitivity(self, temp_project_dir):
        """Test case sensitivity in pattern matching."""
        # Create files with different cases
        (temp_project_dir / "Test.PY").write_text("# Test file")
        (temp_project_dir / "TEST.py").write_text("# Test file")

        root, tree = build_python_project_tree(str(temp_project_dir))

        tree_names = [item[0] for item in tree]

        # Both should be included (not ignored)
        assert "Test.PY" in tree_names or "TEST.py" in tree_names

    def test_relative_vs_absolute_paths(self, temp_project_dir):
        """Test that relative and absolute paths produce same results."""
        # Build with absolute path
        root1, tree1 = build_python_project_tree(str(temp_project_dir.resolve()))

        # Build with relative path (if possible)
        try:
            rel_path = os.path.relpath(temp_project_dir)
            root2, tree2 = build_python_project_tree(rel_path)

            # Results should be equivalent
            assert root1 == root2
            assert len(tree1) == len(tree2)
        except ValueError:
            # Skip if relative path cannot be computed
            pass
