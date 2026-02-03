import pathlib
import tempfile
from unittest.mock import MagicMock

import pytest
from pathspec import PathSpec

from hbllmutils.meta.code.tree import is_file_should_ignore, build_python_project_tree, get_python_project_tree_text


@pytest.fixture
def sample_paths():
    return [
        "main.py",
        "src/module.py",
        "tests/test_main.py",
        "__pycache__/main.cpython-39.pyc",
        "dist/package-1.0.tar.gz",
        ".vscode/settings.json",
        "requirements.txt"
    ]


@pytest.fixture
def extra_patterns():
    return ["*.txt", "temp/", "*.log"]


@pytest.fixture
def empty_extra_patterns():
    return []


@pytest.fixture
def none_extra_patterns():
    return None


@pytest.fixture
def mock_pathspec():
    mock = MagicMock(spec=PathSpec)
    mock.match_file.return_value = True
    return mock


@pytest.fixture
def sorted_extra_patterns():
    return "*.bak", "*.log", "*.txt", "temp/"


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory with a sample project structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = pathlib.Path(temp_dir) / "test_project"
        project_dir.mkdir()

        # Create files and directories
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
        (project_dir / "__pycache__").mkdir()
        (project_dir / "__pycache__" / "main.cpython-39.pyc").write_text("binary")

        (project_dir / ".vscode").mkdir()
        (project_dir / ".vscode" / "settings.json").write_text("{}")

        (project_dir / "build").mkdir()
        (project_dir / "build" / "lib").mkdir()
        (project_dir / "build" / "lib" / "package.py").write_text("# Build file")

        # Create empty directory that should be filtered out
        (project_dir / "empty_dir").mkdir()

        # Create directory with only ignored files
        ignored_only_dir = project_dir / "ignored_only"
        ignored_only_dir.mkdir()
        (ignored_only_dir / "file.pyc").write_text("binary")

        yield project_dir


@pytest.fixture
def pathlib_paths():
    """Fixture providing pathlib.Path objects for testing."""
    return [
        pathlib.Path("main.py"),
        pathlib.Path("src/module.py"),
        pathlib.Path("__pycache__/test.pyc"),
        pathlib.Path(".vscode/settings.json"),
        pathlib.Path("requirements.txt")
    ]


@pytest.fixture
def focus_items_dict():
    """Fixture for focus items dictionary."""
    return {
        "entry": "src/module.py",
        "test": "tests/test_main.py",
        "config": "requirements.txt"
    }


@pytest.fixture
def temp_single_file():
    """Create a temporary single file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "single_file.py"
        file_path.write_text("# Single file content")
        yield file_path


@pytest.mark.unittest
class TestFileIgnorePatterns:
    def test_is_file_should_ignore_with_none_extra_patterns(self, none_extra_patterns):
        # Should ignore Python cache files
        assert is_file_should_ignore("__pycache__/test.pyc", none_extra_patterns)

        # Should not ignore regular Python files
        assert not is_file_should_ignore("main.py", none_extra_patterns)

    def test_is_file_should_ignore_with_empty_extra_patterns(self, empty_extra_patterns):
        # Should ignore Python cache files
        assert is_file_should_ignore("__pycache__/test.pyc", empty_extra_patterns)

        # Should not ignore regular Python files
        assert not is_file_should_ignore("main.py", empty_extra_patterns)

    def test_is_file_should_ignore_with_extra_patterns(self, extra_patterns):
        # Should ignore files matching extra patterns
        assert is_file_should_ignore("test.txt", extra_patterns)
        assert is_file_should_ignore("temp/file.py", extra_patterns)
        assert is_file_should_ignore("error.log", extra_patterns)

        # Should still ignore default patterns
        assert is_file_should_ignore("__pycache__/test.pyc", extra_patterns)

        # Should not ignore unmatched files
        assert not is_file_should_ignore("main.py", extra_patterns)

    def test_is_file_should_ignore_with_pathlib_path(self, pathlib_paths):
        """Test is_file_should_ignore with pathlib.Path objects."""
        for path in pathlib_paths:
            if "__pycache__" in str(path) or ".vscode" in str(path):
                assert is_file_should_ignore(path), f"Expected {path} to be ignored"
            else:
                assert not is_file_should_ignore(path), f"Expected {path} to not be ignored"

    def test_is_file_should_ignore_pathlib_with_extra_patterns(self, extra_patterns):
        """Test pathlib.Path with extra patterns."""
        path = pathlib.Path("test.txt")
        assert is_file_should_ignore(path, extra_patterns)

        path = pathlib.Path("temp/file.py")
        assert is_file_should_ignore(path, extra_patterns)

        path = pathlib.Path("main.py")
        assert not is_file_should_ignore(path, extra_patterns)

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

    def test_build_python_project_tree_with_focus_items_relative_paths(self, temp_project_dir, focus_items_dict):
        """Test build_python_project_tree with focus items using relative paths."""
        root, tree = build_python_project_tree(str(temp_project_dir), focus_items=focus_items_dict)

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

        # The root focus should not appear in the tree structure since we return tree[1]
        # But it should not raise an error
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
        assert len(tree) == 0  # Single file should result in empty tree

    def test_build_python_project_tree_single_file_with_focus(self, temp_single_file):
        """Test build_python_project_tree with a single file and focus items."""
        focus_items = {"target": str(temp_single_file)}

        root, tree = build_python_project_tree(str(temp_single_file), focus_items=focus_items)

        assert isinstance(tree, list)
        assert len(tree) == 0  # Single file should result in empty tree even with focus

    def test_build_python_project_tree_single_ignored_file(self, temp_single_file):
        """Test build_python_project_tree with a single file that should be ignored."""
        # Create a file that should be ignored
        ignored_file = temp_single_file.parent / "test.pyc"
        ignored_file.write_text("binary")

        root, tree = build_python_project_tree(str(ignored_file))

        assert isinstance(tree, list)
        assert len(tree) == 0  # Ignored file should result in empty tree

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

    def test_build_python_project_tree_nested_structure(self, temp_project_dir):
        """Test build_python_project_tree with nested directory structure."""
        # Create deeply nested structure
        deep_dir = temp_project_dir / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep_file.py").write_text("# Deep file")

        root, tree = build_python_project_tree(str(temp_project_dir))

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

    def test_get_python_project_tree_text_single_file(self, temp_single_file):
        """Test get_python_project_tree_text with a single file."""
        result = get_python_project_tree_text(str(temp_single_file))

        assert isinstance(result, str)
        # Single file results in empty tree, so the result should be minimal
        assert temp_single_file.name in result or result.strip() == temp_single_file.name

    def test_get_python_project_tree_text_single_file_with_focus(self, temp_single_file):
        """Test get_python_project_tree_text with a single file and focus items."""
        focus_items = {"target": str(temp_single_file)}

        result = get_python_project_tree_text(str(temp_single_file), focus_items=focus_items)

        assert isinstance(result, str)
        # Single file with focus should still result in minimal output

    def test_get_python_project_tree_text_single_ignored_file(self, temp_single_file):
        """Test get_python_project_tree_text with a single file that should be ignored."""
        # Create a file that should be ignored
        ignored_file = temp_single_file.parent / "test.pyc"
        ignored_file.write_text("binary")

        result = get_python_project_tree_text(str(ignored_file))

        assert isinstance(result, str)
        # Ignored single file should result in minimal output

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
        assert not is_file_should_ignore(file_path), f"Expected {file_path} to NOT be ignored"

    @pytest.mark.parametrize("patterns1,patterns2,patterns3", [
        (["*.txt", "*.log", "temp/"], ["temp/", "*.log", "*.txt"], ["*.log", "temp/", "*.txt"]),
        (["*.bak", "cache/"], ["cache/", "*.bak"], ["*.bak", "cache/"]),
        (["debug/", "*.tmp", "*.old"], ["*.old", "debug/", "*.tmp"], ["*.tmp", "*.old", "debug/"]),
    ])
    def test_pattern_sorting_consistency(self, patterns1, patterns2, patterns3):
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
        result = is_file_should_ignore(pattern)
        assert result == should_ignore, f"Pattern {pattern} should {'be' if should_ignore else 'not be'} ignored"

    @pytest.mark.parametrize("file_path,extra_patterns,expected", [
        ("test.txt", ["*.txt"], True),
        ("debug.log", ["*.log"], True),
        ("temp/file.py", ["temp/"], True),
        ("main.py", ["*.txt"], False),
        ("__pycache__/test.pyc", ["*.txt"], True),  # Should still match default patterns
        ("custom.xyz", ["*.xyz"], True),
        ("normal.py", ["*.xyz"], False),
        ("backup/data.json", ["backup/"], True),
        ("src/backup.py", ["backup/"], False),  # File named backup, not directory
    ])
    def test_extra_patterns_behavior(self, file_path, extra_patterns, expected):
        result = is_file_should_ignore(file_path, extra_patterns)
        assert result == expected, f"File {file_path} with patterns {extra_patterns} should {'be' if expected else 'not be'} ignored"
