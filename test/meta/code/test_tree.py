from unittest.mock import MagicMock

import pytest
from pathspec import PathSpec

from hbllmutils.meta.code.tree import is_file_should_ignore


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
    return ("*.bak", "*.log", "*.txt", "temp/")


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
