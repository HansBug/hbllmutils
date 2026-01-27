import os
import tempfile

import pytest

from hbllmutils.meta.code.module import get_pythonpath_of_source_file


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_package_structure(temp_dir):
    # 创建简单包结构: package/__init__.py, package/module.py
    package_dir = os.path.join(temp_dir, 'package')
    os.makedirs(package_dir)

    init_file = os.path.join(package_dir, '__init__.py')
    module_file = os.path.join(package_dir, 'module.py')

    with open(init_file, 'w') as f:
        f.write('')
    with open(module_file, 'w') as f:
        f.write('')

    return temp_dir, package_dir, init_file, module_file


@pytest.fixture
def nested_package_structure(temp_dir):
    # 创建嵌套包结构: root/package/subpackage/__init__.py, root/package/subpackage/module.py
    root_dir = os.path.join(temp_dir, 'root')
    package_dir = os.path.join(root_dir, 'package')
    subpackage_dir = os.path.join(package_dir, 'subpackage')
    os.makedirs(subpackage_dir)

    root_init = os.path.join(root_dir, '__init__.py')
    package_init = os.path.join(package_dir, '__init__.py')
    subpackage_init = os.path.join(subpackage_dir, '__init__.py')
    module_file = os.path.join(subpackage_dir, 'module.py')

    for init_file in [root_init, package_init, subpackage_init]:
        with open(init_file, 'w') as f:
            f.write('')

    with open(module_file, 'w') as f:
        f.write('')

    return temp_dir, root_dir, package_dir, subpackage_dir, module_file


@pytest.fixture
def no_init_structure(temp_dir):
    # 创建没有__init__.py的目录结构
    dir_path = os.path.join(temp_dir, 'no_package')
    os.makedirs(dir_path)

    module_file = os.path.join(dir_path, 'module.py')
    with open(module_file, 'w') as f:
        f.write('')

    return temp_dir, dir_path, module_file


@pytest.fixture
def mixed_separators_structure(temp_dir):
    # 创建用于测试路径分隔符的结构
    package_dir = os.path.join(temp_dir, 'package')
    subdir = os.path.join(package_dir, 'subdir')
    os.makedirs(subdir)

    package_init = os.path.join(package_dir, '__init__.py')
    subdir_init = os.path.join(subdir, '__init__.py')
    module_file = os.path.join(subdir, 'module.py')

    for init_file in [package_init, subdir_init]:
        with open(init_file, 'w') as f:
            f.write('')

    with open(module_file, 'w') as f:
        f.write('')

    return temp_dir, package_dir, subdir, module_file


@pytest.fixture
def file_with_extension_structure(temp_dir):
    # 创建带不同扩展名的文件
    package_dir = os.path.join(temp_dir, 'package')
    os.makedirs(package_dir)

    init_file = os.path.join(package_dir, '__init__.py')
    py_file = os.path.join(package_dir, 'module.py')
    pyx_file = os.path.join(package_dir, 'cython_module.pyx')
    no_ext_file = os.path.join(package_dir, 'no_extension')

    with open(init_file, 'w') as f:
        f.write('')

    for file_path in [py_file, pyx_file, no_ext_file]:
        with open(file_path, 'w') as f:
            f.write('')

    return temp_dir, package_dir, py_file, pyx_file, no_ext_file


@pytest.fixture
def root_level_file_structure(temp_dir):
    # 创建根级别文件（没有包结构）
    module_file = os.path.join(temp_dir, 'standalone.py')
    with open(module_file, 'w') as f:
        f.write('')

    return temp_dir, module_file


@pytest.mark.unittest
class TestGetPythonpathOfSourceFile:

    def test_simple_package_structure(self, simple_package_structure):
        temp_dir, package_dir, init_file, module_file = simple_package_structure

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == temp_dir
        assert module_text == 'package.module'

    def test_nested_package_structure(self, nested_package_structure):
        temp_dir, root_dir, package_dir, subpackage_dir, module_file = nested_package_structure

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == temp_dir
        assert module_text == 'root.package.subpackage.module'

    def test_no_init_file_structure(self, no_init_structure):
        temp_dir, dir_path, module_file = no_init_structure

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == dir_path
        assert module_text == 'module'

    def test_root_level_file(self, root_level_file_structure):
        temp_dir, module_file = root_level_file_structure

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == temp_dir
        assert module_text == 'standalone'

    def test_init_file_itself(self, simple_package_structure):
        temp_dir, package_dir, init_file, module_file = simple_package_structure

        module_dir, module_text = get_pythonpath_of_source_file(init_file)

        assert module_dir == temp_dir
        assert module_text == 'package.__init__'

    def test_different_file_extensions(self, file_with_extension_structure):
        temp_dir, package_dir, py_file, pyx_file, no_ext_file = file_with_extension_structure

        # Test .py file
        module_dir, module_text = get_pythonpath_of_source_file(py_file)
        assert module_dir == temp_dir
        assert module_text == 'package.module'

        # Test .pyx file
        module_dir, module_text = get_pythonpath_of_source_file(pyx_file)
        assert module_dir == temp_dir
        assert module_text == 'package.cython_module'

        # Test file without extension
        module_dir, module_text = get_pythonpath_of_source_file(no_ext_file)
        assert module_dir == temp_dir
        assert module_text == 'package.no_extension'

    def test_path_separator_normalization(self, mixed_separators_structure):
        temp_dir, package_dir, subdir, module_file = mixed_separators_structure

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == temp_dir
        assert module_text == 'package.subdir.module'
        # 确保路径分隔符被正确转换为点号
        assert '/' not in module_text
        assert '\\' not in module_text

    def test_absolute_path_handling(self, simple_package_structure):
        temp_dir, package_dir, init_file, module_file = simple_package_structure

        # 使用绝对路径
        abs_module_file = os.path.abspath(module_file)
        module_dir, module_text = get_pythonpath_of_source_file(abs_module_file)

        assert module_dir == temp_dir
        assert module_text == 'package.module'

    def test_deeply_nested_structure(self, temp_dir):
        # 创建深度嵌套的包结构
        deep_path = temp_dir
        path_segments = ['level1', 'level2', 'level3', 'level4']

        for segment in path_segments:
            deep_path = os.path.join(deep_path, segment)
            os.makedirs(deep_path)
            init_file = os.path.join(deep_path, '__init__.py')
            with open(init_file, 'w') as f:
                f.write('')

        module_file = os.path.join(deep_path, 'deep_module.py')
        with open(module_file, 'w') as f:
            f.write('')

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        assert module_dir == temp_dir
        assert module_text == 'level1.level2.level3.level4.deep_module'

    def test_partial_package_structure(self, temp_dir):
        # 创建部分包结构（中间某层没有__init__.py）
        level1 = os.path.join(temp_dir, 'level1')
        level2 = os.path.join(level1, 'level2')  # 这层没有__init__.py
        level3 = os.path.join(level2, 'level3')

        os.makedirs(level3)

        # 只在level1和level3创建__init__.py
        init1 = os.path.join(level1, '__init__.py')
        init3 = os.path.join(level3, '__init__.py')

        with open(init1, 'w') as f:
            f.write('')
        with open(init3, 'w') as f:
            f.write('')

        module_file = os.path.join(level3, 'module.py')
        with open(module_file, 'w') as f:
            f.write('')

        module_dir, module_text = get_pythonpath_of_source_file(module_file)

        # 应该停在level2，因为level2没有__init__.py
        assert module_dir == level2
        assert module_text == 'level3.module'
