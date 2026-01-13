import argparse
import ast
import os
import pathlib
from typing import List, Dict, Any

from natsort import natsorted
from sphinx.util.rst import escape


def rst_to_text(text: str):
    return escape(text)


class PublicMemberExtractor(ast.NodeVisitor):
    """提取Python代码中的公有成员（类、函数、变量）"""

    def __init__(self):
        self.public_classes = []
        self.public_functions = []
        self.public_variables = []

    @classmethod
    def is_private(cls, name: str) -> bool:
        """判断是否为私有成员（双下划线开头）"""
        return name.startswith('__') and not (name.startswith('__') and name.endswith('__'))

    @classmethod
    def is_protected(cls, name: str) -> bool:
        """判断是否为保护成员（单下划线开头）"""
        return name.startswith('_') and not name.startswith('__')

    @classmethod
    def is_magic_method(cls, name: str) -> bool:
        """判断是否为魔法方法（双下划线开头和结尾）"""
        return name.startswith('__') and name.endswith('__') and len(name) > 4

    @classmethod
    def is_public_or_magic(cls, name: str) -> bool:
        """判断是否为公有成员或魔法方法"""
        return not cls.is_private(name) and not cls.is_protected(name) or cls.is_magic_method(name)

    def extract_class_members(self, node: ast.ClassDef) -> Dict[str, Any]:
        """提取类的公有成员和魔法方法"""
        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if self.is_public_or_magic(item.name):
                    method_info = {
                        'name': item.name,
                        'type': 'method',
                        'args': self.extract_function_args(item),
                        'decorators': [self.get_decorator_name(dec) for dec in item.decorator_list],
                        'docstring': ast.get_docstring(item),
                        'lineno': item.lineno,
                        'is_magic': self.is_magic_method(item.name)
                    }
                    methods.append(method_info)

            elif isinstance(item, ast.Assign):
                # 提取类变量
                for target in item.targets:
                    if isinstance(target, ast.Name) and self.is_public_or_magic(target.id):
                        attr_info = {
                            'name': target.id,
                            'type': 'class_variable',
                            'lineno': item.lineno,
                            'value': self.get_node_source(item.value) if hasattr(item, 'value') else None
                        }
                        attributes.append(attr_info)

            elif isinstance(item, ast.AnnAssign):
                # 提取带类型注解的类变量
                if isinstance(item.target, ast.Name) and self.is_public_or_magic(item.target.id):
                    attr_info = {
                        'name': item.target.id,
                        'type': 'annotated_variable',
                        'annotation': self.get_node_source(item.annotation),
                        'lineno': item.lineno,
                        'value': self.get_node_source(item.value) if item.value else None
                    }
                    attributes.append(attr_info)

        return {
            'methods': methods,
            'attributes': attributes
        }

    def extract_function_args(self, node: ast.FunctionDef) -> List[str]:
        """提取函数参数"""
        args = []

        # 普通参数
        for arg in node.args.args:
            args.append(arg.arg)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        return args

    def get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self.get_node_source(decorator.value)}.{decorator.attr}"
        else:
            return self.get_node_source(decorator)

    def get_node_source(self, node) -> str:
        """获取AST节点的源代码表示"""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Attribute):
                return f"{self.get_node_source(node.value)}.{node.attr}"
            elif isinstance(node, ast.List):
                elements = [self.get_node_source(elt) for elt in node.elts]
                return f"[{', '.join(elements)}]"
            elif isinstance(node, ast.Dict):
                pairs = []
                for k, v in zip(node.keys, node.values):
                    key = self.get_node_source(k) if k else None
                    value = self.get_node_source(v)
                    pairs.append(f"{key}: {value}" if key else f"**{value}")
                return f"{{{', '.join(pairs)}}}"
            else:
                # 对于复杂表达式，返回类型信息
                return f"<{type(node).__name__}>"
        except:
            return "<unknown>"

    def visit_ClassDef(self, node: ast.ClassDef):
        """访问类定义"""
        if self.is_public_or_magic(node.name):
            # 只处理顶层的公有类
            class_info = {
                'name': node.name,
                'type': 'class',
                'bases': [self.get_node_source(base) for base in node.bases],
                'decorators': [self.get_decorator_name(dec) for dec in node.decorator_list],
                'docstring': ast.get_docstring(node),
                'lineno': node.lineno,
                'members': self.extract_class_members(node)
            }
            self.public_classes.append(class_info)

        # 不递归访问嵌套类

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """访问函数定义"""
        if self.is_public_or_magic(node.name):
            # 只处理顶层的公有函数
            func_info = {
                'name': node.name,
                'type': 'function',
                'args': self.extract_function_args(node),
                'decorators': [self.get_decorator_name(dec) for dec in node.decorator_list],
                'docstring': ast.get_docstring(node),
                'lineno': node.lineno,
                'returns': self.get_node_source(node.returns) if node.returns else None
            }
            self.public_functions.append(func_info)

        # self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """访问赋值语句（变量定义）"""
        # 只处理顶层变量
        for target in node.targets:
            if isinstance(target, ast.Name) and self.is_public_or_magic(target.id):
                var_info = {
                    'name': target.id,
                    'type': 'variable',
                    'lineno': node.lineno,
                    'value': self.get_node_source(node.value)
                }
                self.public_variables.append(var_info)

        # self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """访问带注解的赋值语句"""
        if isinstance(node.target, ast.Name) and self.is_public_or_magic(node.target.id):
            var_info = {
                'name': node.target.id,
                'type': 'annotated_variable',
                'annotation': self.get_node_source(node.annotation),
                'lineno': node.lineno,
                'value': self.get_node_source(node.value) if node.value else None
            }
            self.public_variables.append(var_info)

        # self.generic_visit(node)


def extract_public_members(source_code: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    从Python源代码中提取公有成员

    Args:
        source_code: Python源代码字符串

    Returns:
        包含classes, functions, variables三个键的字典
    """
    tree = ast.parse(source_code)
    extractor = PublicMemberExtractor()
    extractor.visit(tree)

    return {
        'classes': extractor.public_classes,
        'functions': extractor.public_functions,
        'variables': extractor.public_variables
    }


def extract_public_members_from_file(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    从Python文件中提取公有成员

    Args:
        file_path: Python文件路径

    Returns:
        包含classes, functions, variables三个键的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    return extract_public_members(source_code)


def print_extracted_members(f, members: Dict[str, List[Dict[str, Any]]]):
    """打印提取的成员信息"""

    for var in members['variables']:
        print(f'{rst_to_text(var["name"])}', file=f)
        print(f'-----------------------------------------------------', file=f)
        print(f'', file=f)
        print(f'.. autodata:: {var["name"]}', file=f)
        print(f'', file=f)
        print(f'', file=f)

    for cls in members['classes']:
        print(f'{rst_to_text(cls["name"])}', file=f)
        print(f'-----------------------------------------------------', file=f)
        print(f'', file=f)
        print(f'.. autoclass:: {cls["name"]}', file=f)
        member_names = []
        for method in cls['members']['methods']:
            member_names.append(method['name'])
        for attr in cls['members']['attributes']:
            member_names.append(attr['name'])
        if member_names:
            print(f'    :members: {",".join(member_names)}', file=f)
        print(f'', file=f)
        print(f'', file=f)

    for func in members['functions']:
        print(f'{rst_to_text(func["name"])}', file=f)
        print(f'-----------------------------------------------------', file=f)
        print(f'', file=f)
        print(f'.. autofunction:: {func["name"]}', file=f)
        print(f'', file=f)
        print(f'', file=f)


def convert_code_to_rst(code_file: str, rst_file: str, lib_dir: str = '.'):
    if os.path.dirname(rst_file):
        os.makedirs(os.path.dirname(rst_file), exist_ok=True)
    members = extract_public_members(pathlib.Path(code_file).read_text())

    with open(rst_file, 'w') as f:
        rel_file = os.path.relpath(os.path.abspath(code_file), os.path.abspath(lib_dir))
        rel_segs = os.path.splitext(rel_file)[0]
        module_name = rel_segs.replace('/', '.').replace('\\', '.')
        if module_name.split('.')[-1] == '__init__':
            module_name = '.'.join(module_name.split('.')[:-1])

        print(f'{rst_to_text(module_name)}', file=f)
        print(f'========================================================', file=f)
        print(f'', file=f)

        print(f'.. currentmodule:: {module_name}', file=f)
        print(f'', file=f)
        print(f'.. automodule:: {module_name}', file=f)
        print(f'', file=f)
        print(f'', file=f)

        if os.path.basename(code_file) != '__init__.py':
            print_extracted_members(f, members)
        else:
            code_rels = []
            for code_rel_file in os.listdir(os.path.dirname(code_file)):
                code_rel_base = os.path.splitext(code_rel_file)[0]
                if code_rel_file.endswith('.py') and \
                        not (code_rel_base.startswith('__') and code_rel_base.endswith('__')):
                    code_rels.append(code_rel_base)

            if code_rels:
                code_rels = natsorted(code_rels)
                print(f'.. toctree::', file=f)
                print(f'    :maxdepth: 3', file=f)
                print(f'', file=f)
                for code_rel_base in code_rels:
                    print(f'    {code_rel_base}', file=f)
                print(f'', file=f)


def main():
    parser = argparse.ArgumentParser(description='Auto create rst docs for python code file')
    parser.add_argument('-i', '--input', required=True, help='Input python code file')
    parser.add_argument('-o', '--output', required=True, help='Output rst doc file')
    args = parser.parse_args()

    convert_code_to_rst(
        code_file=args.input,
        rst_file=args.output,
        lib_dir='.'
    )


if __name__ == "__main__":
    main()
