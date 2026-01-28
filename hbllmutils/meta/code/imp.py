import ast
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class ImportStatement:
    """表示 import 语句的数据类"""
    module: str
    alias: Optional[str] = None
    line: int = 0
    col_offset: int = 0

    def __repr__(self) -> str:
        """返回Python可读的import语句"""
        if self.alias:
            return f"import {self.module} as {self.alias}"
        else:
            return f"import {self.module}"


@dataclass
class FromImportStatement:
    """表示 from import 语句的数据类"""
    module: str
    name: str
    alias: Optional[str] = None
    level: int = 0
    line: int = 0
    col_offset: int = 0

    def __repr__(self) -> str:
        """返回Python可读的from import语句"""
        # 构建相对导入的点号前缀
        level_str = "." * self.level

        # 构建模块路径
        if self.module:
            module_str = f"{level_str}{self.module}"
        else:
            module_str = level_str if level_str else ""

        # 构建别名部分
        alias_str = f" as {self.alias}" if self.alias else ""

        # 构建完整的from import语句
        if module_str:
            return f"from {module_str} import {self.name}{alias_str}"
        else:
            return f"from . import {self.name}{alias_str}"


ImportStatementTyping = Union[ImportStatement, FromImportStatement]


class ImportVisitor(ast.NodeVisitor):
    """
    自定义访问器类，专门用于收集import信息
    """

    def __init__(self):
        self.imports: List[ImportStatementTyping] = []

    def visit_Import(self, node):
        """访问import节点"""
        for alias in node.names:
            import_stmt = ImportStatement(
                module=alias.name,
                alias=alias.asname,
                line=node.lineno,
                col_offset=node.col_offset
            )
            self.imports.append(import_stmt)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """访问from import节点"""
        module = node.module or ''
        level = node.level

        for alias in node.names:
            from_import_stmt = FromImportStatement(
                module=module,
                name=alias.name,
                alias=alias.asname,
                level=level,
                line=node.lineno,
                col_offset=node.col_offset
            )
            self.imports.append(from_import_stmt)
        self.generic_visit(node)


def analyze_imports(code_text) -> List[ImportStatementTyping]:
    tree = ast.parse(code_text)
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports

