import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator

from hbutils.model import IComparable
from natsort import natsorted


class MatcherMeta(type):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.__regexp_pattern__, instance.__fields__, instance.__field_names__ = \
            cls._cls_init(instance.__pattern__, instance.__annotations__)
        instance.__field_names_set__ = set(instance.__field_names__)
        return instance

    @classmethod
    def _cls_init(cls, pattern: str, annotations: Dict[str, type]) -> tuple[str, Dict[str, type], List[str]]:
        fields = {}
        # 查找所有的占位符 <field_name>
        placeholder_pattern = r'<(\w+)>'
        placeholders = re.findall(placeholder_pattern, pattern)
        annotations = {key: value for key, value in annotations.items()
                       if not (key.startswith('__') and key.endswith('__'))}

        # 构建正则表达式
        regex_pattern = pattern
        if set(annotations.keys()) != set(placeholders):
            if set(annotations.keys()) - set(placeholders):
                raise NameError(f'Field {", ".join(natsorted(set(annotations.keys()) - set(placeholders)))} '
                                f'not included in pattern {pattern!r}.')
            if set(placeholders) - set(annotations.keys()):
                raise NameError(f'Placeholder {", ".join(natsorted(set(placeholders) - set(annotations.keys())))} '
                                f'not included in fields {annotations!r}.')
        for placeholder in placeholders:
            field_type = annotations.get(placeholder, str)
            fields[placeholder] = field_type

            # 根据类型生成对应的正则表达式
            if field_type == int:
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'(\d+?)')
            elif field_type == float:
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'(\d+\.?\d*?)')
            else:  # str 或其他类型
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'([^/\\]+?)')

        # 转义特殊字符，但保留我们的捕获组
        # 先临时替换捕获组
        temp_markers = {}
        group_count = 0
        for match in re.finditer(r'\([^)]+\)', regex_pattern):
            marker = f"__TEMP_GROUP_{group_count}__"
            temp_markers[marker] = match.group()
            regex_pattern = regex_pattern.replace(match.group(), marker, 1)
            group_count += 1

        # 转义特殊字符
        regex_pattern = re.escape(regex_pattern)

        # 恢复捕获组
        for marker, group in temp_markers.items():
            regex_pattern = regex_pattern.replace(marker, group)

        return regex_pattern, fields, placeholders


class BaseMatcher(IComparable, metaclass=MatcherMeta):
    """文件匹配器基类"""

    __pattern__: str = ""
    __recursively__: bool = False

    def __init__(self, full_path: str, **kwargs):
        """
        初始化匹配器实例

        Args:
            full_path: 匹配到的文件完整路径
            **kwargs: 从文件名中提取的字段值
        """
        self.full_path = full_path
        self.file_name = os.path.basename(full_path)
        self.dir_path = os.path.dirname(full_path)

        unknown_fields = {}
        excluded_fields = set(self.__field_names_set__)
        for key, value in kwargs.items():
            if key not in self.__field_names_set__:
                unknown_fields[key] = value
            else:
                excluded_fields.remove(key)

        if unknown_fields:
            raise ValueError(f'Unknown fields for class {self.__class__.__name__}: {unknown_fields!r}.')
        if excluded_fields:
            raise ValueError(f'Non-included fields of class {self.__class__.__name__}: {natsorted(excluded_fields)!r}.')

        # 设置从模式中提取的字段
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _convert_value(cls, value: str, target_type: type) -> Any:
        """
        将字符串值转换为目标类型

        Args:
            value: 字符串值
            target_type: 目标类型

        Returns:
            转换后的值
        """
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == str:
            return value
        else:
            raise TypeError(f'Unsupported target type - {target_type!r}.')

    @classmethod
    def _yield_match(cls, directory: Union[str, Path]) -> Iterator['BaseMatcher']:
        directory = Path(directory)
        if not directory.exists():
            return

        # regex_pattern, fields, field_order = cls._parse_pattern()
        regex_pattern, fields, field_order = cls.__regexp_pattern__, cls.__fields__, cls.__field_names__
        compiled_pattern = re.compile(regex_pattern)

        recursively = getattr(cls, '__recursively__', False)

        # 构建搜索模式
        search_pattern = "**/*" if recursively else "*"

        for file_path in natsorted(directory.glob(search_pattern)):
            if file_path.is_file():
                file_name = file_path.name
                match = compiled_pattern.match(file_name)

                if match:
                    # 提取字段值
                    field_values = {}
                    for i, field_name in enumerate(field_order):
                        raw_value = match.group(i + 1)
                        field_type = fields[field_name]
                        try:
                            converted_value = cls._convert_value(raw_value, field_type)
                        except (ValueError, TypeError):
                            # 类型转换失败，跳过这个文件
                            continue
                        else:
                            field_values[field_name] = converted_value

                    # 创建实例
                    instance = cls(str(file_path), **field_values)
                    yield instance

    @classmethod
    def match(cls, directory: Union[str, Path]) -> Optional['BaseMatcher']:
        """
        在指定目录中匹配第一个符合模式的文件

        Args:
            directory: 搜索目录

        Returns:
            匹配的文件实例，如果没有找到则返回None
        """
        iterable = cls._yield_match(directory)
        try:
            return next(iterable)
        except StopIteration:
            return None

    @classmethod
    def match_all(cls, directory: Union[str, Path]) -> List['BaseMatcher']:
        """
        在指定目录中匹配所有符合模式的文件

        Args:
            directory: 搜索目录

        Returns:
            匹配的文件实例列表
        """
        return list(cls._yield_match(directory))

    @classmethod
    def exists(cls, directory: Union[str, Path]) -> bool:
        """
        检查指定目录中是否存在符合模式的文件

        Args:
            directory: 搜索目录

        Returns:
            是否存在匹配的文件
        """
        return cls.match(directory) is not None

    def __str__(self) -> str:
        """字符串表示"""
        field_info = []
        annotations = getattr(self.__class__, '__annotations__', {})

        for field_name in annotations:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                field_info.append(f"{field_name}={value!r}")
        field_info.append(f"full_path={self.full_path!r}")

        field_str = ", ".join(field_info)
        return f"{self.__class__.__name__}({field_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def tuple(self):
        return tuple(getattr(self, name) for name in self.__field_names__)

    def dict(self):
        return {name: getattr(self, name) for name in self.__field_names__}

    def __hash__(self):
        return hash(self.tuple())

    def _cmpkey(self):
        return self.tuple()
