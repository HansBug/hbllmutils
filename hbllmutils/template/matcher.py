"""
File pattern matching module for extracting structured information from filenames.

This module provides a metaclass-based pattern matching system that allows defining
file matchers using template patterns with typed placeholders. It supports automatic
field extraction, type conversion, and file discovery in directories.

The main components are:
- MatcherMeta: Metaclass that processes pattern templates and generates regex patterns
- BaseMatcher: Base class for creating custom file matchers with pattern matching capabilities

Example::
    >>> class ImageMatcher(BaseMatcher):
    ...     __pattern__ = "image_<id>_<name>.png"
    ...     id: int
    ...     name: str
    >>> matcher = ImageMatcher.match("/path/to/images")
    >>> print(matcher.id, matcher.name)
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator, Tuple

from hbutils.model import IComparable
from natsort import natsorted


class _MatcherMeta(type):
    """
    Metaclass for creating pattern-based file matchers.
    
    This metaclass processes the __pattern__ attribute and type annotations to generate
    a compiled regular expression pattern and field metadata. It automatically converts
    template patterns like "file_<id>_<name>.txt" into proper regex patterns with
    appropriate capture groups based on field types.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new matcher class with processed pattern and field information.
        
        :param args: Positional arguments for type creation
        :type args: tuple
        :param kwargs: Keyword arguments for type creation
        :type kwargs: dict
        
        :return: New matcher class instance with processed metadata
        :rtype: type
        """
        instance = super().__new__(cls, *args, **kwargs)
        instance.__regexp_pattern__, instance.__fields__, instance.__field_names__ = \
            cls._cls_init(instance.__pattern__, getattr(instance, '__annotations__') or {})
        instance.__field_names_set__ = set(instance.__field_names__)
        return instance

    @classmethod
    def _cls_init(cls, pattern: str, annotations: Dict[str, type]) -> Tuple[str, Dict[str, type], List[str]]:
        """
        Initialize class-level pattern and field information.
        
        Parses the template pattern to extract placeholders, validates them against
        type annotations, and generates a compiled regex pattern with appropriate
        capture groups for each field type.
        
        :param pattern: Template pattern with placeholders like "file_<id>_<name>.txt"
        :type pattern: str
        :param annotations: Type annotations for fields
        :type annotations: Dict[str, type]
        
        :return: Tuple of (regex_pattern, fields_dict, field_names_list)
        :rtype: Tuple[str, Dict[str, type], List[str]]
        
        :raises NameError: If placeholders don't match annotations or vice versa
        
        Example::
            >>> pattern = "image_<id>_<name>.png"
            >>> annotations = {'id': int, 'name': str}
            >>> regex, fields, names = _MatcherMeta._cls_init(pattern, annotations)
            >>> print(names)
            ['id', 'name']
        """
        fields = {}
        # Find all placeholders <field_name>
        placeholder_pattern = r'<(\w+)>'
        placeholders = re.findall(placeholder_pattern, pattern)
        annotations = {key: value for key, value in annotations.items()
                       if not (key.startswith('__') and key.endswith('__'))}

        # Build regular expression
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

            # Generate corresponding regex based on type
            if field_type == int:
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'(\d+?)')
            elif field_type == float:
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'(\d+\.?\d*?)')
            else:  # str or other types
                regex_pattern = regex_pattern.replace(f'<{placeholder}>', r'([^/\\]+?)')

        # Escape special characters but preserve capture groups
        # Temporarily replace capture groups
        temp_markers = {}
        group_count = 0
        for match in re.finditer(r'\([^)]+\)', regex_pattern):
            marker = f"__TEMP_GROUP_{group_count}__"
            temp_markers[marker] = match.group()
            regex_pattern = regex_pattern.replace(match.group(), marker, 1)
            group_count += 1

        # Escape special characters
        regex_pattern = re.escape(regex_pattern)

        # Restore capture groups
        for marker, group in temp_markers.items():
            regex_pattern = regex_pattern.replace(marker, group)

        return regex_pattern, fields, placeholders


class BaseMatcher(IComparable, metaclass=_MatcherMeta):
    """
    Base class for file pattern matchers.
    
    This class provides functionality to match files based on template patterns and
    automatically extract typed fields from filenames. Subclasses should define
    __pattern__ and type-annotated fields.
    
    :cvar __pattern__: Template pattern with placeholders (e.g., "file_<id>_<name>.txt")
    :vartype __pattern__: str
    :cvar __recursively__: Whether to search directories recursively
    :vartype __recursively__: bool
    
    Example::
        >>> class LogMatcher(BaseMatcher):
        ...     __pattern__ = "log_<date>_<level>.txt"
        ...     date: str
        ...     level: str
        >>> matcher = LogMatcher.match("/var/logs")
        >>> if matcher:
        ...     print(f"Found log: {matcher.date} - {matcher.level}")
    """

    __pattern__: str = ""
    __recursively__: bool = False

    def __init__(self, full_path: str, **kwargs):
        """
        Initialize matcher instance with extracted field values.
        
        :param full_path: Complete path to the matched file
        :type full_path: str
        :param kwargs: Extracted field values from the filename
        :type kwargs: Any
        
        :raises ValueError: If unknown fields are provided or required fields are missing
        
        Example::
            >>> matcher = ImageMatcher("/path/image_001_test.png", id=1, name="test")
            >>> print(matcher.id, matcher.name)
            1 test
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

        # Set fields extracted from pattern
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _convert_value(cls, value: str, target_type: type) -> Any:
        """
        Convert string value to target type.
        
        :param value: String value to convert
        :type value: str
        :param target_type: Target type for conversion
        :type target_type: type
        
        :return: Converted value
        :rtype: Any
        
        :raises TypeError: If target type is not supported
        
        Example::
            >>> BaseMatcher._convert_value("123", int)
            123
            >>> BaseMatcher._convert_value("3.14", float)
            3.14
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
        """
        Yield all matching file instances in the specified directory.
        
        :param directory: Directory to search for matching files
        :type directory: Union[str, Path]
        
        :return: Iterator of matched file instances
        :rtype: Iterator[BaseMatcher]
        
        Example::
            >>> for matcher in ImageMatcher._yield_match("/path/to/images"):
            ...     print(matcher.id, matcher.name)
        """
        directory = Path(directory)
        if not directory.exists():
            return

        regex_pattern, fields, field_order = cls.__regexp_pattern__, cls.__fields__, cls.__field_names__
        compiled_pattern = re.compile(regex_pattern)

        recursively = getattr(cls, '__recursively__', False)

        # Build search pattern
        search_pattern = "**/*" if recursively else "*"

        for file_path in natsorted(directory.glob(search_pattern)):
            if file_path.is_file():
                file_name = file_path.name
                match = compiled_pattern.match(file_name)

                if match:
                    # Extract field values
                    field_values = {}
                    for i, field_name in enumerate(field_order):
                        raw_value = match.group(i + 1)
                        field_type = fields[field_name]
                        try:
                            converted_value = cls._convert_value(raw_value, field_type)
                        except (ValueError, TypeError):
                            # Type conversion failed, skip this file
                            continue
                        else:
                            field_values[field_name] = converted_value

                    # Create instance
                    instance = cls(str(file_path), **field_values)
                    yield instance

    @classmethod
    def match(cls, directory: Union[str, Path]) -> Optional['BaseMatcher']:
        """
        Match the first file that conforms to the pattern in the specified directory.
        
        :param directory: Directory to search
        :type directory: Union[str, Path]
        
        :return: Matched file instance, or None if not found
        :rtype: Optional[BaseMatcher]
        
        Example::
            >>> matcher = ImageMatcher.match("/path/to/images")
            >>> if matcher:
            ...     print(f"Found: {matcher.full_path}")
        """
        iterable = cls._yield_match(directory)
        try:
            return next(iterable)
        except StopIteration:
            return None

    @classmethod
    def match_all(cls, directory: Union[str, Path]) -> List['BaseMatcher']:
        """
        Match all files that conform to the pattern in the specified directory.
        
        :param directory: Directory to search
        :type directory: Union[str, Path]
        
        :return: List of matched file instances
        :rtype: List[BaseMatcher]
        
        Example::
            >>> matchers = ImageMatcher.match_all("/path/to/images")
            >>> print(f"Found {len(matchers)} images")
        """
        return list(cls._yield_match(directory))

    @classmethod
    def exists(cls, directory: Union[str, Path]) -> bool:
        """
        Check if any file matching the pattern exists in the specified directory.
        
        :param directory: Directory to search
        :type directory: Union[str, Path]
        
        :return: True if matching file exists, False otherwise
        :rtype: bool
        
        Example::
            >>> if ImageMatcher.exists("/path/to/images"):
            ...     print("Images found!")
        """
        return cls.match(directory) is not None

    def __str__(self) -> str:
        """
        Get string representation of the matcher instance.
        
        :return: String representation showing field values and full path
        :rtype: str
        """
        field_info = []
        annotations = getattr(self.__class__, '__annotations__') or {}

        for field_name in annotations:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                field_info.append(f"{field_name}={value!r}")
        field_info.append(f"full_path={self.full_path!r}")

        field_str = ", ".join(field_info)
        return f"{self.__class__.__name__}({field_str})"

    def __repr__(self) -> str:
        """
        Get representation string of the matcher instance.
        
        :return: Representation string
        :rtype: str
        """
        return self.__str__()

    def tuple(self):
        """
        Get field values as a tuple.
        
        :return: Tuple of field values in definition order
        :rtype: tuple
        
        Example::
            >>> matcher = ImageMatcher("/path/image_001_test.png", id=1, name="test")
            >>> matcher.tuple()
            (1, 'test')
        """
        return tuple(getattr(self, name) for name in self.__field_names__)

    def dict(self):
        """
        Get field values as a dictionary.
        
        :return: Dictionary mapping field names to values
        :rtype: dict
        
        Example::
            >>> matcher = ImageMatcher("/path/image_001_test.png", id=1, name="test")
            >>> matcher.dict()
            {'id': 1, 'name': 'test'}
        """
        return {name: getattr(self, name) for name in self.__field_names__}

    def __hash__(self):
        """
        Get hash value of the matcher instance.
        
        :return: Hash value based on field values
        :rtype: int
        """
        return hash(self.tuple())

    def _cmpkey(self):
        """
        Get comparison key for ordering instances.
        
        :return: Tuple of field values used for comparison
        :rtype: tuple
        """
        return self.tuple()
