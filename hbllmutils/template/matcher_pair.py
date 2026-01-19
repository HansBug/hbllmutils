"""
This module provides a matcher pair system for matching and organizing files based on multiple matcher criteria.

The module implements a metaclass-based system that allows defining matcher pairs with multiple matcher fields
and value fields. It enables matching files in directories using multiple matchers simultaneously and organizing
the results into structured pair objects.

Classes:
    _MatcherPairMeta: Metaclass for creating matcher pair classes with field validation
    BaseMatcherPair: Base class for defining and working with matcher pairs

Example::
    >>> class MyMatcherPair(BaseMatcherPair):
    ...     matcher1: SomeMatcher
    ...     matcher2: AnotherMatcher
    >>> pairs = MyMatcherPair.match_all('/path/to/directory')
    >>> for pair in pairs:
    ...     print(pair.values_dict())
"""

from pathlib import Path
from typing import List, Tuple, Type, Dict, Optional, Any, Union

from hbutils.model import IComparable
from natsort import natsorted

from .matcher import BaseMatcher


class _MatcherPairMeta(type):
    """
    Metaclass for creating matcher pair classes with automatic field initialization and validation.
    
    This metaclass processes class annotations to extract matcher fields and their corresponding
    value fields, ensuring consistency across all matchers in a pair.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new matcher pair class with initialized field metadata.
        
        :param args: Positional arguments for class creation
        :type args: tuple
        :param kwargs: Keyword arguments for class creation
        :type kwargs: dict
        
        :return: New matcher pair class instance with field metadata
        :rtype: type
        """
        instance = super().__new__(cls, *args, **kwargs)
        instance.__fields__, instance.__field_names__, \
            instance.__value_fields__, instance.__value_field_names__ = cls._cls_init(instance.__annotations__)
        instance.__field_names_set__ = set(instance.__field_names__)
        instance.__value_field_names_set__ = set(instance.__value_field_names__)
        return instance

    @classmethod
    def _cls_init(cls, annotations) -> Tuple[Dict[str, Type[BaseMatcher]], List[str], Dict[str, type], List[str]]:
        """
        Initialize class fields from annotations.
        
        Processes class annotations to extract matcher fields and validate that all matchers
        have consistent value field definitions.
        
        :param annotations: Class annotations dictionary
        :type annotations: dict
        
        :return: Tuple containing (fields dict, field names list, value fields dict, value field names list)
        :rtype: Tuple[Dict[str, Type[BaseMatcher]], List[str], Dict[str, type], List[str]]
        
        :raises NameError: If a field is not a BaseMatcher subclass
        :raises TypeError: If matchers have inconsistent value field definitions
        
        Example::
            >>> annotations = {'matcher1': SomeMatcher, 'matcher2': AnotherMatcher}
            >>> fields, names, value_fields, value_names = _MatcherPairMeta._cls_init(annotations)
        """
        fields, field_names = {}, []
        annotations = {key: value for key, value in annotations.items()
                       if not (key.startswith('__') and key.endswith('__'))}
        value_fields: Optional[Dict[str, type]] = None
        value_field_names: Optional[List[str]] = None
        for field_name, field_type in annotations.items():
            if not (isinstance(field_type, type) and issubclass(field_type, BaseMatcher)):
                raise NameError(f'Field {field_name!r} is not a matcher, but {field_type!r} found.')
            field_name: str
            field_type: Type[BaseMatcher]
            fields[field_name] = field_type
            field_names.append(field_name)
            if value_fields is None:
                value_fields = field_type.__fields__
                value_field_names = field_type.__field_names__
            else:
                if value_fields != field_type.__fields__:
                    raise TypeError(f'Field not match, {value_fields!r} vs {field_type.__fields__!r}')

        value_fields: Dict[str, type] = value_fields or {}
        value_field_names: List[str] = value_field_names or []
        return fields, field_names, value_fields, value_field_names


class BaseMatcherPair(IComparable, metaclass=_MatcherPairMeta):
    """
    Base class for matcher pairs that group multiple matchers with shared value fields.
    
    A matcher pair represents a collection of matchers that all match files with the same
    set of identifying values (e.g., same ID, version, etc.). This class provides functionality
    to match files in directories and organize them into structured pairs.
    
    :ivar __fields__: Dictionary mapping field names to matcher types
    :vartype __fields__: Dict[str, Type[BaseMatcher]]
    :ivar __field_names__: List of matcher field names
    :vartype __field_names__: List[str]
    :ivar __value_fields__: Dictionary mapping value field names to types
    :vartype __value_fields__: Dict[str, type]
    :ivar __value_field_names__: List of value field names
    :vartype __value_field_names__: List[str]
    
    Example::
        >>> class ImagePair(BaseMatcherPair):
        ...     image: ImageMatcher
        ...     thumbnail: ThumbnailMatcher
        >>> pairs = ImagePair.match_all('/path/to/images')
        >>> for pair in pairs:
        ...     print(f"ID: {pair.id}, Image: {pair.image}, Thumbnail: {pair.thumbnail}")
    """

    def __init__(self, values: Dict[str, Any], instances: Dict[str, BaseMatcher]):
        """
        Initialize a matcher pair with values and matcher instances.
        
        :param values: Dictionary of value field names to their values
        :type values: Dict[str, Any]
        :param instances: Dictionary of matcher field names to matcher instances
        :type instances: Dict[str, BaseMatcher]
        
        :raises ValueError: If unknown fields are provided or required fields are missing
        
        Example::
            >>> pair = ImagePair(
            ...     values={'id': '001'},
            ...     instances={'image': image_matcher, 'thumbnail': thumb_matcher}
            ... )
        """
        unknown_fields = {}
        excluded_fields = set(self.__field_names_set__)
        for key, value in instances.items():
            if key not in self.__field_names_set__:
                unknown_fields[key] = value
            else:
                excluded_fields.remove(key)
        if unknown_fields:
            raise ValueError(f'Unknown fields for class {self.__class__.__name__}: {unknown_fields!r}.')
        if excluded_fields:
            raise ValueError(f'Non-included fields of class {self.__class__.__name__}: {natsorted(excluded_fields)!r}.')
        for key, value in instances.items():
            setattr(self, key, value)

        unknown_value_fields = {}
        excluded_value_fields = set(self.__value_field_names_set__)
        for key, value in values.items():
            if key not in self.__value_field_names_set__:
                unknown_value_fields[key] = value
            else:
                excluded_value_fields.remove(key)
        if unknown_value_fields:
            raise ValueError(f'Unknown value fields for class {self.__class__.__name__}: {unknown_value_fields!r}.')
        if excluded_value_fields:
            raise ValueError(
                f'Non-included value fields of class {self.__class__.__name__}: {natsorted(excluded_value_fields)!r}.')
        for key, value in values.items():
            setattr(self, key, value)

    @classmethod
    def match_all(cls, directory: Union[str, Path]) -> List['BaseMatcherPair']:
        """
        Match all files in a directory and group them into matcher pairs.
        
        This method uses all defined matchers to find matching files in the directory,
        then groups files with the same identifying values into pairs.
        
        :param directory: Path to the directory to search
        :type directory: Union[str, Path]
        
        :return: List of matcher pairs found in the directory, sorted naturally
        :rtype: List[BaseMatcherPair]
        
        Example::
            >>> pairs = ImagePair.match_all('/path/to/images')
            >>> print(f"Found {len(pairs)} image pairs")
            Found 5 image pairs
        """
        d_fields, s_tuples = {}, None
        for field_name, field_type in cls.__fields__.items():
            d_fields[field_name] = {
                x.tuple(): x for x in field_type.match_all(directory)
            }
            tpls = set(d_fields[field_name].keys())
            if s_tuples is None:
                s_tuples = tpls
            else:
                s_tuples = s_tuples & tpls

        tuples = natsorted(s_tuples)
        retval = []
        for tpl in tuples:
            d_instances, d_values = {}, None
            for field_name in cls.__field_names__:
                instance = d_fields[field_name][tpl]
                d_instances[field_name] = instance
                if d_values is None:
                    d_values = instance.dict()

            retval.append(cls(
                values=d_values,
                instances=d_instances,
            ))

        return retval

    def __str__(self) -> str:
        """
        Get string representation of the matcher pair.
        
        :return: String representation showing all value and matcher fields
        :rtype: str
        
        Example::
            >>> str(pair)
            'ImagePair(id='001', image=ImageMatcher(...), thumbnail=ThumbnailMatcher(...))'
        """
        field_info = []
        for value_field_name in self.__value_field_names__:
            field_info.append(f'{value_field_name}={getattr(self, value_field_name)!r}')
        for field_name in self.__field_names__:
            field_info.append(f'{field_name}={getattr(self, field_name)!r}')

        field_str = ", ".join(field_info)
        return f"{self.__class__.__name__}({field_str})"

    def __repr__(self) -> str:
        """
        Get detailed string representation of the matcher pair.
        
        :return: String representation showing all value and matcher fields
        :rtype: str
        """
        return self.__str__()

    def values_tuple(self):
        """
        Get tuple of value field values.
        
        :return: Tuple containing values of all value fields in order
        :rtype: tuple
        
        Example::
            >>> pair.values_tuple()
            ('001', 'v1')
        """
        return tuple(getattr(self, name) for name in self.__value_field_names__)

    def values_dict(self):
        """
        Get dictionary of value field names to values.
        
        :return: Dictionary mapping value field names to their values
        :rtype: dict
        
        Example::
            >>> pair.values_dict()
            {'id': '001', 'version': 'v1'}
        """
        return {name: getattr(self, name) for name in self.__value_field_names__}

    def tuple(self):
        """
        Get tuple of all field values (both value fields and matcher fields).
        
        :return: Tuple containing all field values in order
        :rtype: tuple
        
        Example::
            >>> pair.tuple()
            ('001', 'v1', ImageMatcher(...), ThumbnailMatcher(...))
        """
        return tuple(getattr(self, name) for name in [*self.__value_field_names__, *self.__field_names__])

    def dict(self):
        """
        Get dictionary of all field names to values (both value fields and matcher fields).
        
        :return: Dictionary mapping all field names to their values
        :rtype: dict
        
        Example::
            >>> pair.dict()
            {'id': '001', 'version': 'v1', 'image': ImageMatcher(...), 'thumbnail': ThumbnailMatcher(...)}
        """
        return {name: getattr(self, name) for name in [*self.__value_field_names__, *self.__field_names__]}

    def __hash__(self):
        """
        Get hash value of the matcher pair instance.

        :return: Hash value based on all field values
        :rtype: int
        
        Example::
            >>> hash(pair)
            123456789
        """
        return hash(self.tuple())

    def _cmpkey(self):
        """
        Get comparison key for ordering instances.

        :return: Tuple of all field values used for comparison
        :rtype: tuple
        
        Example::
            >>> pair1._cmpkey() < pair2._cmpkey()
            True
        """
        return self.tuple()
