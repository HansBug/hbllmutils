from pathlib import Path
from typing import List, Tuple, Type, Dict, Optional, Any, Union

from hbutils.model import IComparable
from natsort import natsorted

from .matcher import BaseMatcher


class _MatcherPairMeta(type):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.__fields__, instance.__field_names__, \
            instance.__value_fields__, instance.__value_field_names__ = cls._cls_init(instance.__annotations__)
        instance.__field_names_set__ = set(instance.__field_names__)
        instance.__value_field_names_set__ = set(instance.__value_field_names__)
        return instance

    @classmethod
    def _cls_init(cls, annotations) -> Tuple[Dict[str, Type[BaseMatcher]], List[str], Dict[str, type], List[str]]:
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
    def __init__(self, values: Dict[str, Any], instances: Dict[str, BaseMatcher]):
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
        field_info = []
        for value_field_name in self.__value_field_names__:
            field_info.append(f'{value_field_name}={getattr(self, value_field_name)!r}')
        for field_name in self.__field_names__:
            field_info.append(f'{field_name}={getattr(self, field_name)!r}')

        field_str = ", ".join(field_info)
        return f"{self.__class__.__name__}({field_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def values_tuple(self):
        return tuple(getattr(self, name) for name in self.__value_field_names__)

    def values_dict(self):
        return {name: getattr(self, name) for name in self.__value_field_names__}

    def tuple(self):
        return tuple(getattr(self, name) for name in [*self.__value_field_names__, *self.__field_names__])

    def dict(self):
        return {name: getattr(self, name) for name in [*self.__value_field_names__, *self.__field_names__]}

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
