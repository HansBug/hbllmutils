"""
Object hashability conversion utilities.

This module provides utilities for converting Python objects into hashable
representations. It handles common data structures including primitives,
collections, and nested structures by recursively transforming them into
immutable, hashable equivalents.

The module contains the following main components:

* :func:`obj_hashable` - Convert objects to hashable representations

This is particularly useful for creating dictionary keys, set members, or
cache keys from complex data structures that may contain mutable elements
like lists or dictionaries.

.. note::
   The conversion process uses natural sorting for dictionary keys to ensure
   consistent ordering across different Python versions and implementations.

.. warning::
   Objects that are not explicitly handled (custom classes, etc.) are returned
   as-is without conversion. Ensure such objects are inherently hashable or
   implement appropriate hash methods.

Example::

    >>> from hbllmutils.utils.hashable import obj_hashable
    >>> 
    >>> # Convert a list to hashable tuple
    >>> data = [1, 2, 3]
    >>> hashable_data = obj_hashable(data)
    >>> print(type(hashable_data))
    <class 'tuple'>
    >>> 
    >>> # Convert nested structures
    >>> complex_data = {'key': [1, 2], 'nested': {'a': 1, 'b': 2}}
    >>> hashable_complex = obj_hashable(complex_data)
    >>> # Can now be used as dictionary key
    >>> cache = {hashable_complex: 'cached_value'}

"""

from typing import Any, Union


def obj_hashable(obj: Any) -> Union[None, int, float, str, tuple]:
    """
    Convert an object to a hashable representation.

    This function recursively transforms mutable data structures (lists, dicts)
    into immutable, hashable equivalents (tuples). It handles nested structures
    and ensures consistent ordering for dictionaries using natural sorting.

    The conversion follows these rules:
    
    * ``None`` remains ``None``
    * Primitives (int, float, str) are returned unchanged
    * Lists and tuples are converted to tuples with recursively converted elements
    * Dictionaries are converted to tuples of (key, value) pairs with naturally sorted keys
    * Other objects are returned as-is (assumed to be already hashable)

    :param obj: Object to convert to hashable form. Can be any Python object including
                None, primitives, lists, tuples, dictionaries, or nested combinations thereof.
    :type obj: Any
    :return: Hashable representation of the input object. Returns None for None,
             primitives unchanged, tuples for lists, and sorted tuples of 
             key-value pairs for dictionaries. Other objects are returned as-is.
    :rtype: Union[None, int, float, str, tuple]
    :raises TypeError: If the object or any of its nested components cannot be hashed
                      and is not a handled type (list, dict, tuple)

    .. note::
       Dictionary keys are sorted using natural sorting (natsort) to ensure
       consistent ordering regardless of insertion order or Python version.
       This is especially important for keys that mix numbers and strings.

    .. note::
       The function is recursive and will traverse deeply nested structures,
       converting all mutable containers encountered along the way.

    .. warning::
       Custom objects not explicitly handled by this function are returned
       unchanged. Ensure they implement ``__hash__`` if hashability is required.
       Attempting to use non-hashable custom objects as dictionary keys or
       set members will raise a ``TypeError``.

    .. warning::
       Very deeply nested structures may cause recursion depth issues.
       Python's default recursion limit is typically around 1000 levels.

    Example::

        >>> # Handle None
        >>> obj_hashable(None)
        None

        >>> # Primitives remain unchanged
        >>> obj_hashable(42)
        42
        >>> obj_hashable(3.14)
        3.14
        >>> obj_hashable("hello")
        'hello'

        >>> # Lists converted to tuples
        >>> obj_hashable([1, 2, 3])
        (1, 2, 3)

        >>> # Tuples are also processed (elements converted)
        >>> obj_hashable((1, [2, 3], 4))
        (1, (2, 3), 4)

        >>> # Nested lists
        >>> obj_hashable([[1, 2], [3, 4]])
        ((1, 2), (3, 4))

        >>> # Dictionaries to sorted tuples
        >>> obj_hashable({'b': 2, 'a': 1})
        (('a', 1), ('b', 2))

        >>> # Natural sorting handles mixed keys
        >>> obj_hashable({'item10': 1, 'item2': 2, 'item1': 3})
        (('item1', 3), ('item2', 2), ('item10', 1))

        >>> # Complex nested structures
        >>> data = {'list': [1, 2], 'dict': {'x': 10}}
        >>> result = obj_hashable(data)
        >>> print(result)
        (('dict', (('x', 10),)), ('list', (1, 2)))

        >>> # Use as dictionary key
        >>> cache_key = obj_hashable({'query': 'test', 'limit': 10})
        >>> cache = {cache_key: 'result'}
        >>> print(cache[cache_key])
        'result'

        >>> # Use in sets
        >>> set_of_configs = {
        ...     obj_hashable({'mode': 'train', 'epochs': 10}),
        ...     obj_hashable({'mode': 'test', 'epochs': 5})
        ... }
        >>> len(set_of_configs)
        2

    """
    if obj is None:
        return None
    elif isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, (tuple, list)):
        return tuple(obj_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        from natsort import natsorted
        # noinspection PyTypeChecker
        sorted_items = natsorted(obj.items(), key=lambda x: str(x[0]))
        return tuple((k, obj_hashable(v)) for k, v in sorted_items)
    else:
        try:
            hash(obj)
        except TypeError as e:
            raise TypeError(f"Object of type {type(obj).__name__!r} is not hashable") from e
        return obj
