"""
Automatic text decoding utilities with emphasis on Chinese encodings.

This module provides a small, focused API for decoding byte strings when the
source encoding is unknown. The decoding strategy prioritizes encodings commonly
used on Chinese Windows systems and falls back to the system default encoding.
An additional heuristic uses :mod:`chardet` to detect likely encodings.

The module contains the following public components:

* :data:`windows_chinese_encodings` - Ordered list of commonly used Chinese encodings
* :func:`auto_decode` - Robust decoder that tries multiple encodings

.. note::
   The detection order changes for short inputs to reduce mis-detection.
   For data shorter than 30 bytes, the module tries common encodings first.

Example::

    >>> from hbllmutils.template.decode import auto_decode
    >>> text_bytes = b'\\xc4\\xe3\\xba\\xc3'  # "你好" in GBK
    >>> auto_decode(text_bytes)
    '你好'

"""

import sys
from typing import Optional, Union

import chardet
from hbutils.collection import unique

windows_chinese_encodings = [
    'utf-8',  # UTF-8 encoding, Unicode standard
    'gbk',  # Most common default encoding for Chinese Windows
    'gb2312',  # Common encoding for Simplified Chinese, subset of GBK
    'gb18030',  # Chinese national standard encoding, includes all Chinese characters
    'big5',  # Common encoding for Traditional Chinese (Taiwan, Hong Kong)
    'cp936',  # Windows code page for Simplified Chinese, essentially an alias for GBK
    'cp950',  # Windows code page for Traditional Chinese, approximately equivalent to Big5
    'hz',  # Early Chinese character encoding
    # 'iso-2022-cn',  # ISO standard encoding for Chinese
    'euc-cn',  # Extended Unix Code for Chinese
    'utf-16',  # Default Unicode encoding used by Windows Notepad
    'utf-16-le',  # Little-endian UTF-16 encoding, commonly used in Windows
    'utf-16-be',  # Big-endian UTF-16 encoding
    'utf-32',  # 32-bit Unicode encoding
    'utf-32-le',  # Little-endian UTF-32 encoding
    'utf-32-be'  # Big-endian UTF-32 encoding
]
"""
Ordered list of encodings commonly found in Chinese Windows environments.

This list is used by :func:`auto_decode` to attempt decoding when the encoding
is unknown. The order is chosen to prioritize modern and frequently used
encodings.

:type: list[str]
"""


def _decode(data: bytes, encoding: str) -> str:
    """
    Decode bytes data using the specified encoding.

    :param data: The bytes data to decode.
    :type data: bytes
    :param encoding: The encoding to use for decoding.
    :type encoding: str
    :return: The decoded string.
    :rtype: str
    :raises UnicodeDecodeError: If the data cannot be decoded with the specified encoding.
    """
    return data.decode(encoding)


def auto_decode(data: Union[bytes, bytearray]) -> str:
    """
    Automatically decode bytes data by trying multiple encodings.

    This function attempts to decode the input data using multiple encodings in
    the following order:

    1. The encoding detected by :mod:`chardet` (for inputs >= 30 bytes)
    2. Common Chinese encodings used in Windows
    3. The default system encoding
    4. The encoding detected by :mod:`chardet` (for inputs < 30 bytes)

    The function tries each encoding until successful decoding is achieved. If
    all encodings fail, it raises the :class:`UnicodeDecodeError` that decoded
    the longest prefix before failing.

    :param data: The bytes data to decode.
    :type data: Union[bytes, bytearray]
    :return: The decoded string.
    :rtype: str
    :raises UnicodeDecodeError: If the data cannot be decoded with any of the attempted encodings.

    Example::

        >>> text_bytes = b'\\xc4\\xe3\\xba\\xc3'  # "你好" in GBK encoding
        >>> auto_decode(text_bytes)
        '你好'
    """
    if len(data) >= 30:
        _elist = list(filter(bool, unique([
            chardet.detect(data)['encoding'],
            *windows_chinese_encodings,
            sys.getdefaultencoding(),
        ])))
    else:
        _elist = list(filter(bool, unique([
            *windows_chinese_encodings,
            sys.getdefaultencoding(),
            chardet.detect(data)['encoding'],
        ])))

    last_err: Optional[UnicodeDecodeError] = None
    for enc in _elist:
        try:
            text = _decode(data, enc)
        except UnicodeDecodeError as err:
            if last_err is None or err.start > last_err.start:
                last_err = err
        else:
            return text

    raise last_err
