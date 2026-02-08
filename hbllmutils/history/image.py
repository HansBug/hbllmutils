"""
Image blob URL utilities.

This module provides helper functions for converting :class:`PIL.Image.Image`
objects to ``data:`` blob URLs that can be embedded directly in HTML or CSS.
It focuses on safe in-memory conversion with base64 encoding and simple MIME
type handling.

The module contains the following main components:

* :func:`to_blob_url` - Convert a PIL image into a base64-encoded data URL

Example::

    >>> from PIL import Image
    >>> from hbllmutils.history.image import to_blob_url
    >>> img = Image.new("RGB", (2, 2), color="red")
    >>> url = to_blob_url(img, format="png")
    >>> url.startswith("data:image/png;base64,")
    True

.. note::
   The returned blob URL string can be large for high-resolution images.
   Consider optimizing or resizing images before conversion for bandwidth
   efficiency.
"""

import base64
from io import BytesIO
from typing import Any

from PIL import Image

_FORMAT_REPLACE = {'JPG': 'JPEG'}


def to_blob_url(image: Image.Image, format: str = 'jpg', **save_kwargs: Any) -> str:
    """
    Convert a PIL Image to a blob URL string.

    This function encodes an image into a base64 data URL that can be embedded
    directly in HTML or CSS. The image is saved to an in-memory buffer in the
    specified format, then base64-encoded and wrapped in a data URL.

    :param image: The PIL Image object to convert.
    :type image: PIL.Image.Image
    :param format: The desired image format for the blob URL (e.g., ``"jpg"``,
                   ``"png"``, ``"webp"``), defaults to ``"jpg"``.
    :type format: str
    :param save_kwargs: Additional keyword arguments passed to
                        :meth:`PIL.Image.Image.save` (e.g., ``quality``,
                        ``optimize``).
    :type save_kwargs: Any
    :return: A blob URL string in the format
             ``"data:{mime_type};base64,{encoded_data}"``.
    :rtype: str

    Example::

        >>> from PIL import Image
        >>> img = Image.new("RGB", (1, 1), color="white")
        >>> blob_url = to_blob_url(img, format="png")
        >>> blob_url.startswith("data:image/png;base64,")
        True
        >>> # Use higher quality JPEG
        >>> blob_url = to_blob_url(img, format="jpg", quality=95, optimize=True)

    """
    format = (_FORMAT_REPLACE.get(format.upper(), format)).upper()
    with BytesIO() as buffer:
        image.save(buffer, **{'format': format, **save_kwargs})
        buffer.seek(0)
        mime_type = Image.MIME.get(format.upper(), f'image/{format.lower()}')
        base64_str = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f"data:{mime_type};base64,{base64_str}"
