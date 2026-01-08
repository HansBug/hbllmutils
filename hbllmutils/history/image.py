"""
This module provides utilities for handling image blob URLs, including conversion between images and blob URLs,
and validation of blob URL format.

The module supports:

- Converting images to blob URLs with specified formats
- Loading images from blob URLs
- Validating image blob URL format
- Handling various image formats and MIME types
"""

import base64
from io import BytesIO

from PIL import Image

_FORMAT_REPLACE = {'JPG': 'JPEG'}


def to_blob_url(image: Image.Image, format: str = 'jpg', **save_kwargs) -> str:
    """
    Convert a PIL Image to a blob URL string.

    This function encodes an image into a base64 data URL that can be embedded directly in HTML or CSS.
    The image is saved to a buffer in the specified format, then base64-encoded and wrapped in a data URL.

    :param image: The PIL Image object to convert
    :type image: Image.Image
    :param format: The desired image format for the blob URL (e.g., 'jpg', 'png', 'webp'), defaults to 'jpg'
    :type format: str
    :param save_kwargs: Additional keyword arguments passed to PIL Image.save() method (e.g., quality, optimize)
    :type save_kwargs: dict
    :return: A blob URL string in the format 'data:{mime_type};base64,{encoded_data}'
    :rtype: str

    Example::
        >>> from PIL import Image
        >>> img = Image.open('test.jpg')
        >>> blob_url = to_blob_url(img, format='png', quality=95)
        >>> print(blob_url[:50])  # data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
        >>> # Use with higher quality JPEG
        >>> blob_url = to_blob_url(img, format='jpg', quality=95, optimize=True)
    """
    format = (_FORMAT_REPLACE.get(format.upper(), format)).upper()
    with BytesIO() as buffer:
        image.save(buffer, **{'format': format, **save_kwargs})
        buffer.seek(0)
        mime_type = Image.MIME.get(format.upper(), f'image/{format.lower()}')
        base64_str = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f"data:{mime_type};base64,{base64_str}"
