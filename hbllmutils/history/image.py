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
    Convert an image to a blob URL string.

    :param image: The input image, can be PIL Image, numpy array, or file path
    :type image: ImageTyping
    :param format: The desired image format for the blob URL, defaults to 'jpg'
    :type format: str
    :param save_kwargs: Additional keyword arguments passed to PIL Image.save()
    :return: A blob URL string containing the encoded image data
    :rtype: str

    :example:
        >>> img = Image.open('test.jpg')
        >>> blob_url = to_blob_url(img, format='png', quality=95)
        >>> print(blob_url)  # data:image/png;base64,...</pre>
    """
    format = (_FORMAT_REPLACE.get(format.upper(), format)).upper()
    with BytesIO() as buffer:
        image.save(buffer, **{'format': format, **save_kwargs})
        buffer.seek(0)
        mime_type = Image.MIME.get(format.upper(), f'image/{format.lower()}')
        base64_str = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f"data:{mime_type};base64,{base64_str}"
