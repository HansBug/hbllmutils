import pathlib

import pytest
from PIL import Image
from hbutils.testing import tmatrix

from hbllmutils.history import to_blob_url
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataBlob:
    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'mostima_post.jpg',
            'soldiers.jpg',
        ],
        ('format', 'mimetype'): [
            ('jpg', 'image/jpeg'),
            ('jpeg', 'image/jpeg'),
            ('png', 'image/png'),
            ('webp', 'image/webp'),
        ]
    }, mode='matrix'))
    def test_to_blob_url_format_check(self, filename, format, mimetype):
        original_image = Image.open(get_testfile(filename)).convert('RGB')
        blob_url = to_blob_url(original_image, format=format)
        assert blob_url.startswith(f'data:{mimetype};base64,')
        assert pathlib.Path(get_testfile(f'{filename}-{format}.txt')).read_text().strip() == blob_url
