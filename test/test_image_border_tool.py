from pathlib import Path

from PIL import Image
import pytest

from image_border_tool import BorderSize, detect_border


def _test_data(path: str) -> Path:
    return Path(__file__).parent / 'data' / path


@pytest.mark.parametrize('file', ['no_border_1.jpg', 'no_border_2.png'])
def test_detect_border_no_border(file: str) -> None:
    img = Image.open(_test_data(file))
    border = detect_border(img)
    assert border == BorderSize(0, 0, 0, 0)


def test_detect_border_white_border() -> None:
    img = Image.open(_test_data('border_white.jpg'))
    border = detect_border(img)
    assert border == BorderSize(top=149, bottom=151, left=274, right=274)


def test_detect_border_black_border() -> None:
    img = Image.open(_test_data('border_black.jpg'))
    border = detect_border(img)
    assert border == BorderSize(top=149, bottom=151, left=349, right=351)
