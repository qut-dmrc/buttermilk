
from buttermilk._core.runner_types import RecordInfo
from buttermilk.data.toxic import ImplicitHatePipe, DragQueens


def test_implicit_hate():
    pipe = ImplicitHatePipe()
    for example in pipe:
        assert example
        assert isinstance(example, RecordInfo)
        assert len(example.text) > 10
        break
    pass

def test_drag_images():
    pipe = DragQueens()
    for example in pipe:
        assert example
        assert isinstance(example, RecordInfo)
        assert str.lower(example.source) == "drag queens and white supremacists"
        assert example.text

        image = read_image(example.path)
        assert isinstance(image, ImageRecord)
        assert len(image.as_base64()) > 1024
        break
    pass
