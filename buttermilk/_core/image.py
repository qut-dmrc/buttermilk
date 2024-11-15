import base64
from io import BytesIO
from pathlib import Path

from cloudpathlib import AnyPath, CloudPath
from PIL import Image
from pydantic import BaseModel, model_validator

from buttermilk.utils.utils import read_file


# Create a new class to contain the PIL Image, a URI, and base64 encoding
class ImageRecord(BaseModel):
    image: Image.Image | None = None
    uri: str | None = None
    image_b64: str | None = None
    _image_bytes: bytes | None = None
    description: str | None = None
    alt_text: str | None = None
    source_uri: str | None = None
    model: str | None = None
    params: dict | None = None
    error: str | dict | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    @classmethod
    def check_fields(cls, obj):
        if obj.image is None:
            if obj.error is not None:
                # add a sad robot
                obj.image = Image.open("datatools/resources/sad_robot.png")
            else:
                raise ValueError(
                    "Image is required, unless an error has occured and the 'error' field is set.",
                )

        if isinstance(obj.error, str):
            obj.error = {"message": obj.error}

        return obj

    def as_url(self):
        return (f"data:image/png;base64,{self.as_base64()}",)

    def as_bytestream(self):
        output = BytesIO()
        self.image.save(output, format="png")
        output.seek(0)
        return output

    def as_base64(self, longest_edge=None, shortest_edge=None):
        return image_to_b64(
            self.image,
            longest_edge=longest_edge,
            shortest_edge=shortest_edge,
        )

    def as_bytes(self):
        output = self.as_bytestream()
        return output.read()

    def save(self, path: str | CloudPath | Path) -> str:
        if isinstance(path, str):
            path = AnyPath(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            self.image.save(f, format="png")

        return path.as_uri()


def read_image(
    path=None,
    data=None,
    image_b64=None,
    shortest_edge=768,
    longest_edge=None,
    token=None,
) -> ImageRecord:
    """Get image data from either a URI or a file upload."""
    if path:
        data = read_file(path, auth_token=token)
    elif image_b64:
        data = base64.b64decode(image_b64)

    if not data:
        raise ValueError("No valid image filename or binary data provided.")

    data = BytesIO(data)

    img = Image.open(data)

    # Check the image is valid
    img.verify()

    # reopen
    img = Image.open(data)
    image_b64 = image_to_b64(
        img,
        longest_edge=longest_edge,
        shortest_edge=shortest_edge,
    )
    data.seek(0)  # go to the start of the image again
    img = ImageRecord(
        image=img,
        uri=path,
        image_b64=image_b64,
        _image_bytes=data.read(),
    )

    return img


def image_to_b64(img: Image, longest_edge=-1, shortest_edge=-1) -> str:
    if longest_edge and longest_edge > 0:
        # resize the image to max 1024 on the longest edge
        if img.width > longest_edge or img.height > longest_edge:
            if img.width > img.height:
                new_width = longest_edge
                new_height = int(longest_edge * img.height / img.width)
            else:
                new_height = longest_edge
                new_width = int(longest_edge * img.width / img.height)
            img = img.resize((new_width, new_height))
    elif shortest_edge and shortest_edge > 0:
        # resize the image to max 1024 on the shortest edge
        if img.width > shortest_edge or img.height > shortest_edge:
            if img.width > img.height:
                new_height = shortest_edge
                new_width = int(shortest_edge * img.width / img.height)
            else:
                new_width = shortest_edge
                new_height = int(shortest_edge * img.height / img.width)
            img = img.resize((new_width, new_height))

    # Convert image to base64 string
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return image_b64


def image_to_byte_array(image: Image.Image) -> bytes:
    imgByteArr = BytesIO()

    # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr
