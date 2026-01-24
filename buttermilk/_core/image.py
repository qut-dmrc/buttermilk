import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from cloudpathlib import AnyPath, CloudPath
from google import genai
from PIL import Image
from pydantic import BaseModel, model_validator

from buttermilk._core.types import Record
from buttermilk.utils.utils import read_file


# Create a new class to contain the PIL Image, a URI, and base64 encoding
class ImageRecord(BaseModel):
    image: Image.Image | None = None
    uri: str | None = None
    image_b64: str | None = None
    _image_bytes: bytes | None = None
    description: str | None = None
    alt_text: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    enhanced_prompt: str | None = None
    source_uri: str | None = None
    model: str | None = None
    parameters: dict | None = None
    error: str | dict | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @staticmethod
    def _is_refusal_error(error: str | dict | None) -> bool:
        """Determine if an error is a content policy refusal.

        A refusal error is when the content was rejected for policy reasons,
        not when there was a technical failure (network, timeout, invalid credentials, etc.).

        Args:
            error: Error information (either string message or dict with error details)

        Returns:
            True if this is a content policy refusal, False otherwise
        """
        if error is None:
            return False

        error_str = error if isinstance(error, str) else (
            error.get("message", "") if isinstance(error, dict) else str(error)
        )

        # Normalize to lowercase for case-insensitive matching
        error_lower = error_str.lower()

        # Common refusal keywords from various APIs
        refusal_keywords = [
            "refus",  # "refused", "refusal", "refuses"
            "content policy",
            "not allowed",
            "not permitted",
            "violates",
            "inappropriate",
            "blocked",  # Content was blocked by policy
            "cannot generate",  # Usually due to policy
            "unable to generate",  # Usually due to policy
            "request was refused",
            "prompt violates",
            "unsafe",  # When it's about policy violation
            "not suitable",
            "prohibited",
        ]

        return any(keyword in error_lower for keyword in refusal_keywords)

    @model_validator(mode="after")
    @classmethod
    def check_fields(cls, obj):
        if obj.image is None:
            if obj.error is not None:
                # Only add sad robot for content policy refusals, not general API errors
                if cls._is_refusal_error(obj.error):
                    obj.image = Image.open("tests/data/sadrobot.jpg")
                else:
                    raise ValueError(
                        "Image generation failed due to a technical error. "
                        f"Error details: {obj.error}"
                    )
            else:
                raise ValueError(
                    "Image is required, unless an error has occured and the 'error' field is set.",
                )
        if isinstance(obj.image, genai.types.Image):
            # convert to PIL Image
            obj.image = google_genai_image_to_pil(obj.image)
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
    *,
    path=None,
    data=None,
    image_b64=None,
    shortest_edge=768,
    longest_edge=None,
    token=None,
) -> Record:
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
    data.seek(0)  # go to the start of the image again
    rec = Record(content=[img], metadata={"uri": path})

    return rec


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


def google_genai_image_to_pil(gimg: Any) -> Image.Image:
    """
    Convert google.genai.types.Image to a PIL Image.
    """

    data = BytesIO(gimg.image_bytes)
    try:
        img = Image.open(data)
        img.load()  # force read before closing buffer
        return img
    finally:
        data.close()
