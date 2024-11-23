import base64

from bs4 import BeautifulSoup
from cloudpathlib import AnyPath
from pydantic import AnyUrl, validate_call

from buttermilk._core.runner_types import MediaObj
from buttermilk.utils.utils import download_limited_async


@validate_call
async def validate_uri_extract_text(value: AnyUrl | str | None) -> str | None:
    if value:
        try:
            _ = AnyUrl(value)
            assert _.scheme and _.host
        except:
            return value

        # It's a URL, go fetch
        obj, mimetype = await download_limited_async(value)

        # try to extract text from object
        if mimetype.startswith("text/html"):
            soup = BeautifulSoup(obj, "html.parser")
            value = soup.get_text()
        else:
            value = obj.decode()
    return value


def is_b64(value: str) -> bool:
    # Check if the string is a valid base64-encoded string
    try:
        base64.b64decode(value, validate=True)
        return True
    except:
        return False


@validate_call
async def validate_uri_or_b64(value: AnyUrl | str | None) -> MediaObj | None:
    if value:
        if is_b64(value):
            return MediaObj(base_64=value)

        try:
            if isinstance(value, AnyUrl) or AnyUrl(value):
                # It's a URL, go fetch and encode it
                obj = await download_limited_async(value)
                value = base64.b64encode(obj).decode("utf-8")
                return MediaObj(base_64=value)
        except Exception:
            raise ValueError("Invalid URI or base64-encoded string")
    return None


async def download_and_convert(input_path: str | AnyPath):
    path = AnyPath(input_path) if not isinstance(input_path, AnyPath) else input_path

    extension = path.suffix

    # Guess mime type from extension
    # ...
    mimetype = ""

    if isinstance(path, AnyUrl) or AnyUrl(path):
        # It's a URL, go fetch and encode it
        obj = await download_limited_async(path)
        value = base64.b64encode(obj).decode("utf-8")
        return MediaObj(base_64=value, mime=mimetype)
    # TODO: finish this.
