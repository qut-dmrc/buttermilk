import base64

import regex as re
from bs4 import BeautifulSoup
from cloudpathlib import AnyPath
from pydantic import AnyUrl, validate_call

from buttermilk._core.runner_types import MediaObj
from buttermilk.utils.utils import download_limited_async


def extract_main_content(html: bytes | str) -> str:
    """Extract and clean main content from webpage"""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup.find_all([
        "script",
        "style",
        "header",
        "footer",
        "nav",
        "iframe",
        "aside",
    ]):
        element.decompose()

    # Try to find main content
    # this is an ordered list, stop when we find a match.
    main_content = None
    content_elements = [
        soup.find(id=re.compile(r".*(content|main|article).*(text|body|main).*", re.I)),
        soup.find("main"),
        soup.find("article"),
        soup.find(id=re.compile(r".*(content|main|article).*", re.I)),
        soup.find(class_=re.compile(r".*(content|main|article).*", re.I)),
    ]

    for element in content_elements:
        if element:
            main_content = element
            break

    # Fallback to body if no main content found
    if not main_content:
        main_content = soup.body or soup

    # Extract and clean text
    text = " ".join(
        line.strip()
        for line in main_content.get_text(separator=" ").splitlines()
        if line.strip()
    )

    # Clean extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
            return extract_main_content(obj)
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
