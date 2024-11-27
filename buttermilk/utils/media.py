import base64
from typing import Any

import regex as re
from bs4 import BeautifulSoup
from pydantic import validate_call

from buttermilk._core.runner_types import MediaObj
from buttermilk.utils.utils import download_limited_async, is_b64, is_uri


@validate_call
async def download_and_convert(obj: Any, mimetype: str | None = None) -> MediaObj:
    # If we have a URI, download it.
    # If it's a webpage, extract the text.
    # If it's a binary object, convert it to base64.
    # Try to guess the mime type from the extension if possible.
    mimetype = mimetype or "application/octet-stream"
    if is_b64(obj):
        return MediaObj(base_64=obj, mime=mimetype)

    if is_uri(obj):
        obj, detected_mimetype = await download_limited_async(obj)
        if not mimetype or mimetype == "application/octet-stream":
            mimetype = detected_mimetype

    if mimetype.startswith("text/html"):
        # try to extract text from object
        obj = extract_main_content(obj)
        mimetype = "text/plain"

    if mimetype.startswith("text/"):
        return MediaObj(text=obj, mime=mimetype)

    # By default, encode the object as base64
    return MediaObj(mime=mimetype, base_64=base64.b64encode(obj).decode("utf-8"))


def extract_main_content(html: bytes | str) -> str:
    """Extract and clean main content from webpage."""
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
