import contextlib
from typing import Any

import regex as re
from bs4 import BeautifulSoup
from readabilipy import simple_json_from_html_string

from buttermilk._core.runner_types import MediaObj, RecordInfo
from buttermilk.utils.utils import download_limited_async, is_b64, is_uri


async def download_and_convert(
    obj: Any,
    label: str | None = None,
    mime: str = "application/octet-stream",
    allow_arbitrarily_large_downloads: bool = False,
    max_size: int = 1024 * 1024 * 10,
    token: str | None = None,
    alt_text: str | None = None,
    ground_truth: Any = None,
    metadata: dict = {},
    **kwargs: Any,
) -> RecordInfo:
    # If we have a URI, download it.
    # If it's a binary object, convert it to base64.
    # Try to guess the mime type from the extension if possible.

    uri = None

    if is_uri(obj):
        uri = obj
        obj, detected_mimetype = await download_limited_async(
            uri,
            allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads,
            token=token,
            max_size=max_size,
        )

        # Replace mimetype if default or none was passed in
        if detected_mimetype and (not mime or mime == "application/octet-stream"):
            mime = detected_mimetype

        obj_list = []  # List of component media objects

        with contextlib.suppress(Exception):
            obj = obj.decode("utf-8")

    if mime.startswith("text/html"):
        # try to extract text from web page
        obj_list, retrieved_metadata = extract_main_content(
            obj,
            metadata=metadata,
        )
        metadata.update(retrieved_metadata)

    else:
        b64 = None
        if is_b64(obj):
            b64 = obj
            obj = None
        obj_list = [
            MediaObj(
                label=label,
                content=obj,
                base_64=b64,
                mime=mime,
            ),
        ]

    return RecordInfo(
        data=obj_list,
        metadata=metadata,
        alt_text=alt_text,
        ground_truth=ground_truth,
    )


def extract_main_content(html: str, **kwargs) -> tuple[MediaObj, dict]:
    doc = simple_json_from_html_string(html, use_readability=True)

    paragraphs = [
        MediaObj(content=para["text"], label="paragraph", mime="text/plain")
        for para in doc.pop("plain_text")
    ]
    del doc["plain_content"]
    del doc["content"]
    doc.update(kwargs)
    doc = {k: v for k, v in doc.items() if v}

    return paragraphs, doc


def extract_main_content_bs(html: bytes | str) -> str:
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
