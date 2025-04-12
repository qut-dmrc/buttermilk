import contextlib
from typing import Any

import regex as re
from bs4 import BeautifulSoup
from readabilipy import simple_json_from_html_string

from buttermilk._core.types import MediaObj, Record
from buttermilk.utils.utils import (
    download_limited_async,
    is_filepath,
    is_uri,
    read_file,
)


async def download_and_convert(
    obj: bytes | Any | None = None,
    mime: str = "application/octet-stream",
    *,
    uri: str | None = None,
    b64: str | None = None,
    html: str | None = None,
    text: str | None = None,
    label: str | None = None,
    filepath: str | None = None,
    allow_arbitrarily_large_downloads: bool = False,
    max_size: int = 1024 * 1024 * 10,
    token: str | None = None,
    **kwargs: Any,
) -> Record:
    # If we have a URI, download it.
    # If it's a binary object, convert it to base64.
    # Try to guess the mime type from the extension if possible.

    if not obj and not uri and not b64 and not text:
        # Nothing passed in, return None
        return None
    # clean up kwargs to remove defaults
    kwargs = {k: v for k, v in kwargs.items() if v}

    with contextlib.suppress(Exception):
        obj = obj.decode("utf-8")

    if uri or is_uri(obj):
        uri = uri or obj
        obj, detected_mimetype = await download_limited_async(
            uri,
            allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads,
            token=token,
            max_size=max_size,
        )
        kwargs["uri"] = uri

        # Replace mimetype if default or none was passed in
        if detected_mimetype and (not mime or mime == "application/octet-stream"):
            mime = detected_mimetype
        with contextlib.suppress(Exception):
            obj = obj.decode("utf-8")

    elif filepath or is_filepath(obj):
        filepath = filepath or obj
        obj = read_file(filepath)
        kwargs["uri"] = filepath

    obj_list = []  # List of component media objects

    if html or mime.startswith("text/html"):
        # try to extract text from web page
        html = html or obj
        obj_list, retrieved_metadata = extract_main_content(
            html=html,
        )
        # Add additional metadata to the record
        kwargs.update(retrieved_metadata)
    elif b64:
        obj_list = [
            MediaObj(
                label=label,
                _base_64=b64,
                mime=mime,
            ),
        ]
    elif text or isinstance(obj, str):
        text = text or obj
        if not mime or mime == "application/octet-stream":
            mime = "text/plain"
        obj_list = [
            MediaObj(
                label=label,
                content=text,
                mime=mime,
            ),
        ]
    else:
        obj_list = [
            MediaObj(
                label=label,
                content=obj,
                mime=mime,
            ),
        ]

    record = Record(
        components=obj_list,
        **kwargs,
    )
    return record


def get_news_record_from_uri(uri: str) -> Record:
    import newspaper

    article = newspaper.article(uri)
    paragraphs = []
    chunks = []
    for para in article.text.splitlines():
        if para := para.strip():
            # Text exists, add it to current chunk
            chunks.append(para)
        # New paragraph
        elif chunks:
            para = "\n".join(chunks)
            paragraphs.append(
                MediaObj(content=para, label="paragraph", mime="text/plain"),
            )
            chunks = []
    if chunks:
        para = "\n".join(chunks)
        paragraphs.append(MediaObj(content=para, label="paragraph", mime="text/plain"))
        chunks = []

    metadata = dict(
        title=article.title,
        keywords=article.keywords,
        authors=article.authors,
        publish_date=article.publish_date,
    )

    record = Record(uri=uri, components=paragraphs, metadata=metadata)

    return record


def extract_main_content(html: str, **kwargs) -> tuple[list[MediaObj], dict]:
    doc = simple_json_from_html_string(html, use_readability=True)

    paragraphs = []
    chunks = []
    for span in doc.pop("plain_text", []):
        if chunk := span.get("text").strip():
            # Text exists, add it to current chunk
            chunks.append(chunk)
        # New paragraph
        elif chunks:
            para = "\n".join(chunks)
            paragraphs.append(
                MediaObj(content=para, label="paragraph", mime="text/plain"),
            )
            chunks = []
    if chunks:
        para = "\n".join(chunks)
        paragraphs.append(MediaObj(content=para, label="paragraph", mime="text/plain"))
        chunks = []

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
