"""Utilities for downloading, converting, and extracting content from various media types.

This module provides functions to:
- Download content from URIs (web URLs, file paths, cloud storage).
- Convert different input types (bytes, base64, HTML, text) into a standardized
  `Record` object.
- Extract article content from news web pages using the `newspaper` library.
- Extract main textual content from HTML using `readabilipy` or `BeautifulSoup`.
"""

import contextlib  # For suppressing exceptions in specific blocks
from typing import Any

import regex as re  # For regular expression operations
from bs4 import BeautifulSoup  # For HTML parsing
from readabilipy import simple_json_from_html_string  # For extracting main content from HTML

from buttermilk._core.image import read_image  # Utility for reading image data
from buttermilk._core.log import logger  # Centralized logger
from buttermilk._core.types import MediaObj, Record  # Buttermilk core data types
from buttermilk.utils.utils import (  # General utilities
    download_limited_async,  # Async download with size limit
    is_filepath,  # Check if string is a file path
    is_uri,  # Check if string is a URI
    read_file,  # Read local file content
)


async def download_and_convert(
    obj: bytes | str | Any | None = None,  # Changed: obj can also be str if it's a URI/filepath
    mime: str = "application/octet-stream",
    *,  # Keyword-only arguments follow
    uri: str | None = None,
    b64: str | None = None,
    html: str | None = None,
    text: str | None = None,
    label: str | None = None,  # Added label to docstring, though not directly used to create Record
    filepath: str | None = None,
    allow_arbitrarily_large_downloads: bool = False,
    max_size: int = 1024 * 1024 * 10,  # 10 MB default max size
    token: str | None = None,  # Auth token for downloads
    **kwargs: Any,  # Additional metadata for the Record
) -> Record | None:
    """Downloads content from a URI or processes provided data, converting it into a `Record`.

    This versatile function handles various input types:
    - If `uri` is provided (or `obj` is a URI string), it downloads the content.
    - If `filepath` is provided (or `obj` is a filepath string), it reads the local file.
    - If `html` string is provided, it extracts main content using `extract_main_content`.
    - If `b64` (base64 string) is provided, it's processed as an image.
    - If `text` string is provided, it's used as direct text content.
    - If `obj` is bytes, it's processed as an image.
    - If `obj` is a string (not URI/filepath), it's treated as text.

    It attempts to detect MIME types and populates a `Record` object with the
    processed content and metadata.

    Args:
        obj (bytes | str | Any | None): The primary input data. Can be raw bytes,
            a string (URI, filepath, HTML, plain text, or base64 if `b64` is not set),
            or None if other specific arguments like `uri` or `b64` are used.
        mime (str): The MIME type of the input `obj` if known, or the desired
            MIME type. Defaults to "application/octet-stream". This might be
            overridden if a more specific type is detected (e.g., from URI download).
        uri (str | None): A URI (URL, GCS path, etc.) to download content from.
            Takes precedence if `obj` is also URI-like.
        b64 (str | None): A base64 encoded string, typically for an image.
        html (str | None): Raw HTML content as a string.
        text (str | None): Plain text content as a string.
        label (str | None): An optional label for the content (not directly used in
            Record creation by this function but might be contextually relevant).
        filepath (str | None): A local file path to read content from.
        allow_arbitrarily_large_downloads (bool): If True, bypasses `max_size`
            check for downloads. Defaults to False.
        max_size (int): Maximum size in bytes for downloads. Defaults to 10MB.
        token (str | None): Optional authentication token for downloading from URIs
            that require authorization.
        **kwargs (Any): Additional keyword arguments are collected into the
            `metadata` dictionary of the resulting `Record`.

    Returns:
        Record | None: A `Record` object containing the processed content, metadata,
        detected MIME type, and source URI (if applicable). Returns `None` if no
        input data (`obj`, `uri`, `b64`, `text`, `html`, `filepath`) is provided.

    Raises:
        Various exceptions from underlying download or file reading functions if
        they fail (e.g., network errors, file not found).

    """
    if not obj and not uri and not b64 and not text and not html and not filepath:
        logger.debug("download_and_convert: No input data provided (obj, uri, b64, text, html, or filepath). Returning None.")
        return None

    # Initialize metadata from kwargs, filtering out None values.
    # `label` is not directly used in Record creation here but is part of kwargs.
    active_metadata = {k: v for k, v in kwargs.items() if v is not None}
    active_mime = mime  # Use provided mime as default

    # Attempt to decode obj if it's bytes (e.g. from a file read or download)
    # This is a general attempt; specific handling below might re-process `obj`.
    if isinstance(obj, bytes):
        with contextlib.suppress(Exception):  # Suppress decoding errors if it's not valid UTF-8
            obj = obj.decode("utf-8")

    # Handle URI or URI-like string in obj
    if uri or (isinstance(obj, str) and is_uri(obj)):
        final_uri = uri or str(obj)  # Prioritize explicit uri argument
        logger.debug(f"download_and_convert: Processing URI: {final_uri}")
        downloaded_content, detected_mimetype = await download_limited_async(
            final_uri,
            allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads,
            token=token,
            max_size=max_size,
        )
        # Update active_mime if a more specific one was detected and current is generic
        if detected_mimetype and (not active_mime or active_mime == "application/octet-stream"):
            active_mime = detected_mimetype

        obj = downloaded_content  # The downloaded content becomes the new obj to process
        # Attempt to decode if it's bytes (common for downloads)
        if isinstance(obj, bytes):
            with contextlib.suppress(Exception):
                obj = obj.decode("utf-8", errors="replace")  # Replace errors to avoid crash
        uri = final_uri  # Ensure uri field in Record is set to what was downloaded

    # Handle filepath or filepath-like string in obj
    elif filepath or (isinstance(obj, str) and is_filepath(obj)):
        final_filepath = filepath or str(obj)
        logger.debug(f"download_and_convert: Processing filepath: {final_filepath}")
        obj = read_file(final_filepath)  # obj becomes file content (bytes or str)
        uri = Path(final_filepath).as_uri()  # Store original filepath as URI
        # Try to guess MIME from filepath extension if current is generic
        import mimetypes
        guessed_type, _ = mimetypes.guess_type(final_filepath)
        if guessed_type and (not active_mime or active_mime == "application/octet-stream"):
            active_mime = guessed_type

    # Determine final content and process based on type/mime
    final_content: Any
    if html or (isinstance(obj, str) and (active_mime and active_mime.startswith("text/html"))):
        logger.debug("download_and_convert: Processing as HTML content.")
        html_content_to_parse = html or str(obj)
        # Extract main content and metadata from HTML
        doc_metadata = extract_main_content(html=html_content_to_parse)
        "\n".join(doc_metadata.pop("paragraphs", []))
        # Add additional metadata to the record
        metadata.update({k: doc_metadata.get(k) for k in ["title", "keywords", "byline", "authors", "date", "publish_date"] if doc_metadata.get(k)})

        final_content = doc_metadata.pop("plain_text", "") if isinstance(doc_metadata, dict) else ""
        if isinstance(doc_metadata, dict): active_metadata.update(doc_metadata)  # Add extracted HTML metadata
        if not active_mime or active_mime == "application/octet-stream":  # Ensure mime is text/html
            active_mime = "text/html"

    elif b64:  # Explicit base64 input
        logger.debug("download_and_convert: Processing as explicit base64 image.")
        # read_image handles base64 and returns an ImageRecord (which is a Record)
        # We want to create a new Record, so extract info from ImageRecord if needed.
        image_rec_from_b64 = read_image(data=b64)  # Assuming read_image can take b64 string
        final_content = image_rec_from_b64.content  # This might be complex (list of MediaObj)
        if image_rec_from_b64.mime: active_mime = image_rec_from_b64.mime
        active_metadata.update(image_rec_from_b64.metadata)

    elif text or (isinstance(obj, str) and (not uri or active_mime.startswith("text/"))):  # Explicit text or obj is string and looks like text
        logger.debug("download_and_convert: Processing as plain text.")
        final_content = text or str(obj)
        if not active_mime or active_mime == "application/octet-stream":
            active_mime = "text/plain"

    elif isinstance(obj, bytes) or (isinstance(obj, str) and is_b64(obj)):  # Raw bytes or obj is a b64 string (implicit image)
        logger.debug(f"download_and_convert: Processing as implicit image/binary data (type: {type(obj)}).")
        # read_image handles bytes or base64 string and returns an ImageRecord
        image_rec_from_obj = read_image(data=obj)
        final_content = image_rec_from_obj.content
        if image_rec_from_obj.mime: active_mime = image_rec_from_obj.mime
        active_metadata.update(image_rec_from_obj.metadata)

    elif obj is not None:  # Fallback for other types of obj not caught above
        logger.debug(f"download_and_convert: Processing obj of type {type(obj)} as generic content.")
        final_content = str(obj)  # Convert to string as a last resort
        if not active_mime or active_mime == "application/octet-stream":
             active_mime = "text/plain"  # Assume text if unknown
    else:  # Should have been caught by the initial check, but as a safeguard
        logger.error("download_and_convert: Reached final content determination with no valid data.")
        return None

    return Record(content=final_content, metadata=active_metadata, uri=uri, mime=active_mime)


def get_news_record_from_uri(uri: str) -> Record:
    """Extracts article content from a news web page URI using the `newspaper3k` library.

    Downloads the article from the given URI, parses it to extract title, authors,
    publication date, keywords, and main text. The main text is split into
    paragraphs and stored as `MediaObj` components within the returned `Record`.

    Args:
        uri (str): The URL of the news article to process.

    Returns:
        Record: A `Record` object populated with the extracted article information.
                The `record.content` will be a list of `MediaObj` instances,
                each representing a paragraph of the article text.

    Raises:
        newspaper.article.ArticleException: If `newspaper3k` fails to download or parse the article.
        ImportError: If `newspaper3k` library is not installed.

    """
    try:
        import newspaper  # Dynamically import to keep as optional dependency
    except ImportError as e:
        logger.error("The 'newspaper3k' library is required for get_news_record_from_uri. Please install it.")
        raise ImportError("newspaper3k library not found. Please install with `pip install newspaper3k`.") from e

    article = newspaper.Article(uri)  # Initialize Article object
    article.download()  # Download HTML content
    article.parse()    # Parse content to extract elements

    media_components: list[MediaObj] = []
    current_paragraph_chunks: list[str] = []

    # Process extracted text, splitting into paragraphs
    # article.text often contains newline characters separating paragraphs.
    for line in (article.text or "").splitlines():
        stripped_line = line.strip()
        if stripped_line:  # If line has content, add to current paragraph chunk
            current_paragraph_chunks.append(stripped_line)
        elif current_paragraph_chunks:  # Empty line signifies end of a paragraph
            paragraph_text = "\n".join(current_paragraph_chunks)  # Join chunks with newlines for multi-line para
            media_components.append(
                MediaObj(content=paragraph_text, label="paragraph", mime="text/plain"),
            )
            current_paragraph_chunks = []  # Reset for next paragraph

    # Add any remaining text as the last paragraph
    if current_paragraph_chunks:
        paragraph_text = "\n".join(current_paragraph_chunks)
        media_components.append(
            MediaObj(content=paragraph_text, label="paragraph", mime="text/plain"),
        )

    # Populate metadata
    record_metadata = {
        "title": article.title or "N/A",
        "keywords": article.keywords or [],
        "authors": article.authors or [],
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
        "source_type": "newspaper3k_article",
    }
    # Filter out None metadata values for cleanliness
    record_metadata = {k: v for k, v in record_metadata.items() if v is not None}

    # Create Record with MediaObj components for content
    return Record(uri=uri, content=media_components, metadata=record_metadata, mime="multipart/mixed")  # mime for list of media


def extract_main_content(html: str, **kwargs: Any) -> dict[str, Any]:
    """Extracts the main textual content and some metadata from an HTML string.

    Uses the `readabilipy` library (which itself often uses Mozilla's Readability.js)
    to identify and extract the primary article content from a potentially cluttered
    HTML page. It also attempts to gather paragraphs.

    Args:
        html (str): The HTML content as a string.
        **kwargs: Additional keyword arguments passed to
                  `readabilipy.simple_json_from_html_string` (e.g., `use_readability`).

    Returns:
        dict[str, Any]: A dictionary containing extracted information. Key fields include:
            - "title" (str): The main title of the page.
            - "byline" (str): Author information, if found.
            - "plain_text" (str): The extracted main content as plain text.
            - "plain_html" (str): The extracted main content with simplified HTML.
            - "paragraphs" (list[str]): A list of strings, where each string is a
              paragraph from the extracted plain text.
            Other fields from `readabilipy` might also be present.

    """
    # Ensure use_readability is True by default if not specified, as it's key for main content
    kwargs.setdefault("use_readability", True)
    doc_parts = simple_json_from_html_string(html, **kwargs)

    # Extract paragraphs from the 'plain_text' field provided by readabilipy
    # readabilipy's plain_text is a list of dicts like [{"text": "paragraph1"}, {"text": "paragraph2"}]
    extracted_paragraphs: list[str] = []
    current_paragraph_buffer: list[str] = []
    if "plain_text" in doc_parts and isinstance(doc_parts["plain_text"], list):
        for text_span in doc_parts["plain_text"]:
            if isinstance(text_span, dict) and "text" in text_span:
                chunk = text_span.get("text", "").strip()
                if chunk:  # Add non-empty text to current paragraph buffer
                    current_paragraph_buffer.append(chunk)
                elif current_paragraph_buffer:  # Empty chunk signifies paragraph break if buffer has content
                    extracted_paragraphs.append(" ".join(current_paragraph_buffer))
                    current_paragraph_buffer = []  # Reset buffer

    # Add any remaining text in the buffer as the last paragraph
    if current_paragraph_buffer:
        extracted_paragraphs.append(" ".join(current_paragraph_buffer))

    doc_parts["paragraphs"] = extracted_paragraphs  # Add the processed paragraphs list to the output

    return doc_parts


def extract_main_content_bs(html: bytes | str) -> str:
    """Extract and clean main content from webpage."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup.find_all(
        [
            "script",
            "style",
            "header",
            "footer",
            "nav",
            "iframe",
            "aside",
        ],
    ):
        element_type.decompose()  # Remove the element and its content

    # Attempt to find the main content container using common tags and attributes
    main_content_element: Any = None  # Can be Tag or NavigableString
    # Ordered list of selectors to try for finding the main content
    potential_main_selectors = [
        # More specific selectors first
        {"id": re.compile(r".*(content|main|article).*(text|body|main).*", re.IGNORECASE)},
        "main",
        "article",
        {"id": re.compile(r".*(content|main|article).*", re.IGNORECASE)},
        {"class_": re.compile(r".*(content|main|article).*", re.IGNORECASE)},  # class_ for BeautifulSoup
    ]

    for selector in potential_main_selectors:
        if isinstance(selector, str):  # Tag name selector
            main_content_element = soup.find(selector)
        elif isinstance(selector, dict):  # Attribute selector (id or class)
            main_content_element = soup.find(**selector)

        if main_content_element:  # Found a candidate
            break

    # Fallback to using the whole body if no specific main content element is found
    if not main_content_element:
        main_content_element = soup.body or soup  # Use soup itself if body is also missing

    if not main_content_element:  # Should not happen if HTML is valid, but safeguard
        return ""

    # Extract text, joining lines and cleaning whitespace
    # get_text(separator=" ") joins text nodes with a space, then splitlines handles various newline types
    text_lines = [line.strip() for line in main_content_element.get_text(separator=" ").splitlines() if line.strip()]
    cleaned_text = " ".join(text_lines)

    # Further reduce multiple spaces to single spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text
